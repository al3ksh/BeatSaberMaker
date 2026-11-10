"""Data-driven calibration of the playability thresholds."""
from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bsmaker.formats.events import Event, to_events
from bsmaker.formats.reader import read_map
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

PASS_HARD_RATE = 0.01
TARGET_PASS_SHARE = 0.95
MAX_ROUNDS = 6
HARD_RULE_NAMES = ("rare_token", "hand_speed", "vision_block", "same_cell", "crossover",
                   "handclap", "parity")
_ANGLE_STEP, _ANGLE_CAP = 22.5, 90.0
_SLOW_STEP, _SLOW_FLOOR = 0.05, 0.15
_VISION_STEP, _VISION_FLOOR = 0.03, 0.03


@dataclass(frozen=True)
class CalibrationResult:
    n_files: int
    pass_share: float
    rule_note_share: dict[str, float]
    hand_gap_p01_s: float
    rare_tokens: tuple[frozenset[int], frozenset[int]]
    config: PlayabilityConfig


def _hand_gaps(events: Sequence[Event]) -> list[float]:
    gaps: list[float] = []
    last: list[float | None] = [None, None]
    for e in events:
        for hand, token in ((0, e.left), (1, e.right)):
            if token == NONE_TOKEN:
                continue
            if last[hand] is not None:
                gaps.append(e.time_s - last[hand])
            last[hand] = e.time_s
    return gaps


def measure(event_lists: Sequence[Sequence[Event]], cfg: PlayabilityConfig) -> CalibrationResult:
    """Check every list with `cfg`; pass = hard_rate <= 0.01."""
    passed = 0
    notes = 0
    counts: Counter[str] = Counter()
    gaps: list[float] = []
    for events in event_lists:
        report = check(events, cfg)
        passed += report.hard_rate <= PASS_HARD_RATE
        notes += report.n_notes
        counts.update(v.rule for v in report.hard)
        gaps.extend(_hand_gaps(events))
    n = len(event_lists)
    share = {r: counts.get(r, 0) / max(notes, 1) for r in HARD_RULE_NAMES}
    p01 = float(np.percentile(gaps, 1)) if gaps else 0.0
    return CalibrationResult(n, passed / n if n else 0.0, share, p01, cfg.rare_tokens, cfg)


def rare_tokens_from(
    event_lists: Sequence[Sequence[Event]], min_freq: float = 1e-4
) -> tuple[frozenset[int], frozenset[int]]:
    """Per hand, tokens whose frequency among that hand's notes is below `min_freq` (or zero)."""
    counts: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
    for events in event_lists:
        for e in events:
            if e.left != NONE_TOKEN:
                counts[0][e.left] += 1
            if e.right != NONE_TOKEN:
                counts[1][e.right] += 1
    out = []
    for c in counts:
        total = sum(c.values())
        out.append(frozenset(t for t in range(1, VOCAB_SIZE)
                             if total == 0 or c.get(t, 0) / total < min_freq))
    return out[0], out[1]


def _relax(rule: str, cfg: PlayabilityConfig, p01: float) -> tuple[PlayabilityConfig, str] | None:
    """One relaxation step for `rule`, or None when it is never relaxed or has no step left."""
    if rule == "parity":
        if cfg.strict_angle_deg < _ANGLE_CAP:
            v = min(_ANGLE_CAP, cfg.strict_angle_deg + _ANGLE_STEP)
            return replace(cfg, strict_angle_deg=v), f"parity: strict_angle_deg -> {v}"
        if cfg.slow_gap_s > _SLOW_FLOOR + 1e-9:
            v = round(max(_SLOW_FLOOR, cfg.slow_gap_s - _SLOW_STEP), 4)
            return replace(cfg, slow_gap_s=v), f"parity: slow_gap_s -> {v}"
    elif rule == "hand_speed":
        if p01 < cfg.min_hand_gap_s:
            return (replace(cfg, min_hand_gap_s=p01),
                    f"hand_speed: min_hand_gap_s -> {p01:.4f} (1st percentile of hand gaps)")
        if cfg.min_hand_gap_per_cell_s != 0:
            return (replace(cfg, min_hand_gap_per_cell_s=0.0),
                    "hand_speed: min_hand_gap_per_cell_s -> 0.0")
    elif rule == "crossover":
        if cfg.forbid_crossover:
            return replace(cfg, forbid_crossover=False), "crossover: forbid_crossover -> False"
    elif rule == "vision_block":
        if cfg.vision_gap_s > _VISION_FLOOR + 1e-9:
            v = round(max(_VISION_FLOOR, cfg.vision_gap_s - _VISION_STEP), 4)
            return replace(cfg, vision_gap_s=v), f"vision_block: vision_gap_s -> {v}"
    return None


def calibrate(
    event_lists: Sequence[Sequence[Event]],
    start: PlayabilityConfig,
    max_rounds: int = MAX_ROUNDS,
    progress: Callable[[str], None] | None = None,
) -> tuple[CalibrationResult, list[str]]:
    """Set rare tokens, then relax one rule by one step per round until pass_share >= 0.95."""
    say = progress or (lambda _msg: None)
    cfg = replace(start, rare_tokens=rare_tokens_from(event_lists))
    result = measure(event_lists, cfg)
    say(f"round 0 (rare tokens): pass_share={result.pass_share:.4f}")
    steps: list[str] = []
    for rnd in range(1, max_rounds + 1):
        if result.pass_share >= TARGET_PASS_SHARE:
            break
        ranked = sorted((r for r in HARD_RULE_NAMES if result.rule_note_share[r] > 0),
                        key=lambda r: -result.rule_note_share[r])
        for rule in ranked:
            step = _relax(rule, result.config, result.hand_gap_p01_s)
            if step is not None:
                break
        else:
            say(f"round {rnd}: no rule has a relaxation step left, stopping")
            break
        cfg, text = step
        steps.append(text)
        result = measure(event_lists, cfg)
        say(f"round {rnd}: {text}; pass_share={result.pass_share:.4f}")
    return result, steps


def write_playability_toml(cfg: PlayabilityConfig, path: Path) -> None:
    lines = []
    for name in ("strict_angle_deg", "lenient_angle_deg", "slow_gap_s", "reset_gap_s",
                 "min_hand_gap_s", "min_hand_gap_per_cell_s", "vision_gap_s"):
        lines.append(f"{name} = {float(getattr(cfg, name))!r}")
    lines.append(f"forbid_crossover = {'true' if cfg.forbid_crossover else 'false'}")
    for name, tokens in zip(("rare_tokens_left", "rare_tokens_right"), cfg.rare_tokens):
        lines.append(f"{name} = [{', '.join(str(t) for t in sorted(tokens))}]")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_event_lists(conn: sqlite3.Connection) -> list[list[Event]]:
    """Event lists of local Standard difficulties with no requirements and clean flags."""
    lists: list[list[Event]] = []
    for row in conn.execute("SELECT folder FROM maps WHERE source = 'local' ORDER BY key"):
        try:
            bm = read_map(Path(row["folder"]))
        except Exception:  # noqa: BLE001 - untrusted map, skip it
            continue
        if bm.requirements:
            continue
        for d in bm.difficulties:
            f = d.flags
            if d.characteristic != "Standard" or f.has_offgrid_notes or f.has_custom_note_data \
                    or f.has_bpm_changes or f.has_time_offset:
                continue
            events, _ = to_events(d.notes)
            if events:
                lists.append(events)
    return lists


def render_report(before: CalibrationResult, after: CalibrationResult,
                  steps: Sequence[str] = ()) -> str:
    c = after.config
    lines = ["# Playability calibration", "", "| metric | before | after |", "|---|---|---|",
             f"| files | {before.n_files} | {after.n_files} |",
             f"| pass_share | {before.pass_share:.4f} | {after.pass_share:.4f} |"]
    lines += [f"| note share: {r} | {before.rule_note_share[r]:.5f} | "
              f"{after.rule_note_share[r]:.5f} |" for r in HARD_RULE_NAMES]
    lines += ["", "## Relaxation steps", ""]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)] or ["(none)"]
    lines += ["", "## Final thresholds", ""]
    for name in ("strict_angle_deg", "lenient_angle_deg", "slow_gap_s", "reset_gap_s",
                 "min_hand_gap_s", "min_hand_gap_per_cell_s", "vision_gap_s", "forbid_crossover"):
        lines.append(f"- {name}: {getattr(c, name)}")
    lines.append(f"- blocked tokens (left / right): {len(c.rare_tokens[0])} / {len(c.rare_tokens[1])}")
    lines.append(f"- hand-gap 1st percentile (before): {before.hand_gap_p01_s:.4f} s")
    ok = after.pass_share >= TARGET_PASS_SHARE
    lines += ["", f"Result: {'target reached' if ok else 'FAILED'} "
              f"(pass_share {after.pass_share:.4f}, target {TARGET_PASS_SHARE})."]
    if not ok:
        lines.append("Thresholds were not loosened further; this decision goes back to the user.")
    return "\n".join(lines) + "\n"
