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
PASS_HARD_RATE_02 = 0.02
TARGET_PASS_SHARE = 0.95
HARD_RULE_NAMES = ("rare_token", "hand_speed", "vision_block", "same_cell", "crossover",
                   "handclap", "parity")
# Fixed relaxation steps for the data profile, applied in this order.
LADDER: tuple[tuple[str, dict[str, float | bool]], ...] = (
    ("reset_gap_s = 0.25", {"reset_gap_s": 0.25}),
    ("forbid_crossover = False", {"forbid_crossover": False}),
    ("strict_angle_deg = 90.0, lenient_angle_deg = 90.0",
     {"strict_angle_deg": 90.0, "lenient_angle_deg": 90.0}),
    ("slider_gap_s = 0.12", {"slider_gap_s": 0.12}),
    ("min_hand_gap_s = 0.06, min_hand_gap_per_cell_s = 0.0",
     {"min_hand_gap_s": 0.06, "min_hand_gap_per_cell_s": 0.0}),
)
_FIELDS = ("strict_angle_deg", "lenient_angle_deg", "slow_gap_s", "reset_gap_s", "min_hand_gap_s",
           "min_hand_gap_per_cell_s", "vision_gap_s", "slider_gap_s", "forbid_crossover")


@dataclass(frozen=True)
class CalibrationResult:
    n_files: int
    pass_share: float
    pass_share_02: float
    rule_note_share: dict[str, float]
    hand_gap_p01_s: float
    rare_tokens: tuple[frozenset[int], frozenset[int]]
    config: PlayabilityConfig


@dataclass(frozen=True)
class Calibration:
    mask: CalibrationResult
    data: CalibrationResult
    steps: list[tuple[str, CalibrationResult]]
    start: CalibrationResult  # data profile before any ladder step


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
    """Check every list with `cfg`; pass = hard_rate <= 0.01 (also reports the share <= 0.02)."""
    passed = 0
    passed_02 = 0
    notes = 0
    counts: Counter[str] = Counter()
    gaps: list[float] = []
    for events in event_lists:
        report = check(events, cfg)
        passed += report.hard_rate <= PASS_HARD_RATE
        passed_02 += report.hard_rate <= PASS_HARD_RATE_02
        notes += report.n_notes
        counts.update(v.rule for v in report.hard)
        gaps.extend(_hand_gaps(events))
    n = len(event_lists)
    share = {r: counts.get(r, 0) / max(notes, 1) for r in HARD_RULE_NAMES}
    p01 = float(np.percentile(gaps, 1)) if gaps else 0.0
    return CalibrationResult(n, passed / n if n else 0.0, passed_02 / n if n else 0.0, share, p01,
                             cfg.rare_tokens, cfg)


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


def calibrate(
    event_lists: Sequence[Sequence[Event]],
    progress: Callable[[str], None] | None = None,
) -> Calibration:
    """Measure the strict mask profile, then relax the data profile along LADDER until 0.95."""
    say = progress or (lambda _msg: None)
    mask_cfg = replace(PlayabilityConfig(), rare_tokens=rare_tokens_from(event_lists, 1e-4))
    mask = measure(event_lists, mask_cfg)
    say(f"mask: pass_share={mask.pass_share:.4f}")
    cfg = PlayabilityConfig()
    data = measure(event_lists, cfg)
    say(f"data start: pass_share={data.pass_share:.4f}")
    start = data
    steps: list[tuple[str, CalibrationResult]] = []
    for label, values in LADDER:
        if data.pass_share >= TARGET_PASS_SHARE:
            break
        cfg = replace(cfg, **values)
        data = measure(event_lists, cfg)
        steps.append((label, data))
        say(f"data step {label}: pass_share={data.pass_share:.4f}")
    return Calibration(mask, data, steps, start)


def _table(name: str, cfg: PlayabilityConfig) -> list[str]:
    lines = [f"[{name}]"]
    for field in _FIELDS:
        v = getattr(cfg, field)
        lines.append(f"{field} = {'true' if v else 'false'}" if isinstance(v, bool)
                     else f"{field} = {float(v)!r}")
    for key, tokens in zip(("rare_tokens_left", "rare_tokens_right"), cfg.rare_tokens):
        lines.append(f"{key} = [{', '.join(str(t) for t in sorted(tokens))}]")
    return lines


def write_playability_toml(mask: PlayabilityConfig, data: PlayabilityConfig, path: Path) -> None:
    """Write both profiles as [mask] and [data] tables."""
    lines = _table("mask", mask) + [""] + _table("data", data)
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


def render_report(cal: Calibration) -> str:
    lines = ["# Playability calibration", "",
             f"Files: {cal.mask.n_files}. Pass share = share of files with hard rate below the bound.",
             "", "| profile / step | pass share <= 0.01 | pass share <= 0.02 | "
             + " | ".join(f"note share: {r}" for r in HARD_RULE_NAMES) + " |",
             "|---|---|---|" + "---|" * len(HARD_RULE_NAMES)]

    def row(label: str, r: CalibrationResult) -> str:
        shares = " | ".join(f"{r.rule_note_share[n]:.5f}" for n in HARD_RULE_NAMES)
        return f"| {label} | {r.pass_share:.4f} | {r.pass_share_02:.4f} | {shares} |"

    lines.append(row("mask", cal.mask))
    lines.append(row("data: start (no relaxation)", cal.start))
    lines += [row(f"data: {i}. {label}", r) for i, (label, r) in enumerate(cal.steps, 1)]
    for name, res in (("mask", cal.mask), ("data", cal.data)):
        lines += ["", f"## Final values: {name}", ""]
        lines += [f"- {f}: {getattr(res.config, f)}" for f in _FIELDS]
        lines.append(f"- blocked tokens (left / right): "
                     f"{len(res.config.rare_tokens[0])} / {len(res.config.rare_tokens[1])}")
    ok = cal.data.pass_share >= TARGET_PASS_SHARE
    lines += ["", f"Result: {'target reached' if ok else 'FAILED'} "
              f"(data profile pass share {cal.data.pass_share:.4f}, target {TARGET_PASS_SHARE})."]
    if not ok:
        lines.append("The ladder is exhausted; thresholds were not loosened further.")
    return "\n".join(lines) + "\n"
