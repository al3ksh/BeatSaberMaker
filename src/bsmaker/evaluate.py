"""Evaluate the trained models on held-out songs and render a markdown report."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np

from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.formats.events import Event, to_events
from bsmaker.generate import check_exact
from bsmaker.formats.tokens import NONE_TOKEN, decode_token
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.models.timing.decode import LoadedTiming, make_timing_eval, predict_events
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

DENSITY_TOLERANCE = 0.10
AXES = {"column": 4, "row": 3, "direction": 9, "hand": 2}
AXIS_LABELS = {
    "column": ("0", "1", "2", "3"),
    "row": ("0", "1", "2"),
    "direction": ("up", "down", "left", "right", "up-left", "up-right", "down-left",
                  "down-right", "dot"),
    "hand": ("left", "right"),
}


def _spaced(entries: Sequence[Entry], n: int) -> list[Entry]:
    """At most `n` entries, deterministic and evenly spaced."""
    entries = list(entries)
    if n <= 0 or not entries:
        return []
    if n >= len(entries):
        return entries
    return [entries[i] for i in np.unique(np.linspace(0, len(entries) - 1, n).astype(int))]


def _measured_nps(notes) -> float:
    if len(notes) < 2:
        return 0.0
    span = notes[-1].time_s - notes[0].time_s
    return len(notes) / span if span > 0 else 0.0


def _one_per_song(entries: Sequence[Entry]) -> list[Entry]:
    """Per song the difficulty with the median nps (ties: the lower)."""
    by_song: dict[str, list[Entry]] = {}
    for e in entries:
        by_song.setdefault(e.map_key, []).append(e)
    out = []
    for group in by_song.values():
        group = sorted(group, key=lambda e: (e.nps, e.i))
        out.append(group[(len(group) - 1) // 2])
    return out


def _mel(features_dir: Path, e: Entry) -> np.ndarray:
    mel = np.load(Path(features_dir) / f"{e.map_key}.npy", allow_pickle=False)
    return mel[:min(int(e.n_frames), mel.shape[0])]


def human_events(cache: EventCache, e: Entry) -> list[Event]:
    t, left, right = cache.events(e.i)
    return [Event(float(a), int(b), int(c)) for a, b, c in zip(t, left, right)]


def _bin_label(edges: Sequence[float], nps: float) -> str:
    b = min(max(int(np.searchsorted(edges, nps, side="right")) - 1, 0), len(edges) - 2)
    return f"{edges[b]:g}-{edges[b + 1]:g}"


def eval_timing(loaded: LoadedTiming, cache: EventCache, features_dir: Path, split: str,
                n: int, device: str) -> dict:
    """Mean onset P/R/F1 at the human event count, overall and by nps bin."""
    entries = _spaced(cache.select(split), n)
    edges = list(loaded.stats.get("nps_bins", []))
    scores: list[tuple[float, float, float]] = []
    labels: list[str] = []
    order: dict[str, int] = {}
    for e in entries:
        # the training-time whole-song eval, run on one song at a time
        fn = make_timing_eval(cache, [e], features_dir, loaded.stats, lambda m, loader: {},
                              n_songs=1)
        m = fn(loaded.model, None)
        scores.append((m["precision"], m["recall"], m["f1"]))
        lab = _bin_label(edges, e.nps) if len(edges) >= 2 else "all"
        order.setdefault(lab, len(edges) and int(np.searchsorted(edges, e.nps, side="right")))
        labels.append(lab)
    arr = np.asarray(scores, dtype=np.float64).reshape(-1, 3)
    by_bin: dict[str, dict] = {}
    for lab in sorted(set(labels), key=lambda k: order[k]):
        sel = np.asarray([x == lab for x in labels])
        by_bin[lab] = {"n": int(sel.sum()), "f1": float(arr[sel, 2].mean())}
    nan = float("nan")
    return {"n": len(entries), "precision": float(arr[:, 0].mean()) if len(arr) else nan,
            "recall": float(arr[:, 1].mean()) if len(arr) else nan,
            "f1": float(arr[:, 2].mean()) if len(arr) else nan, "f1_by_nps_bin": by_bin}


def eval_pattern(loaded: LoadedPattern, cache: EventCache, features_dir: Path, split: str,
                 n: int, device: str) -> dict:
    """Cross-entropy on one fixed run per difficulty (same construction as validation)."""
    from torch.utils.data import DataLoader

    from bsmaker.models.pattern.dataset import PatternSequences
    from bsmaker.models.pattern.model import pattern_eval

    entries = _spaced(cache.select(split), n)
    ds = PatternSequences(cache, entries, features_dir=features_dir, stats=loaded.stats,
                          context=loaded.config.context, seed=1, fixed=True)
    m = pattern_eval(loaded.model, DataLoader(ds, batch_size=16, num_workers=0))
    m = {k: float(v) for k, v in m.items()}
    m["n"] = len(entries)
    return m


def distribution(event_lists: Sequence[Sequence[Event]]) -> dict:
    """Shares over all notes by column, row, direction and hand, plus the double share."""
    counts = {a: np.zeros(k) for a, k in AXES.items()}
    n_events = n_double = 0
    for events in event_lists:
        for e in events:
            n_events += 1
            n_double += int(e.left != NONE_TOKEN and e.right != NONE_TOKEN)
            for hand, tok in enumerate((e.left, e.right)):
                if tok == NONE_TOKEN:
                    continue
                x, y, d = decode_token(tok)
                counts["column"][x] += 1
                counts["row"][y] += 1
                counts["direction"][d] += 1
                counts["hand"][hand] += 1
    out: dict = {}
    for a, c in counts.items():
        total = c.sum()
        out[a] = (c / total if total else c).tolist()
    out["n_notes"] = int(counts["hand"].sum())
    out["double_share"] = n_double / n_events if n_events else 0.0
    return out


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def _soft_rates(event_lists: Sequence[Sequence[Event]], cfg: PlayabilityConfig) -> tuple[dict, int, int]:
    """(soft violations per 1000 notes by rule, notes, hard violations)."""
    soft: Counter[str] = Counter()
    notes = hard = 0
    for ev in event_lists:
        rep = check(ev, cfg)
        notes += rep.n_notes
        hard += len(rep.hard)
        soft.update(v.rule for v in rep.soft)
    return ({r: 1000.0 * c / notes for r, c in sorted(soft.items())} if notes else {}), notes, hard


def eval_generated(timing: LoadedTiming, pattern: LoadedPattern, cache: EventCache,
                   features_dir: Path, mask_cfg: PlayabilityConfig, split: str, n: int,
                   seed: int, device: str) -> dict:
    """Generate for `n` test difficulties (one per song) and compare with the human maps.

    No audio is decoded: the cached mel is used, there is no beat grid and no snapping, and
    there is a single density round.
    """
    entries = _spaced(_one_per_song(cache.select(split)), n)
    gen_lists: list[list[Event]] = []
    hum_lists: list[list[Event]] = []
    n_events = n_skipped = n_fallback = n_collisions = 0
    masses: list[float] = []
    hits = 0
    for e in entries:
        mel = _mel(features_dir, e)
        times = predict_events(timing, mel, e.nps, e.styles, device)
        notes, rep = sample_notes(pattern, times, mel, e.nps, e.styles, seed, mask_cfg, device)
        events, dropped = to_events(notes)
        if dropped:
            raise RuntimeError(f"{dropped} generated notes dropped by to_events in {e.map_key} (bug)")
        if check(events, mask_cfg).hard:
            raise RuntimeError(f"generated map for {e.map_key} breaks a hard rule "
                               f"({len(check(events, mask_cfg).hard)} violations; bug)")
        if check_exact(notes, mask_cfg).hard:
            raise RuntimeError(f"generated map for {e.map_key} breaks a hard rule on exact "
                               "times (bug)")
        gen_lists.append(events)
        hum_lists.append(human_events(cache, e))
        n_events += rep.n_events
        n_skipped += rep.n_skipped
        n_collisions += rep.n_frame_collisions
        n_fallback += rep.n_single_hand_fallback
        if rep.n_events:
            masses.append(rep.masked_mass_mean)
        m = _measured_nps(notes)
        hits += int(m > 0 and abs(m / e.nps - 1.0) <= DENSITY_TOLERANCE)
    gen_soft, gen_notes, gen_hard = _soft_rates(gen_lists, mask_cfg)
    hum_soft, hum_notes, hum_hard = _soft_rates(hum_lists, mask_cfg)
    gd, hd = distribution(gen_lists), distribution(hum_lists)
    return {
        "n": len(entries), "dropped_notes": 0,
        "nps_range": [float(x) for x in np.percentile([e.nps for e in entries], [0, 50, 100])]
        if entries else [0.0, 0.0, 0.0], "generated": gd, "human": hd,
        "tv": {a: tv_distance(gd[a], hd[a]) for a in AXES},
        "soft_per_1000": {"generated": gen_soft, "human": hum_soft},
        "hard_violations_generated": gen_hard,
        "hard_rate_human_per_1000": 1000.0 * hum_hard / hum_notes if hum_notes else 0.0,
        "skipped_rate": n_skipped / n_events if n_events else 0.0,
        "frame_collision_rate": n_collisions / n_events if n_events else 0.0,
        "single_hand_fallback": n_fallback,
        "masked_mass_mean": float(np.mean(masses)) if masses else 0.0,
        "density_hit_share": hits / len(entries) if entries else 0.0,
    }


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def _summary(r: dict) -> list[str]:
    t, p, g = r["timing"], r["pattern"], r["generated"]
    return [
        "## Summary", "",
        f"- Timing onset F1 ({t['n']} songs, human event count): {_f(t['f1'])} "
        f"(precision {_f(t['precision'])}, recall {_f(t['recall'])})",
        f"- Pattern cross-entropy, both hands (joint, per event): {_f(p['ce'])} "
        f"(left hand {_f(p['ce_left'])}, right hand given left {_f(p['ce_right'])})",
        f"- Generated maps ({g['n']}): {g['hard_violations_generated']} hard violations, "
        f"density within 10% after a single round: {100 * g['density_hit_share']:.0f}%",
        f"- Double share: generated {_f(g['generated']['double_share'])} vs "
        f"human {_f(g['human']['double_share'])}",
        f"- Skipped-event rate: {100 * g['skipped_rate']:.1f}% "
        f"(frame collisions, counted separately: {100 * g['frame_collision_rate']:.1f}%)",
    ]


def render_summary(results: dict) -> str:
    return "\n".join(_summary(results)) + "\n"


def render_eval_report(results: dict) -> str:
    t, p, g, s = results["timing"], results["pattern"], results["generated"], results["setup"]
    L = [f"# Evaluation report ({s['split']})", ""] + _summary(results) + [""]
    L += ["## Timing", "", "| | value |", "|---|---|",
          f"| precision | {_f(t['precision'])} |", f"| recall | {_f(t['recall'])} |",
          f"| F1 | {_f(t['f1'])} |", "",
          "F1 by notes-per-second bin:", "", "| nps bin | songs | F1 |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {_f(v['f1'])} |" for k, v in t["f1_by_nps_bin"].items()]
    L += ["", "## Pattern", "", "| | cross-entropy | perplexity |", "|---|---|---|"]
    for name, key in (("both hands (joint, per event)", "ce"), ("left hand", "ce_left"),
                      ("right hand given left", "ce_right")):
        v = p[key]
        L.append(f"| {name} | {_f(v)} | {_f(float(np.exp(v)) if v == v else v)} |")
    L += ["", "## Generated vs human", "",
          f"{g['n']} test difficulties, one per song. Generation uses the cached mel (no audio is "
          "decoded), no beat-grid detection and no snapping, and a single density round.", ""]
    for axis, labels in AXIS_LABELS.items():
        L += [f"| {axis} | generated | human |", "|---|---|---|"]
        L += [f"| {lab} | {_f(a)} | {_f(b)} |"
              for lab, a, b in zip(labels, g["generated"][axis], g["human"][axis])]
        L.append("")
    L += ["| axis | total-variation distance |", "|---|---|"]
    L += [f"| {a} | {_f(v)} |" for a, v in g["tv"].items()]
    L += ["", f"Double share: generated {_f(g['generated']['double_share'])}, "
              f"human {_f(g['human']['double_share'])}.", "",
          "Soft-rule violations per 1 000 notes:", "", "| rule | generated | human |", "|---|---|---|"]
    gs, hs = g["soft_per_1000"]["generated"], g["soft_per_1000"]["human"]
    L += [f"| {r} | {_f(gs.get(r, 0.0), 1)} | {_f(hs.get(r, 0.0), 1)} |"
          for r in sorted(set(gs) | set(hs))] or []
    L += ["", f"- Notes dropped by event grouping in generated maps: {g['dropped_notes']} (must be 0)",
          f"- Hard violations in generated maps: {g['hard_violations_generated']} (must be 0)",
          f"- Hard violations in human maps under the mask profile (information): "
          f"{_f(g['hard_rate_human_per_1000'], 1)} per 1 000 notes",
          f"- Skipped-event rate (playability mask): {100 * g['skipped_rate']:.2f}%",
          f"- Frame-collision rate (events sharing a mel frame, not mask skips): "
          f"{100 * g['frame_collision_rate']:.2f}%",
          f"- Mean probability mass removed by the mask: {_f(g['masked_mass_mean'], 4)}",
          f"- Share of maps within 10% of the target nps after a single round: "
          f"{100 * g['density_hit_share']:.0f}%", ""]
    lo, med, hi = g["nps_range"]
    L += ["## Setup", "", f"- Timing checkpoint: {s['timing_checkpoint']} (step {s['timing_step']})",
          f"- Pattern checkpoint: {s['pattern_checkpoint']} (step {s['pattern_step']})",
          f"- Split: {s['split']}", f"- n (timing, pattern): {s['n']}; n generated: {s['n_generated']}",
          f"- Generated difficulties' nps (min / median / max): {lo:.2f} / {med:.2f} / {hi:.2f}",
          f"- Seed: {s['seed']}", f"- Date: {s.get('date', date.today().isoformat())}", ""]
    return "\n".join(L)
