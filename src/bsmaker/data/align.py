"""Audio-alignment score: do the notes of a difficulty land on onsets of the song?"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from bsmaker.data.events_cache import training_events
from bsmaker.data.manifest import get_meta, set_meta
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.model import Note
from bsmaker.formats.reader import read_map, read_map_with_tempo
from bsmaker.formats.tempo import TempoMap

MIN_NOTES = 20
ALIGN_SHIFTS_S = (-0.31, -0.23, -0.17, -0.11, 0.11, 0.17, 0.23, 0.31)
MAX_UNCOVERED_SHARE = 0.02
THRESHOLD_KEY = "align_threshold"
NEGATIVE_PASS_KEY = "align_negative_pass_share"
PERCENTILES_KEY = "align_calibration_percentiles"
CALIBRATION_N_KEY = "align_calibration_n"
MIN_SANE_THRESHOLD = 1.2
MAX_NEGATIVE_PASS = 0.05
NEGATIVE_STRETCH = 1.015
TIE_EPS = 0.01
PERCENTILES = (5, 25, 50, 75, 95)


def onset_envelope(mel: np.ndarray) -> np.ndarray:
    """Positive spectral flux of a log-mel, summed over bands, minus its minimum."""
    mel = np.asarray(mel, dtype=np.float32)
    flux = np.zeros(mel.shape[0], dtype=np.float32)
    if mel.shape[0] > 1:
        flux[1:] = np.maximum(mel[1:] - mel[:-1], 0.0).sum(axis=1)
    return flux - flux.min() if flux.size else flux


def _mean_peak(frames: np.ndarray, env: np.ndarray) -> tuple[float, int]:
    """Mean envelope maximum over frame +-1 for in-range frames; returns (mean, count)."""
    ok = frames[(frames >= 0) & (frames < env.size)]
    if ok.size == 0:
        return float("nan"), 0
    padded = np.pad(env, 1, constant_values=-np.inf)
    peaks = np.maximum(np.maximum(padded[ok], padded[ok + 1]), padded[ok + 2])
    return float(peaks.mean()), int(ok.size)


def alignment_score(times_s: np.ndarray, env: np.ndarray) -> float:
    """A(0) / median(A(s) for s in ALIGN_SHIFTS_S), A(s) = mean envelope peak at t + s.

    The shifted copies of the same events are the baseline, so dense maps do not score high
    just because the envelope is busy. NaN: fewer than 20 usable events, a flat envelope, a
    non-positive baseline, or more than MAX_UNCOVERED_SHARE of events beyond the envelope.
    """
    env = np.asarray(env, dtype=np.float32)
    times = np.asarray(times_s, dtype=np.float64)
    times = times[np.isfinite(times)]
    if env.size == 0 or times.size == 0 or float(env.max()) - float(env.min()) <= 1e-9:
        return float("nan")

    def frames_at(shift: float) -> np.ndarray:
        return np.rint((times + shift) * FRAMES_PER_SECOND).astype(np.int64)

    f0 = frames_at(0.0)
    a0, n0 = _mean_peak(f0, env)
    if n0 < MIN_NOTES or (times.size - n0) / times.size > MAX_UNCOVERED_SHARE:
        return float("nan")
    base = [_mean_peak(frames_at(s), env)[0] for s in ALIGN_SHIFTS_S]
    base = [b for b in base if math.isfinite(b)]
    if not base:
        return float("nan")
    median = float(np.median(base))
    if not math.isfinite(median) or median <= 0:
        return float("nan")
    return a0 / median


def _envelope(features_dir: Path, key: str) -> np.ndarray | None:
    path = Path(features_dir) / f"{key}.npy"
    if not path.is_file():
        return None
    mel = np.load(path, mmap_mode="r", allow_pickle=False)
    return onset_envelope(np.asarray(mel, dtype=np.float32))


def _event_times(notes: list[Note]) -> np.ndarray:
    return np.array([e.time_s for e in training_events(notes)], dtype=np.float64)


def _retimed(notes: list[Note], to_seconds) -> list[Note]:
    return [Note(to_seconds(n.beat), n.beat, n.x, n.y, n.color, n.direction) for n in notes]


def _const_notes(notes: list[Note], bpm: float) -> list[Note]:
    return _retimed(notes, lambda beat: beat * 60.0 / bpm)


def _usable_tempo(tempo: TempoMap | None) -> bool:
    return tempo is not None and all(math.isfinite(b) and math.isfinite(m) and m > 0
                                     for b, m in tempo.points)


def _find(bm, characteristic: str, rank: str):
    return next((x for x in bm.difficulties
                 if x.characteristic == characteristic and x.rank == rank), None)


def calibrate_threshold(conn: sqlite3.Connection, features_dir: Path, n: int = 1000,
                        seed: int = 0, min_scores: int = 50) -> float:
    """5th percentile of the score over accepted constant-BPM train difficulties; stored in meta."""
    rows = conn.execute(
        "SELECT d.map_key, d.characteristic, d.rank, m.folder FROM difficulties d "
        "JOIN maps m ON m.key = d.map_key WHERE d.accepted = 1 AND d.split = 'train' "
        "AND (d.tempo_kind IS NULL OR d.tempo_kind = 'none') "
        "ORDER BY d.map_key, d.characteristic, d.rank").fetchall()
    rows = [r for r in rows if (Path(features_dir) / f"{r['map_key']}.npy").is_file()]
    order = np.random.default_rng(seed).permutation(len(rows))
    scores: list[float] = []
    stretched: list[float] = []
    for i in order:
        if len(scores) >= n:
            break
        r = rows[int(i)]
        try:
            d = _find(read_map(Path(r["folder"])), r["characteristic"], r["rank"])
            env = _envelope(features_dir, r["map_key"])
            if d is None or env is None:
                continue
            times = _event_times(d.notes)
            s = alignment_score(times, env)
            neg = alignment_score(times * NEGATIVE_STRETCH, env)
        except (sqlite3.Error, MemoryError):
            raise
        except Exception:  # noqa: BLE001 - one unreadable map is skipped
            continue
        if math.isfinite(s):
            scores.append(s)
            stretched.append(neg)
    if len(scores) < min_scores:
        raise ValueError(f"only {len(scores)} usable alignment scores, need {min_scores}")
    threshold = float(np.percentile(scores, 5))
    set_meta(conn, THRESHOLD_KEY, repr(threshold))
    # negative control: the same events stretched by 1.5 % should NOT look aligned
    passed = sum(1 for v in stretched if math.isfinite(v) and v >= threshold)
    set_meta(conn, NEGATIVE_PASS_KEY, repr(passed / len(stretched)))
    set_meta(conn, PERCENTILES_KEY, json.dumps(
        {str(p): float(np.percentile(scores, p)) for p in PERCENTILES}))
    set_meta(conn, CALIBRATION_N_KEY, str(len(scores)))
    return threshold


def calibration_summary(conn: sqlite3.Connection) -> dict:
    """What the last calibration stored: threshold, score percentiles, negative-control share."""
    def _float(key: str) -> float | None:
        raw = get_meta(conn, key)
        try:
            return None if raw is None else float(raw)
        except ValueError:
            return None

    pct = get_meta(conn, PERCENTILES_KEY)
    try:
        percentiles = {int(k): float(v) for k, v in json.loads(pct).items()} if pct else {}
    except (ValueError, AttributeError, TypeError):
        percentiles = {}
    n = _float(CALIBRATION_N_KEY)
    return {"threshold": _float(THRESHOLD_KEY), "percentiles": percentiles,
            "negative_pass_share": _float(NEGATIVE_PASS_KEY),
            "n": None if n is None else int(n)}


def _blank() -> dict[str, int]:
    return {"accepted": 0, "failed": 0, "missing": 0, "unreadable": 0, "nan": 0}


def verify_tempo(conn: sqlite3.Connection, cfg, *, write: bool = True) -> dict:
    """Score difficulties whose only reason is `tempo_unverified`; accept those above threshold.

    With write=False nothing is stored or accepted (a dry run); the counts are the same.
    `align_score_const` holds the score of the OTHER variant: constant BPM for kind "events",
    the tempo-map variant for kind "editor" (whose `align_score` uses constant BPM).

    Returns {"threshold", "dry_run", "kinds": {kind: {accepted, failed, missing, unreadable, nan}},
    "events_scored", "ties", "const_higher" (kind "events"; const_higher counts non-tie pairs
    only), "editor_scored", "editor_ties", "editor_map_higher", "calibration"}.
    """
    raw = get_meta(conn, THRESHOLD_KEY)
    if raw is None:
        raise ValueError(f"no {THRESHOLD_KEY} in the manifest; calibrate first")
    try:
        threshold = float(raw)
    except ValueError:
        raise ValueError(f"manifest meta {THRESHOLD_KEY!r} is not a number: {raw!r}") from None
    out: dict = {"threshold": threshold, "dry_run": not write, "kinds": {},
                 "events_scored": 0, "ties": 0, "const_higher": 0,
                 "editor_scored": 0, "editor_ties": 0, "editor_map_higher": 0,
                 "calibration": calibration_summary(conn)}
    rows = conn.execute(
        "SELECT d.map_key, d.characteristic, d.rank, d.tempo_kind, m.folder FROM difficulties d "
        "JOIN maps m ON m.key = d.map_key WHERE d.reject_reasons = ? AND d.accepted = 0 "
        "ORDER BY d.map_key, d.characteristic, d.rank",
        (json.dumps(["tempo_unverified"]),)).fetchall()
    by_map: dict[str, list] = {}
    for r in rows:
        by_map.setdefault(r["map_key"], []).append(r)
    for key, diffs in by_map.items():
        env = _envelope(cfg.features_dir, key)
        bm = None
        tempos: dict = {}
        if env is not None:
            try:
                bm, tempos = read_map_with_tempo(Path(diffs[0]["folder"]))
            except sqlite3.Error:
                raise
            except Exception:  # noqa: BLE001 - unreadable map: counted as missing
                bm = None
        updates = []
        for r in diffs:
            kind = r["tempo_kind"] or "none"
            c = out["kinds"].setdefault(kind, _blank())
            if env is None:
                c["missing"] += 1
                continue
            d = None if bm is None else _find(bm, r["characteristic"], r["rank"])
            if d is None:
                c["unreadable"] += 1
                continue
            score = alignment_score(_event_times(d.notes), env)
            other = None
            if kind == "events":
                other = alignment_score(_event_times(_const_notes(d.notes, bm.bpm)), env)
                if math.isfinite(score) and math.isfinite(other):
                    out["events_scored"] += 1
                    if abs(score - other) < TIE_EPS:
                        out["ties"] += 1
                    else:
                        out["const_higher"] += int(other > score)
            elif kind == "editor":
                tempo = tempos.get((r["characteristic"], r["rank"]))
                if _usable_tempo(tempo):
                    other = alignment_score(
                        _event_times(_retimed(d.notes, tempo.to_seconds)), env)
                    if math.isfinite(score) and math.isfinite(other):
                        out["editor_scored"] += 1
                        if abs(score - other) < TIE_EPS:
                            out["editor_ties"] += 1
                        else:
                            out["editor_map_higher"] += int(other > score)
            ok = math.isfinite(score) and score >= threshold
            if not math.isfinite(score):
                c["nan"] += 1
            c["accepted" if ok else "failed"] += 1
            updates.append((r, score, other, ok))
        if not write:
            continue
        with conn:  # one transaction per map
            for r, score, other, ok in updates:
                s = score if math.isfinite(score) else None
                cs = other if other is not None and math.isfinite(other) else None
                where = (r["map_key"], r["characteristic"], r["rank"])
                if ok:
                    conn.execute("UPDATE difficulties SET align_score=?, align_score_const=?, "
                                 "accepted=1, reject_reasons='[]' WHERE map_key=? AND "
                                 "characteristic=? AND rank=?", (s, cs, *where))
                else:
                    conn.execute("UPDATE difficulties SET align_score=?, align_score_const=? "
                                 "WHERE map_key=? AND characteristic=? AND rank=?", (s, cs, *where))
    return out


def _pair_line(label: str, higher: int, scored: int, ties: int, what: str) -> str:
    non_tie = scored - ties
    share = higher / non_tie if non_tie else 0.0
    return (f"{label}: {scored} scored, {ties} ties (|diff| < {TIE_EPS}); {what} in "
            f"{higher} of {non_tie} non-tie pairs ({share:.1%})")


def render_report(res: dict, calibrated: bool) -> str:
    lines = []
    if res.get("dry_run"):
        lines.append("DRY RUN: nothing was written to the manifest (no scores stored, "
                     "nothing accepted)")
    lines.append(f"threshold: {res['threshold']:.4f} "
                 f"({'calibrated now' if calibrated else 'read from the manifest'})")
    cal = res.get("calibration") or {}
    if cal.get("percentiles"):
        pct = ", ".join(f"p{p} {v:.3f}" for p, v in sorted(cal["percentiles"].items()))
        lines.append(f"calibration scores ({cal.get('n')} difficulties): {pct}")
    neg = cal.get("negative_pass_share")
    if neg is not None:
        lines.append(f"negative control (event times x {NEGATIVE_STRETCH}): {neg:.1%} reach "
                     "the threshold")
    if res["threshold"] < MIN_SANE_THRESHOLD:
        lines.append(f"WARNING: threshold {res['threshold']:.3f} is below {MIN_SANE_THRESHOLD}; "
                     "the gate may not separate aligned from misaligned maps")
    if neg is not None and neg > MAX_NEGATIVE_PASS:
        lines.append(f"WARNING: {neg:.1%} of stretched negative controls pass (limit "
                     f"{MAX_NEGATIVE_PASS:.0%}); the gate is too lenient")
    for kind in sorted(set(res["kinds"]) | {"editor", "events"}):
        c = res["kinds"].get(kind, _blank())
        lines.append(f"{kind}: accepted {c['accepted']}, failed {c['failed']}, "
                     f"missing features {c['missing']}, unreadable {c['unreadable']}, "
                     f"NaN {c['nan']}")
    lines.append(_pair_line("events", res.get("const_higher", 0), res.get("events_scored", 0),
                            res.get("ties", 0), "constant BPM scores higher than the tempo map"))
    lines.append(_pair_line("editor", res.get("editor_map_higher", 0), res.get("editor_scored", 0),
                            res.get("editor_ties", 0),
                            "the tempo map scores higher than constant BPM"))
    lines.append("legend: align_score_const is the score of the other variant (constant BPM for "
                 "events, the tempo map for editor)")
    return "\n".join(lines)
