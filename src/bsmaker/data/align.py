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
from bsmaker.formats.reader import read_map

MIN_NOTES = 20
ALIGN_SHIFTS_S = (-0.31, -0.23, -0.17, -0.11, 0.11, 0.17, 0.23, 0.31)
MAX_UNCOVERED_SHARE = 0.02
THRESHOLD_KEY = "align_threshold"


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


def _const_notes(notes: list[Note], bpm: float) -> list[Note]:
    return [Note(n.beat * 60.0 / bpm, n.beat, n.x, n.y, n.color, n.direction) for n in notes]


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
    for i in order:
        if len(scores) >= n:
            break
        r = rows[int(i)]
        try:
            d = _find(read_map(Path(r["folder"])), r["characteristic"], r["rank"])
            env = _envelope(features_dir, r["map_key"])
            if d is None or env is None:
                continue
            s = alignment_score(_event_times(d.notes), env)
        except (sqlite3.Error, MemoryError):
            raise
        except Exception:  # noqa: BLE001 - one unreadable map is skipped
            continue
        if math.isfinite(s):
            scores.append(s)
    if len(scores) < min_scores:
        raise ValueError(f"only {len(scores)} usable alignment scores, need {min_scores}")
    threshold = float(np.percentile(scores, 5))
    set_meta(conn, THRESHOLD_KEY, repr(threshold))
    return threshold


def _blank() -> dict[str, int]:
    return {"accepted": 0, "failed": 0, "missing": 0, "unreadable": 0, "nan": 0}


def verify_tempo(conn: sqlite3.Connection, cfg) -> dict:
    """Score difficulties whose only reason is `tempo_unverified`; accept those above threshold.

    Returns {"threshold", "kinds": {kind: {accepted, failed, missing, nan}},
    "events_scored": n, "const_higher": n} (the last two for kind "events").
    """
    raw = get_meta(conn, THRESHOLD_KEY)
    if raw is None:
        raise ValueError(f"no {THRESHOLD_KEY} in the manifest; calibrate first")
    try:
        threshold = float(raw)
    except ValueError:
        raise ValueError(f"manifest meta {THRESHOLD_KEY!r} is not a number: {raw!r}") from None
    out: dict = {"threshold": threshold, "kinds": {}, "events_scored": 0, "const_higher": 0}
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
        if env is not None:
            try:
                bm = read_map(Path(diffs[0]["folder"]))
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
            const = None
            if kind == "events":
                const = alignment_score(_event_times(_const_notes(d.notes, bm.bpm)), env)
                if math.isfinite(score) and math.isfinite(const):
                    out["events_scored"] += 1
                    out["const_higher"] += int(const > score)
            ok = math.isfinite(score) and score >= threshold
            if not math.isfinite(score):
                c["nan"] += 1
            c["accepted" if ok else "failed"] += 1
            updates.append((r, score, const, ok))
        with conn:  # one transaction per map
            for r, score, const, ok in updates:
                s = score if math.isfinite(score) else None
                cs = const if const is not None and math.isfinite(const) else None
                where = (r["map_key"], r["characteristic"], r["rank"])
                if ok:
                    conn.execute("UPDATE difficulties SET align_score=?, align_score_const=?, "
                                 "accepted=1, reject_reasons='[]' WHERE map_key=? AND "
                                 "characteristic=? AND rank=?", (s, cs, *where))
                else:
                    conn.execute("UPDATE difficulties SET align_score=?, align_score_const=? "
                                 "WHERE map_key=? AND characteristic=? AND rank=?", (s, cs, *where))
    return out


def render_report(res: dict, calibrated: bool) -> str:
    lines = [f"threshold: {res['threshold']:.4f} "
             f"({'calibrated now' if calibrated else 'read from the manifest'})"]
    for kind in sorted(set(res["kinds"]) | {"editor", "events"}):
        c = res["kinds"].get(kind, _blank())
        lines.append(f"{kind}: accepted {c['accepted']}, failed {c['failed']}, "
                     f"missing features {c['missing']}, unreadable {c['unreadable']}, "
                     f"NaN {c['nan']}")
    n = res["events_scored"]
    share = res["const_higher"] / n if n else 0.0
    lines.append(f"events: {res['const_higher']} of {n} ({share:.1%}) score higher with "
                 "constant BPM than with the tempo map")
    return "\n".join(lines)
