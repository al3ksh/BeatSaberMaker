"""Audio file in, playable Beat Saber map folder out."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bsmaker.audio.beats import Grid, align_grid, detect_grid, snap_times, to_beats
from bsmaker.audio.decode import SAMPLE_RATE, decode_mono
from bsmaker.audio.mel import log_mel
from bsmaker.data.stats import lookup
from bsmaker.formats.events import Event, to_events
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
from bsmaker.formats.model import Note
from bsmaker.formats.writer import write_map
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.models.timing.decode import LoadedTiming, predict_events
from bsmaker.playability.checker import Report, check
from bsmaker.playability.config import PlayabilityConfig

MIN_AUDIO_S = 30.0
MIN_EVENTS = 20
SILENCE_PEAK = 1e-3
DENSITY_TOLERANCE = 0.10
MAX_DENSITY_ROUNDS = 3
_RANKS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")


class GenerateError(Exception):
    """The song cannot be turned into a map (too short, no usable events, ...)."""


@dataclass(frozen=True)
class GenerateResult:
    folder: Path
    n_notes: int
    nps: float
    bpm: float
    grid_steady: bool
    rank: str
    n_skipped: int
    density_rounds: int
    warnings: tuple[str, ...]


def safe_title(text: str) -> str:
    """A title usable in a Windows folder name."""
    t = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t[:80].rstrip(". ")
    return t or "untitled"


STATS_KEYS = ("coverage_median", "mel_mean", "mel_std", "nps_bins", "njs_by_bin",
              "njs_offset_by_bin", "double_share_by_bin")


def require_stats(stats: dict, keys: Sequence[str] = STATS_KEYS) -> None:
    """Raise GenerateError naming the dataset-stats keys a checkpoint is missing."""
    missing = [k for k in keys if k not in (stats or {})]
    if missing:
        raise GenerateError(f"checkpoint has no dataset stats ({', '.join(missing)}); "
                            "retrain or use a newer checkpoint")


def pick_rank(nps: float, stats: dict) -> str:
    medians = stats.get("rank_median_nps", {})
    known = [r for r in _RANKS if r in medians]
    if not known:
        return "Expert"
    return min(known, key=lambda r: abs(float(medians[r]) - nps))


def pick_njs(nps: float, stats: dict) -> tuple[float, float]:
    return lookup(stats, "njs_by_bin", nps), lookup(stats, "njs_offset_by_bin", nps)


def _measured_nps(notes: Sequence[Note]) -> float:
    if len(notes) < 2:
        return 0.0
    span = notes[-1].time_s - notes[0].time_s
    return len(notes) / span if span > 0 else 0.0


def check_exact(notes: Sequence[Note], cfg: PlayabilityConfig) -> Report:
    """Playability check on the exact note times: group identical times, no frame rounding."""
    groups: dict[float, dict[int, int]] = {}
    for n in notes:
        groups.setdefault(n.time_s, {})[n.color] = encode_token(n.x, n.y, n.direction)
    events = [Event(t, g.get(0, NONE_TOKEN), g.get(1, NONE_TOKEN)) for t, g in sorted(groups.items())]
    return check(events, cfg)


def generate_map(audio_path: Path, *, nps: float, styles: Sequence[str], seed: int,
                 out_dir: Path, timing: LoadedTiming, pattern: LoadedPattern,
                 mask_cfg: PlayabilityConfig, stats: dict, device: str,
                 overwrite: bool = False, title: str | None = None,
                 extra_warnings: Sequence[str] = ()) -> GenerateResult:
    audio_path = Path(audio_path)
    audio = decode_mono(audio_path)
    if len(audio) / SAMPLE_RATE < MIN_AUDIO_S:
        raise GenerateError(f"audio is shorter than {MIN_AUDIO_S:g} s")
    if float(np.abs(audio).max()) < SILENCE_PEAK:
        raise GenerateError("audio is silent")
    mel = log_mel(audio)
    grid: Grid = detect_grid(audio)

    count_scale = 1.0
    best = None  # (relative error, notes, planned, report, aligned grid)
    rounds = 0
    early_stop = ""
    for rnd in range(1, MAX_DENSITY_ROUNDS + 1):
        rounds = rnd
        events = predict_events(timing, mel, nps, styles, device, count_scale)
        if len(events) < MIN_EVENTS:
            if best is None:
                raise GenerateError(f"only {len(events)} events found; the audio looks silent "
                                    "or has no usable rhythm")
            early_stop = f"density search stopped early: only {len(events)} events in round {rnd}"
            break
        round_grid = align_grid(grid, events)
        planned = snap_times(events, round_grid)
        notes, report = sample_notes(pattern, planned, mel, nps, styles, seed, mask_cfg, device)
        measured = _measured_nps(notes)
        err = abs(measured / nps - 1.0) if measured > 0 else float("inf")
        if best is None or err < best[0]:
            best = (err, notes, planned, report, round_grid)
        if err <= DENSITY_TOLERANCE:
            break
        count_scale *= nps / measured if measured > 0 else 2.0
    _, notes, planned, report, grid = best  # the grid aligned in the kept round
    if len(notes) < 2:
        raise GenerateError("no notes could be placed")

    notes = [replace(n, beat=float(to_beats(n.time_s, grid.bpm))) for n in notes]
    final_events, dropped = to_events(notes)
    if dropped or check(final_events, mask_cfg).hard:
        raise RuntimeError("generated map breaks a hard playability rule on frame times (bug)")
    if check_exact(notes, mask_cfg).hard:
        raise RuntimeError("generated map breaks a hard playability rule on exact times (bug)")
    measured = _measured_nps(notes)

    warnings: list[str] = list(extra_warnings)
    if not grid.steady:
        warnings.append("no steady beat grid found; notes are not snapped to beats")
    if report.n_skipped:
        warnings.append(f"{report.n_skipped} events skipped by the playability mask")
    if report.n_frame_collisions:
        warnings.append(f"{report.n_frame_collisions} planned events dropped: "
                        "two events fell into one mel frame")
    if early_stop:
        warnings.append(early_stop)
    if abs(measured / nps - 1.0) > DENSITY_TOLERANCE:
        warnings.append(f"density {measured:.2f} nps is outside 10% of the requested {nps:g}")

    rank = pick_rank(nps, stats)
    njs, njs_offset = pick_njs(nps, stats)
    raw = title or audio_path.stem
    name = re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()[:120] or "untitled"
    style = "+".join(styles) or "unknown"
    folder = Path(out_dir) / f"BSM - {safe_title(name)} [nps {nps:g}, {style}]"
    try:
        write_map(folder, title=name, artist="Unknown", bpm=grid.bpm, rank=rank, njs=njs,
                  njs_offset=njs_offset, notes=notes, audio_src=audio_path, overwrite=overwrite)
    except FileExistsError:
        raise
    except OSError as e:
        raise GenerateError(f"cannot write the map to {out_dir}: {e}") from e
    return GenerateResult(folder, len(notes), measured, grid.bpm, grid.steady, rank,
                          report.n_skipped, rounds, tuple(warnings))
