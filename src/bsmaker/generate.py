"""Audio file in, playable Beat Saber map folder out."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from bsmaker.audio.bar_similarity import BarRepeats, bar_repeats
from bsmaker.audio.beats import Grid, align_grid, detect_grid, onset_envelope, snap_times, to_beats
from bsmaker.audio.decode import SAMPLE_RATE, decode_mono
from bsmaker.audio.mel import log_mel
from bsmaker.data.stats import lookup
from bsmaker.errors import GenerateCancelled
from bsmaker.flow_rules import FlowReport, apply_flow, final_flow_ok
from bsmaker.formats.events import to_events
from bsmaker.formats.layer_tokens import SUPPORTED_ENVIRONMENTS
from bsmaker.formats.model import Arc, Bomb, Chain, Layers, Note, Obstacle
from bsmaker.formats.writer import LevelData, write_map_multi
from bsmaker.layers_generate import LayerModels, generate_decor, generate_lights
from bsmaker.models.ornaments.sample import OrnamentReport, sample_ornaments
from bsmaker.models.pattern.sample import LoadedPattern, SampleReport, sample_notes
from bsmaker.models.timing.decode import LoadedTiming, TimingCurve, predict_events
from bsmaker.phrase import plan_rhythm
from bsmaker.playability.checker import Report, check, exact_events
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.layers import check_layers

MIN_AUDIO_S = 30.0
MIN_EVENTS = 20
SILENCE_PEAK = 1e-3
DENSITY_TOLERANCE = 0.10
MAX_DENSITY_ROUNDS = 3
_RANKS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
_RANK_BY_KEY = {r.lower(): r for r in _RANKS}
_RANK_ORDER = {r: i for i, r in enumerate(_RANKS)}
_ENV_BY_SHORT = {e.removesuffix("Environment").lower(): e for e in SUPPORTED_ENVIRONMENTS}
# Median reaction time (seconds from spawn to the hit) of human maps per nps bin: 30 576 accepted
# train difficulties. 10th-90th percentile spans roughly 0.8x-1.4x of each value.
REACTION_NPS_EDGES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 12.0)
REACTION_S_BY_BIN = (1.200, 1.043, 0.863, 0.714, 0.625, 0.580, 0.528, 0.483)
MIN_HALF_JUMP_BEATS = 0.25  # the game's own lower limit


class GenerateError(Exception):
    """The song cannot be turned into a map (too short, no usable events, ...)."""


ProgressCallback = Callable[[str, int], None]
CancelCheck = Callable[[], bool]


def _check_cancel(is_cancelled: CancelCheck | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")


class _ProgressTracker:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._last_pct = 0

    def emit(self, stage: str, pct: int) -> None:
        clamped = max(self._last_pct, min(100, int(pct)))
        self._last_pct = clamped
        if self._callback is not None:
            self._callback(stage, clamped)


@dataclass(frozen=True)
class LevelResult:
    rank: str
    target_nps: float
    n_notes: int
    nps: float
    n_skipped: int
    density_rounds: int
    n_arcs: int = 0
    n_chains: int = 0
    ornament_report: OrnamentReport | None = None
    n_walls: int = 0
    n_bombs: int = 0
    flow_report: FlowReport | None = None


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
    levels: tuple[LevelResult, ...] = ()
    n_walls: int = 0
    n_bombs: int = 0
    n_lights: int = 0
    environment: str = "DefaultEnvironment"
    decor_report: dict | None = None
    lights_report: dict | None = None


@dataclass
class _BuiltLevel:
    rank: str
    target_nps: float
    notes: list[Note]
    measured_nps: float
    grid: Grid
    report: SampleReport
    rounds: int
    early_stop: str
    njs: float
    njs_offset: float
    arcs: list[Arc]
    chains: list[Chain]
    ornament_report: OrnamentReport
    obstacles: list[Obstacle] = field(default_factory=list)
    bombs: list[Bomb] = field(default_factory=list)
    decor_report: dict = field(default_factory=dict)
    flow_report: FlowReport | None = None


def _merge_decor_reports(reps: list[dict]) -> dict:
    """Sum per-level decor reports (counts add up, flags are OR-ed)."""
    if len(reps) == 1:
        return reps[0]
    out: dict = {}
    for rep in reps:
        for k, v in rep.items():
            out[k] = (out.get(k, False) or v) if isinstance(v, bool) else out.get(k, 0) + v
    return out


def parse_levels(levels: Sequence[str] | str) -> tuple[str, ...]:
    """Validate and canonicalise difficulty names into ascending rank order."""
    raw = levels.split(",") if isinstance(levels, str) else list(levels)
    if not raw:
        raise ValueError("at least one difficulty level is required")
    seen: set[str] = set()
    parsed: list[str] = []
    for item in raw:
        key = item.strip().lower()
        if not key:
            raise ValueError("empty difficulty name in levels")
        rank = _RANK_BY_KEY.get(key)
        if rank is None:
            raise ValueError(
                f"unknown level {item.strip()!r}; expected one of easy, normal, hard, expert, expertplus"
            )
        if rank in seen:
            raise ValueError(f"duplicate level: {rank}")
        seen.add(rank)
        parsed.append(rank)
    return tuple(sorted(parsed, key=lambda r: _RANK_ORDER[r]))


def parse_environment(name: str) -> str:
    """Resolve a case-insensitive short environment name (without 'Environment') to Info name."""
    env = _ENV_BY_SHORT.get(name.strip().lower())
    if env is None:
        short_names = ", ".join(e.removesuffix("Environment") for e in SUPPORTED_ENVIRONMENTS)
        raise ValueError(f"unknown environment {name!r}; expected one of {short_names}")
    return env


def auto_ornaments_level(nps: float) -> int:
    """Auto ornament/decor level: 2 below 4.5 nps, 1 for 4.5-7 nps, 0 above."""
    if nps < 4.5:
        return 2
    if nps <= 7.0:
        return 1
    return 0


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


def base_half_jump_beats(bpm: float, njs: float) -> float:
    """The game's half jump duration in beats before the start-beat offset is applied."""
    beat = 60.0 / bpm
    hjd = 4.0
    while njs * beat * hjd > 17.999:
        hjd /= 2.0
    return hjd


def reaction_offset(nps: float, njs: float, bpm: float) -> float:
    """Start-beat offset giving the human-median reaction time for this density.

    An offset in beats means a different time at every BPM, and the detected BPM is often half
    the real one, so the offset is solved from a reaction time in seconds instead.
    """
    i = int(np.clip(np.searchsorted(REACTION_NPS_EDGES, nps, side="right") - 1,
                    0, len(REACTION_S_BY_BIN) - 1))
    wanted = REACTION_S_BY_BIN[i] / (60.0 / bpm)
    base = base_half_jump_beats(bpm, njs)
    return round(max(wanted, MIN_HALF_JUMP_BEATS) - base, 3)


def _measured_nps(notes: Sequence[Note]) -> float:
    if len(notes) < 2:
        return 0.0
    span = notes[-1].time_s - notes[0].time_s
    return len(notes) / span if span > 0 else 0.0


def check_exact(notes: Sequence[Note], cfg: PlayabilityConfig) -> Report:
    """Playability check on the exact note times: group identical times, no frame rounding."""
    return check(exact_events(notes), cfg)


def _generate_level_notes(
    mel: np.ndarray,
    base_grid: Grid,
    target_nps: float,
    styles: Sequence[str],
    seed: int,
    timing: LoadedTiming,
    pattern: LoadedPattern,
    mask_cfg: PlayabilityConfig,
    device: str,
    *,
    repeats: BarRepeats | None = None,
    is_cancelled: CancelCheck | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Note], Grid, SampleReport, int, str, float]:
    count_scale = 1.0
    best = None  # (relative error, notes, planned, report, aligned grid)
    rounds = 0
    early_stop = ""
    for rnd in range(1, MAX_DENSITY_ROUNDS + 1):
        _check_cancel(is_cancelled)
        rounds = rnd
        curve_out: list[TimingCurve] = []
        events = predict_events(
            timing, mel, target_nps, styles, device, count_scale,
            bar_s=4 * 60.0 / base_grid.bpm if base_grid.steady and base_grid.bpm > 0 else None,
            curve_out=curve_out,
        )
        _check_cancel(is_cancelled)
        if len(events) < MIN_EVENTS:
            if best is None:
                raise GenerateError(f"only {len(events)} events found; the audio looks silent "
                                    "or has no usable rhythm")
            early_stop = f"density search stopped early: only {len(events)} events in round {rnd}"
            break
        round_grid = align_grid(base_grid, events)
        planned = snap_times(events, round_grid)
        planned, hand_rhythm = plan_rhythm(planned, curve_out[-1] if curve_out else None, repeats)
        notes, report = sample_notes(
            pattern, planned, mel, target_nps, styles, seed, mask_cfg, device,
            hand_rhythm=hand_rhythm, is_cancelled=is_cancelled, on_progress=on_progress,
        )
        measured = _measured_nps(notes)
        err = abs(measured / target_nps - 1.0) if measured > 0 else float("inf")
        if best is None or err < best[0]:
            best = (err, notes, planned, report, round_grid)
        if err <= DENSITY_TOLERANCE:
            break
        count_scale *= target_nps / measured if measured > 0 else 2.0
    _, notes, _, report, grid = best  # the grid aligned in the kept round
    if len(notes) < 2:
        raise GenerateError("no notes could be placed")

    notes = [replace(n, beat=float(to_beats(n.time_s, grid.bpm))) for n in notes]
    final_events, dropped = to_events(notes)
    if dropped or check(final_events, mask_cfg).hard:
        raise RuntimeError("generated map breaks a hard playability rule on frame times (bug)")
    if check_exact(notes, mask_cfg).hard:
        raise RuntimeError("generated map breaks a hard playability rule on exact times (bug)")
    return notes, grid, report, rounds, early_stop, _measured_nps(notes)


def generate_map(
    audio_path: Path,
    *,
    nps: float | None = None,
    styles: Sequence[str],
    seed: int,
    out_dir: Path,
    timing: LoadedTiming,
    pattern: LoadedPattern,
    mask_cfg: PlayabilityConfig,
    stats: dict,
    device: str,
    overwrite: bool = False,
    unique_folder: bool = False,
    title: str | None = None,
    artist: str = "Unknown",
    cover_src: Path | None = None,
    extra_warnings: Sequence[str] = (),
    levels: Sequence[str] | None = None,
    ornaments: int | None = 0,
    lights: int = 0,
    flow: int = 0,
    environment: str = "DefaultEnvironment",
    layer_models: LayerModels | None = None,
    layer_stats: dict | None = None,
    lights_rules: bool = False,
    ffmpeg_path: Path | None = None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> GenerateResult:
    if (nps is None) == (levels is None):
        raise ValueError("pass either nps or levels, not both or neither")
    if nps is not None and not 0 < nps < float("inf"):
        raise ValueError(f"nps must be positive, got {nps}")
    if ornaments is not None and ornaments not in (0, 1, 2, 3):
        raise ValueError(f"ornaments must be 0-3 or None, got {ornaments}")
    if lights not in (0, 1, 2, 3):
        raise ValueError(f"lights must be 0-3, got {lights}")
    if flow not in (0, 1, 2, 3):
        raise ValueError(f"flow must be 0-3, got {flow}")
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            f"unsupported environment {environment!r}; expected one of {', '.join(SUPPORTED_ENVIRONMENTS)}"
        )

    if levels is not None:
        ranks = parse_levels(levels)
        medians = stats.get("rank_median_nps", {})
        missing_ranks = [r for r in ranks if r not in medians]
        if missing_ranks:
            raise GenerateError(f"checkpoint stats missing median nps for rank(s): {', '.join(missing_ranks)}")
        targets = [(r, float(medians[r])) for r in ranks]
        if len({t for _, t in targets}) != len(targets):
            raise GenerateError("selected levels map to identical median nps")
    else:
        assert nps is not None
        targets = [(pick_rank(nps, stats), float(nps))]

    tracker = _ProgressTracker(progress)
    _check_cancel(is_cancelled)
    tracker.emit("analyzing", 0)

    audio_path = Path(audio_path)
    audio = decode_mono(audio_path, ffmpeg_path=ffmpeg_path)
    _check_cancel(is_cancelled)
    tracker.emit("analyzing", 5)

    duration_s = len(audio) / SAMPLE_RATE
    if duration_s < MIN_AUDIO_S:
        raise GenerateError(f"audio is shorter than {MIN_AUDIO_S:g} s")
    if float(np.abs(audio).max()) < SILENCE_PEAK:
        raise GenerateError("audio is silent")
    mel = log_mel(audio)
    _check_cancel(is_cancelled)
    tracker.emit("analyzing", 10)

    base_grid: Grid = detect_grid(audio)
    repeats = (bar_repeats(audio, base_grid.bpm, offset_s=base_grid.offset_s)
               if base_grid.steady and base_grid.bpm > 0 else None)
    onset_env, onset_hop_s = onset_envelope(audio) if flow > 0 else (np.zeros(0, np.float32), 0.0)
    _check_cancel(is_cancelled)
    tracker.emit("analyzing", 15)

    models = layer_models or LayerModels()
    built: list[_BuiltLevel] = []
    n_targets = len(targets)
    for idx, (rank, target_nps) in enumerate(targets):
        _check_cancel(is_cancelled)
        band_start = 15 + int(round(60 * idx / n_targets))
        band_end = 15 + int(round(60 * (idx + 1) / n_targets))
        tracker.emit("notes", band_start)

        def _on_note_progress(done: int, total: int, *, _s=band_start, _e=band_end) -> None:
            if total > 0:
                tracker.emit("notes", _s + int(round((_e - _s) * done / total)))

        notes, grid, report, rounds, early_stop, measured = _generate_level_notes(
            mel, base_grid, target_nps, styles, seed, timing, pattern, mask_cfg, device,
            repeats=repeats, is_cancelled=is_cancelled, on_progress=_on_note_progress,
        )
        if flow > 0:
            pre_flow = notes
            notes, flow_report = apply_flow(
                notes, bpm=grid.bpm, onset_env=onset_env, onset_hop_s=onset_hop_s,
                level=flow, seed=seed, mask_cfg=mask_cfg, repeats=repeats,
            )
            # Without the added stack dots the map must pass the mask profile like any base map;
            # with them (as slider tails) it must add no hard violation.
            if not final_flow_ok(pre_flow, notes, mask_cfg):
                raise RuntimeError("flow layer broke a hard playability rule (bug)")
            measured = _measured_nps(notes)
        else:
            flow_report = FlowReport(level=0)
        tracker.emit("notes", band_end)

        njs, _ = pick_njs(target_nps, stats)
        njs_offset = reaction_offset(target_nps, njs, grid.bpm)
        orn_level = auto_ornaments_level(target_nps) if ornaments is None else ornaments
        _check_cancel(is_cancelled)
        if orn_level > 0 and models.ornaments is not None and layer_stats is not None:
            arcs, chains, orn_rep = sample_ornaments(
                models.ornaments, notes, mel, target_nps, orn_level,
                layer_stats, grid.bpm, seed, device, is_cancelled=is_cancelled,
            )
        else:
            arcs, chains, orn_rep = [], [], OrnamentReport(0, 0, 0, 0, 0, 0)
        built.append(_BuiltLevel(
            rank=rank, target_nps=target_nps, notes=notes, measured_nps=measured,
            grid=grid, report=report, rounds=rounds, early_stop=early_stop,
            njs=njs, njs_offset=njs_offset, arcs=arcs, chains=chains,
            ornament_report=orn_rep, flow_report=flow_report,
        ))

    _check_cancel(is_cancelled)
    tracker.emit("layers", 80)

    densest = max(built, key=lambda lv: (lv.measured_nps, _RANK_ORDER[lv.rank]))
    for i, lv in enumerate(built):
        _check_cancel(is_cancelled)
        decor_level = auto_ornaments_level(lv.target_nps) if ornaments is None else ornaments
        lv.obstacles, lv.bombs, lv.decor_report = generate_decor(
            models, mel, lv.notes, lv.target_nps, base_grid.bpm, duration_s,
            decor_level, layer_stats, seed, device, arcs=lv.arcs, chains=lv.chains,
            is_cancelled=is_cancelled,
        )
        tracker.emit("layers", 80 + int(round(6 * (i + 1) / len(built))))
    decor_rep = _merge_decor_reports([lv.decor_report for lv in built])

    _check_cancel(is_cancelled)
    light_events, lights_rep = generate_lights(
        models, mel, densest.notes, densest.target_nps, base_grid.bpm,
        densest.grid.offset_s, duration_s, lights, layer_stats, seed, device,
        use_rules=lights_rules, is_cancelled=is_cancelled,
    )

    for lv in built:
        lv_layers = Layers(
            tuple(lv.bombs), tuple(lv.obstacles), tuple(lv.arcs), tuple(lv.chains),
            tuple(light_events), False, 0,
        )
        if check_layers(lv.notes, lv_layers):
            raise RuntimeError(f"generated map ({lv.rank}) breaks a hard layer rule (bug)")
    tracker.emit("layers", 92)

    warnings: list[str] = list(extra_warnings)
    if not base_grid.steady:
        warnings.append("no steady beat grid found; notes are not snapped to beats")
    multi = levels is not None
    for lv in built:
        prefix = f"{lv.rank}: " if multi else ""
        if lv.report.n_skipped:
            warnings.append(f"{prefix}{lv.report.n_skipped} events skipped by the playability mask")
        if lv.report.n_frame_collisions:
            warnings.append(
                f"{prefix}{lv.report.n_frame_collisions} planned events dropped: "
                "two events fell into one mel frame"
            )
        if lv.early_stop:
            warnings.append(f"{prefix}{lv.early_stop}")
        if abs(lv.measured_nps / lv.target_nps - 1.0) > DENSITY_TOLERANCE:
            warnings.append(
                f"{prefix}density {lv.measured_nps:.2f} nps is outside 10% of the requested "
                f"{lv.target_nps:g}"
            )

    raw = title or audio_path.stem
    name = re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()[:120] or "untitled"
    clean_artist = re.sub(r"[\x00-\x1f\x7f]", "", artist).strip()[:120] or "Unknown"
    style = "+".join(styles) or "unknown"
    tag = f"nps {nps:g}" if levels is None else "+".join(lv.rank for lv in built)
    base_name = f"BSM - {safe_title(name)} [{tag}, {style}]"
    folder = Path(out_dir) / base_name
    if unique_folder and not overwrite and folder.exists():
        n = 2
        while (Path(out_dir) / f"{base_name} ({n})").exists():
            n += 1
        folder = Path(out_dir) / f"{base_name} ({n})"
    level_data = [
        LevelData(
            rank=lv.rank, njs=lv.njs, njs_offset=lv.njs_offset, notes=lv.notes,
            bombs=lv.bombs, obstacles=lv.obstacles, arcs=lv.arcs, chains=lv.chains,
        )
        for lv in built
    ]
    _check_cancel(is_cancelled)
    tracker.emit("writing", 92)
    try:
        write_map_multi(
            folder, title=name, artist=clean_artist, bpm=base_grid.bpm,
            levels=level_data, lights=light_events, environment=environment,
            audio_src=audio_path, overwrite=overwrite,
            cover_src=cover_src, ffmpeg_path=ffmpeg_path,
        )
    except FileExistsError:
        raise
    except OSError as e:
        raise GenerateError(f"cannot write the map to {out_dir}: {e}") from e
    tracker.emit("writing", 100)

    level_results = tuple(
        LevelResult(
            rank=lv.rank, target_nps=lv.target_nps, n_notes=len(lv.notes),
            nps=lv.measured_nps, n_skipped=lv.report.n_skipped, density_rounds=lv.rounds,
            n_arcs=len(lv.arcs), n_chains=len(lv.chains), ornament_report=lv.ornament_report,
            n_walls=len(lv.obstacles), n_bombs=len(lv.bombs), flow_report=lv.flow_report,
        )
        for lv in built
    )
    return GenerateResult(
        folder=folder,
        n_notes=len(densest.notes),
        nps=densest.measured_nps,
        bpm=base_grid.bpm,
        grid_steady=base_grid.steady,
        rank=densest.rank,
        n_skipped=sum(lv.report.n_skipped for lv in built),
        density_rounds=max(lv.rounds for lv in built),
        warnings=tuple(warnings),
        levels=level_results,
        n_walls=len(densest.obstacles),
        n_bombs=len(densest.bombs),
        n_lights=len(light_events),
        environment=environment,
        decor_report=decor_rep,
        lights_report=lights_rep,
    )
