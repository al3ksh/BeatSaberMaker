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
from bsmaker.formats.layer_tokens import SUPPORTED_ENVIRONMENTS
from bsmaker.formats.model import Arc, Bomb, Chain, Layers, Note, Obstacle
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
from bsmaker.formats.writer import LevelData, write_map_multi
from bsmaker.layers_generate import LayerModels, generate_decor, generate_lights
from bsmaker.models.ornaments.sample import OrnamentReport, sample_ornaments
from bsmaker.models.pattern.sample import LoadedPattern, SampleReport, sample_notes
from bsmaker.models.timing.decode import LoadedTiming, predict_events
from bsmaker.playability.checker import Report, check
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.layers import NoteIndex, bomb_mask_ok, check_layers, wall_ok

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
    groups: dict[float, dict[int, int]] = {}
    for n in notes:
        groups.setdefault(n.time_s, {})[n.color] = encode_token(n.x, n.y, n.direction)
    events = [Event(t, g.get(0, NONE_TOKEN), g.get(1, NONE_TOKEN)) for t, g in sorted(groups.items())]
    return check(events, cfg)


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
) -> tuple[list[Note], Grid, SampleReport, int, str, float]:
    count_scale = 1.0
    best = None  # (relative error, notes, planned, report, aligned grid)
    rounds = 0
    early_stop = ""
    for rnd in range(1, MAX_DENSITY_ROUNDS + 1):
        rounds = rnd
        events = predict_events(timing, mel, target_nps, styles, device, count_scale)
        if len(events) < MIN_EVENTS:
            if best is None:
                raise GenerateError(f"only {len(events)} events found; the audio looks silent "
                                    "or has no usable rhythm")
            early_stop = f"density search stopped early: only {len(events)} events in round {rnd}"
            break
        round_grid = align_grid(base_grid, events)
        planned = snap_times(events, round_grid)
        notes, report = sample_notes(pattern, planned, mel, target_nps, styles, seed, mask_cfg, device)
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


def _filter_shared_decor(
    obstacles: list[Obstacle],
    bombs: list[Bomb],
    decor_rep: dict,
    other_idxs: Sequence[NoteIndex],
) -> tuple[list[Obstacle], list[Bomb], dict]:
    """Drop shared walls/bomb groups that violate hard rules on another difficulty level."""
    rep = dict(decor_rep)
    if obstacles:
        kept_obs: list[Obstacle] = []
        dropped_obs: list[Obstacle] = []
        for o in obstacles:
            if all(wall_ok(o, idx) for idx in other_idxs):
                kept_obs.append(o)
            else:
                dropped_obs.append(o)
        if dropped_obs:
            dropped_gp = sum(1 for o in dropped_obs if o.w > 1)
            dropped_decor = len(dropped_obs) - dropped_gp
            obstacles = kept_obs
            rep["decor_walls_placed"] = max(0, rep["decor_walls_placed"] - dropped_decor)
            rep["gameplay_walls_placed"] = max(0, rep["gameplay_walls_placed"] - dropped_gp)
            rep["walls_placed"] = len(kept_obs)
            rep["walls_skipped"] += len(dropped_obs)
            rep["skipped_by_rules"] += len(dropped_obs)
    if bombs:
        by_time: dict[float, list[Bomb]] = {}
        for b in bombs:
            by_time.setdefault(b.time_s, []).append(b)
        kept_bombs: list[Bomb] = []
        groups_dropped = 0
        for t_s, grp in by_time.items():
            mask = sum(1 << (b.y * 4 + b.x) for b in grp)
            if all(bomb_mask_ok(mask, idx, t_s) for idx in other_idxs):
                kept_bombs.extend(grp)
            else:
                groups_dropped += 1
        if groups_dropped:
            bombs = kept_bombs
            rep["bomb_groups_placed"] = max(0, rep["bomb_groups_placed"] - groups_dropped)
            rep["bombs_placed"] = len(kept_bombs)
            rep["bombs_skipped"] += groups_dropped
            rep["skipped_by_rules"] += groups_dropped
    return obstacles, bombs, rep


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
    title: str | None = None,
    extra_warnings: Sequence[str] = (),
    levels: Sequence[str] | None = None,
    ornaments: int | None = 0,
    lights: int = 0,
    environment: str = "DefaultEnvironment",
    layer_models: LayerModels | None = None,
    layer_stats: dict | None = None,
    lights_rules: bool = False,
) -> GenerateResult:
    if (nps is None) == (levels is None):
        raise ValueError("pass either nps or levels, not both or neither")
    if nps is not None and not 0 < nps < float("inf"):
        raise ValueError(f"nps must be positive, got {nps}")
    if ornaments is not None and ornaments not in (0, 1, 2, 3):
        raise ValueError(f"ornaments must be 0-3 or None, got {ornaments}")
    if lights not in (0, 1, 2, 3):
        raise ValueError(f"lights must be 0-3, got {lights}")
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

    audio_path = Path(audio_path)
    audio = decode_mono(audio_path)
    duration_s = len(audio) / SAMPLE_RATE
    if duration_s < MIN_AUDIO_S:
        raise GenerateError(f"audio is shorter than {MIN_AUDIO_S:g} s")
    if float(np.abs(audio).max()) < SILENCE_PEAK:
        raise GenerateError("audio is silent")
    mel = log_mel(audio)
    base_grid: Grid = detect_grid(audio)

    models = layer_models or LayerModels()
    built: list[_BuiltLevel] = []
    for rank, target_nps in targets:
        notes, grid, report, rounds, early_stop, measured = _generate_level_notes(
            mel, base_grid, target_nps, styles, seed, timing, pattern, mask_cfg, device,
        )
        njs, _ = pick_njs(target_nps, stats)
        njs_offset = reaction_offset(target_nps, njs, grid.bpm)
        orn_level = auto_ornaments_level(target_nps) if ornaments is None else ornaments
        if orn_level > 0 and models.ornaments is not None and layer_stats is not None:
            arcs, chains, orn_rep = sample_ornaments(
                models.ornaments, notes, mel, target_nps, orn_level,
                layer_stats, grid.bpm, seed, device,
            )
        else:
            arcs, chains, orn_rep = [], [], OrnamentReport(0, 0, 0, 0, 0, 0)
        built.append(_BuiltLevel(
            rank=rank, target_nps=target_nps, notes=notes, measured_nps=measured,
            grid=grid, report=report, rounds=rounds, early_stop=early_stop,
            njs=njs, njs_offset=njs_offset, arcs=arcs, chains=chains,
            ornament_report=orn_rep,
        ))

    densest = max(built, key=lambda lv: (lv.measured_nps, _RANK_ORDER[lv.rank]))
    decor_level = auto_ornaments_level(densest.target_nps) if ornaments is None else ornaments
    obstacles, bombs, decor_rep = generate_decor(
        models, mel, densest.notes, densest.target_nps, base_grid.bpm, duration_s,
        decor_level, layer_stats, seed, device,
    )
    if len(built) > 1 and (obstacles or bombs):
        other_idxs = [NoteIndex(lv.notes) for lv in built if lv is not densest]
        obstacles, bombs, decor_rep = _filter_shared_decor(obstacles, bombs, decor_rep, other_idxs)

    light_events, lights_rep = generate_lights(
        models, mel, densest.notes, densest.target_nps, base_grid.bpm,
        densest.grid.offset_s, duration_s, lights, layer_stats, seed, device,
        use_rules=lights_rules,
    )

    for lv in built:
        lv_layers = Layers(
            tuple(bombs), tuple(obstacles), tuple(lv.arcs), tuple(lv.chains),
            tuple(light_events), False, 0,
        )
        if check_layers(lv.notes, lv_layers):
            raise RuntimeError(f"generated map ({lv.rank}) breaks a hard layer rule (bug)")

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
    style = "+".join(styles) or "unknown"
    tag = f"nps {nps:g}" if levels is None else "+".join(lv.rank for lv in built)
    folder = Path(out_dir) / f"BSM - {safe_title(name)} [{tag}, {style}]"
    level_data = [
        LevelData(
            rank=lv.rank, njs=lv.njs, njs_offset=lv.njs_offset, notes=lv.notes,
            bombs=bombs, obstacles=obstacles, arcs=lv.arcs, chains=lv.chains,
        )
        for lv in built
    ]
    try:
        write_map_multi(
            folder, title=name, artist="Unknown", bpm=base_grid.bpm,
            levels=level_data, lights=light_events, environment=environment,
            audio_src=audio_path, overwrite=overwrite,
        )
    except FileExistsError:
        raise
    except OSError as e:
        raise GenerateError(f"cannot write the map to {out_dir}: {e}") from e

    level_results = tuple(
        LevelResult(
            rank=lv.rank, target_nps=lv.target_nps, n_notes=len(lv.notes),
            nps=lv.measured_nps, n_skipped=lv.report.n_skipped, density_rounds=lv.rounds,
            n_arcs=len(lv.arcs), n_chains=len(lv.chains), ornament_report=lv.ornament_report,
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
        n_walls=len(obstacles),
        n_bombs=len(bombs),
        n_lights=len(light_events),
        environment=environment,
        decor_report=decor_rep,
        lights_report=lights_rep,
    )
