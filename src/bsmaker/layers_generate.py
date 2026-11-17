"""Layer checkpoint loading and generation of decoration (walls, bombs) and lights."""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bsmaker.data.layer_stats import level_rate
from bsmaker.errors import GenerateCancelled
from bsmaker.formats.layer_tokens import (
    WALL_VOCAB, mask_to_cells, token_to_light_event, token_to_wall)
from bsmaker.formats.model import Arc, Bomb, Chain, LightEvent, Note, Obstacle
from bsmaker.lights_rules import rule_lights
from bsmaker.models.layers.spec import LIGHTS
from bsmaker.models.layers.what import LoadedWhat, load_what, sample_layer_values
from bsmaker.models.layers.when import LoadedWhen, load_when, predict_layer_events
from bsmaker.models.ornaments.sample import LoadedOrnaments, load_ornaments
from bsmaker.obstacles_rules import gameplay_obstacles
from bsmaker.playability.layers import NoteIndex, bomb_mask_ok, ornament_points, wall_ok, wall_on_ornament

log = logging.getLogger(__name__)

_LEFT_SIDE = 1
_RIGHT_SIDE = 2


@dataclass(frozen=True)
class LayerModels:
    """Loaded optional layer checkpoints (`None` when a checkpoint is absent)."""

    ornaments: LoadedOrnaments | None = None
    decor_when: LoadedWhen | None = None
    decor_what: LoadedWhat | None = None
    lights_when: LoadedWhen | None = None
    lights_what: LoadedWhat | None = None


def load_layer_models(checkpoints_dir: Path, device: str) -> tuple[LayerModels, list[str]]:
    """Load layer checkpoints from `<checkpoints_dir>/layers/<name>/best.pt`, warning if missing."""
    layers_dir = Path(checkpoints_dir) / "layers"
    specs = (
        ("ornaments", "ornaments", load_ornaments, None),
        ("decor_when", "decor-when", load_when, "decor"),
        ("decor_what", "decor-what", load_what, "decor"),
        ("lights_when", "lights-when", load_when, "lights"),
        ("lights_what", "lights-what", load_what, "lights"),
    )
    loaded: dict[str, object] = {}
    warnings: list[str] = []
    for field, name, loader, expected_layer in specs:
        ckpt = layers_dir / name / "best.pt"
        if not ckpt.is_file():
            warnings.append(f"missing {name} checkpoint: {ckpt}")
            loaded[field] = None
            continue
        model = loader(ckpt, device)
        actual_layer = getattr(model, "layer", "")
        if expected_layer is not None and actual_layer != expected_layer:
            warnings.append(
                f"{name} checkpoint has layer {actual_layer!r}, expected {expected_layer!r}: {ckpt}"
            )
            loaded[field] = None
        else:
            loaded[field] = model
    return LayerModels(**loaded), warnings  # type: ignore[arg-type]


def _check_level(level: int) -> None:
    if level not in (0, 1, 2, 3):
        raise ValueError(f"level must be 0-3, got {level}")


def _flatten_events(times_per_channel: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    chans: list[np.ndarray] = []
    for ch, ts in enumerate(times_per_channel):
        if len(ts):
            times.append(np.asarray(ts, dtype=np.float64))
            chans.append(np.full(len(ts), ch, dtype=np.int64))
    if not times:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(times), np.concatenate(chans)


def _wall_sides(ob: Obstacle) -> int:
    sides = 0
    if ob.x < 2:
        sides |= _LEFT_SIDE
    if ob.x + ob.w > 2:
        sides |= _RIGHT_SIDE
    return sides


def _overlaps_gameplay(decor_wall: Obstacle, gameplay_walls: Sequence[Obstacle]) -> bool:
    d_sides = _wall_sides(decor_wall)
    d0 = decor_wall.time_s
    d1 = decor_wall.time_s + max(decor_wall.duration_s, 0.0)
    for g in gameplay_walls:
        if (d_sides & _wall_sides(g)) == 0:
            continue
        g0 = g.time_s
        g1 = g.time_s + max(g.duration_s, 0.0)
        if d0 < g1 and g0 < d1:
            return True
    return False


def _decor_allowed_masks(
    times: np.ndarray,
    channels: np.ndarray,
    note_idx: NoteIndex,
    bpm: float,
    vocab_size: int,
    bomb_vocab: Sequence[int],
) -> np.ndarray:
    masks = np.zeros((len(times), vocab_size), dtype=bool)
    n_wall_tokens = min(WALL_VOCAB, vocab_size)
    for i, (t, ch) in enumerate(zip(times.tolist(), channels.tolist())):
        if ch < 2:
            for tok in range(n_wall_tokens):
                masks[i, tok] = wall_ok(token_to_wall(ch, tok, t, bpm), note_idx)
        else:
            for k, cell_mask in enumerate(bomb_vocab):
                tok = WALL_VOCAB + k
                if tok < vocab_size:
                    masks[i, tok] = bomb_mask_ok(cell_mask, note_idx, t)
    return masks


def _decor_report(
    *,
    walls_requested: int = 0,
    bomb_groups_requested: int = 0,
    decor_walls_placed: int = 0,
    gameplay_walls_placed: int = 0,
    bomb_groups_placed: int = 0,
    bombs_placed: int = 0,
    walls_skipped: int = 0,
    bombs_skipped: int = 0,
    dropped_by_gameplay: int = 0,
    missing_models: bool = False,
    missing_stats: bool = False,
) -> dict:
    return {
        "walls_requested": walls_requested,
        "bomb_groups_requested": bomb_groups_requested,
        "walls_placed": decor_walls_placed + gameplay_walls_placed,
        "decor_walls_placed": decor_walls_placed,
        "gameplay_walls_placed": gameplay_walls_placed,
        "bomb_groups_placed": bomb_groups_placed,
        "bombs_placed": bombs_placed,
        "walls_skipped": walls_skipped,
        "bombs_skipped": bombs_skipped,
        "skipped_by_rules": walls_skipped + bombs_skipped,
        "dropped_by_gameplay": dropped_by_gameplay,
        "missing_models": missing_models,
        "missing_stats": missing_stats,
    }


def _lights_report(
    *,
    requested: int = 0,
    placed: int = 0,
    skipped_by_rules: int = 0,
    used_rules: bool = False,
    missing_models: bool = False,
    missing_stats: bool = False,
) -> dict:
    return {
        "requested": requested,
        "placed": placed,
        "skipped_by_rules": skipped_by_rules,
        "used_rules": used_rules,
        "missing_models": missing_models,
        "missing_stats": missing_stats,
    }


@torch.inference_mode()
def generate_decor(
    models: LayerModels,
    mel: np.ndarray,
    notes: Sequence[Note],
    nps: float,
    bpm: float,
    duration_s: float,
    level: int,
    layer_stats: dict | None,
    seed: int,
    device: str,
    *,
    arcs: Sequence[Arc] = (),
    chains: Sequence[Chain] = (),
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[Obstacle], list[Bomb], dict]:
    """Generate rule-safe decorative walls, bomb groups and gameplay obstacles."""
    _check_level(level)
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    if level == 0:
        return [], [], _decor_report()
    missing_models = models.decor_when is None or models.decor_what is None
    missing_stats = layer_stats is None
    if bpm <= 0 or duration_s <= 0:
        return [], [], _decor_report(missing_models=missing_models, missing_stats=missing_stats)

    gp_walls = gameplay_obstacles(notes, bpm, duration_s, level, seed, arcs, chains)
    if missing_models or missing_stats:
        obstacles = sorted(gp_walls, key=lambda o: (o.time_s, o.x, o.y))
        return obstacles, [], _decor_report(
            gameplay_walls_placed=len(gp_walls),
            missing_models=missing_models,
            missing_stats=missing_stats,
        )

    minutes = duration_s / 60.0
    total_walls = int(round(level_rate(layer_stats, "walls", nps, level) * minutes))
    n_bombs = int(round(level_rate(layer_stats, "bomb_groups", nps, level) * minutes))
    n_left = total_walls // 2
    n_right = total_walls - n_left
    counts = (n_left, n_right, n_bombs)

    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    times_per_ch = predict_layer_events(models.decor_when, mel, notes, nps, counts, device)
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    times, channels = _flatten_events(times_per_ch)
    bomb_vocab = (
        [int(m) for m in models.decor_what.bomb_masks]
        if models.decor_what.bomb_masks is not None
        else [int(m) for m in layer_stats.get("bomb_vocab", ())]
    )
    note_idx = NoteIndex(notes)
    allowed_masks = _decor_allowed_masks(
        times, channels, note_idx, bpm, models.decor_what.config.vocab_size, bomb_vocab
    )
    tokens, _ = sample_layer_values(
        models.decor_what,
        times,
        channels,
        mel,
        notes,
        nps,
        seed,
        device,
        allowed=lambda idx, _t, _ch: allowed_masks[idx],
        is_cancelled=is_cancelled,
    )

    decor_walls: list[Obstacle] = []
    bombs: list[Bomb] = []
    walls_skipped = 0
    bombs_skipped = 0
    bomb_groups_placed = 0
    for t, ch, tok in zip(times.tolist(), channels.tolist(), tokens.tolist()):
        if tok < 0:
            if ch < 2:
                walls_skipped += 1
            else:
                bombs_skipped += 1
            continue
        if ch < 2:
            decor_walls.append(token_to_wall(ch, tok, t, bpm))
        else:
            cell_mask = bomb_vocab[tok - WALL_VOCAB]
            bomb_groups_placed += 1
            beat = t * bpm / 60.0
            for x, y in mask_to_cells(cell_mask):
                bombs.append(Bomb(t, beat, x, y))

    # walls are decoration: ornaments (bombs, arcs, chains) win over them
    points = ornament_points(bombs, arcs, chains)
    decor_walls = [w for w in decor_walls if not wall_on_ornament(w, points)]
    gp_walls = [w for w in gp_walls if not wall_on_ornament(w, points)]
    kept_decor = [w for w in decor_walls if not _overlaps_gameplay(w, gp_walls)]
    dropped_by_gameplay = len(decor_walls) - len(kept_decor)
    obstacles = sorted([*kept_decor, *gp_walls], key=lambda o: (o.time_s, o.x, o.y))
    bombs.sort(key=lambda b: (b.time_s, b.x, b.y))

    if walls_skipped or bombs_skipped:
        log.warning(
            "decor: skipped %d wall(s) and %d bomb group(s) by playability rules "
            "(placed %d/%d decor walls, %d/%d bomb groups)",
            walls_skipped,
            bombs_skipped,
            len(kept_decor),
            total_walls,
            bomb_groups_placed,
            n_bombs,
        )

    return obstacles, bombs, _decor_report(
        walls_requested=total_walls,
        bomb_groups_requested=n_bombs,
        decor_walls_placed=len(kept_decor),
        gameplay_walls_placed=len(gp_walls),
        bomb_groups_placed=bomb_groups_placed,
        bombs_placed=len(bombs),
        walls_skipped=walls_skipped,
        bombs_skipped=bombs_skipped,
        dropped_by_gameplay=dropped_by_gameplay,
    )


@torch.inference_mode()
def generate_lights(
    models: LayerModels,
    mel: np.ndarray,
    notes: Sequence[Note],
    nps: float,
    bpm: float,
    offset_s: float,
    duration_s: float,
    level: int,
    layer_stats: dict | None,
    seed: int,
    device: str,
    use_rules: bool = False,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[LightEvent], dict]:
    """Generate lightshow events via the learned models or `rule_lights` fallback."""
    _check_level(level)
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    if level == 0:
        return [], _lights_report()
    missing_models = models.lights_when is None or models.lights_what is None
    missing_stats = layer_stats is None
    if bpm <= 0 or duration_s <= 0:
        return [], _lights_report(
            used_rules=use_rules or missing_models or missing_stats,
            missing_models=missing_models,
            missing_stats=missing_stats,
        )
    if use_rules or missing_models or missing_stats:
        events = rule_lights(mel, notes, bpm, offset_s, duration_s, level, seed)
        return events, _lights_report(
            requested=len(events),
            placed=len(events),
            used_rules=True,
            missing_models=missing_models,
            missing_stats=missing_stats,
        )

    minutes = duration_s / 60.0
    counts = [
        int(round(level_rate(layer_stats, el, nps, level) * minutes))
        for el in LIGHTS.rate_elements
    ]
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    times_per_ch = predict_layer_events(models.lights_when, mel, notes, nps, counts, device)
    if is_cancelled is not None and is_cancelled():
        raise GenerateCancelled("cancelled")
    times, channels = _flatten_events(times_per_ch)
    tokens, skipped = sample_layer_values(
        models.lights_what, times, channels, mel, notes, nps, seed, device,
        is_cancelled=is_cancelled,
    )

    events: list[LightEvent] = []
    for t, ch, tok in zip(times.tolist(), channels.tolist(), tokens.tolist()):
        if tok < 0:
            continue
        ev_type, ev_val = token_to_light_event(ch, tok)
        events.append(LightEvent(t, t * bpm / 60.0, ev_type, ev_val, 1.0))
    events.sort(key=lambda e: (e.time_s, e.type))

    return events, _lights_report(
        requested=sum(counts),
        placed=len(events),
        skipped_by_rules=int(skipped),
        used_rules=False,
        missing_models=False,
    )
