"""Shared model-loading and map-generation pipeline for CLI and desktop engine."""
from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from bsmaker.audio.decode import require_ffmpeg
from bsmaker.config import Config
from bsmaker.data.layer_stats import load_layer_stats
from bsmaker.generate import (
    CancelCheck,
    GenerateError,
    GenerateResult,
    ProgressCallback,
    auto_ornaments_level,
    generate_map,
    parse_levels,
    require_stats,
)
from bsmaker.layers_generate import LayerModels, load_layer_models
from bsmaker.models.pattern.sample import LoadedPattern, load_pattern
from bsmaker.models.timing.decode import LoadedTiming, load_timing
from bsmaker.playability.config import PlayabilityConfig, load_playability_config

NO_PLAYABILITY_WARNING = (
    "playability.toml not found next to bsmaker.toml; using built-in mask defaults"
)


class MissingModelsError(GenerateError):
    """Required timing or pattern checkpoint is missing."""


@dataclass
class LoadedPipeline:
    timing: LoadedTiming
    pattern: LoadedPattern
    mask_cfg: PlayabilityConfig
    layer_models: LayerModels
    layer_warnings: tuple[str, ...]
    layer_stats: dict | None
    layer_stats_missing_path: Path | None
    playability_missing: bool
    device: str


def resolve_device(device: str | None) -> str:
    """Resolve and validate compute device ('cuda', 'cpu', or None for auto)."""
    if device not in (None, "cuda", "cpu"):
        raise ValueError(f"unsupported device {device!r}; expected cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise GenerateError("--device cuda requested but CUDA is not available; use --device cpu")
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_pipeline(
    cfg: Config,
    device: str | None = None,
    *,
    timing_ckpt: Path | None = None,
    pattern_ckpt: Path | None = None,
    load_layers: bool = True,
) -> LoadedPipeline:
    """Load timing, pattern, playability config, and optional layer models/stats."""
    if device not in (None, "cuda", "cpu"):
        raise ValueError(f"unsupported device {device!r}; expected cuda or cpu")

    timing_ck = timing_ckpt or cfg.checkpoints_dir / "timing" / "best.pt"
    pattern_ck = pattern_ckpt or cfg.checkpoints_dir / "pattern" / "best.pt"
    for ck in (timing_ck, pattern_ck):
        if not ck.is_file():
            raise MissingModelsError(f"checkpoint not found: {ck}; train the models first")

    dev = resolve_device(device)
    timing = load_timing(timing_ck, dev)
    pattern = load_pattern(pattern_ck, dev)
    require_stats(timing.stats)
    require_stats(pattern.stats, ("mel_mean", "mel_std"))

    play_path = cfg.playability_path
    playability_missing = play_path is None
    mask_cfg = load_playability_config(play_path, "mask")

    if load_layers:
        layer_models, ckpt_warnings = load_layer_models(cfg.checkpoints_dir, dev)
        layer_warnings = tuple(ckpt_warnings)
        ls_path = cfg.layer_stats_path
        if ls_path is not None:
            layer_stats = load_layer_stats(ls_path)
            layer_stats_missing_path = None
        else:
            layer_stats = None
            layer_stats_missing_path = (
                cfg._layer_stats_path if cfg._layer_stats_path is not None
                else (cfg.base_dir / "layer_stats.json")
            )
    else:
        layer_models = LayerModels()
        layer_warnings = ()
        layer_stats = None
        layer_stats_missing_path = None

    return LoadedPipeline(
        timing=timing,
        pattern=pattern,
        mask_cfg=mask_cfg,
        layer_models=layer_models,
        layer_warnings=layer_warnings,
        layer_stats=layer_stats,
        layer_stats_missing_path=layer_stats_missing_path,
        playability_missing=playability_missing,
        device=dev,
    )


def pipeline_to_device(pipeline: LoadedPipeline, device: str) -> None:
    """Move all loaded models in `pipeline` to `device` and update `pipeline.device`."""
    dev = torch.device(device)
    pipeline.timing.model.to(dev)
    pipeline.pattern.model.to(dev)
    lm = pipeline.layer_models
    for slot in (lm.ornaments, lm.decor_when, lm.decor_what, lm.lights_when, lm.lights_what):
        if slot is not None:
            slot.model.to(dev)
    pipeline.device = device


def pipeline_to_cpu(pipeline: LoadedPipeline) -> None:
    """Offload all pipeline models to CPU and release CUDA cache if available."""
    pipeline_to_device(pipeline, "cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_generate(
    pipeline: LoadedPipeline,
    audio_path: Path,
    *,
    out_dir: Path,
    nps: float | None = None,
    levels: Sequence[str] | None = None,
    styles: Sequence[str] = (),
    seed: int = 0,
    overwrite: bool = False,
    unique_folder: bool = False,
    title: str | None = None,
    artist: str = "Unknown",
    cover_src: Path | None = None,
    ornaments: int | None = None,
    lights: int = 0,
    flow: int | None = None,
    environment: str = "DefaultEnvironment",
    lights_rules: bool = False,
    ffmpeg_path: Path | None = None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> GenerateResult:
    """Run map generation using a pre-loaded `LoadedPipeline`."""
    require_ffmpeg(ffmpeg_path)
    if (nps is None) == (levels is None):
        raise ValueError("pass either nps or levels, not both or neither")

    parsed_levels: tuple[str, ...] | None = None
    if levels is not None:
        parsed_levels = parse_levels(levels)
        medians = pipeline.timing.stats.get("rank_median_nps", {})
        missing_r = [r for r in parsed_levels if r not in medians]
        if missing_r:
            raise GenerateError(
                f"checkpoint stats missing median nps for rank(s): {', '.join(missing_r)}"
            )
        target_nps_list = [float(medians[r]) for r in parsed_levels]
    else:
        assert nps is not None
        target_nps_list = [float(nps)]

    need_orn_decor = (ornaments > 0) if ornaments is not None else any(
        auto_ornaments_level(x) > 0 for x in target_nps_list
    )
    need_lights_model = lights > 0 and not lights_rules
    need_layer_stats = need_orn_decor or need_lights_model

    lm = pipeline.layer_models
    active_lm = LayerModels(
        ornaments=lm.ornaments if need_orn_decor else None,
        decor_when=lm.decor_when if need_orn_decor else None,
        decor_what=lm.decor_what if need_orn_decor else None,
        lights_when=lm.lights_when if need_lights_model else None,
        lights_what=lm.lights_what if need_lights_model else None,
    )

    extra: list[str] = []
    if pipeline.playability_missing:
        extra.append(NO_PLAYABILITY_WARNING)

    needed_slots = (
        (("ornaments", "decor-when", "decor-what") if need_orn_decor else ())
        + (("lights-when", "lights-what") if need_lights_model else ())
    )
    for w in pipeline.layer_warnings:
        if any(
            w.startswith((f"missing {slot} checkpoint:", f"{slot} checkpoint "))
            for slot in needed_slots
        ):
            extra.append(w)

    if need_layer_stats and pipeline.layer_stats is None and pipeline.layer_stats_missing_path is not None:
        extra.append(
            f"missing layer_stats.json: {pipeline.layer_stats_missing_path}; learned layers switched off"
        )

    return generate_map(
        audio_path,
        nps=nps,
        styles=tuple(styles),
        seed=seed,
        out_dir=out_dir,
        timing=pipeline.timing,
        pattern=pipeline.pattern,
        mask_cfg=pipeline.mask_cfg,
        stats=pipeline.timing.stats,
        device=pipeline.device,
        overwrite=overwrite,
        unique_folder=unique_folder,
        title=title,
        artist=artist,
        cover_src=cover_src,
        extra_warnings=extra,
        levels=parsed_levels,
        ornaments=ornaments,
        lights=lights,
        flow=flow,
        environment=environment,
        layer_models=active_lm,
        layer_stats=pipeline.layer_stats,
        lights_rules=lights_rules,
        ffmpeg_path=ffmpeg_path,
        progress=progress,
        is_cancelled=is_cancelled,
    )
