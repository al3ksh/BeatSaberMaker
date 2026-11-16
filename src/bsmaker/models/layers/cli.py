"""`train layer-when` and `train layer-what`; wired into the CLI with `register`."""
from __future__ import annotations

from pathlib import Path

import typer

from bsmaker.config import load_config

LAYERS = ("lights", "decor")
VAL_CAP = 256


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _weights(entries: list, stats: dict, weighting: str):
    import numpy as np

    from bsmaker.models.sampling import entry_weights

    if weighting not in ("none", "quality"):
        _fail("--weighting must be none or quality", 2)
    print(f"weighting: {weighting}", flush=True)
    if weighting == "none":
        return None
    w = entry_weights(entries, stats)
    nz = w[w > 0]
    if len(nz) == 0:
        _fail("every training entry has sampling weight 0 (all excluded); "
              "check the quality fields in the event cache", 2)
    print(f"weighting: {len(w) - len(nz)} of {len(w)} entries excluded (weight 0); "
          f"non-zero weights min {nz.min():.3f} median {np.median(nz):.3f} max {nz.max():.3f}",
          flush=True)
    return w


def _prepare(layer: str, lr: float, device: str):
    """Validate the arguments and load caches, stats and the layer spec."""
    import torch

    from bsmaker.data.events_cache import EventCache
    from bsmaker.data.layer_stats import load_layer_stats
    from bsmaker.data.layers_cache import LayerCache
    from bsmaker.data.stats import load_stats
    from bsmaker.models.layers.spec import get_spec

    if layer not in LAYERS:
        _fail(f"--layer must be one of {', '.join(LAYERS)}", 2)
    if not 0 < lr < float("inf"):
        _fail("--lr must be positive", 2)
    cfg = load_config()
    stats_path = cfg.base_dir / "dataset_stats.json"
    ls_path = cfg.base_dir / "layer_stats.json"
    if not (cfg.events_dir / "index.json").is_file() or not (cfg.events_dir / "events.npz").is_file():
        _fail(f"event cache not found in {cfg.events_dir}; run `bsmaker data build-events`")
    if not (cfg.layers_dir / "index.json").is_file() or not (cfg.layers_dir / "layers.npz").is_file():
        _fail(f"layer cache not found in {cfg.layers_dir}; run `bsmaker data build-layers`")
    if not stats_path.is_file():
        _fail(f"{stats_path} not found; run `bsmaker data build-events`")
    if layer == "decor" and not ls_path.is_file():
        _fail(f"{ls_path} not found; run `bsmaker data build-layers`")
    if device == "cuda" and not torch.cuda.is_available():
        _fail("--device cuda requested but CUDA is not available; use --device cpu")
    event_cache = EventCache.load(cfg.events_dir)
    layer_cache = LayerCache.load(cfg.layers_dir)
    stats = load_stats(stats_path)
    layer_stats = load_layer_stats(ls_path) if ls_path.is_file() else None
    spec = get_spec(layer, layer_stats)
    if layer == "decor":
        stats = dict(stats, bomb_masks=[int(m) for m in layer_stats["bomb_vocab"]])
    return cfg, event_cache, layer_cache, stats, spec


def _entries(event_cache, layer_cache, spec):
    from bsmaker.models.layers.spec import eligible_entries, evenly_spaced

    train = eligible_entries(event_cache.entries, layer_cache, spec, "train")
    val = evenly_spaced(eligible_entries(event_cache.entries, layer_cache, spec, "val"), VAL_CAP)
    if not train or not val:
        _fail(f"no eligible train or val difficulties for layer {spec.name}")
    return train, val


def register(train_app: typer.Typer) -> None:
    """Add `layer-when` and `layer-what` to the `train` typer app."""

    @train_app.command("layer-when")
    def train_layer_when(
            layer: str = typer.Option(..., "--layer", help="lights or decor."),
            max_steps: int = typer.Option(30000, "--max-steps"),
            batch_size: int = typer.Option(32, "--batch-size"),
            num_workers: int = typer.Option(4, "--num-workers"),
            device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
            eval_every: int = typer.Option(2000, "--eval-every"),
            save_every: int = typer.Option(1000, "--save-every"),
            weighting: str = typer.Option("quality", "--weighting", help="none or quality."),
            lr: float = typer.Option(3e-4, "--lr"),
            out: Path | None = typer.Option(None, "--out", help="Checkpoint folder (default: "
                                            "<data_dir>/checkpoints/layers/<layer>-when).")
    ) -> None:
        """Train the multi-channel layer timing model; resumes from the checkpoint folder."""
        from dataclasses import asdict
        from functools import partial

        from bsmaker.models.layers.when import (
            WhenConfig, WhenModel, WhenWindows, when_better, when_eval, when_step)
        from bsmaker.models.train_loop import TrainSettings, run_training

        cfg, event_cache, layer_cache, stats, spec = _prepare(layer, lr, device)
        train_entries, val_entries = _entries(event_cache, layer_cache, spec)
        mcfg = WhenConfig(n_channels=spec.n_channels)
        weights = _weights(train_entries, stats, weighting)
        common = dict(features_dir=cfg.features_dir, stats=stats, spec=spec, window=mcfg.window,
                      layers_dir=cfg.layers_dir, events_dir=cfg.events_dir)
        train_ds = WhenWindows(layer_cache, event_cache, train_entries, seed=0, **common)
        val_ds = WhenWindows(layer_cache, event_cache, val_entries, seed=1, fixed=True, **common)
        settings = TrainSettings(max_steps=max_steps, batch_size=batch_size,
                                 num_workers=num_workers, device=device, eval_every=eval_every,
                                 save_every=save_every, lr=lr)
        options = {"layer": layer, "weighting": weighting, "lr": lr, "max_steps": max_steps,
                   "batch_size": batch_size}
        model = WhenModel(mcfg)
        print(f"layer {layer}: {sum(p.numel() for p in model.parameters()):,} parameters; "
              f"{len(train_entries)} train, {len(val_entries)} val difficulties", flush=True)
        result = run_training(
            model, train_ds, val_ds, when_step,
            partial(when_eval, min_gap_frames=spec.min_gap_frames),
            out if out is not None else cfg.checkpoints_dir / "layers" / f"{layer}-when",
            settings, asdict(mcfg), when_better, lambda m: print(m, flush=True), stats=stats,
            sample_weights=weights, train_options=options)
        print(f"done at step {result['step']}, best {result['best']}")

    @train_app.command("layer-what")
    def train_layer_what(
            layer: str = typer.Option(..., "--layer", help="lights or decor."),
            max_steps: int = typer.Option(40000, "--max-steps"),
            batch_size: int = typer.Option(32, "--batch-size"),
            num_workers: int = typer.Option(4, "--num-workers"),
            device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
            eval_every: int = typer.Option(2000, "--eval-every"),
            save_every: int = typer.Option(1000, "--save-every"),
            weighting: str = typer.Option("quality", "--weighting", help="none or quality."),
            lr: float = typer.Option(3e-4, "--lr"),
            out: Path | None = typer.Option(None, "--out", help="Checkpoint folder (default: "
                                            "<data_dir>/checkpoints/layers/<layer>-what).")
    ) -> None:
        """Train the layer value model; resumes from the checkpoint folder."""
        from dataclasses import asdict

        from bsmaker.models.layers.spec import channel_mask
        from bsmaker.models.layers.what import (
            WhatConfig, WhatModel, WhatSequences, what_better, what_eval, what_step)
        from bsmaker.models.train_loop import TrainSettings, run_training

        cfg, event_cache, layer_cache, stats, spec = _prepare(layer, lr, device)
        train_entries, val_entries = _entries(event_cache, layer_cache, spec)
        mcfg = WhatConfig(n_channels=spec.n_channels, vocab_size=spec.vocab_size)
        weights = _weights(train_entries, stats, weighting)
        common = dict(features_dir=cfg.features_dir, stats=stats, spec=spec, context=mcfg.context,
                      layers_dir=cfg.layers_dir, events_dir=cfg.events_dir)
        train_ds = WhatSequences(layer_cache, event_cache, train_entries, seed=0, **common)
        val_ds = WhatSequences(layer_cache, event_cache, val_entries, seed=1, fixed=True, **common)
        settings = TrainSettings(max_steps=max_steps, batch_size=batch_size,
                                 num_workers=num_workers, device=device, eval_every=eval_every,
                                 save_every=save_every, lr=lr)
        options = {"layer": layer, "weighting": weighting, "lr": lr, "max_steps": max_steps,
                   "batch_size": batch_size}
        model = WhatModel(mcfg, channel_mask(spec))
        print(f"layer {layer}: {sum(p.numel() for p in model.parameters()):,} parameters; "
              f"{len(train_entries)} train, {len(val_entries)} val difficulties", flush=True)
        result = run_training(
            model, train_ds, val_ds, what_step, what_eval,
            out if out is not None else cfg.checkpoints_dir / "layers" / f"{layer}-what",
            settings, asdict(mcfg), what_better, lambda m: print(m, flush=True), stats=stats,
            sample_weights=weights, train_options=options)
        print(f"done at step {result['step']}, best {result['best']}")
