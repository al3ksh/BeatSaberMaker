"""`bsmaker train ornaments` (registered on the train app by the CLI module)."""
from __future__ import annotations

from pathlib import Path

import typer

from bsmaker.config import load_config


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def register(train_app: typer.Typer) -> None:
    @train_app.command("ornaments")
    def train_ornaments(
            max_steps: int = typer.Option(30000, "--max-steps", help="Optimizer steps."),
            batch_size: int = typer.Option(32, "--batch-size"),
            num_workers: int = typer.Option(4, "--num-workers"),
            device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
            eval_every: int = typer.Option(2000, "--eval-every"),
            save_every: int = typer.Option(1000, "--save-every"),
            weighting: str = typer.Option("quality", "--weighting", help="none or quality."),
            lr: float = typer.Option(3e-4, "--lr", help="Peak learning rate."),
            out: Path | None = typer.Option(None, "--out", help="Checkpoint folder (default: "
                                            "<data_dir>/checkpoints/layers/ornaments).")) -> None:
        """Train the ornament tagger on the layer cache; resumes from the checkpoint folder."""
        from dataclasses import asdict

        import torch

        from bsmaker.data.events_cache import EventCache
        from bsmaker.data.layers_cache import LayerCache
        from bsmaker.data.stats import load_stats
        from bsmaker.models.ornaments.dataset import OrnamentNotes
        from bsmaker.models.ornaments.model import (
            OrnamentConfig, OrnamentModel, ornament_better, ornament_eval, ornament_step)
        from bsmaker.models.sampling import entry_weights
        from bsmaker.models.train_loop import TrainSettings, run_training

        if not 0 < lr < float("inf"):
            _fail("--lr must be positive", 2)
        if weighting not in ("none", "quality"):
            _fail("--weighting must be none or quality", 2)
        cfg = load_config()
        stats_path = cfg.base_dir / "dataset_stats.json"
        layers_dir = cfg.layers_dir
        if not (cfg.events_dir / "index.json").is_file() or not (cfg.events_dir / "events.npz").is_file():
            _fail(f"event cache not found in {cfg.events_dir}; run `bsmaker data build-events`")
        if not (layers_dir / "index.json").is_file() or not (layers_dir / "layers.npz").is_file():
            _fail(f"layer cache not found in {layers_dir}; run `bsmaker data build-layers`")
        if not stats_path.is_file():
            _fail(f"{stats_path} not found; run `bsmaker data build-events`")
        if device == "cuda" and not torch.cuda.is_available():
            _fail("--device cuda requested but CUDA is not available; use --device cpu")
        events = EventCache.load(cfg.events_dir)
        layers = LayerCache.load(layers_dir)
        stats = load_stats(stats_path)
        mcfg = OrnamentConfig()
        common = dict(features_dir=cfg.features_dir, stats=stats, context=mcfg.context,
                      layers_dir=layers_dir)
        train_ds = OrnamentNotes(layers, events.select("train"), seed=0, **common)
        val_all = OrnamentNotes(layers, events.select("val"), seed=1, fixed=True, **common).entries
        if not len(train_ds) or not val_all:
            _fail("no train or val difficulties with arcs or chains in the layer cache")
        stride = max(1, len(val_all) // 256)
        val_ds = OrnamentNotes(layers, val_all[::stride][:256], seed=1, fixed=True, **common)
        weights = None
        print(f"weighting: {weighting}", flush=True)
        if weighting == "quality":
            weights = entry_weights(train_ds.entries, stats)
            if not (weights > 0).any():
                _fail("every training entry has sampling weight 0 (all excluded)", 2)
            print(f"weighting: {int((weights == 0).sum())} of {len(weights)} entries excluded",
                  flush=True)
        settings = TrainSettings(max_steps=max_steps, batch_size=batch_size,
                                 num_workers=num_workers, device=device, eval_every=eval_every,
                                 save_every=save_every, lr=lr)
        options = {"weighting": weighting, "lr": lr, "max_steps": max_steps,
                   "batch_size": batch_size}
        model = OrnamentModel(mcfg)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"ornaments: {n_par:,} parameters; train {len(train_ds)} val {len(val_ds)}; "
              + ", ".join(f"{k}={v}" for k, v in options.items()), flush=True)
        result = run_training(
            model, train_ds, val_ds, ornament_step, ornament_eval,
            out if out is not None else cfg.checkpoints_dir / "layers" / "ornaments", settings,
            asdict(mcfg), ornament_better, lambda m: print(m, flush=True), stats=stats,
            sample_weights=weights, train_options=options)
        print(f"done at step {result['step']}, best {result['best']}")
