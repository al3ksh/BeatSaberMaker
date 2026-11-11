"""Generic training loop shared by the timing and note models."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from bsmaker.models.common import load_checkpoint, save_checkpoint


@dataclass
class TrainSettings:
    max_steps: int
    batch_size: int = 32
    lr: float = 3e-4
    warmup: int = 1000
    eval_every: int = 2000
    save_every: int = 1000
    num_workers: int = 4
    amp: bool = True
    device: str = "cuda"


def _lr_factor(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _load_resume(out_dir: Path, device: str) -> dict | None:
    errors = []
    for name in ("last.pt", "prev.pt"):
        p = out_dir / name
        if not p.exists():
            continue
        try:
            return load_checkpoint(p, map_location=device)
        except Exception as exc:  # noqa: BLE001 - truncated or corrupt file: try the older one
            errors.append(f"{name}: {type(exc).__name__}")
    if errors:
        raise RuntimeError(f"no loadable checkpoint in {out_dir} ({'; '.join(errors)})")
    return None


def _batches(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def run_training(model: nn.Module, train_ds: Dataset, val_ds: Dataset,
                 step_fn: Callable[[nn.Module, dict], Tensor],
                 eval_fn: Callable[[nn.Module, DataLoader], dict[str, float]],
                 out_dir: Path, settings: TrainSettings, model_config: dict,
                 better: Callable[[dict, dict | None], bool],
                 log: Callable[[str], None], stats: dict | None = None) -> dict:
    """Train until `settings.max_steps`; returns `{"step": int, "best": dict | None}`."""
    s = settings
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(s.device)
    use_amp = s.amp and device.type == "cuda"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s.lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step = 0
    best: dict | None = None
    ckpt = _load_resume(out_dir, s.device)
    stats = dict(stats or {})
    if ckpt is not None:
        if ckpt["model_config"] != model_config:
            raise ValueError("checkpoint model_config differs from the requested model")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if use_amp and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        step, best = int(ckpt["step"]), ckpt["best"]
        log(f"resumed from step {step}")

    def payload() -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "model_config": model_config,
                "stats": stats, "best": best}

    def save_last() -> None:
        last = out_dir / "last.pt"
        if last.exists():
            (out_dir / "prev.pt").unlink(missing_ok=True)
            last.replace(out_dir / "prev.pt")
        save_checkpoint(last, payload())

    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(step)
    loader = DataLoader(train_ds, batch_size=s.batch_size, shuffle=True, drop_last=True,
                        num_workers=s.num_workers, persistent_workers=s.num_workers > 0,
                        pin_memory=pin, generator=gen)
    if len(loader) == 0:
        raise ValueError("training set is smaller than one batch")
    val_loader = DataLoader(val_ds, batch_size=s.batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)

    def evaluate() -> None:
        nonlocal best
        model.eval()
        metrics = eval_fn(model, val_loader)
        model.train()
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **metrics}) + "\n")
        log(f"eval step {step} " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
        if better(metrics, best):
            best = dict(metrics)
            save_checkpoint(out_dir / "best.pt", payload())

    model.train()
    batches = _batches(loader)
    t0 = time.monotonic()
    t_log, n_log = t0, 0
    saved_at = step
    while step < s.max_steps:
        batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
        for g in opt.param_groups:
            g["lr"] = s.lr * _lr_factor(step, s.warmup, s.max_steps)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = step_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        n_log += 1
        if step == 1 or step % 100 == 0 or step == s.max_steps:
            now = time.monotonic()
            rate = n_log / max(now - t_log, 1e-9)
            log(f"step {step} loss {loss.item():.4f} {rate:.2f} steps/s")
            t_log, n_log = now, 0
        if step % s.eval_every == 0 or step == s.max_steps:
            evaluate()
        if step % s.save_every == 0 or step == s.max_steps:
            save_last()
            saved_at = step
    if saved_at != step or not (out_dir / "last.pt").exists():
        save_last()
    return {"step": step, "best": best}
