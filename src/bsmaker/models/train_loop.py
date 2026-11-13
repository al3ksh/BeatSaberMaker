"""Generic training loop shared by the timing and note models."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def _load_resume(out_dir: Path, log: Callable[[str], None]) -> dict | None:
    """Newest loadable checkpoint (read on the CPU).

    An unloadable `last.pt` is never deleted: when `prev.pt` is used instead, it is renamed to
    `last.corrupt.pt`, so the next save does not rotate it over the good `prev.pt`.
    """
    errors = []
    for name in ("last.pt", "prev.pt"):
        p = out_dir / name
        if not p.exists():
            continue
        try:
            ckpt = load_checkpoint(p, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - truncated or corrupt file: try the older one
            errors.append(f"{name}: {type(exc).__name__}")
            continue
        if errors and name == "prev.pt":
            (out_dir / "last.pt").replace(out_dir / "last.corrupt.pt")
            log("last.pt could not be loaded; kept as last.corrupt.pt, resuming from prev.pt")
        return ckpt
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
                 log: Callable[[str], None], stats: dict | None = None,
                 sample_weights: np.ndarray | None = None,
                 train_options: dict | None = None) -> dict:
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
    ckpt = _load_resume(out_dir, log)
    stats = dict(stats or {})
    train_options = dict(train_options or {})
    if ckpt is not None:
        if ckpt["model_config"] != model_config:
            raise ValueError("checkpoint model_config differs from the requested model")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if use_amp and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        if "train_options" in ckpt:
            old = ckpt["train_options"]
            diffs = [f"{k}: checkpoint {old.get(k)!r}, requested {train_options.get(k)!r}"
                     for k in sorted(set(old) | set(train_options))
                     if old.get(k) != train_options.get(k)]
            if diffs:
                raise ValueError("checkpoint train_options differ from the requested ones ("
                                 + "; ".join(diffs) + "); use another --out")
        else:
            log("checkpoint has no train_options (older checkpoint); continuing")
        step, best = int(ckpt["step"]), ckpt["best"]
        log(f"resumed from step {step}")
    try:  # a best.pt of the same model config that beats the resumed record is kept
        saved_ck = load_checkpoint(out_dir / "best.pt", map_location="cpu")
    except Exception:  # noqa: BLE001 - missing or unreadable: ignore
        saved_ck = None
    if saved_ck is not None and saved_ck.get("best") is not None:
        if saved_ck.get("model_config") != model_config:
            log("ignoring best.pt: it was saved with a different model_config")
        elif best is None or better(saved_ck["best"], best):
            best = saved_ck["best"]

    def payload() -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "model_config": model_config,
                "stats": stats, "best": best, "train_options": train_options}

    def save_last() -> None:
        last = out_dir / "last.pt"
        if last.exists():
            (out_dir / "prev.pt").unlink(missing_ok=True)
            last.replace(out_dir / "prev.pt")
        save_checkpoint(last, payload())

    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(
        int(np.random.SeedSequence([0, step]).generate_state(1, dtype=np.uint64)[0] >> 1))
    if sample_weights is None:
        order: dict = {"shuffle": True}
    else:
        order = {"sampler": WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ds),
            replacement=True, generator=gen)}
    loader = DataLoader(train_ds, batch_size=s.batch_size, drop_last=True, **order,
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
