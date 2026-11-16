"""Strip optimizer and scaler state from checkpoints for distribution."""
from __future__ import annotations

import argparse
from pathlib import Path

from bsmaker.models.common import load_checkpoint, save_checkpoint

KEEP_KEYS = ("model", "model_config", "stats", "step", "best", "train_options")
LAYER_NAMES = ("ornaments", "decor-when", "decor-what", "lights-when", "lights-what")


def _slim_one(src: Path, dst: Path) -> None:
    payload = load_checkpoint(src, map_location="cpu")
    slimmed = {k: payload[k] for k in KEEP_KEYS if k in payload}
    save_checkpoint(dst, slimmed)


def slim_checkpoints(
    src_dir: Path,
    dst_dir: Path,
    *,
    allow_missing_layers: bool = True,
) -> dict[str, list[str]]:
    """Copy and slim required core checkpoints and optional layer checkpoints."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    slimmed: list[str] = []
    missing_layers: list[str] = []

    for core in ("timing", "pattern"):
        ck_src = src_dir / core / "best.pt"
        if not ck_src.is_file():
            raise FileNotFoundError(f"required checkpoint not found: {ck_src}")
        _slim_one(ck_src, dst_dir / core / "best.pt")
        slimmed.append(core)

    for layer in LAYER_NAMES:
        rel = f"layers/{layer}"
        ck_src = src_dir / "layers" / layer / "best.pt"
        if not ck_src.is_file():
            missing_layers.append(rel)
            continue
        _slim_one(ck_src, dst_dir / "layers" / layer / "best.pt")
        slimmed.append(rel)

    if not allow_missing_layers and missing_layers:
        raise FileNotFoundError(f"missing required layer checkpoints: {', '.join(missing_layers)}")

    return {"slimmed": slimmed, "missing_layers": missing_layers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strip optimizer state from checkpoints.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--strict-layers", action="store_true")
    args = parser.parse_args(argv)
    report = slim_checkpoints(args.src, args.dst, allow_missing_layers=not args.strict_layers)
    print(f"slimmed={report['slimmed']} missing_layers={report['missing_layers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
