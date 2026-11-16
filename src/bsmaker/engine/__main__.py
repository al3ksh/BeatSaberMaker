"""Standalone entry point: python -m bsmaker.engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bsmaker.engine.server import EngineConfig, run_engine


def _reconfigure_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_utf8_stdio()
    parser = argparse.ArgumentParser(description="BeatSaberMaker JSON-lines engine server")
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--playability", type=Path, default=None)
    parser.add_argument("--layer-stats", type=Path, default=None)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--ytdlp", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = EngineConfig(
        checkpoints_dir=args.checkpoints_dir,
        playability_path=args.playability,
        layer_stats_path=args.layer_stats,
        ffmpeg_path=args.ffmpeg,
        ytdlp_path=args.ytdlp,
        device=None if args.device == "auto" else args.device,
        log_path=args.log_file,
    )
    return run_engine(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
