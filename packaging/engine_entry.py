"""PyInstaller entry point for bsmaker-engine.exe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bsmaker.engine.server import EngineConfig, run_engine


def _reconfigure_utf8() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _default_pack_root() -> Path:
    return Path(sys.executable).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    _reconfigure_utf8()
    parser = argparse.ArgumentParser(description="BeatSaberMaker frozen engine server")
    parser.add_argument("--resources-dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--playability", type=Path, default=None)
    parser.add_argument("--layer-stats", type=Path, default=None)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--ytdlp", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.resources_dir or _default_pack_root()
    models_dir = root / "models" if (root / "models").is_dir() else root
    tools_dir = root / "tools"

    checkpoints_dir = args.checkpoints_dir or (models_dir / "checkpoints")
    playability = args.playability or (
        models_dir / "playability.toml"
        if (models_dir / "playability.toml").is_file()
        else None
    )
    layer_stats = args.layer_stats or (
        models_dir / "layer_stats.json"
        if (models_dir / "layer_stats.json").is_file()
        else None
    )
    ffmpeg = args.ffmpeg or (
        tools_dir / "ffmpeg.exe" if (tools_dir / "ffmpeg.exe").is_file() else None
    )
    ytdlp = args.ytdlp or (
        tools_dir / "yt-dlp.exe" if (tools_dir / "yt-dlp.exe").is_file() else None
    )

    return run_engine(
        EngineConfig(
            checkpoints_dir=checkpoints_dir,
            playability_path=playability,
            layer_stats_path=layer_stats,
            ffmpeg_path=ffmpeg,
            ytdlp_path=ytdlp,
            device=None if args.device == "auto" else args.device,
            log_path=args.log_file,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
