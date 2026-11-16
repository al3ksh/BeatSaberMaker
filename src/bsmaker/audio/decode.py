"""Audio decoding through ffmpeg. Input files are treated as untrusted."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
_TIMEOUT_S = 300

_SAFE_INPUT = ["-nostdin", "-protocol_whitelist", "file,pipe"]


class AudioDecodeError(Exception):
    """Raised when an audio file cannot be decoded."""


class FfmpegMissingError(Exception):
    """ffmpeg or ffprobe cannot be run: an environment problem, never a verdict on a file."""


def _win_no_window() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def require_ffmpeg(ffmpeg_path: Path | None = None) -> str:
    """Verify ffmpeg is usable and return its executable string.

    When `ffmpeg_path` is None, checks that both `ffmpeg` and `ffprobe` are on PATH.
    When `ffmpeg_path` is given, verifies that exact file exists on disk.
    """
    if ffmpeg_path is not None:
        p = Path(ffmpeg_path)
        if not p.is_file():
            raise FfmpegMissingError(f"ffmpeg not found: {ffmpeg_path}")
        return str(p)
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FfmpegMissingError(f"{' and '.join(missing)} not found on PATH")
    return "ffmpeg"


def _run(args: list[str], path: Path, *, ffmpeg_path: Path | None = None) -> bytes:
    exe = str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg"
    cmd = [exe, "-hide_banner", "-loglevel", "error", *args]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_TIMEOUT_S,
            shell=False,
            **_win_no_window(),
        )
    except FileNotFoundError as exc:
        msg = f"ffmpeg not found: {ffmpeg_path}" if ffmpeg_path is not None else "ffmpeg not found on PATH"
        raise FfmpegMissingError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"ffmpeg timed out decoding {path}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise AudioDecodeError(f"ffmpeg failed on {path} (exit {proc.returncode}): {err}")
    return proc.stdout


def decode_mono(path: Path, *, ffmpeg_path: Path | None = None) -> np.ndarray:
    """Decode to mono float32 at 22 050 Hz."""
    path = Path(path)
    out = _run(
        [*_SAFE_INPUT, "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "pipe:1"],
        path,
        ffmpeg_path=ffmpeg_path,
    )
    audio = np.frombuffer(out, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise AudioDecodeError(f"no audio samples decoded from {path}")
    return audio


def write_compact(path_in: Path, path_out: Path, *, ffmpeg_path: Path | None = None) -> None:
    """Write mono 22 050 Hz Ogg Vorbis at about 64 kb/s."""
    path_in = Path(path_in)
    _run(
        [*_SAFE_INPUT, "-i", str(path_in), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg", "-y", str(path_out)],
        path_in,
        ffmpeg_path=ffmpeg_path,
    )
