"""Render a map's song with a click at every note, to check sync by ear."""
from pathlib import Path

import numpy as np
import soundfile as sf

from bsmaker.audio.decode import SAMPLE_RATE, decode_mono
from bsmaker.formats.reader import read_map

CLICK_S = 0.015
FADE_S = 0.001
CLICK_AMP = 0.5
PEAK = 0.9
LEFT_HZ = 1000.0
RIGHT_HZ = 1500.0
_RANK_ORDER = {"easy": 0, "normal": 1, "hard": 2, "expert": 3, "expertplus": 4, "expert+": 4}


def _rank_key(rank: str) -> int:
    return _RANK_ORDER.get(rank.strip().lower(), -1)


def _click(freq: float) -> np.ndarray:
    n = int(round(CLICK_S * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    burst = CLICK_AMP * np.sin(2 * np.pi * freq * t)
    f = int(round(FADE_S * SAMPLE_RATE))
    ramp = np.linspace(0.0, 1.0, f)
    burst[:f] *= ramp
    burst[n - f:] *= ramp[::-1]
    return burst.astype(np.float32)


def render_preview(folder: Path, out_path: Path, characteristic: str = "Standard",
                   rank: str | None = None) -> Path:
    """Write song + clicks (1000 Hz left, 1500 Hz right) as a 16-bit WAV; return out_path."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"map folder not found: {folder}")
    bm = read_map(folder)
    if bm.audio_path is None:
        raise ValueError("map has no audio file")
    cands = [d for d in bm.difficulties if d.characteristic == characteristic]
    if not cands:
        raise ValueError(f"map has no {characteristic} difficulty")
    if rank is None:
        diff = max(cands, key=lambda d: _rank_key(d.rank))
    else:
        match = [d for d in cands if d.rank.lower() == rank.lower()
                 or _rank_key(d.rank) == _rank_key(rank) >= 0]
        if not match:
            raise ValueError(f"map has no {characteristic} {rank} difficulty")
        diff = match[0]
    mix = decode_mono(bm.audio_path).astype(np.float32).copy()
    clicks = {0: _click(LEFT_HZ), 1: _click(RIGHT_HZ)}
    for note in diff.notes:
        start = int(round(note.time_s * SAMPLE_RATE))
        if start < 0 or start >= len(mix):
            continue
        c = clicks[0 if note.color == 0 else 1]
        seg = c[: len(mix) - start]
        mix[start:start + len(seg)] += seg
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    mix = mix * (PEAK / peak) if peak > 0 else np.zeros_like(mix)
    out_path = Path(out_path)
    sf.write(str(out_path), mix, SAMPLE_RATE, subtype="PCM_16")
    return out_path
