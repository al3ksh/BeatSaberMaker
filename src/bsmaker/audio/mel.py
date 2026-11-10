"""Log-mel spectrogram and its on-disk cache."""
from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import numpy as np

from bsmaker.audio.decode import SAMPLE_RATE, decode_mono

HOP = 256
N_FFT = 1024
N_MELS = 80
FRAMES_PER_SECOND = SAMPLE_RATE / HOP


def log_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32), sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, center=True, power=2.0,
    )
    return np.log(mel + 1e-5).T.astype(np.float16)


def time_to_frame(time_s: float) -> int:
    return round(time_s * SAMPLE_RATE / HOP)


def _meta(audio_path: Path) -> dict:
    st = audio_path.stat()
    return {"sr": SAMPLE_RATE, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def cached_log_mel(audio_path: Path, cache_path: Path) -> np.ndarray:
    audio_path, cache_path = Path(audio_path), Path(cache_path)
    sidecar = cache_path.with_name(cache_path.name + ".json")
    meta = _meta(audio_path)
    try:
        if json.loads(sidecar.read_text("utf-8")) == meta:
            return np.load(cache_path)
    except (OSError, ValueError):
        pass
    mel = log_mel(decode_mono(audio_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Sidecar is removed first and written last, so a partial write is a miss.
    sidecar.unlink(missing_ok=True)
    tmp_npy = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_npy, "wb") as f:
        np.save(f, mel)
    os.replace(tmp_npy, cache_path)
    tmp_json = sidecar.with_name(sidecar.name + ".tmp")
    tmp_json.write_text(json.dumps(meta), "utf-8")
    os.replace(tmp_json, sidecar)
    return mel
