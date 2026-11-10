# Environment

Windows 11, Python 3.13.2 (only interpreter installed), ffmpeg 8.1, NVIDIA RTX 4060 Ti 16 GB.

## Install commands
```
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install numpy librosa soundfile httpx typer pytest
.venv/Scripts/python -m pip install beat_this
.venv/Scripts/python -m pip install -e .
```

## Versions
- torch 2.11.0+cu128 (CUDA 12.8), `torch.cuda.is_available()` = True, device "NVIDIA GeForce RTX 4060 Ti"
- torchaudio 2.11.1 (pulled in by beat_this)
- numpy 2.5.4, librosa 1.0.0, soundfile 0.14.0, httpx 0.28.1, typer 0.29.0, pytest 9.2.0
- beat_this 1.1.0 (with einops 0.8.2, rotary-embedding-torch 0.9.2)

## Beat tracker
`beat_this` works: installed from PyPI and `from beat_this.inference import File2Beats` imports cleanly.
Fallback if ever needed: `librosa.beat.beat_track`.
