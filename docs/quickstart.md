# Quickstart for a new contributor

This gets you from a fresh clone to a generated Beat Saber map. You do not need the training data for that — only the two trained model files.

## 1. Requirements

- Windows 10/11 (Linux should work, but is untested).
- Python 3.13.
- ffmpeg and ffprobe on `PATH` (`ffmpeg -version` must work in a terminal).
- An NVIDIA GPU is optional. Generation takes about half a minute per song on a GPU and works on CPU with `--device cpu`, only slower.

## 2. Install

From the repository root:

```
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e .
```

Without an NVIDIA GPU, install plain `torch` instead of the `cu128` build:

```
.venv/Scripts/python -m pip install torch
```

Check the install:

```
.venv/Scripts/python -m pytest -q
```

All tests should pass; they use small synthetic data and need neither the models nor the dataset.

## 3. Configure paths

Edit `bsmaker.toml` in the repository root. It holds the paths of the machine it was last committed from, so both lines need changing:

```
custom_levels = 'C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels'
data_dir = 'D:\BeatSaberMakerData'
```

- `custom_levels` — the Beat Saber custom levels folder. Generated maps are written here. Without the game installed, point it at any empty folder.
- `data_dir` — any folder with a few hundred MB free. Models and reports live here.

Please do not commit your own paths.

## 4. Get the models

The trained models are not in git (`*.pt` is ignored). Ask the repository owner for the two files and place them exactly here:

```
<data_dir>\checkpoints\timing\best.pt     (about 60 MB)
<data_dir>\checkpoints\pattern\best.pt    (about 180 MB)
```

## 5. Generate a map

```
.venv/Scripts/bsmaker generate "C:\path\to\song.mp3" --nps 4
```

- `--nps` — notes per second, 0.5 to 12. Roughly: 2.3 Normal, 3.1 Hard, 4 Expert, 4.9 Expert+.
- `--seed 1`, `2`, ... — a different map for the same song.
- `--overwrite` — replace an existing generated map of the same name.
- `--device cpu` — run without a GPU.

The map appears in `custom_levels` as `BSM - <file name> [nps 4, unknown]`; the title comes from the audio file name. Only folders named `BSM - ...` are ever written or replaced.

Useful afterwards:

```
.venv/Scripts/bsmaker check "<map folder>"
.venv/Scripts/bsmaker preview "<map folder>" --out preview.wav
```

`check` reports playability rule violations; `preview` renders the song with a click on every note. To watch a map without a headset, zip the files of the map folder and drop the zip on ArcViewer in a browser.

## 6. Working on the code

- Tests: `.venv/Scripts/python -m pytest -q`.
- Layout: `src/bsmaker/audio` (decoding, mel, beat grid), `formats` (reading and writing maps), `playability` (rules and the mask), `models/timing` and `models/pattern` (the two models), `generate.py`, `preview.py`, `evaluate.py`, `cli.py`.
- Design and plans: `docs/superpowers/specs/` and `docs/superpowers/plans/`. Play-test notes: `docs/feedback/`.
- Work on a branch and open a pull request against `main`.

## 7. Training (optional)

Training and `bsmaker eval` need the dataset (about 40 GB packed: features and the event cache), which is not distributed with the repository. If you have it, see `docs/data-pipeline.md` and `docs/colab.md`.
