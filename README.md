# BeatSaberMaker

Generates playable Beat Saber maps from a song. Two small models are trained on human maps: a timing model decides where notes go, a pattern model decides which notes. A playability mask keeps hard rule violations out of the result.

## Setup

New here? Follow `docs/quickstart.md`: install, configure paths, get the model files, generate a map.

Details of the reference machine are in `docs/environment.md` (Python venv, PyTorch with CUDA, ffmpeg).

## Data pipeline

See `docs/data-pipeline.md` for indexing maps, building features and the event cache.

## Training

```
bsmaker train timing
bsmaker train pattern
```

Checkpoints go to `<data_dir>/checkpoints/<model>/best.pt`. To train on Colab instead, follow `docs/colab.md`.

## Usage

Generate a map folder from a song:

```
bsmaker generate "song.mp3" --nps 4 --style tech --seed 1
```

Check sync by ear with a click track (written outside the map folder):

```
bsmaker preview "BSM - song [nps 4, tech]" --out preview.wav
```

Evaluate the models on held-out songs; writes `<data_dir>/reports/eval-val.md`:

```
bsmaker eval --split val --n 200 --n-generated 10
```

Generated folders are always named `BSM - ...`, and only such folders are ever replaced (with `--overwrite`). Notes from VR play-tests go to `docs/feedback/`.
