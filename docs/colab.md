# Training on Google Colab

## What to upload

Put two files in Google Drive under `MyDrive/BeatSaberMaker/`:

- `bsmaker-train-data.tar` (about 40 GB, uncompressed tar whose top-level entries are `events/` and `features/`).
- `bsmaker-code.zip` (the code bundle, see below).

## Building the code bundle

Commit your changes, then run:

    .venv/Scripts/python scripts/make_colab_bundle.py [--out PATH]

It packs `src/`, `pyproject.toml`, `dataset_stats.json` and `playability.toml` as tracked at `HEAD`
(default output `F:\BeatSaberMakerData\upload\bsmaker-code.zip`), adds `BUNDLE_COMMIT.txt` and prints the
path, size and commit. It refuses to run if `src/` has uncommitted changes.

## Running

1. Open `notebooks/colab_train.ipynb` in Colab.
2. Runtime -> Change runtime type -> GPU.
3. Runtime -> Run all. Parameters (steps, batch size, workers) are in the config cell near the top.

After a disconnect, run all again: data unpacking is repeated (it needs about 45 GB of free local disk and is
skipped only if the `.done` marker exists; an interrupted unpack is removed and redone), and both trainings resume from `last.pt` in the Drive checkpoint folders.

## Bringing the checkpoints back

Download from `MyDrive/BeatSaberMaker/checkpoints/` and place them at:

- `F:\BeatSaberMakerData\checkpoints\timing\best.pt`
- `F:\BeatSaberMakerData\checkpoints\pattern\best.pt`

Before downloading checkpoints or closing the runtime, run the last cell (`drive.flush_and_unmount()`) so
buffered writes reach Drive.

## Known limits

- Colab sessions disconnect (idle timeout, usage limits, about 12 h maximum); progress is kept only as far as the
  last checkpoint (`--save-every`, 1000 steps by default).
- Checkpoints are written straight to Drive, which has write latency and may sync with a delay; wait for the sync
  before downloading.
- `pyproject.toml` requires Python >= 3.13, but Colab ships 3.11/3.12. The package is therefore not pip-installed:
  the notebook puts `src` on `PYTHONPATH` and installs only `librosa`, `soundfile`, `typer`, `httpx`. The source
  was checked to parse as Python 3.11 syntax; a runtime use of 3.12+/3.13-only library features would still show
  up only on Colab.
- The notebook has not been run in CI; there is no Colab in the test environment.
