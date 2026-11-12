# Plan 3: second training round — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover maps with BPM changes, weight training samples, harden the pattern model against its own mistakes, train several model sizes through the Colab CLI and pick the winners by validation.

**Architecture:** The map reader gains a tempo map, so note times in seconds are right for variable-BPM maps; recovered difficulties enter the dataset only after an audio-alignment check. The event cache carries quality fields that feed a weighted sampler. Training gets model-size presets and previous-token corruption. Colab is driven from the user's Raspberry Pi with the `colab` CLI and two VM-side scripts.

**Tech Stack:** Python 3.13 (local), 3.12 (Colab), PyTorch, numpy, librosa, sqlite3, typer, pytest, `google-colab-cli` 0.8.0 on the Pi.

**Spec:** `docs/superpowers/specs/2026-11-13-etap2-trening-design.md` (Polish). Earlier spec: `docs/superpowers/specs/2026-11-10-nuty-v1-design.md`.

## Global Constraints

- Splits are frozen: never run `bsmaker data resplit`; a difficulty keeps the split stored in the manifest.
- Generation-1 checkpoints (`<data_dir>/checkpoints/timing/best.pt`, `.../pattern/best.pt`) are never overwritten; new runs write to `<data_dir>/checkpoints/gen2/<run-name>/`.
- The test split is evaluated once, at the end, for the chosen pair only.
- Everything under the data dir (`F:\BeatSaberMakerData`), the Beat Saber `CustomLevels` folder and `Downloads` is data: tests use synthetic data only; real-data runs happen in the controller tasks (9, 10), from the repo root, as `.venv/Scripts/python -I -m bsmaker ...`.
- No new Python dependencies.
- Reject-reason strings are exact: `bpm_trick`, `tempo_external`, `tempo_unverified`. The old reason `bpm_changes` is no longer produced.
- BPM trick limits: any tempo value `< 20` or `> 1000` BPM.
- Alignment threshold: the 5th percentile of the score over 1 000 accepted train difficulties (seed 0).
- Quality weight: `0.25 + 0.75 * clip((score - 0.80) / 0.15, 0, 1)`; missing score → `0.6`. Density weight: `(median bin count / bin count) ** 0.5` clipped to `[0.5, 3.0]`, bins from `stats["nps_bins"]`.
- Previous-token corruption default `p = 0.1`.
- Size presets — pattern: `s` = 384/8/6, `m` = 512/12/8, `l` = 768/12/12 (d_model / n_layers / n_heads); timing: `s` = 256/6/4, `m` = 384/8/6.
- Commits: `git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit`; never push.

## Review Focus

1. A v3 map whose first `bpmEvents` entry is not at beat 0, or has entries out of order or with duplicate beats → times must still be monotonic in beat and finite (Task 1 tests).
2. A tempo value of 0, negative, non-numeric or missing in the file → the difficulty is rejected as `bpm_trick`, never a crash or a division by zero (Tasks 1, 2).
3. A candidate difficulty whose feature file is missing or shorter than the notes → `verify-tempo` leaves it `tempo_unverified` and reports it; no crash (Task 3).
4. An index written by the old code (no `score` / `upvotes` fields) → `EventCache.load` still works and weights fall back to the missing-score value (Task 4).
5. Resuming a run in a folder whose checkpoint was made with another size preset or another `corrupt_p` → a clear error naming the difference, not silent training of a different model (Tasks 5, 6).

---

### Task 1: Tempo map in the reader

**Files:**
- Create: `src/bsmaker/formats/tempo.py`
- Modify: `src/bsmaker/formats/reader.py` (`_bpm_changes`, `_read_difficulty`), `src/bsmaker/formats/model.py` (`DifficultyFlags`)
- Test: `tests/formats/test_tempo.py`, existing reader tests

**Interfaces:**
- Produces:
  - `TempoMap(info_bpm: float, points: tuple[tuple[float, float], ...])` (frozen dataclass; points are `(beat, bpm)`, sorted by beat, duplicates of a beat keep the last one). `to_seconds(beat: float) -> float`: piecewise linear, `info_bpm` before the first point. `bpm_range() -> tuple[float, float]` over `info_bpm` and all points.
  - `read_tempo(data: dict, schema: str, info_bpm: float, folder: Path) -> tuple[str, TempoMap]` returning the kind and the map. Kinds: `"none"`; `"editor"` (v2 `_customData._BPMChanges` / `_bpmChanges` only: points are recorded for `bpm_range`, but `to_seconds` must not be used — the reader converts with the constant Info BPM); `"events"` (v2 `_events` of `_type` 100 with `_floatValue`, or v3 `bpmEvents` `b`/`m`, when they amount to a change by today's rule in `_bpm_changes`); `"external"` (only `BPMInfo.dat` / `AudioData.dat` signal a change). A v3 file with `bpmEvents` and a `BPMInfo.dat` is `"events"`. A non-numeric, zero or negative tempo value is kept as `float("nan")` in the point so that `bpm_range` reports it.
  - `DifficultyFlags` gains `tempo_kind: str = "none"`, `bpm_min: float = 0.0`, `bpm_max: float = 0.0`; `has_bpm_changes` keeps today's meaning (`tempo_kind != "none"`).
- Reader: `Note.time_s = tempo.to_seconds(beat)` for kind `"events"` when every point is finite and positive; otherwise `beat * 60 / info_bpm` as today.

- [ ] **Step 1: tests** in `tests/formats/test_tempo.py`: constant map → `to_seconds(8) == 8*60/120`; two points `(0,120),(8,60)` → `to_seconds(8) == 4.0`, `to_seconds(10) == 6.0`; first point at beat 4 with info 120 → beats before 4 at 120; unsorted and duplicate-beat points; `read_tempo` for each kind from small dicts (v2 `_BPMChanges`, v2 type 100, v3 one event equal to Info BPM → `"none"`, v3 two events → `"events"`, BPMInfo file only → `"external"`); a value of 0 and of `"abc"` → NaN in `bpm_range`; a synthetic v3 map folder read through `read_map` has note times following the tempo map, and a v2 `_BPMChanges` folder keeps constant-BPM times with `tempo_kind == "editor"`.
- [ ] **Step 2:** run them, expect failures.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** `.venv/Scripts/python -m pytest tests/formats -q`, then the full suite.
- [ ] **Step 5:** commit `feat(formats): tempo map for maps with BPM changes`.

### Task 2: Filters, manifest columns and `data reindex-tempo`

**Files:**
- Modify: `src/bsmaker/data/filters.py`, `src/bsmaker/data/manifest.py`, `src/bsmaker/data/index_local.py`, `src/bsmaker/cli.py`
- Test: `tests/data/` (filters, manifest, a new `test_reindex_tempo.py`)

**Interfaces:**
- Consumes: Task 1 flags.
- Produces:
  - `reject_reasons(...)`: instead of `bpm_changes` — `tempo_external` for kind `"external"`; `bpm_trick` when `bpm_min`/`bpm_max` is NaN or outside `[20, 1000]` (kinds `"editor"` and `"events"`); otherwise, for kinds `"editor"` and `"events"`, `tempo_unverified` unless the caller passes `tempo_verified=True` (new keyword, default `False`).
  - Manifest: `difficulties` gains `tempo_kind TEXT`, `align_score REAL`, `align_score_const REAL` (added with `ALTER TABLE` like `song_id`, schema version bumped). The indexer writes `tempo_kind`; a difficulty with a stored `align_score` at or above the stored threshold (see Task 3) is indexed with `tempo_verified=True`.
  - `reindex_tempo(conn, cfg) -> dict[str, int]` in `index_local.py` and CLI `bsmaker data reindex-tempo`: re-reads every map that has a difficulty whose `reject_reasons` contain `bpm_changes`, recomputes that map's difficulty rows through the normal indexing path (keeping `split`, `song_id` and all `bs_*` fields), and prints counts per new reason. Rows of maps without such a difficulty are untouched.

- [ ] **Step 1: tests:** reasons for each kind and for trick values (0, NaN, 19.9, 1000.1; 20 and 1000 pass); `tempo_verified=True` removes `tempo_unverified` only; migration adds the three columns to an old-schema database and is idempotent; `reindex_tempo` on a temp manifest with three synthetic map folders (constant, v2 `_BPMChanges`, v3 `bpmEvents`) turns `["bpm_changes"]` into `["tempo_unverified"]`, keeps the stored split, leaves the constant map's row byte-identical.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat(data): tempo-aware reject reasons and reindex-tempo`.

### Task 3: Alignment check, candidate features, `data verify-tempo`

**Files:**
- Create: `src/bsmaker/data/align.py`
- Modify: `src/bsmaker/data/features.py`, `src/bsmaker/cli.py`
- Test: `tests/data/test_align.py`

**Interfaces:**
- Produces:
  - `onset_envelope(mel: np.ndarray) -> np.ndarray`: positive spectral flux of the log-mel summed over bands, one value per frame, minus its minimum.
  - `alignment_score(times_s: np.ndarray, env: np.ndarray) -> float`: mean over notes of the maximum of `env` in frames `round(t * FRAMES_PER_SECOND) ± 1`, divided by `env.mean()`; notes beyond the envelope are ignored; fewer than 20 usable notes or a flat envelope → `float("nan")`.
  - `calibrate_threshold(conn, features_dir, n=1000, seed=0) -> float`: 5th percentile of the score over `n` accepted train difficulties (constant-BPM maps only, `tempo_kind` NULL or `"none"`); stored in a manifest key-value table `meta(key TEXT PRIMARY KEY, value TEXT)` under `align_threshold`.
  - `verify_tempo(conn, cfg) -> dict`: for every difficulty with reasons exactly `["tempo_unverified"]` and an existing feature file: score with the reader's times → `align_score`; for kind `"events"` also the constant-BPM times → `align_score_const`; `align_score >= threshold` → `accepted = 1`, reasons `[]`. Returns counts (accepted, failed, missing features, NaN) and, for kind `"events"`, how many score higher with constant BPM.
  - `build_features(..., tempo_candidates: bool = False)` and CLI `bsmaker data features --tempo-candidates`: also builds features for maps that have a difficulty with reasons exactly `["tempo_unverified"]`. CLI `bsmaker data verify-tempo` calibrates if no threshold is stored, runs `verify_tempo`, prints the counts and writes `<reports>/verify-tempo.txt`.

- [ ] **Step 1: tests** (synthetic mel: flat background with bright frames at chosen times): notes on the bright frames score well above 1; the same notes shifted by 150 ms score about 1; notes stretched by 3% (wrong tempo) score clearly lower than aligned; NaN cases; `verify_tempo` on a temp manifest accepts the aligned difficulty, keeps the misaligned one, and reports a missing feature file without failing.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat(data): audio-alignment check for tempo-map difficulties`.

### Task 4: Quality fields in the event cache and weighted sampling

**Files:**
- Modify: `src/bsmaker/data/events_cache.py` (`Entry`, `build_event_cache`, `EventCache.load`), `src/bsmaker/models/train_loop.py`, `src/bsmaker/cli.py` (both train commands)
- Create: `src/bsmaker/models/sampling.py`
- Test: `tests/data/test_events_cache.py`, `tests/models/test_sampling.py`, train-loop tests

**Interfaces:**
- Produces:
  - `Entry` gains `score: float | None = None`, `upvotes: int | None = None` (from `maps.bs_score`, `maps.bs_upvotes`), written to and read from `index.json`; an index without them loads with `None`.
  - `entry_weights(entries: Sequence[Entry], stats: dict) -> np.ndarray` (float64, length `len(entries)`), per the Global Constraints formula; bins counted over the given entries with `np.searchsorted` on `stats["nps_bins"]`, values outside the range go to the nearest bin.
  - `run_training(..., sample_weights: np.ndarray | None = None)`: `None` keeps `shuffle=True`; otherwise a `WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True, generator=gen)` replaces shuffling.
  - CLI `--weighting [none|quality]` on `train timing` and `train pattern`, default `quality`; the choice is logged at start and stored in the checkpoint payload under `"train_options"` (a dict that Tasks 5 and 6 extend).

- [ ] **Step 1: tests:** weight values for scores `None`, 0.5, 0.80, 0.875, 0.95, 0.99 (0.6, 0.25, 0.25, 0.625, 1.0, 1.0 before the density factor); density factor for a three-bin toy distribution, including the clips; old-format index loads; a run with weights `[1, 0, 0, …]` only ever yields index 0; `--weighting none` gives the same first batches as before the change.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat(models): quality and density weighted sampling`.

### Task 5: Previous-token corruption for the pattern model

**Files:**
- Modify: `src/bsmaker/models/pattern/dataset.py` (`PatternSequences`), `src/bsmaker/cli.py` (`train pattern`)
- Test: `tests/models/test_pattern_dataset.py`

**Interfaces:**
- Produces: `PatternSequences(..., corrupt_p: float = 0.0)`. In `__getitem__`, when not `fixed`: for every valid position `j >= 1` of the run independently with probability `corrupt_p`, the pair `(prev_l[j], prev_r[j])` is replaced by the true `(left, right)` pair of another position of the same run drawn uniformly (not `j - 1`); position 0 (BOS or the true predecessor) is never touched; targets are unchanged. `fixed=True` (validation) never corrupts. CLI `--corrupt-p FLOAT` (default 0.1, must be in `[0, 0.5]`), stored in `"train_options"`.

- [ ] **Step 1: tests:** `corrupt_p=0` → identical batch to today for the same seed; `corrupt_p=0.5` → roughly half of the positions differ from the true predecessors (binomial bounds), every replaced pair exists elsewhere in the run, targets equal the uncorrupted ones; a run of length 1 or 2 does not crash; `fixed=True` ignores `corrupt_p`; resuming with a different `corrupt_p` than the checkpoint's `train_options` fails with a message naming both values.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat(models): previous-token corruption in pattern training`.

### Task 6: Size presets and learning rate option

**Files:**
- Modify: `src/bsmaker/cli.py` (both train commands), `src/bsmaker/models/train_loop.py` (resume check message)
- Test: CLI and train-loop tests

**Interfaces:**
- Produces: `--size [s|m|l]` on `train pattern` (default `s`) and `--size [s|m]` on `train timing` (default `s`), mapping to the presets in Global Constraints through module constants `PATTERN_SIZES` / `TIMING_SIZES`; `--lr FLOAT` on both (default 3e-4). On start each command logs the size, the parameter count and the options. Resuming in a folder whose checkpoint has another `model_config` stops with: `checkpoint in <dir> was made with <old>; this run asks for <new>; use another --out`.

- [ ] **Step 1: tests:** each preset builds a model whose config matches and whose parameter count is within 15% of the spec's figure (15M / 38M / 85M; 5M / 14M); an unknown size is a usage error; the resume mismatch message; `--lr` reaches `TrainSettings`.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat(cli): model size presets and --lr for training`.

### Task 7: Evaluation extensions

**Files:**
- Modify: `src/bsmaker/evaluate.py`, `src/bsmaker/cli.py` (`eval`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - CLI options: `--timing-ckpt PATH`, `--pattern-ckpt PATH` (defaults as today), `--entries-from PATH` (an `index.json`; only difficulties whose `(map_key, characteristic, rank)` appear in it are used), `--tag NAME` (report file `eval-<split>-<tag>.md`, also `eval-<split>-<tag>.json` with the raw numbers).
  - `eval_timing` also returns F1 at 10 ms (`f1_10`), shown next to the 25 ms figure and in the per-bin table.
  - `eval_pattern_sampled(pattern, cache, features_dir, mask_cfg, split, n, seed, device, entries=None) -> dict`: `sample_notes` on the human event times of `n` difficulties (one per song, spread evenly; default `n = 40`), then: joint row × direction shares for human and generated with their total-variation distance (`tv_row_dir`), share of upward cuts (up, up-left, up-right) in the bottom row and of downward cuts in the top row, hard violations (must be 0). A new report section "Pattern on human event times".

- [ ] **Step 1: tests** with the tiny synthetic models the existing eval tests use: `--entries-from` restricts the set; `f1_10 <= f1_25`; `tv_row_dir` is 0 for identical event lists and 1 for disjoint ones; the JSON file holds every number of the summary; custom checkpoint paths are used and named in the Setup section.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat: eval with checkpoint paths, old-entry lists and sampled pattern metrics`.

### Task 8: Colab CLI scripts, delta bundle, documentation

**Files:**
- Create: `scripts/colab/vm_setup.py`, `scripts/colab/vm_train.py`, `docs/colab-cli.md`
- Modify: `scripts/make_colab_bundle.py`, `tests/test_colab_bundle.py`

**Interfaces:**
- Produces:
  - `make_colab_bundle.py --delta --base-index PATH --data-dir PATH --out-dir PATH`: writes `bsmaker-train-delta.tar` containing `events/` (whole), `dataset_stats.json` (if present in the data dir) and `features/<key>.npy` + `.json` for every map key in the current `events/index.json` that is absent from the base index; prints the file count and size; refuses to overwrite an existing tar. Without `--delta` it behaves as today.
  - `vm_setup.py` (runs on the Colab VM, plain Python standard library, arguments `--drive-dir /content/drive/MyDrive/BeatSaberMaker --work /content`): unpacks `bsmaker-train-data.tar` then `bsmaker-train-delta.tar` into `<work>/data` (delta overwrites `events/`), each guarded by its own `.done` marker and a `.partial` folder as in the notebook; unzips `bsmaker-code.zip` to `<work>/bsmaker`; installs `librosa soundfile typer httpx` with pip; writes `<work>/bsmaker/bsmaker.toml`; prints the number of feature files and of index entries. Safe to run twice.
  - `vm_train.py` (on the VM): `--model timing|pattern --name RUN --drive-dir ... --work /content` plus pass-through options after `--`; runs `python -m bsmaker train <model> --out <drive-dir>/checkpoints/gen2/<RUN> ...` with `PYTHONPATH=<work>/bsmaker/src`, streaming output line by line and appending it to `<out>/console.log`; exit code propagated.
  - `docs/colab-cli.md`: the exact command sequence from the Pi (`colab new -s train --gpu A100 --high-mem`; the user's `colab drivemount -s train`; `colab exec -s train -f vm_setup.py --timeout 0`; training inside `tmux` with `colab exec ... --timeout 0`; `colab status`, `colab usage`, `colab stop -s train`), how resuming works, and the rule that a session is always stopped after a run.

- [ ] **Step 1: tests:** delta tar from a temp data dir holds exactly the new keys' features plus `events/`; refusal to overwrite; `vm_setup.py` against a temp "drive" with tiny tars and a tiny zip (pip step skippable with `--no-pip`) produces the layout, honours `.done`, and recovers from a leftover `.partial`; `vm_train.py --dry-run` prints the exact command line it would run.
- [ ] **Step 2–4:** fail, implement, pass, full suite.
- [ ] **Step 5:** commit `feat: Colab CLI driver scripts and delta data bundle`.

### Task 9: Data run (controller, real data)

Not for a subagent implementer; the controller runs it and records numbers in the ledger.

- [ ] Confirm with the user that `F:\BeatSaberMakerData\upload` is deleted and that the GPU and RAM are free (no game running).
- [ ] Copy `events/index.json` to `events/index.gen1.json` (the old validation list).
- [ ] `bsmaker data reindex-tempo` → record counts per reason.
- [ ] `bsmaker data features --tempo-candidates` → record computed / failed and free space.
- [ ] `bsmaker data verify-tempo` → record the threshold, accepted / failed counts and, for kind `events`, how many score higher with constant BPM. **Stop rule:** if more than 30% of `events` difficulties score higher with constant BPM, or fewer than 60% of `editor` difficulties pass, the conversion or the editor-only assumption is wrong: do not build the cache; open a fix task instead.
- [ ] Spot check: print title, tempo kind, change points, `align_score` and `align_score_const` for 5 accepted `events` maps and 5 accepted `editor` maps (seed 0) into the ledger.
- [ ] `bsmaker data build-events` → record difficulties and events per split, old versus new.
- [ ] Build the code bundle and the delta tar (`--base-index events/index.gen1.json`); tell the user the two files to upload to `MyDrive/BeatSaberMaker/`.

### Task 10: Training runs, selection, hand-over (controller)

- [ ] From the Pi: session `train` on A100 high-mem; the user runs `colab drivemount -s train`; `vm_setup.py`; a 20-step probe of each model.
- [ ] Runs, each in `tmux`, each followed by `colab usage`: `pat-s` (`--size s --corrupt-p 0.1`), `pat-s-p0` (`--corrupt-p 0`), `pat-m` (`--size m`), `tim-s`, `tim-m`; `pat-l` only if `pat-m` beats `pat-s` on validation CE or `tv_row_dir`. Stop the session when no run is active. Budget: 120 compute units in total; stop and report if the next run would exceed it.
- [ ] Download each `best.pt` to `<data_dir>/checkpoints/gen2/<run>/best.pt`.
- [ ] `bsmaker eval --split val --tag <run>` for every run with `--entries-from events/index.gen1.json`, plus generation 1 as `--tag gen1`; one comparison table in the ledger.
- [ ] Choose per the spec (pattern: lowest `tv_row_dir` among runs within 0.03 CE of the best; timing: highest F1@25). If no run beats generation 1 on its criterion, keep generation 1 for that model and say so.
- [ ] `bsmaker eval --split test` once for the chosen pair.
- [ ] Generate the user's nine songs with the chosen pair into `CustomLevels` under the same names (`--overwrite` only after copying the current nine zips to `previews/gen1/`), run `check`, zip, and hand the list to the user for a preview or VR pass.
- [ ] Record the user's verdict in `docs/feedback/`; only on a positive verdict copy the chosen checkpoints over the default paths (after renaming the generation-1 files to `best.gen1.pt`).
