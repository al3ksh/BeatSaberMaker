# BeatSaberMaker Plan 2 — Models and Generation (M5–M8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the timing model and the pattern model on the indexed dataset and ship `bsmaker generate`, `preview` and `eval`, ending with the first generated maps playable in VR.

**Architecture:** A song-level split and a packed event cache feed two PyTorch models: a bidirectional transformer that marks event frames on log-mel windows, and a causal transformer that picks a (left, right) token pair per event under the playability mask. `generate` chains decode → timing → pattern sampling → beat-grid snap → `write_map`. Long trainings are started by the controller in the background, not by implementers.

**Tech Stack:** Python 3.13, PyTorch 2.11 + CUDA (RTX 4060 Ti 16 GB, mixed precision), numpy, librosa (mel, beat tracking), soundfile, typer, pytest, SQLite.

**Spec:** `docs/superpowers/specs/2026-11-10-nuty-v1-design.md` — sections 5, 7, 8, 9 and milestones M5–M8. Read it before starting any task.

## Measured dataset (2026-11-11)

| | |
|---|---|
| Accepted difficulties / maps | 33 971 / 14 976 |
| Audio | 807 h, features 38 GB (`features\<key>.npy`, float16, shape `[frames, 80]`) |
| Events / notes | 21.1 M / 26.1 M, mean 4.2 notes/s |
| Maps sharing normalised title+artist with another map | 3 493 (23%) |
| Maps with empty normalised title | 244 |
| Accepted under the strict `mask` profile (≤ 1% hard) | about 47% of files |

## Global Constraints

- Everything in Plan 1's Global Constraints still holds (paths from `bsmaker.toml`, untrusted map data, venv commands, note encoding, mel parameters, type hints).
- Tests must not need the network, the GPU, the F: drive or the real CustomLevels folder. Models in tests are tiny (`d_model ≤ 32`, ≤ 2 layers) and run on CPU in seconds.
- Real-data commands run in isolated mode from the repo root: `.venv/Scripts/python -I -m bsmaker ...`.
- One bulk writer on the manifest at a time. Trainings only read the event cache and features, never the manifest.
- **Implementers never start a real training, never run `generate` against the real CustomLevels and never download anything.** The controller runs trainings in the background after the task review; smoke runs by implementers use `--max-steps 50` at most and `--out` in a temp folder.
- The only writes to CustomLevels are new folders named `BSM - …` created by `generate`.
- No new dependencies. Beat tracking uses `librosa.beat.beat_track` (`beat_this` needs a weights download; switching to it is a later, separately approved change).
- All randomness goes through an explicit seed (`torch.Generator`, `numpy.random.default_rng`); same seed + same checkpoint + same audio → same map.
- Checkpoints and generated files are written to a temp name and `os.replace`d.
- Commit with `git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit ...`; end messages with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. Never push.

## Decisions fixed by this plan

- **Song identity.** `song_id` groups maps that are the same song: normalised title + artist equal, or normalised title equal and durations within 2 s. A map with an empty normalised title is its own song. The split is a hash of the group's smallest key.
- **Training sequences** are `to_events(notes)` then `drop_slider_tails(events, 0.12)`. Density everywhere in this plan is `nps = notes after that / (last event time − first event time)`.
- **The mask is not part of the training loss.** The pattern model learns from all accepted human maps (`data` profile); the strict `mask` profile is applied only when sampling. `eval` reports how much probability mass the mask removes.
- **Recovering maps with BPM changes** (7 882 rejected difficulties) is out of scope; it goes to Stage 2.

## Review Focus

1. **Silent or near-silent audio** — the timing model finds almost no peaks: `generate` exits non-zero with a clear message when fewer than 20 events survive, it does not write an empty map or loop. (Task 8)
2. **Song titles that are not valid Windows folder names** (`AC/DC: Live?`, trailing dot, 200 characters, emoji): the folder name is sanitised, non-empty and at most 80 characters. (Task 8)
3. **A run of dead ends in the pattern sampler** (mask blocks everything for many events in a row): events are skipped and counted, sampling terminates, the result still has zero hard violations. (Task 6)
4. **Training interrupted during a checkpoint write** — a truncated `last.pt` must never be loaded: writes are atomic and resume falls back to the previous file. (Task 3)
5. **Density extremes** (`--nps 0.5`, `--nps 12`, and values outside 0.5–12): in-range values produce a map within tolerance or a printed warning; out-of-range values are a usage error. (Task 8)

## Task order

Strictly sequential: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10. Controller runs between tasks are marked **[controller]**.

---

### Task 1: Song-level split and manifest schema version

**Files:**
- Create: `src/bsmaker/data/songs.py`
- Modify: `src/bsmaker/data/manifest.py`, `src/bsmaker/data/index_local.py`, `src/bsmaker/cli.py`, `docs/data-pipeline.md`, `docs/superpowers/specs/2026-11-10-nuty-v1-design.md` (section 4: two playability profiles, song-level split)
- Test: `tests/data/test_songs.py`, `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: `open_manifest`, `assign_split(key: str) -> str`.
- Produces: `normalise_title(text: str) -> str` (NFKC, casefold, bracketed parts `(...)`/`[...]` removed, everything except Unicode letters and digits removed); `assign_song_ids(rows: Sequence[tuple[str, str, str, float | None]]) -> dict[str, str]` (rows are `(key, title, artist, duration_s)`; value is the smallest key of the group); `resplit(conn) -> dict[str, int]` (sets `maps.song_id` and every `difficulties.split` to `assign_split(song_id)`, one transaction, returns counts per split).
- Manifest: `SCHEMA_VERSION = 2` stored in `PRAGMA user_version`; `open_manifest` migrates version 0/1 by adding `maps.song_id TEXT`; a file with a higher version raises `RuntimeError` naming both versions.
- `index_folder` keeps an existing `difficulties.split` on re-index (new rows get `assign_split(key)` until the next `resplit`).
- CLI: `bsmaker data resplit` prints counts per split before and after and the number of songs.

- [ ] **Step 1: Failing tests** — `test_normalise_title` (`"Odo (TV Size) [Full]"` → `"odo"`, `"ＯＤＯ"` → `"odo"`, `"うっせぇわ"` stays non-empty, `"!!!"` → `""`); `test_same_title_and_artist_grouped`; `test_same_title_close_duration_grouped` (artists differ, durations 180.0 and 181.5 → one song; 180 and 190 → two); `test_empty_title_is_own_song`; `test_grouping_is_transitive`; `test_song_id_is_smallest_key`; `test_resplit_puts_all_difficulties_of_a_song_in_one_split`; `test_resplit_is_idempotent`; `test_migration_adds_song_id_and_sets_version`; `test_newer_schema_refused`; `test_reindex_keeps_split`.
- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/data -q` — new tests fail.
- [ ] **Step 3:** Implement (union-find over the two grouping keys).
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest -q` — all pass.
- [ ] **Step 5: Real run.** `.venv/Scripts/python -I -m bsmaker data resplit`, output saved to `F:\BeatSaberMakerData\reports\resplit.txt`. Report songs, the largest five groups (keys and titles) and counts per split.
- [ ] **Step 6: Commit** — `feat(data): song-level split and manifest schema version`.

### Task 2: Event cache and dataset statistics

**Files:**
- Create: `src/bsmaker/data/events_cache.py`, `src/bsmaker/data/stats.py`
- Modify: `src/bsmaker/config.py` (`events_dir`), `src/bsmaker/cli.py`
- Generated and committed: `dataset_stats.json` (repo root)
- Test: `tests/data/test_events_cache.py`, `tests/data/test_stats.py`

**Interfaces:**
- Consumes: `iter_accepted`, `read_map`, `to_events`, `drop_slider_tails`, `styles_from_tags`, `STYLES`.
- Produces: `SLIDER_GAP_S = 0.12`; `training_events(notes: Sequence[Note]) -> list[Event]`; `build_event_cache(conn, events_dir: Path, features_dir: Path, progress_every: int = 0) -> int` writing `events.npz` (`time_s float32[N]`, `left uint8[N]`, `right uint8[N]`, `offsets int64[D+1]`) and `index.json` (list of D objects: `map_key`, `characteristic`, `rank`, `split`, `nps`, `styles` (list), `njs`, `njs_offset`, `n_frames`); a difficulty whose feature file is missing or which has fewer than 20 events after tail dropping is left out and counted.
- `class EventCache`: `EventCache.load(events_dir: Path)`, `.entries: list[Entry]` (frozen dataclass with the index fields plus `i: int`), `.events(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, `.select(split: str) -> list[Entry]`.
- `compute_stats(cache: EventCache, features_dir: Path, sample_maps: int = 500, seed: int = 0) -> dict` with keys: `rank_median_nps` (per rank, train split), `nps_bins` (edges `[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 12]`), `njs_by_bin`, `njs_offset_by_bin`, `double_share_by_bin` (events with both hands / events), `coverage_median` (event span / audio duration), `mel_mean`, `mel_std` (scalars over the sampled train maps). `load_stats(path: Path) -> dict`. `lookup(stats: dict, table: str, nps: float) -> float` (value of `njs_by_bin`, `njs_offset_by_bin` or `double_share_by_bin` for the bin holding `nps`, clamped to the first/last bin).
- CLI: `bsmaker data build-events` (builds the cache, then writes `dataset_stats.json` next to `bsmaker.toml`).

- [ ] **Step 1: Failing tests** — `test_cache_roundtrip` (two synthetic maps → same events back, offsets correct); `test_slider_tails_dropped_in_cache`; `test_nps_uses_event_span`; `test_short_or_featureless_difficulty_left_out`; `test_select_by_split`; `test_stats_double_share` (hand-built cache with known shares); `test_stats_bins_cover_range` (every bin has a value: empty bins take the nearest non-empty bin's value); `test_stats_json_roundtrip`; `test_lookup_clamps_outside_bins`.
- [ ] **Step 2:** Run the two test files — fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Whole suite passes.
- [ ] **Step 5: Real run.** `.venv/Scripts/python -I -m bsmaker data build-events`. Report difficulties cached and left out, total events, file sizes, and the contents of `dataset_stats.json`.
- [ ] **Step 6: Commit** — `feat(data): packed event cache and dataset statistics` (includes `dataset_stats.json`).

### Task 3: Timing model, dataset and training loop

**Files:**
- Create: `src/bsmaker/models/__init__.py`, `src/bsmaker/models/common.py`, `src/bsmaker/models/train_loop.py`, `src/bsmaker/models/timing/__init__.py`, `src/bsmaker/models/timing/model.py`, `src/bsmaker/models/timing/dataset.py`
- Modify: `src/bsmaker/cli.py` (`train timing`)
- Test: `tests/models/test_common.py`, `tests/models/test_train_loop.py`, `tests/models/test_timing_model.py`

**Interfaces:**
- `common.py`: `STYLE_DIM = 8` (the 7 `STYLES` then `unknown`); `style_vector(styles: Iterable[str]) -> np.ndarray` (float32 multi-hot; empty or only unknown names → `unknown` set); `class Conditioning(nn.Module)`: `__init__(d_model: int)`, `forward(log_nps: Tensor[B], styles: Tensor[B, 8]) -> Tensor[B, d_model]`; `save_checkpoint(path: Path, payload: dict) -> None` (atomic); `load_checkpoint(path: Path, map_location) -> dict` (`torch.load(..., weights_only=True)`).
- `timing/model.py`: `@dataclass(frozen=True) class TimingConfig: n_mels: int = 80; d_model: int = 256; n_layers: int = 6; n_heads: int = 4; window: int = 1024; dropout: float = 0.1`; `class TimingModel(nn.Module)`: `forward(mel: Tensor[B, T, 80], log_nps: Tensor[B], styles: Tensor[B, 8]) -> Tensor[B, T]` (logits; conv stem over the frequency axis keeps T; bidirectional encoder; conditioning added to every frame); `timing_loss(logits, targets, pos_weight: float = 5.0) -> Tensor`.
- `timing/dataset.py`: `frame_targets(times_s: np.ndarray, n_frames: int) -> np.ndarray` (float32; 1.0 at `time_to_frame(t)`, 0.25 at the two neighbours, max where they overlap); `class TimingWindows(torch.utils.data.Dataset)`: `__init__(cache: EventCache, entries: Sequence[Entry], features_dir: Path, stats: dict, window: int, seed: int, style_dropout: float = 0.15, fixed: bool = False)`; each item is one random window of one difficulty (`fixed=True` → the same window every time, for validation): `{"mel": [window, 80] normalised with mel_mean/mel_std, "target": [window], "log_nps": scalar, "styles": [8]}`; features opened with `np.load(..., mmap_mode="r")`; windows shorter than `window` are zero-padded with a `"valid"` mask the loss respects.
- `train_loop.py`: `@dataclass class TrainSettings: max_steps: int; batch_size: int = 32; lr: float = 3e-4; warmup: int = 1000; eval_every: int = 2000; save_every: int = 1000; num_workers: int = 4; amp: bool = True; device: str = "cuda"`; `run_training(model, train_ds, val_ds, step_fn: Callable[[nn.Module, dict], Tensor], eval_fn: Callable[[nn.Module, DataLoader], dict[str, float]], out_dir: Path, settings: TrainSettings, model_config: dict, better: Callable[[dict, dict | None], bool], log: Callable[[str], None]) -> dict` — AdamW, linear warmup then cosine, mixed precision on CUDA, `last.pt` every `save_every` (previous one kept as `prev.pt`), `best.pt` when `better` says so, resumes from `last.pt` (falls back to `prev.pt` if `last.pt` does not load), appends one JSON line per eval to `log.jsonl`.
- Checkpoint payload: `{"model": state_dict, "optimizer": ..., "scaler": ..., "step": int, "model_config": dict, "stats": dict, "best": dict | None}`.
- CLI: `bsmaker train timing [--max-steps 40000] [--batch-size 32] [--num-workers 4] [--device cuda]` → `checkpoints\timing\`; validation = 512 fixed windows from the `val` split; `eval_fn` returns `{"loss": ...}` here (Task 4 adds F1); `better` = lower loss.

- [ ] **Step 1: Failing tests** — `test_style_vector` (multi-hot, unknown fallback); `test_conditioning_shape`; `test_timing_forward_shape` (`[2, 128, 80]` → `[2, 128]`); `test_frame_targets_fuzzy` (one event → `[.., 0.25, 1.0, 0.25, ..]`; events one frame apart keep 1.0 on both); `test_window_padding_masked_in_loss`; `test_style_dropout_rate` (seeded, share of unknown-only vectors near 0.15 for styled entries); `test_overfit_two_maps` (tiny model, synthetic mel with energy bursts at event frames, loss below 0.05 within 300 CPU steps); `test_checkpoint_atomic_and_resume` (train 20 steps, resume, step continues at 20, weights equal); `test_resume_falls_back_when_last_is_truncated` (Review Focus 4); `test_load_checkpoint_weights_only`.
- [ ] **Step 2:** Run `tests/models` — fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Whole suite passes.
- [ ] **Step 5: Smoke run.** `.venv/Scripts/python -I -m bsmaker train timing --max-steps 50` on the real cache: report steps per second, GPU memory, the loss at steps 1 and 50, then delete `checkpoints\timing\`.
- [ ] **Step 6: Commit** — `feat(models): timing model, window dataset and training loop`.

### Task 4: Timing decoding and onset F1

**Files:**
- Create: `src/bsmaker/models/timing/decode.py`, `src/bsmaker/models/timing/metrics.py`
- Modify: `src/bsmaker/cli.py` (F1 inside `train timing`'s `eval_fn`; `better` = higher F1)
- Test: `tests/models/test_timing_decode.py`

**Interfaces:**
- `decode.py`: `MIN_FIRST_EVENT_S = 1.5`; `frame_probs(model: TimingModel, mel: np.ndarray, nps: float, styles: Iterable[str], stats: dict, device: str, hop: int | None = None) -> np.ndarray` (float32 `[frames]`; overlapping windows with hop = `window // 2`, overlaps averaged with a triangular weight); `pick_peaks(probs: np.ndarray, threshold: float, min_gap_frames: int = 3) -> np.ndarray` (frame indices of local maxima above the threshold; of two peaks closer than the gap the higher stays); `threshold_for_count(probs: np.ndarray, n_events: int, min_gap_frames: int = 3) -> float` (binary search, 30 iterations, over `[0, 1]` for the threshold whose peak count is closest to `n_events`); `class LoadedTiming` with `.model`, `.config`, `.stats`; `load_timing(path: Path, device: str) -> LoadedTiming`; `predict_events(loaded: LoadedTiming, mel: np.ndarray, nps: float, styles: Iterable[str], device: str, count_scale: float = 1.0) -> np.ndarray` (seconds, float64, sorted; target count = `nps × duration × coverage_median / (1 + lookup(stats, "double_share_by_bin", nps)) × count_scale`; events before `MIN_FIRST_EVENT_S` are dropped).
- `metrics.py`: `onset_f1(pred_s: Sequence[float], true_s: Sequence[float], tol_s: float = 0.025) -> tuple[float, float, float]` (precision, recall, F1; one-to-one greedy matching in time order).
- `eval_fn` for timing: on 64 fixed `val` difficulties run `frame_probs` over the whole song, decode with the count equal to the human event count, return mean `f1`, `precision`, `recall` and `loss`.

- [ ] **Step 1: Failing tests** — `test_pick_peaks_local_maxima_and_gap`; `test_threshold_matches_count` (synthetic probs with 40 clear peaks, asking for 10 / 40 → 10 / 40); `test_threshold_when_fewer_peaks_than_asked` (returns the lowest useful threshold, no error); `test_onset_f1_perfect_and_shifted` (shift 20 ms → F1 1.0, 30 ms → 0.0); `test_onset_f1_one_to_one` (two predictions near one truth → precision 0.5); `test_frame_probs_overlap_covers_tail` (length not a multiple of the hop; output length equals the mel length); `test_predict_drops_events_before_lead_in`; `test_predict_count_scale`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat(models): timing decoding and onset F1`.
- [ ] **[controller] Train timing.** `.venv/Scripts/python -I -u -m bsmaker train timing` in the background, log in `reports\train-timing.log`. Done when `best.pt` has val F1 ≥ 0.70; if the best F1 after 40 000 steps is below 0.70, run 40 000 more steps once, then record the number and go on (spec: 0.70 is the "model works" bar).

### Task 5: Pattern model, dataset and training

**Files:**
- Create: `src/bsmaker/models/pattern/__init__.py`, `src/bsmaker/models/pattern/model.py`, `src/bsmaker/models/pattern/dataset.py`
- Modify: `src/bsmaker/cli.py` (`train pattern`)
- Test: `tests/models/test_pattern_model.py`

**Interfaces:**
- `model.py`: `SLICE_FRAMES = 35` (±0.2 s), `N_NEXT = 4`, `BOS_TOKEN = 109` (input-only id for "no previous event"); `@dataclass(frozen=True) class PatternConfig: d_model: int = 384; n_layers: int = 8; n_heads: int = 6; context: int = 256; dropout: float = 0.1`; `class PatternModel(nn.Module)`:
  - `encode(mel_slices: Tensor[B, L, 35, 80], gaps: Tensor[B, L, 5], prev_left: Tensor[B, L], prev_right: Tensor[B, L], log_nps: Tensor[B], styles: Tensor[B, 8]) -> Tensor[B, L, d]` — causal over L; `gaps[..., 0]` is the gap to the previous event and `[..., 1:5]` the gaps to the next four, each as `log1p(seconds)`, 0 where there is none; conditioning enters as a prefix token;
  - `left_logits(h: Tensor[B, L, d]) -> Tensor[B, L, 109]`;
  - `right_logits(h: Tensor[B, L, d], left: Tensor[B, L]) -> Tensor[B, L, 109]`.
  - `pattern_loss(model, batch) -> tuple[Tensor, dict[str, float]]` (cross-entropy of left plus right given the true left, averaged over valid positions; dict has `ce_left`, `ce_right`).
- `dataset.py`: `class PatternSequences(Dataset)`: `__init__(cache: EventCache, entries, features_dir: Path, stats: dict, context: int, seed: int, style_dropout: float = 0.15, fixed: bool = False)`; an item is a random run of up to `context` consecutive events of one difficulty with the tensors `encode` needs, plus `left`, `right` (targets) and `valid`; `event_inputs(times_s: np.ndarray, mel: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]` (slices `[L, 35, 80]` normalised and zero-padded at song edges, gaps `[L, 5]`) — shared with the sampler in Task 6.
- CLI: `bsmaker train pattern [--max-steps 60000] [--batch-size 32] [--num-workers 4] [--device cuda]` → `checkpoints\pattern\`; validation = 256 fixed sequences from `val`; `eval_fn` returns `ce`, `ce_left`, `ce_right`; `better` = lower `ce`.

- [ ] **Step 1: Failing tests** — `test_pattern_shapes`; `test_encode_is_causal` (changing inputs at position k leaves outputs before k unchanged); `test_right_depends_on_left`; `test_gaps_layout` (first event has previous gap 0, last event has zeros for missing next gaps); `test_slices_padded_at_edges`; `test_prev_tokens_are_shifted_targets` (position 0 gets `BOS_TOKEN`); `test_overfit_two_sequences` (tiny model, loss below 0.1 within 400 CPU steps); `test_no_event_with_both_hands_empty_in_dataset`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Smoke run.** `train pattern --max-steps 50` on the real cache: steps per second, GPU memory, loss at steps 1 and 50; delete `checkpoints\pattern\`.
- [ ] **Step 6: Commit** — `feat(models): pattern model, sequence dataset and training`.
- [ ] **[controller] Train pattern** in the background (`reports\train-pattern.log`), after the timing training has finished (one GPU job at a time).

### Task 6: Masked sampling

**Files:**
- Create: `src/bsmaker/models/pattern/sample.py`
- Test: `tests/models/test_pattern_sample.py`

**Interfaces:**
- Consumes: `allowed(state, time_s, hand, cfg, other_token)`, `advance`, `initial_state`, `check`, `decode_token`, `event_inputs`, the `mask` profile config.
- Produces: `class LoadedPattern`; `load_pattern(path: Path, device: str) -> LoadedPattern`; `@dataclass(frozen=True) class SampleReport: n_events: int; n_skipped: int; n_single_hand_fallback: int; masked_mass_mean: float`; `sample_notes(loaded: LoadedPattern, event_times_s: np.ndarray, mel: np.ndarray, nps: float, styles: Iterable[str], seed: int, cfg: PlayabilityConfig, device: str, temperature: float = 0.9, top_p: float = 0.95) -> tuple[list[Note], SampleReport]`.
- Per event: left hand first — logits → mask with `allowed(state, t, 0, cfg)` → temperature → top-p → sample; then right hand with `right_logits(h, left)` and `allowed(state, t, 1, cfg, other_token=left)`. `NONE_TOKEN` is always allowed for a hand; if both hands end up `NONE_TOKEN` the right hand is re-sampled once with `NONE_TOKEN` excluded; if nothing else is allowed the event is skipped (`n_skipped`), and the state does not advance. The model sees only kept events as history; context is the last `config.context` kept events.
- `masked_mass_mean`: mean, over sampled tokens, of the model probability on tokens the mask removed.
- Returned notes: `beat = 0.0` (filled at export), `time_s` from the event time, colour 0 for left and 1 for right.

- [ ] **Step 1: Failing tests** (tiny random-weight model) — `test_sampled_map_has_zero_hard_violations` (500 events at mixed gaps, `check(to_events(notes), mask_cfg).hard == ()`); `test_never_returns_masked_token` (monkeypatched `allowed` that permits two tokens only); `test_all_blocked_events_are_skipped_and_terminate` (Review Focus 3: `allowed` permits only `NONE_TOKEN` for 50 events in a row → 50 skipped, function returns); `test_same_seed_same_notes`, `test_different_seed_differs`; `test_no_event_with_both_hands_empty`; `test_history_excludes_skipped_events`; `test_top_p_keeps_at_least_one_token`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat(models): pattern sampling under the playability mask`.

### Task 7: Beat grid

**Files:**
- Create: `src/bsmaker/audio/beats.py`
- Test: `tests/audio/test_beats.py`

**Interfaces:**
- `@dataclass(frozen=True) class Grid: bpm: float; offset_s: float; steady: bool`; `fit_grid(beat_times_s: np.ndarray) -> Grid` (least-squares line through beat index → time; `steady` when at least 16 beats and the median absolute residual is below 20 ms; otherwise `bpm` = 60 / median beat interval, `offset_s` = 0, `steady = False`); `detect_grid(audio: np.ndarray) -> Grid` (`librosa.beat.beat_track` at 22 050 Hz, then `fit_grid`; fewer than 4 beats → `Grid(120.0, 0.0, False)`); `snap_times(times_s: np.ndarray, grid: Grid, max_shift_s: float = 0.025) -> np.ndarray` (nearest position on the 1/16 or 1/12 beat grid measured from `offset_s`, kept unchanged when the shift would exceed the limit or the grid is not steady; two events must never land on the same time — the later one stays unsnapped); `to_beats(times_s: np.ndarray, bpm: float) -> np.ndarray` (`time × bpm / 60`, the game's own conversion with zero song offset).

- [ ] **Step 1: Failing tests** — `test_fit_grid_recovers_bpm_and_offset` (128 BPM, offset 0.37 s, 5 ms jitter); `test_fit_grid_unsteady_on_tempo_drift`; `test_detect_grid_on_click_track` (synthetic clicks at 120 BPM → bpm within 1, or its half/double, steady); `test_detect_grid_on_silence`; `test_snap_moves_within_limit_only`; `test_snap_prefers_nearest_of_sixteenth_and_triplet`; `test_snap_keeps_times_strictly_increasing`; `test_unsteady_grid_leaves_times_alone`; `test_to_beats`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat(audio): beat grid detection and snapping`.

### Task 8: `generate`

**Files:**
- Create: `src/bsmaker/generate.py`
- Modify: `src/bsmaker/cli.py`, `src/bsmaker/formats/writer.py` (only if `write_map` does not already build the folder under a temp name and rename it)
- Test: `tests/test_generate.py`, `tests/test_cli_generate.py`

**Interfaces:**
- Consumes: `decode_mono`, `log_mel`, `load_timing`, `predict_events`, `load_pattern`, `sample_notes`, `detect_grid`, `snap_times`, `to_beats`, `write_map`, `check`, `to_events`, `load_stats`, `load_playability_config(path, "mask")`.
- Produces: `MIN_AUDIO_S = 30.0`, `MIN_EVENTS = 20`, `DENSITY_TOLERANCE = 0.10`, `MAX_DENSITY_ROUNDS = 3`; `class GenerateError(Exception)`; `@dataclass(frozen=True) class GenerateResult: folder: Path; n_notes: int; nps: float; bpm: float; grid_steady: bool; rank: str; n_skipped: int; density_rounds: int; warnings: tuple[str, ...]`; `safe_title(text: str) -> str` (characters `<>:"/\|?*` and control characters removed, whitespace collapsed, trailing dots and spaces stripped, at most 80 characters, `"untitled"` when empty); `pick_rank(nps: float, stats: dict) -> str`; `pick_njs(nps: float, stats: dict) -> tuple[float, float]`; `generate_map(audio_path: Path, *, nps: float, styles: Sequence[str], seed: int, out_dir: Path, timing: LoadedTiming, pattern: LoadedPattern, mask_cfg: PlayabilityConfig, stats: dict, device: str, overwrite: bool = False, title: str | None = None) -> GenerateResult`.
- Flow: decode → `GenerateError` under `MIN_AUDIO_S` → mel → loop up to `MAX_DENSITY_ROUNDS`: `predict_events(count_scale)` → `GenerateError` under `MIN_EVENTS` → `sample_notes` → measured nps; if off by more than 10%, `count_scale *= target / measured` and repeat, keeping the closest attempt → grid → snap → beats → assert `check(...).hard == ()` (a violation is a bug: raise `RuntimeError`) → `write_map` into `out_dir / f"BSM - {safe_title} [nps {nps:g}, {style}]"` (style = the styles joined with `+`, `unknown` when none).
- Warnings (strings in the result and printed): grid not steady, events skipped by the mask, density still outside tolerance after the last round.
- CLI: `bsmaker generate <audio> --nps N [--style S ...] [--seed K] [--out DIR] [--overwrite] [--device cuda|cpu]`. `--nps` outside 0.5–12 or an unknown style → exit 2; missing audio file, missing checkpoint (`checkpoints\timing\best.pt`, `checkpoints\pattern\best.pt`), missing ffmpeg, `GenerateError`, existing folder without `--overwrite` → one-line message and exit 1. Default `--out` is `cfg.custom_levels`; default device is CUDA when available. Prints the result summary.

- [ ] **Step 1: Failing tests** (tiny random-weight checkpoints written by a fixture, synthetic 40 s audio with clicks) — `test_generate_end_to_end` (folder exists, `read_map` parses it with no problems, one Standard difficulty, notes sorted, zero hard violations); `test_generated_folder_has_only_expected_files`; `test_same_seed_same_map`; `test_short_audio_rejected`; `test_silent_audio_rejected` (Review Focus 1); `test_safe_title` (Review Focus 2 cases); `test_existing_folder_needs_overwrite`; `test_density_loop_rescales` (stub `predict_events` returning `count_scale`-proportional events → at most 3 rounds, result within tolerance); `test_density_warning_when_not_reached`; `test_pick_rank_and_njs_from_stats`; CLI: `test_cli_nps_out_of_range_exit_2` (Review Focus 5), `test_cli_unknown_style_exit_2`, `test_cli_missing_checkpoint_exit_1`, `test_cli_generate_to_out_dir`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat: generate command`.

### Task 9: `preview`

**Files:**
- Create: `src/bsmaker/preview.py`
- Modify: `src/bsmaker/cli.py`
- Test: `tests/test_preview.py`

**Interfaces:**
- `render_preview(folder: Path, out_path: Path, characteristic: str = "Standard", rank: str | None = None) -> Path` — decodes the map's audio to mono 22 050 Hz, adds a 15 ms click at every note (1 000 Hz for the left hand, 1 500 Hz for the right), peak-normalises to 0.9 and writes a 16-bit WAV with `soundfile`; `rank=None` picks the map's highest difficulty; a map without audio or without that difficulty → `ValueError`.
- CLI: `bsmaker preview <folder> [--rank R] [--out FILE]`; default output is `preview - <folder name>.wav` in the current directory, never inside the map folder.

- [ ] **Step 1: Failing tests** — `test_clicks_at_note_times` (energy in the click bands rises at each note time and not in between); `test_left_and_right_pitches_differ`; `test_output_not_clipped`; `test_missing_audio_raises`; `test_default_output_outside_map_folder`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat: preview command`.

### Task 10: `eval` and the evaluation report

**Files:**
- Create: `src/bsmaker/evaluate.py`, `docs/feedback/README.md` (how to write VR notes: one file per session, map folder, seed, what felt wrong and when)
- Modify: `src/bsmaker/cli.py`, `README.md` (create if missing; usage of `generate`, `preview`, `eval`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- `eval_timing(loaded: LoadedTiming, cache: EventCache, features_dir: Path, split: str, n: int, device: str) -> dict` (mean precision/recall/F1 at the human event count, plus F1 by nps bin); `eval_pattern(loaded: LoadedPattern, cache, features_dir, split, n, device) -> dict` (`ce`, `ce_left`, `ce_right`); `distribution(event_lists: Sequence[Sequence[Event]]) -> dict` (shares by column, row, direction and hand, double share); `eval_generated(timing, pattern, cache, features_dir, mask_cfg, split: str, n: int, seed: int, device: str) -> dict` (for `n` test difficulties: timing + sampling at the human nps and styles, no file output; returns generated vs human `distribution`, total-variation distance per axis, soft-rule rates per 1 000 notes for both, skipped-event rate, `masked_mass_mean`, share within ±10% of the target nps); `render_eval_report(results: dict) -> str` (markdown).
- CLI: `bsmaker eval [--split val|test] [--n 200] [--n-generated 10]` writes `reports\eval-<split>.md` and prints the headline numbers.

- [ ] **Step 1: Failing tests** — `test_distribution_shares_sum_to_one`; `test_tv_distance_zero_for_identical`; `test_eval_timing_runs_on_tiny_cache`; `test_eval_generated_reports_zero_hard_violations_and_density_share`; `test_render_report_has_all_sections`.
- [ ] **Step 2–4:** fail, implement, whole suite passes.
- [ ] **Step 5: Commit** — `feat: eval command and report`.
- [ ] **[controller] Evaluate and generate.** After both trainings: `bsmaker eval --split val`, then `--split test`; generate three maps from songs the user names (or three test-split songs of different genres) into CustomLevels with `preview` files; hand the report and the folder names to the user for the VR test.

---

## Done criteria for Plan 2

- `.venv/Scripts/python -m pytest -q` passes.
- `checkpoints\timing\best.pt` and `checkpoints\pattern\best.pt` exist; the timing F1 on validation is recorded (target ≥ 0.70).
- `reports\eval-val.md` and `reports\eval-test.md` exist.
- `bsmaker generate` produced at least three `BSM - …` folders that load in Beat Saber 1.37.1 with no mods, each with zero hard violations and density within ±10% or a printed warning.
- The user's VR notes are in `docs/feedback/` (spec success criterion 5) — this one is the user's step.
