# Plan 4: layers (arcs, chains, walls, bombs, lights, multi-difficulty) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generated map carries, besides notes, three optional learned layers — note ornaments (arcs, chains), decoration (walls, bombs, plus rare rule-based gameplay obstacles) and a lightshow with a chosen environment — and can hold several difficulties.

**Architecture:** Layers repeat the proven "when + what" split of the note models. A generic multi-channel frame tagger (the timing model with extra note-track inputs) says when layer events happen; a generic causal sequence model says what each event is; a small bidirectional tagger over the note sequence adds arcs and chains. Layers are generated after the notes and never change them. Hard rules are enforced as sampling masks and re-checked afterwards.

**Tech Stack:** Python 3.13, PyTorch, numpy, typer, pytest; training through the Colab CLI queue from the Pi (`docs/colab-cli.md`).

**Spec:** `docs/superpowers/specs/2026-11-16-etap3-warstwy-design.md` (Polish). Measurements: `docs/superpowers/specs/2026-11-16-etap3-pomiary.md`, `...-etap4-pomiary.md`.

## Global Constraints

- The note models, their checkpoints and the note sampler are not changed by this plan. With every layer off, `generate` output is byte-identical to today's for the same seed (a test pins this).
- Splits are frozen; never run `bsmaker data resplit`. The manifest and the event cache are not rebuilt.
- v4 difficulty files are skipped by the layer parser (counted, not an error).
- Light channels (index: meaning, v2 `_type` / v3 `et`): 0 back lasers (0), 1 ring lights (1), 2 left lasers (2), 3 right lasers (3), 4 centre lights (4), 5 ring spin (8), 6 ring zoom (9), 7 left laser speed (12), 8 right laser speed (13).
- Light values for channels 0–4: token 0 = off; tokens 1–6 = (on, flash, fade) × (colour A = v2 values 5–7 "red", colour B = v2 values 1–3 "blue"), in that order: on-A, flash-A, fade-A, on-B, flash-B, fade-B. v2 value 4 and 8 (transition) and white values 9–12 map to on of the nearest colour (4→B, 8→A, 9–12→A). Channels 7–8: tokens 7–16 = speed 0–9 (larger clipped to 9). Channels 5–6 always token 17.
- Supported environments (exact Info names): `DefaultEnvironment`, `BigMirrorEnvironment`, `NiceEnvironment`, `TriangleEnvironment`, `PanicEnvironment`, `OriginsEnvironment`, `DragonsEnvironment`, `TimbalandEnvironment`, `BTSEnvironment`, `KaleidoscopeEnvironment`. CLI `--tlo` takes the short name without the `Environment` suffix, case-insensitive.
- Lightshow training eligibility: environment in the supported list, no v3 group-lighting boxes, at least 100 light events per minute.
- Wall classes (rules from the survey, in order): flash — duration ≤ 0 or width ≤ 0 or duration < 0.05 s; side — covers neither column 1 nor 2; everything else is "centre" and is excluded from training.
- Wall value token = side variant (0 = edge column, 1 = just outside the grid) × height variant (0 = full: y 0, h 5; 1 = high: y 2, h 3) × duration bucket (index into 1/8, 1/4, 1/2, 1, 2, 4 beats, nearest in log scale) → 2 × 2 × 6 = 24 tokens. Width is always 1.
- Bomb value token = index into a vocabulary of 12-bit cell masks (bit `y * 4 + x`), the most frequent masks covering 95% of bomb groups, capped at 127 entries, built from train difficulties and stored in `layer_stats.json`.
- Hard rules: a wall never covers a cell holding a note from 0.1 s before its start to 0.1 s after its end; a bomb never sits in the cell of a note within ±0.4 s nor in the neighbouring cell along that note's cut direction (both ways); a chain needs the same hand's next note at least 0.25 s later and its cells free of other notes in between; an arc never ends after the same hand's next note.
- Levels: 0 = layer off; 1 / 2 / 3 = 25th / 50th / 75th percentile of the human per-minute rate for the map's nps bin, among train difficulties that have the element. Auto ornaments level: 2 below 4.5 nps, 1 for 4.5–7, 0 above. Default lights level: 2.
- CLI option names are Polish, as in the spec: `--levels`, `--ozdoby`, `--swiatla`, `--tlo`.
- Layer checkpoints live in `<data_dir>/checkpoints/layers/<name>/best.pt` with names `ornaments`, `decor-when`, `decor-what`, `lights-when`, `lights-what`. A missing checkpoint switches that layer off with a warning; it is never an error.
- Tests use synthetic data only; real-data commands are run by the controller (Tasks 15–16), in isolated mode (`.venv/Scripts/python -I -m bsmaker ...`).
- No new dependencies. Commits: `git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit`; never push.

## Execution order

Foundation, one writer: Tasks 1 → 2 → 3. Then three tracks in separate git worktrees, in parallel: **A** Tasks 4 → 5; **B** Tasks 6 → 7; **C** Tasks 8 → 9 → 10. The controller merges the tracks into the plan branch, then one writer does Tasks 11 → 12 → 13 → 14. Tasks 15–16 are the controller's.

## Review Focus

1. A map whose walls, bombs or light events carry non-numeric, missing or huge fields (the data is untrusted) → the layer parser drops that object and counts it; no crash, no NaN time reaches the cache (Task 1).
2. A song where the beat grid is not steady → wall durations and arc lengths (defined in beats) fall back to the detected BPM without snapping; the layers still generate (Tasks 5, 11).
3. A very sparse or very dense map (0.5 nps, 12 nps) and a 31-second song → every layer returns a valid, possibly empty, result (Tasks 5, 11).
4. Requested level counts that cannot be met because the rule mask forbids almost everything → fewer elements and a warning, never a rule violation or an endless loop (Tasks 5, 11).
5. `--levels` with the same level twice, an unknown name, or levels whose target densities fall in the same rank → clear usage error (Task 12).

---

### Task 1: Layer objects in the reader

**Files:** modify `src/bsmaker/formats/model.py`, `src/bsmaker/formats/reader.py`; test `tests/formats/test_reader_layers.py`.

**Interfaces — produces:**
- Frozen dataclasses in `model.py`: `Obstacle(time_s, beat, duration_s, duration_beats, x, y, w, h)`; `Bomb(time_s, beat, x, y)`; `Arc(time_s, beat, color, x, y, d, tail_time_s, tail_beat, tx, ty, td)`; `Chain(time_s, beat, color, x, y, d, tail_time_s, tail_beat, tx, ty, segments, squish)`; `LightEvent(time_s, beat, type, value, float_value)`; `Layers(bombs, obstacles, arcs, chains, lights, group_lighting: bool, dropped: int)` (tuples).
- `Difficulty` gains `layers: Layers | None = None`; `BeatMap` gains `environment: str = ""`.
- `read_map(folder, *, layers: bool = False)`: with `layers=False` nothing changes (indexing speed). With `True`, each v2/v3 difficulty gets `Layers`; v4 gets `None`. Times use the same tempo logic as notes. v2 walls: `_type` 0 → y 0, h 5; 1 → y 2, h 3; other types → dropped. v2 bombs are `_notes` of `_type` 3. Light events exclude BPM events (type 100). `group_lighting` is true when any of the three v3 box-group lists is non-empty. Objects with a non-finite or non-numeric field are dropped and counted in `dropped`. Each list is sorted by time.

- [ ] Tests: one synthetic v2 map and one v3 map with every object kind, checked field by field; tempo-map times for a wall and an arc tail; `layers=False` leaves `layers is None` and the notes unchanged; malformed objects (string time, null x, 1e308 duration) are dropped and counted; v4 → `None`; environment read from Info.
- [ ] Fail, implement, pass, full suite. Commit `feat(formats): parse walls, bombs, arcs, chains and light events`.

### Task 2: Layer tokens, classes, cache and statistics

**Files:** create `src/bsmaker/formats/layer_tokens.py`, `src/bsmaker/data/layer_classes.py`, `src/bsmaker/data/layers_cache.py`, `src/bsmaker/data/layer_stats.py`; modify `src/bsmaker/config.py` (`layers_dir = data_dir / "layers"`), `src/bsmaker/cli.py` (`data build-layers`); tests under `tests/data/`, `tests/formats/`.

**Interfaces — produces:**
- `layer_tokens.py`: constants from Global Constraints (`LIGHT_CHANNELS = 9`, `LIGHT_VOCAB = 18`, `WALL_VOCAB = 24`, `SUPPORTED_ENVIRONMENTS`, `DURATION_BEATS`); `light_event_to_token(type, value) -> tuple[int, int] | None` (channel, token); `token_to_light_event(channel, token) -> tuple[int, int]` (type, value; colour A → red values, B → blue); `wall_to_token(ob: Obstacle, bpm) -> tuple[int, int] | None` (side 0 left / 1 right, token) for flash and side walls only; `token_to_wall(side, token, time_s, bpm) -> Obstacle`; `cells_to_mask(cells) -> int`, `mask_to_cells(mask) -> list[tuple[int, int]]`.
- `layer_classes.py`: `wall_class(ob) -> str` ("flash", "side", "centre").
- `layers_cache.py`: `build_layer_cache(conn, layers_dir, events_dir, progress_every=0) -> dict` iterates the entries of the EXISTING event cache index (same order, same `i`), reads each map once with `layers=True`, and writes `layers.npz` + `index.json` atomically. Arrays, each with an offsets array per difficulty: lights `light_t` (f32 s), `light_ch` (u8), `light_tok` (u8), same (time, channel) duplicates keep the last; walls `wall_t`, `wall_side`, `wall_tok`; bombs grouped by identical time `bomb_t`, `bomb_mask` (u16); ornaments aligned with the difficulty's notes sorted by time then colour: `orn_t`, `orn_hand` (u8), `orn_token` (u8 note token), `orn_chain` (u8: 0 none, else segments clipped to 3–7), `orn_arc_len` (u8: 0 none, 1–6 index into 0.5, 1, 1.5, 2, 3, 4 beats nearest), `orn_arc_cell` (u8 `ty*4+tx`), `orn_arc_dir` (u8). An arc or chain belongs to the note with the same colour, cell and time within 1 ms; unmatched ones are counted and skipped. Index record per entry: `map_key`, `characteristic`, `rank`, `schema`, `environment`, `lights_ok` (eligibility rule), `has_v3` (schema v3), `duration_s`, counts per element, `bpm`.
- `LayerCache.load(layers_dir)`; `.lights(i)`, `.walls(i)`, `.bombs(i)`, `.ornaments(i)` → tuples of arrays; `.meta(i)` → record.
- `layer_stats.py`: `compute_layer_stats(layer_cache, event_cache, stats) -> dict` written to `<repo>/layer_stats.json`: bomb mask vocabulary; for each element (`arcs`, `chains`, `walls`, `bomb_groups`, and each light channel) and each nps bin of `stats["nps_bins"]` the 25/50/75th percentiles of per-minute rates over train difficulties having the element (empty bins filled from the nearest); share of train difficulties having each element. `load_layer_stats(path)`; `level_rate(stats, element, nps, level) -> float` (0 for level 0).
- CLI `bsmaker data build-layers`: builds the cache and stats, prints counts per element, per wall class, dropped objects, v4 skipped, lightshow-eligible difficulties, the bomb vocabulary size and its coverage.

- [ ] Tests: token round trips for every light type/value and every wall variant; duration bucketing; wall classes on hand-made walls; cache built from three synthetic map folders (v2, v3, v4) aligned with a synthetic event cache — arrays and offsets per difficulty, ornaments matched to the right notes, v4 entry present with empty arrays; duplicate light events; stats percentiles on a toy cache; `level_rate` for levels 0–3 and an empty bin.
- [ ] Fail, implement, pass, full suite. Commit `feat(data): layer cache and statistics`.

### Task 3: Writer — layers, environment, several difficulties

**Files:** modify `src/bsmaker/formats/writer.py`; test `tests/formats/test_writer.py`.

**Interfaces — produces:**
- `LevelData(rank, njs, njs_offset, notes, bombs=(), obstacles=(), arcs=(), chains=())` (frozen dataclass).
- `write_map_multi(folder, *, title, artist, bpm, levels: Sequence[LevelData], lights: Sequence[LightEvent] = (), environment: str = "DefaultEnvironment", audio_src, overwrite=False) -> Path`: one v3 difficulty file per level (`<Rank>Standard.dat`), Info with all levels in ascending rank order, the same lights in every file (`basicBeatmapEvents` with `b`, `et`, `i`, `f`), `bombNotes`, `obstacles` (`b`, `x`, `y`, `d`, `w`, `h`), `sliders`, `burstSliders` filled from the objects (beats from `time_s * bpm / 60` when the object's `beat` was not set by the generator — the generator sets seconds; the writer converts). Duplicate ranks → `ValueError`. Existing safety rules (only `BSM - ` folders are replaced) unchanged.
- `write_map(...)` keeps its signature and delegates; its output for notes-only input stays byte-identical (test against a stored expected file content built before the change).

- [ ] Tests: byte-identity of the old path; a two-level map read back with `read_map(layers=True)` equals what was written (times within 1 ms, every field); environment in Info; duplicate rank error; negative-duration wall and out-of-grid bomb rejected with `ValueError`.
- [ ] Fail, implement, pass, full suite. Commit `feat(formats): write layers, environment and several difficulties`.

### Task 4 (track A): Ornament tagger — model, dataset, training

**Files:** create `src/bsmaker/models/ornaments/{__init__,model,dataset}.py`; modify `src/bsmaker/cli.py` (`train ornaments`); tests `tests/models/test_ornaments.py`.

**Interfaces — produces:**
- `OrnamentConfig(d_model=256, n_layers=6, n_heads=4, context=256, dropout=0.1)`; `OrnamentModel`: bidirectional transformer over a run of notes; per-note inputs: hand embedding, note token embedding, mel slice encoder (reuse `_SliceEncoder` and the slice code of the pattern dataset), gaps to the previous and next note of any hand and of the same hand (log1p seconds), `log_nps`; heads `chain` (6 classes: none, 3–7 segments), `arc_len` (7), `arc_cell` (12), `arc_dir` (9). Loss: sum of cross-entropies; the two arc-geometry heads only on notes that have an arc. Class weights for the "none" classes: 0.2.
- `OrnamentNotes(layer_cache, entries, features_dir, stats, context, seed, fixed=False)` over entries with `has_v3` and at least one arc or chain.
- `ornament_step`, `ornament_eval` (returns `loss`, `chain_f1`, `arc_f1` for the any-vs-none decision at threshold 0.5), `ornament_better` (lower loss).
- CLI `bsmaker train ornaments` with the options of the existing train commands (`--max-steps` default 30000, `--batch-size`, `--num-workers`, `--device`, `--eval-every`, `--save-every`, `--out`, `--lr`, `--weighting`), default out `<data_dir>/checkpoints/layers/ornaments`.

- [ ] Tests: shapes; loss decreases on a tiny synthetic set where chains follow a visible pattern (e.g. every note with gap > 1 s is a chain) and the tagger learns it in under 200 CPU steps; masking of padded positions; arc heads ignored where no arc; CLI smoke run of a few steps on a synthetic cache.
- [ ] Fail, implement, pass, full suite. Commit `feat(models): ornament tagger`.

### Task 5 (track A): Ornament sampling

**Files:** create `src/bsmaker/models/ornaments/sample.py`; test `tests/models/test_ornament_sample.py`.

**Interfaces — produces:** `load_ornaments(path, device)`; `sample_ornaments(loaded, notes: Sequence[Note], mel, nps, level: int, layer_stats: dict, bpm: float, seed: int, device) -> tuple[list[Arc], list[Chain], OrnamentReport]`. Level 0 → empty. Target counts = `level_rate(...)` × duration in minutes for `arcs` and `chains`. Candidates = notes passing the hard rules (chain: same-hand next note ≥ 0.25 s later, chain cells — head cell and the next cell along the cut — free of other notes until then, not a dot note; arc: tail time = head + length beats must be ≤ the same hand's next note time, else the longest allowed shorter bucket, else no arc). Among candidates pick the highest any-probability up to the target count (ties by time); a note gets at most one of chain / arc (the more probable). Arc geometry by sampling the two heads at temperature 0.8 with seed; chain tail = one cell along the cut direction clipped to the grid, `squish` 1.0, tail time = head + 1/24 beat. Report: requested and placed counts, rejected by rule.

- [ ] Tests (tiny random model): level 0 empty; counts never exceed the target and equal it when enough candidates exist; every returned object satisfies the hard rules (property test over random note lists, including 0.5 nps, 12 nps and three notes total); determinism per seed; dot notes never chains.
- [ ] Fail, implement, pass, full suite. Commit `feat(models): ornament sampling with rate calibration`.

### Task 6 (track B): Generic "when" model

**Files:** create `src/bsmaker/models/layers/{__init__,spec,when}.py`; modify `src/bsmaker/cli.py` (`train layer-when`); tests `tests/models/test_layer_when.py`.

**Interfaces — produces:**
- `spec.py`: `LayerSpec(name, n_channels, vocab_size, channel_tokens: tuple[tuple[int, ...], ...], min_gap_frames: tuple[int, ...], rate_elements: tuple[str, ...], eligible: Callable[[dict], bool], extract: Callable[[LayerCache, int], tuple[np.ndarray, np.ndarray, np.ndarray]])` returning (time_s, channel, token) sorted by time then channel. Two instances: `LIGHTS` (9 channels, vocab 18, eligible = `lights_ok`, min gap 2 frames) and `DECOR` (3 channels: wall left, wall right, bombs; vocab = 24 wall tokens followed by the bomb vocabulary, size read from `layer_stats`; eligible = has any wall or bomb; min gap 3 frames) via `decor_spec(layer_stats)`; `get_spec(name, layer_stats)`.
- `when.py`: `WhenConfig(n_mels=80, n_channels, d_model=256, n_layers=6, n_heads=4, window=1024, dropout=0.1)`; `WhenModel` = the timing model's stem and encoder with two extra input channels concatenated to the mel along frequency (left-hand and right-hand note onsets per frame, fuzzy like the timing targets) and conditioning on `log_nps`; output logits (B, T, C). Loss: BCE on fuzzy targets, pos_weight 5, per channel. `WhenWindows(layer_cache, event_cache, entries, features_dir, stats, spec, window, seed, fixed=False)`; `when_step`, `when_eval` (loss and mean per-channel F1 at ±25 ms using each difficulty's human count per channel), `when_better` (higher F1).
- `load_when(path, device)`; `predict_layer_events(loaded, mel, notes, nps, counts: Sequence[int], device) -> list[np.ndarray]` — per channel the event times, using `threshold_for_count` / `pick_peaks` per channel with the spec's min gap, nothing before 1.5 s.
- CLI `bsmaker train layer-when --layer lights|decor` (options as other train commands; `--max-steps` default 30000; default out `<data_dir>/checkpoints/layers/<layer>-when`); `train_options` records the layer.

- [ ] Tests: note-track construction; shapes for both specs; a synthetic task where channel 0 fires on left-hand notes and channel 1 on bright mel frames is learned in under 300 CPU steps (F1 > 0.8); `predict_layer_events` returns the requested counts (±1) and respects min gap and the lead-in; zero counts → empty arrays.
- [ ] Fail, implement, pass, full suite. Commit `feat(models): multi-channel layer timing model`.

### Task 7 (track B): Generic "what" model

**Files:** create `src/bsmaker/models/layers/what.py`; modify `src/bsmaker/cli.py` (`train layer-what`); tests `tests/models/test_layer_what.py`.

**Interfaces — produces:**
- `WhatConfig(n_channels, vocab_size, d_model=384, n_layers=8, n_heads=6, context=256, dropout=0.1)`; `WhatModel`: causal transformer over the layer's events in (time, channel) order; per-event inputs: channel embedding, mel slice (same slice encoder), gap to the previous event and to the next four (log1p s), previous event's token (BOS first), note context (counts of left and right notes within ±0.25 s, log1p), `log_nps`; one head over the vocabulary with tokens outside the event's channel set to −inf.
- `WhatSequences(layer_cache, event_cache, entries, features_dir, stats, spec, context, seed, fixed=False)`; `what_step`, `what_eval` (cross-entropy `ce`), `what_better` (lower).
- `load_what(path, device)`; `sample_layer_values(loaded, times, channels, mel, notes, nps, seed, device, allowed: Callable[[int, float, int], np.ndarray] | None = None, temperature=1.0, top_p=0.99) -> tuple[np.ndarray, int]` — tokens per event and the number of events skipped because `allowed(event_index, time_s, channel)` (a boolean mask over the vocabulary, ANDed with the channel mask) left nothing; skipped events are removed from the context.
- CLI `bsmaker train layer-what --layer lights|decor` (`--max-steps` default 40000; default out `.../layers/<layer>-what`).

- [ ] Tests: channel masking (a token of another channel is never sampled); a synthetic rule (token depends on the previous token of the same channel) is learned in under 300 CPU steps; `allowed` is honoured and skipped events are reported; determinism per seed; empty input.
- [ ] Fail, implement, pass, full suite. Commit `feat(models): layer value model`.

### Task 8 (track C): Layer playability rules

**Files:** create `src/bsmaker/playability/layers.py`; modify `src/bsmaker/cli.py` (`check` prints layer counts and violations when the map has layers); tests `tests/playability/test_layers.py`.

**Interfaces — produces:** `wall_cells(ob) -> set[tuple[int, int]]` (clipped to the grid); `wall_ok(ob, notes) -> bool`; `bomb_forbidden_cells(notes, time_s) -> set[tuple[int, int]]`; `bomb_mask_ok(mask, notes, time_s) -> bool`; `chain_ok(chain, notes) -> bool`; `arc_ok(arc, notes) -> bool`; `check_layers(notes, layers: Layers) -> list[Violation]` with rules named `wall_on_note`, `bomb_in_swing`, `chain_conflict`, `arc_overrun`; constants from Global Constraints. Note lookups by time must be O(log n) (sorted arrays + bisect), because masks are evaluated per sampled event.

- [ ] Tests: each rule at its boundary (0.1 s, 0.4 s, 0.25 s, neighbouring cell along the cut for all nine directions including dot = no neighbour); walls outside the grid cover no cell; `check` CLI on a synthetic map with one violation of each kind prints four violations and exits as it does today for note violations.
- [ ] Fail, implement, pass, full suite. Commit `feat(playability): rules for walls, bombs, arcs and chains`.

### Task 9 (track C): Gameplay obstacles from rules

**Files:** create `src/bsmaker/obstacles_rules.py`; test `tests/test_obstacles_rules.py`.

**Interfaces — produces:** `gameplay_obstacles(notes, bpm, duration_s, level: int, seed: int) -> list[Obstacle]`. Level < 2 → empty. At most one sequence per 45 s of song (level 3: per 30 s), none in the first 10 s or last 5 s. Side-step: a wall on two columns of one side (x 0, w 2 or x 2, w 2; y 0, h 5) for 1–2 beats, only where every note from 0.2 s before to 0.2 s after the wall lies in the other two columns and at least two such notes exist. Duck: a high wall across the middle (x 0, w 4, y 2, h 3) for 1 beat, only in a stretch with no notes for at least 2 beats, centred in it. Candidate spots are found deterministically; the seed picks among them; consecutive sequences alternate sides. Every result passes `wall_ok`.

- [ ] Tests: level gating; spacing and edge margins; side-step only with notes on the other side; duck only in gaps; all results pass `wall_ok` (property test over random note lists); no candidates → empty; determinism.
- [ ] Fail, implement, pass, full suite. Commit `feat: rule-based gameplay obstacles`.

### Task 10 (track C): Rule-based lights (fallback and baseline)

**Files:** create `src/bsmaker/lights_rules.py`; test `tests/test_lights_rules.py`.

**Interfaces — produces:** `rule_lights(mel, notes, bpm, offset_s, duration_s, level: int, seed: int) -> list[LightEvent]`. Level 0 → empty. From a spectral-flux onset envelope of the mel (reuse `onset_envelope` from `bsmaker.data.align`): centre lights (type 4) "on" at the strongest onsets, back lasers (type 0) flash on every bar (4 beats), ring lights (type 1) fade on beats, left/right lasers (types 2, 3) follow left/right-hand notes, ring spin (type 8) on bar starts, zoom (type 9) every 8 bars; colour swaps every 8 beats; each group gets an "off" after 1 beat without a new event; laser speeds (types 12, 13) set every 4 bars from local note density (1–7). Level scales how many onsets and beats are used (1: bars only; 2: beats; 3: half-beats and all onsets above the median). Output sorted, no two events with the same (time, type).

- [ ] Tests: level 0 empty; event counts grow with level; only supported types and values; sorted and unique per (time, type); works with no notes and with a 31 s song; determinism.
- [ ] Fail, implement, pass, full suite. Commit `feat: rule-based lightshow`.

### Task 11: Decoration and lights generation

**Files:** create `src/bsmaker/layers_generate.py`; test `tests/test_layers_generate.py`.

**Interfaces — consumes** Tasks 2, 6–10. **Produces:**
- `LayerModels(ornaments=None, decor_when=None, decor_what=None, lights_when=None, lights_what=None)`; `load_layer_models(checkpoints_dir, device) -> tuple[LayerModels, list[str]]` (warnings for missing ones).
- `generate_decor(models, mel, notes, nps, bpm, duration_s, level, layer_stats, seed, device) -> tuple[list[Obstacle], list[Bomb], dict]`: counts per channel from `level_rate` (`walls` split evenly left/right, `bomb_groups`), times from `predict_layer_events`, values from `sample_layer_values` with `allowed` built from `wall_ok` / `bomb_mask_ok` for every token of the channel (precompute per event), tokens decoded with `token_to_wall` / `mask_to_cells`; then `gameplay_obstacles` appended (they must not overlap a decorative wall in time on the same side — drop the decorative one). Returns a report (requested, placed, skipped by rules).
- `generate_lights(models, mel, notes, nps, bpm, offset_s, duration_s, level, layer_stats, seed, device, use_rules=False) -> tuple[list[LightEvent], dict]`: model path as above with the `LIGHTS` spec (no rule mask); `use_rules=True` or missing light models → `rule_lights`.
- Both run under `torch.inference_mode()` and return empty results for level 0.

- [ ] Tests with tiny random models: level 0 → empty; every wall and bomb passes `check_layers`; missing models → empty decor with a report flag, rule lights for lights; gameplay obstacles never overlap decorative walls; an unsteady grid and a 31 s song work.
- [ ] Fail, implement, pass, full suite. Commit `feat: decoration and lights generation`.

### Task 12: `generate` integration and options

**Files:** modify `src/bsmaker/generate.py`, `src/bsmaker/cli.py`; tests `tests/test_generate.py`, `tests/test_cli.py`.

**Interfaces — produces:**
- `generate_map(..., levels: Sequence[str] | None = None, ornaments: int | None = None, lights: int = 0, environment: str = "DefaultEnvironment", layer_models: LayerModels | None = None, layer_stats: dict | None = None, lights_rules: bool = False)`. `levels=None` keeps the single-level `nps` behaviour. With levels, each rank's target nps is `stats["rank_median_nps"][rank]`; notes are generated per level with the existing density loop (same seed); ornaments per level; decoration and lights once, from the densest level's notes; `check_layers` must return no violations (assert, like the note checks); written with `write_map_multi`. Folder name: `BSM - <title> [<ranks joined by +> | nps X, <style>]` — unchanged for the single-level case. `GenerateResult` gains per-level results and layer reports. `ornaments=None` → auto level from the level's nps. Library defaults keep layers off so existing callers are unchanged; the CLI turns them on.
- CLI: `--levels hard,expert` (names case-insensitive: easy, normal, hard, expert, expertplus; mutually exclusive with `--nps`; duplicates and unknown names are usage errors), `--ozdoby 0-3` (default auto), `--swiatla 0-3` (default 2), `--tlo NAME` (default Default; validated), `--swiatla-reguly` (force rule lights). The summary line lists per level notes / nps / arcs / chains and, once, walls / bombs / light events and the environment. Missing layer checkpoints or `layer_stats.json` print one warning each and switch the layer off.

- [ ] Tests: with all layers off the written difficulty file is byte-identical to the pre-change output for the same seed (generate once before changing code and store the bytes in the test as a fixture produced by the tiny test models); two levels produce two files and one Info; layers present and rule-clean with tiny random layer models; option validation errors; missing checkpoints warn and still write a map.
- [ ] Fail, implement, pass, full suite. Commit `feat: generate with levels, ornaments, decoration, lights and environment`.

### Task 13: Evaluation of layers

**Files:** modify `src/bsmaker/evaluate.py`, `src/bsmaker/cli.py` (`eval --layers`); test `tests/test_evaluate.py`.

**Interfaces — produces:** `eval_layers(models, layer_stats, timing, pattern, event_cache, layer_cache, features_dir, mask_cfg, split, n, seed, device, entries=None) -> dict`: on `n` difficulties (one per song, default 30) generate all layers on the HUMAN notes at levels 1, 2, 3 and report, per layer and level: elements per minute generated vs the human percentiles; for lights the share of events within 30 ms of a quarter beat (human BPM) and the onset ratio (human reference 0.76 and 2.5), total-variation distance of the per-channel token distribution vs human, and the same numbers for `rule_lights`; for decoration the wall-token and bomb-mask TV distance; for ornaments chain and arc rates and the TV distance of arc length; rule violations (must be 0); events skipped by rule masks. New report section "Layers"; numbers also in the JSON. `eval` runs it only with `--layers` and only for layers whose checkpoints exist.

- [ ] Tests with tiny models and a synthetic layer cache: section present with `--layers`, absent without; a missing layer is reported as "not trained"; rates increase with level for a model that fires everywhere; violations are zero by construction.
- [ ] Fail, implement, pass, full suite. Commit `feat: layer evaluation`.

### Task 14: Colab bundle and queue for layers

**Files:** modify `scripts/make_colab_bundle.py`, `scripts/colab/vm_setup.py`, `docs/colab-cli.md`; create `scripts/colab/run_queue.sh`; tests `tests/test_colab_bundle.py`.

**Interfaces — produces:**
- `make_colab_bundle.py --layers --data-dir PATH --out-dir PATH` writes `bsmaker-layers.tar` with the `layers/` folder (regular files only, no temp files); the code-zip path list gains `layer_stats.json`.
- `vm_setup.py`: optional third archive `bsmaker-layers.tar` (marker `.layers.done` with size and mtime like the delta); absent → a note, not an error; when present, verifies that the layers index has the same number of entries as the events index.
- `vm_train.py` accepts `--model` values `ornaments`, `layer-when`, `layer-what` (passed through to `bsmaker train <model>`), output folder `<drive-dir>/checkpoints/layers/<RUN>`.
- `scripts/colab/run_queue.sh`: the queue runner used on 2026-11-15 (reads `queue.txt` line by line: `<model> <run-name> <options…>`, up to three tries per run, downloads `best.pt` and `log.jsonl` to `ckpt/<run>/`, then `drive.flush_and_unmount()` and `colab stop`), taken from `~/bsm-colab/run_queue.sh` on the Pi and parameterised by the remote results folder; documented in `docs/colab-cli.md` together with the `BROWSER=echo colab drivemount -s train` note.

- [ ] Tests: layers tar content; setup with and without the layers tar, entry-count mismatch fails before pip; `vm_train --dry-run` command lines for the three new model kinds; `bash -n scripts/colab/run_queue.sh` passes.
- [ ] Fail, implement, pass, full suite. Commit `feat: Colab bundle and queue for layer models`.

### Task 15: Data run (controller, real data)

- [ ] `bsmaker data build-layers`; record every printed count in the ledger. Sanity: lightshow-eligible difficulties ≥ 8 000; bomb vocabulary coverage ≥ 0.95; share of difficulties with walls ≈ 0.9 and bombs ≈ 0.5 (survey values) — a large deviation means a parser bug: stop and fix.
- [ ] Spot check five maps per element against the survey's example keys (print parsed objects for a few seconds of each).
- [ ] Commit `layer_stats.json`; build the code zip and `bsmaker-layers.tar`; tell the user the two files to upload (replace the old code zip).

### Task 16: Training, evaluation, maps (controller)

- [ ] One Colab session (A100 high-mem); the user runs `BROWSER=echo ~/.local/bin/colab drivemount -s train`; `vm_setup.py`; a 20-step probe of each of the five model kinds.
- [ ] Queue: `ornaments orn`, `layer-when lights-when --layer lights`, `layer-what lights-what --layer lights`, `layer-when decor-when --layer decor`, `layer-what decor-what --layer decor`. Budget 70 compute units; report usage before and after; the queue stops the session.
- [ ] Copy each `best.pt` to `<data_dir>/checkpoints/layers/<name>/best.pt`; `bsmaker eval --split val --layers --tag layers`; record the table. Compare model lights with rule lights; if the model is worse on both alignment numbers, default to `--swiatla-reguly` and say so.
- [ ] Regenerate the user's songs with `--levels hard,expert` (and the single-level versions he already knows), layers on, into `CustomLevels`; `bsmaker check`; zips to `previews/`. Keep the previous zips in `previews/gen2-notes/`.
- [ ] Ask the user to judge each layer separately (ornaments, decoration, gameplay obstacles, lights) in the preview and in VR; record in `docs/feedback/`; switch off by default any layer judged bad; then superpowers:finishing-a-development-branch.
