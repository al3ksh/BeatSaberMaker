# Flow layer (snake + dot stacks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rule-based layer that puts swing-extension dot stacks and occasional snake runs into generated maps, without breaking playability.

**Architecture:** A new module `src/bsmaker/flow_rules.py` post-processes the note list of each level. It runs in `generate_map` right after `_generate_level_notes` and before ornaments. Every change is validated with the existing `check()`, using a mask profile with `slider_gap_s = 0.12`; a change that fails is reverted. The CLI gets `--waz 0-3`, defaulting to an auto level.

**Tech Stack:** Python 3.12, numpy, librosa (onset envelope), pytest, typer.

**Spec:** `docs/superpowers/specs/2026-11-17-flow-layer-design.md` (read it first; it holds the reasons and the exact numbers).

## Global Constraints

- Branch `flow-layer` in `C:\Users\alex\Desktop\BeatSaberMaker`. Do not merge, push, or tag.
- Every Python run is prefixed with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.
- Run one pytest process at a time, never `-n`/xdist, never in the background.
- Use focused tests while iterating. Run the full suite (`.venv/Scripts/python -m pytest -q`) once at the end.
- Commit with `git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit`. End each message with `Co-Authored-By: Gemini <noreply@google.com>`.
- Do not change model code, checkpoints, `playability.toml`, or `data` commands. Never run `data resplit`.
- `CustomLevels` is read-only except for folders named `BSM - …`.
- Do not play audio. Set `BSM_MUTE_AUDIO=1` for anything that might play sound.
- Level table (spec): share of runs → 0 / 0.25 / 0.5 / 0.8; stack chance → 0 / 0.15 / 0.3 / 0.5, for levels 0 / 1 / 2 / 3.
- Auto level is 2 when `"tech"` is in the styles, otherwise 1.
- Dot spacing is 1/8 beat. If that is under 0.045 s, use 1/4 beat. If the result is outside [0.045, 0.09] s, skip the stack. Stack span ≤ 0.12 s.
- Validation profile: `dataclasses.replace(mask_cfg, slider_gap_s=0.12)`.

## Review Focus

1. **Half-time BPM** (GHOST is detected at 110 instead of 220). Spacing comes from seconds limits, so dots must still land 0.045–0.09 s apart. Test: `test_spacing_half_time_bpm`.
2. **Dots after the last note of the song, or past the audio end.** No dot may be placed beyond `onset_env` length. Test: `test_no_dot_past_audio_end`.
3. **Doubles inside a fast run** must end the run, never be rewritten. Test: `test_run_breaks_on_double`.
4. **Other hand inside the stack window** (e.g. a fast stream) means no stack. Test: `test_no_stack_when_other_hand_busy`.
5. **Multi-level generation (`--levels hard,expert`).** Each level gets its own layer pass, and Hard at auto level 1 stays sparse. Test in Task 3: `test_generate_map_flow_per_level`.

---

### Task 1: Onset envelope helper + dot stacks

**Files:**
- Create: `src/bsmaker/flow_rules.py`
- Modify: `src/bsmaker/audio/beats.py` (add `onset_envelope`)
- Test: `tests/test_flow_rules.py`

**Interfaces:**
- Produces:
  - `onset_envelope(audio: np.ndarray) -> tuple[np.ndarray, float]` in `audio/beats.py`. It returns (env, hop_s) using `librosa.onset.onset_strength(..., hop_length=ENV_HOP)`, with `hop_s = ENV_HOP / SAMPLE_RATE`.
  - `@dataclass(frozen=True) class FlowReport` with fields `level: int`, `runs_found: int`, `runs_rewritten: int`, `notes_rewritten: int`, `dot_stacks: int`, `dots_added: int`, `reverted: int`, all ints, default 0 except `level`.
  - `RUN_SHARE = (0.0, 0.25, 0.5, 0.8)` and `STACK_CHANCE = (0.0, 0.15, 0.3, 0.5)`.
  - `def stack_spacing_s(bpm: float) -> float | None`.
  - `def validation_cfg(mask_cfg: PlayabilityConfig) -> PlayabilityConfig`.
  - `def window_ok(notes: Sequence[Note], t0: float, t1: float, cfg: PlayabilityConfig) -> bool`. It checks notes in [t0 − 1 s, t1 + 1 s] through `to_events` + `check(...)`, and returns True when there is no hard violation.
  - `def add_dot_stacks(notes: list[Note], *, bpm: float, onset_env: np.ndarray, onset_hop_s: float, chance: float, rng: random.Random, mask_cfg: PlayabilityConfig) -> tuple[list[Note], int, int, int]`. It returns (notes sorted by time, stacks, dots, reverted).

- [ ] **Step 1: Write failing tests** in `tests/test_flow_rules.py`. Build notes by hand with `Note(time_s, beat, x, y, color, direction)`, and use a flat `onset_env = np.ones(n)` unless the test says otherwise. Assertions:
  - `test_stack_spacing`:
    - `stack_spacing_s(150) == pytest.approx(0.05)`;
    - `stack_spacing_s(220) == pytest.approx(60/220/4)` (1/8 beat is 0.034, so 1/4 beat);
    - `stack_spacing_s(60) is None` (0.125 s and 0.25 s are both > 0.09).
  - `test_spacing_half_time_bpm`: `stack_spacing_s(110) == pytest.approx(60/110/8)` (0.068).
  - `test_dots_on_cut_ray`: one up note L at (1,0) with t=1.0, the next L note at t=2.0, chance=1.0 at bpm 150. Every added dot has `color == 0` and `direction == 8`, x == 1, y in {1, 2}, times 1.05 and 1.10 (or just 1.05), and `beat` matches `time_s * bpm / 60` within 1e-6.
  - `test_no_stack_when_ray_leaves_grid`: an up note at y=2 gets no dots.
  - `test_no_stack_when_next_note_close`: next same-hand note 0.3 s later gets no dots.
  - `test_no_stack_when_other_hand_busy`: an R note at t+0.06 gets no dots.
  - `test_onset_required`: the env is zero around the dot times and 1 elsewhere, so no dots.
  - `test_no_dot_past_audio_end`: the env is shorter than the dot time / hop, so no dots.
  - `test_chance_zero_identity`: `chance=0.0` returns the input notes unchanged, and the count is 0.
  - `test_stacks_pass_validation`: with a random 200-note alternating fixture (seeded) at chance 1.0, `window_ok(result, 0, inf, validation_cfg(PlayabilityConfig()))` is True.
- [ ] **Step 2: Run** `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/Scripts/python -m pytest tests/test_flow_rules.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** per spec, section "Dot stacks":
  - Direction vectors: 0 up (0,+1), 1 down (0,−1), 2 left (−1,0), 3 right (+1,0), 4 UL (−1,+1), 5 UR (+1,+1), 6 DL (−1,−1), 7 DR (+1,−1). The grid is x 0–3, y 0–2.
  - Onset test: `env[k-1:k+2].max() >= np.percentile(env, 70)`, where `k = round(t / hop_s)`.
  - Cells: a dot may not share a cell with any other note within ±0.15 s.
  - Revert: after each stack, call `window_ok` with `validation_cfg`. If it fails, remove the stack and count it as reverted.
  - Use 2 dots when the ray has 2 free cells and `rng.random() < 0.5`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(flow): swing-extension dot stacks`.

### Task 2: Snake runs

**Files:**
- Modify: `src/bsmaker/flow_rules.py`
- Test: `tests/test_flow_rules.py`

**Interfaces:**
- Consumes: `window_ok`, `validation_cfg` from Task 1.
- Produces:
  - `def find_runs(notes: Sequence[Note]) -> list[tuple[int, int]]` returns half-open index ranges [i, j) into time-sorted notes, with j − i ≥ 8.
  - `SLITHER = "slither"`, `ROTATION = "rotation"`.
  - `def rewrite_run(notes: Sequence[Note], i: int, j: int, template: str, rng: random.Random) -> list[Note]` returns a full new note list in which only notes i..j-1 have new (x, y, direction).
  - `def apply_snakes(notes: list[Note], *, share: float, rng: random.Random, mask_cfg: PlayabilityConfig) -> tuple[list[Note], int, int, int, int]` returns (notes, runs_found, runs_rewritten, notes_rewritten, reverted).

- [ ] **Step 1: Write failing tests.** Assertions:
  - `test_find_runs_basic`: 10 alternating L/R single notes, 0.2 s apart, give `[(0, 10)]`.
  - `test_run_breaks_on_double`: an L+R double at index 5 of 12 gives no range containing index 5. Ranges shorter than 8 are dropped.
  - `test_run_gap_limits`: a gap of 0.35 s splits the run; a gap of 0.08 s also splits it.
  - `test_slither_shape`: after `rewrite_run(..., SLITHER, rng)`:
    - every note has y == 0;
    - each hand's consecutive directions alternate between 0 and 1;
    - left x is in 0–2 and right x in 1–3;
    - right x ≥ the left x of the previous left note;
    - each hand's x step is ≤ 1.
  - `test_slither_parity_with_previous`: if the hand's note before the run cuts up (0), the hand's first note in the run cuts down (1).
  - `test_rotation_cycle`: a run starting with L gives the cycle `L(0,1,4) R(1,0,6) L(2,0,7) R(3,1,5)`, repeated as (x, y, d).
  - `test_rotation_falls_back`: when the rotation fails `window_ok`, `apply_snakes` keeps the run as a slither or the original, never invalid. The output passes `window_ok` over the whole song.
  - `test_share_zero_identity` and `test_seed_deterministic` (the same rng seed gives the same output twice).
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement** per spec, section "Snake rewrite".
  - Wave walk: start at left x=1, right x=2. At each hand's note, move x by −1, 0, or +1 chosen by `rng`, clamped to the hand's range and to "right ≥ left".
  - Template choice: ROTATION with probability 0.3 when j − i ≥ 8, otherwise SLITHER.
  - Order of trials: chosen template, then SLITHER, then keep the original. Each trial is checked with `window_ok(validation_cfg)` on the run window. A run that ends up as the original counts as reverted.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(flow): snake run rewrite (slither, rotation loop)`.

### Task 3: Wire into generate + CLI

**Files:**
- Modify: `src/bsmaker/flow_rules.py`, `src/bsmaker/generate.py`, `src/bsmaker/cli.py`, `src/bsmaker/pipeline.py` (pass-through only if it forwards generate options)
- Test: `tests/test_flow_rules.py`, `tests/test_generate.py`, `tests/test_cli_generate.py`

**Interfaces:**
- Consumes: Task 1 and Task 2 functions.
- Produces:
  - `def auto_flow_level(styles: Sequence[str]) -> int`.
  - `def apply_flow(notes: list[Note], *, bpm: float, onset_env: np.ndarray, onset_hop_s: float, level: int, seed: int, mask_cfg: PlayabilityConfig) -> tuple[list[Note], FlowReport]`. Snakes run first, then stacks. Use one `random.Random(seed)`.
  - `generate_map(..., flow: int | None = None)`: None means auto. A value outside {0, 1, 2, 3, None} raises ValueError.
  - `LevelResult.flow_report: FlowReport | None = None`.
  - CLI option `--waz` (int, default None, help "Snake and dot-stack level 0-3 (default: auto from style)").
  - CLI print line per level: `flow {rank}: {runs_rewritten} snake runs ({notes_rewritten} notes), {dot_stacks} dot stacks ({dots_added} dots), {reverted} reverted`.

- [ ] **Step 1: Write failing tests.**
  - `test_apply_flow_level0_identity`.
  - `test_auto_flow_level`: `auto_flow_level(["tech"]) == 2` and `auto_flow_level(["dance"]) == 1`.
  - In `tests/test_generate.py`, reuse the file's existing fake-model fixtures:
    - `test_generate_map_flow_per_level`: generate with `levels` of two ranks and `flow=None`. Each `LevelResult.flow_report` is not None, and its level is 1 (no tech style).
    - `test_generate_map_flow_zero_same_as_before`: with `flow=0` there are no `direction == 8` notes that were not there with flow 0 before. In practice, compare against the same call with the layer patched out.
  - In `tests/test_cli_generate.py`, `--waz 5` exits with code 2 and prints `--waz must be 0, 1, 2, or 3`.
- [ ] **Step 2: Run the focused tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  - In `generate_map`, compute `onset_env, hop_s = onset_envelope(audio)` once after `detect_grid`.
  - After `_generate_level_notes`, call `apply_flow(notes, bpm=grid.bpm, ..., level=flow_level, seed=seed, mask_cfg=mask_cfg)`.
  - Then run the whole-level check: `check(to_events(notes)[0], validation_cfg(mask_cfg)).hard`. If it is non-empty, raise `RuntimeError("flow layer broke a hard playability rule (bug)")`.
  - Recompute `measured` nps after the layer.
  - Forward `flow` in `pipeline.py` only if that function already forwards `ornaments`.
- [ ] **Step 4: Run the focused tests**, then the full suite once. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(flow): --waz level in generate and CLI`.

### Task 4: End-to-end check on GHOST

**Files:**
- Create: `.superpowers/sdd/2026-11-17-flow-layer/e2e.md` (results; git-ignored scratch)

- [ ] **Step 1:** Find the GHOST audio: the `song.egg` inside `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels\BSM - Camellia - GHOST (from Cyphisonia E.P.) [nps 7.5, tech]`. Copy it to `.superpowers/sdd/2026-11-17-flow-layer/ghost.ogg` first, because the generator overwrites that folder.
- [ ] **Step 2:** Run `BSM_MUTE_AUDIO=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/Scripts/python -I -X utf8 -m bsmaker generate ".superpowers/sdd/2026-11-17-flow-layer/Camellia - GHOST (from Cyphisonia E.P.).ogg" --nps 7.5 --style tech --seed 0 --overwrite`. Expected: it writes the folder and prints a `flow ExpertPlus:` line.
- [ ] **Step 3:** From `.superpowers/scratch`, run `../../.venv/Scripts/python -I -X utf8 compare_tech.py "TECH75=<Windows path to the new ExpertPlusStandard.dat>@109.999"`. Use a Windows-style path; the comma in the folder name breaks MSYS path conversion.
  - Expected: `dots` 6–15 and `dots_as_extension` ≥ 70.
  - If not, tune only the onset percentile or STACK_CHANCE, record why, and rerun.
- [ ] **Step 4:** Write the metrics and the CLI flow line into `e2e.md`. Commit any tuning with the message `tune(flow): ...`.
