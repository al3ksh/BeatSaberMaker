# Flow layer (snake + dot stacks): design

Date: 2026-11-17. Status: approved by Alex in chat ("Skoro jesteś pewny reguł do robimy").

Revised 2026-11-17 after review (reference-map measurements): stack eligibility, spacing set, stack chances, a loose-dot repair step and an exact final check.

## Why

We compared our Tech 7.5 NPS map (GHOST) with MAYDAY (6b8c) and Astronomia (e2ba) Expert+ using `.superpowers/scratch/compare_tech.py`. Speed, steps and 180° parity are already close. Two things are missing:

| | MAYDAY | Astronomia | ours (tech 7.5) |
|---|---|---|---|
| dots, % of notes | 18 | 11 | 1 |
| dots that extend a swing, % of dots | 80 | 78 | 0 |
| same-hand gaps ≤ 0.1 s | 19% | 14% | 1% |

1. **Dots as extensions.** In good maps a dot almost always continues a swing. For example, a bottom note cut up is followed 1/8 beat later by 1–2 dots stacked above it. Our model either sprays loose dots or (since the dot brake) places almost none.
2. **Snake / weave.** Alex wants, "from time to time", the Rap God pattern (46d4, 4:28). Hands alternate every 1/4 beat, each hand strictly up/down, and both hands drift sideways along the bottom row like a wave, about one cell apart. A rotation loop also appears: L(0,1)UL, R(1,0)DL, L(2,0)DR, R(3,1)UR, repeated.

## Goals

- A deterministic, rule-based post-processing layer on the notes. No model change, no retraining.
- It turns the base model's loose dots into arrows that follow parity.
- It adds swing-extension dot stacks where the music has fast hits.
- It rewrites some fast alternating runs into snake patterns.
- Level 0 returns the notes unchanged. The base model is never touched.
- The output never breaks a hard playability rule.

## Non-goals

- Making weave runs longer. Run length comes from the timing model and stays as it is.
- Memes and gimmicks (Alex: "Memy później").
- Retraining or a new style token.
- An app control in this change. The app gets the auto level through the existing generate path.

## Anti-goals (failure even if it "works")

- Loose dots: any dot that is not on the cut line of the arrow right before it, by the same hand.
- Snakes everywhere, so the map looks samey. Alex said "od czasu do czasu".
- Dot stacks on quiet music, where nothing in the audio justifies extra notes.

## Approaches considered

1. **Rule layer (chosen).** About 2–3 days, deterministic, can be switched off, and fully preserves the base model. Risk: patterns can feel templated, which is mitigated by applying them sparingly.
2. **Retrain the pattern model on curated tech maps.** About a week plus paid Colab, with no guarantee. Kept for later if rules feel too samey.
3. **Loosen the dot brake.** Rejected: it brings back loose dots, which are exactly the problem.

## Design

New module `src/bsmaker/flow_rules.py`. It is called in `generate_map` right after `_generate_level_notes`, before ornaments, so arcs and chains see the final notes. Order inside the layer: loose-dot repair, then snakes, then dot stacks.

```
apply_flow(notes, *, bpm, onset_env, onset_hop_s, level, seed, mask_cfg) -> (notes, FlowReport)
```

**Level.** CLI `--waz 0-3` (Polish, like `--ozdoby` and `--swiatla`). The default is auto: 2 for style `tech`, otherwise 1. The auto level is resolved only in `run_generate` (CLI and app). The library default is `generate_map(..., flow=0)`, so direct callers get byte-identical output to before the layer. Level 0 is the identity; every level ≥ 1 runs the loose-dot repair. Level drives two numbers:

| level | share of eligible runs turned into snakes | chance of a dot stack per eligible arrow |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0.25 | 0.25 |
| 2 | 0.5 | 0.5 |
| 3 | 0.8 | 0.8 |

### Loose-dot repair

- A base-model dot is a **swing extension** when it lies on the cut ray of the same hand's previous arrow (or of the extension dot that continued it) within 0.35 s. This is the definition `compare_tech.py` measures: from the arrow at (x, y) with direction (dx, dy), the offset (rx, ry) satisfies `rx·dy = ry·dx` and `rx·dx + ry·dy > 0`.
- Every other dot whose hand has an earlier swing becomes an arrow. It cuts the direction parity expects (opposite of the hand's previous swing; a dot flips the swing, as in the checker). Candidates are the directions within 45° of that, straight up/down first, then the closest.
- Each candidate is validated like any flow change (below). The first that passes wins. If none passes, the dot stays and counts as reverted.
- Extension status is tracked as the scan goes, so a repaired arrow can make the next dot an extension.

### Dot stacks

- **Eligible arrow:** a non-dot note A of hand h at time t.
  - Its cut direction points into the grid with at least 1 free cell along the ray. For example, up from y=0 gives (x,1) and (x,2).
  - The next note of hand h is at least `max(0.20 s, span + 0.13 s)` later.
  - The other hand may have a note at t itself (a double, |dt| ≤ 0.01 s) but none in `(t + 0.01 s, last dot + 0.05 s]`.
  - The dot neither blocks nor is blocked by the other hand under the vision rule (stack dots are slider tails for the checker, so this is checked explicitly).
  - The arrow is not inside a weave run of ≥ 8 notes (as `compare_tech.py` measures weave: alternating hands 0.05–0.22 s apart, each hand flipping ≥ 135°); only a run's last note may get a stack. Added after the GHOST end-to-end run, where stacks mid-stream cut the longest weave from 29 to 17 notes.
- **Spacing:** the smallest of 1/16, 1/12, 1/8, 1/6, 1/4 beat (in seconds) that is ≥ 0.045 s. If that is > 0.09 s, skip the stack (`to_events` merges same-hand notes closer than 0.040 s).
- **Size:** 1 or 2 dots along the ray (2 when the ray has room, the span fits and the seed picks it). Total span ≤ 0.11 s.
- **Musical support:** each dot needs an onset. The onset envelope (librosa `onset_strength`, the same kind `detect_grid` uses) at the dot time, ±1 hop, must be ≥ its 50th percentile over the song. A dot past the end of the audio never qualifies.
- **New note values:** dots are `Note(direction=8, color=h)`, with time/beat computed from the spacing.

### Snake rewrite

- **Eligible run:** at least 8 consecutive events, each a single note (no doubles). Hands alternate, the inter-event gap is 0.1–0.3 s, and each hand's own gap is ≤ 0.6 s.
- **Selection:** each run is picked with the level's share. A seeded `random.Random(seed)` keeps it deterministic.
- **Templates:**
  1. **Slither:** all notes on y=0. Each hand strictly alternates up/down. Left x walks a wave within 0–2, right x within 1–3, and right x ≥ left x, with a 1-cell step at most.
  2. **Rotation loop:** the four-note cycle above, repeated. If it breaks a hard rule here, fall back to the slither.
- **Phase:** the first note of each hand in the run flips parity versus that hand's previous note. Only (x, y, direction) change; times stay.

### Validation and revert

- **Final check** (whole level, after the layer, in `generate_map` when flow > 0; failure is a bug and raises):
  - (a) The final map minus the dots the layer added passes the original mask profile exactly like a base map: no merged notes, no hard violation on frame times (`check`) or exact times (`check_exact`).
  - (b) The full final map, under a mask-profile copy with `slider_gap_s = 0.12` (stack dots count as slider tails of their arrow), merges no notes and has no hard violation that the map before the layer did not already have.
- **Per-change window** (one repaired dot, one run, or one stack): the same two conditions on a crop. The crop reaches at least 1 s around the change and further out to an arrow of each hand that resets the checker state (an arrow ≥ 0.14 s after the hand's previous note, so every profile keeps it). Only violations from the change onwards count. The checker is causal, so a change that passes the window cannot fail the final check. A failing change is reverted and counted in the report.
- Stack dots also must not share a cell with any other note within ±0.15 s.

### Report

`FlowReport(level, runs_found, runs_rewritten, notes_rewritten, dot_stacks, dots_added, dots_repaired, reverted)` is added to `LevelResult`. The CLI prints one line, e.g. `flow ExpertPlus: 3 snake runs (24 notes), 41 dot stacks (63 dots), 12 loose dots repaired, 5 reverted`.

## Acceptance

- Unit tests:
  - level 0 is the identity;
  - the same seed gives the same output;
  - dot geometry is checked (every added dot lies on its arrow's ray, in the same column/line);
  - stack eligibility is checked (gap, other hand, onset, vision, spacing, span);
  - loose dots are repaired to the parity direction, extensions are kept, failing repairs revert;
  - the final check rejects a map that breaks the mask profile without the added dots;
  - run detection finds and rejects the right runs;
  - each template keeps parity and has no crossover;
  - a failing change is reverted.
- End to end:
  - Regenerate GHOST at `--nps 7.5 --style tech`.
  - In `compare_tech.py`, dots must be 6–15% of notes with ≥ 70% as extensions.
  - Weave runs must still be there.
  - There must be no hard violations.
- Full test suite passes.

## Left to the builder

Exact wave walk, tie-breaking, the onset helper's placement (new function in `audio/` is fine) and report wording.
