import { describe, expect, it } from "vitest";
import {
  beatToSeconds,
  buildTempoMap,
  computeJumpMetrics,
  objectZAtTime,
  secondsToBeat,
} from "./timing";

describe("viewer timing", () => {
  it("computes hjd and jump distance for NJS 14, BPM 128, offset -0.159", () => {
    // Manual calculation:
    // spb = 60 / 128 = 0.46875
    // 14 * 0.46875 * 4 = 26.25 > 17.999 -> hjd = 2
    // 14 * 0.46875 * 2 = 13.125 <= 17.999
    // hjd = max(2 - 0.159, 0.25) = 1.841 beats
    // halfJumpDurationSec = 1.841 * 0.46875 = 0.86296875 s
    // jumpDistance = 14 * 0.46875 * 1.841 * 2 = 24.163125 m
    const m = computeJumpMetrics(128, 14, -0.159, "Hard");
    expect(m.hjdBeats).toBeCloseTo(1.841, 6);
    expect(m.halfJumpDurationSec).toBeCloseTo(0.86296875, 6);
    expect(m.jumpDistance).toBeCloseTo(24.163125, 6);
    expect(m.halfJumpDistance).toBeCloseTo(12.0815625, 6);
  });

  it("clamps hjd to minimum 0.25 beats when offset is strongly negative", () => {
    const m = computeJumpMetrics(120, 10, -5, "Easy");
    expect(m.hjdBeats).toBe(0.25);
    expect(m.halfJumpDurationSec).toBeCloseTo(0.125, 6);
  });

  it("converts beat <-> seconds with constant BPM and offset, and with bpmEvents", () => {
    const constTempo = buildTempoMap(120, 0.25, []);
    expect(beatToSeconds(4, constTempo)).toBeCloseTo(2.25, 6);
    expect(secondsToBeat(2.25, constTempo)).toBeCloseTo(4, 6);

    const varTempo = buildTempoMap(120, 0, [{ b: 4, m: 60 }]);
    // Beats 0..4 at 120 BPM = 2.0s; beats 4..6 at 60 BPM = +2.0s -> 4.0s
    expect(beatToSeconds(4, varTempo)).toBeCloseTo(2.0, 6);
    expect(beatToSeconds(6, varTempo)).toBeCloseTo(4.0, 6);
    expect(secondsToBeat(4.0, varTempo)).toBeCloseTo(6, 6);
  });

  it("positions objects at -halfJumpDistance on spawn and z=0 at target time", () => {
    const m = computeJumpMetrics(128, 14, -0.159, "Hard");
    const targetTime = 10.0;
    const spawnTime = targetTime - m.halfJumpDurationSec;
    expect(objectZAtTime(targetTime, spawnTime, 14)).toBeCloseTo(-m.halfJumpDistance, 5);
    expect(objectZAtTime(targetTime, targetTime, 14)).toBeCloseTo(0, 6);
    expect(objectZAtTime(targetTime, targetTime + 0.5, 14)).toBeCloseTo(7, 6);
  });
});
