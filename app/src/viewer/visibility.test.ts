import { describe, expect, it } from "vitest";
import {
  CUT_EFFECT_DURATION_S,
  computeSpawnFade,
  computeWallZSpan,
  cutEffectProgress,
  findArcVisibleUStart,
  isArcVisible,
  isHitObjectVisible,
  isWallVisible,
} from "./visibility";

describe("viewer hit-time visibility rules", () => {
  it("hides notes and bombs at exact hit time b and prevents objects behind player", () => {
    const hjd = 1.0;
    const hitTime = 10.0;

    expect(isHitObjectVisible(hitTime, 8.99, hjd)).toBe(false);
    expect(isHitObjectVisible(hitTime, 9.0, hjd)).toBe(true);
    expect(isHitObjectVisible(hitTime, 9.999, hjd)).toBe(true);
    // Vanishes at hit time b and never lingers behind player
    expect(isHitObjectVisible(hitTime, 10.0, hjd)).toBe(false);
    expect(isHitObjectVisible(hitTime, 10.1, hjd)).toBe(false);
  });

  it("tracks ~0.2s cut effect progress strictly after hit time", () => {
    const hitTime = 10.0;
    expect(cutEffectProgress(hitTime, 9.99)).toBeNull();
    expect(cutEffectProgress(hitTime, 10.0)).toBeCloseTo(0, 6);
    expect(cutEffectProgress(hitTime, 10.0 + CUT_EFFECT_DURATION_S * 0.5)).toBeCloseTo(0.5, 6);
    expect(cutEffectProgress(hitTime, 10.0 + CUT_EFFECT_DURATION_S)).toBeNull();
  });

  it("keeps walls passing through z=0 until duration ends and clamps front behind player", () => {
    const hjd = 1.0;
    expect(isWallVisible(10.0, 12.0, 8.9, hjd)).toBe(false);
    expect(isWallVisible(10.0, 12.0, 9.0, hjd)).toBe(true);
    expect(isWallVisible(10.0, 12.0, 11.0, hjd)).toBe(true);
    expect(isWallVisible(10.0, 12.0, 11.99, hjd)).toBe(true);
    expect(isWallVisible(10.0, 12.0, 12.0, hjd)).toBe(false);

    // Approaching wall before hit time
    const approaching = computeWallZSpan(10.0, 12.0, 9.5, 14);
    expect(approaching).not.toBeNull();
    expect(approaching!.zFront).toBeCloseTo(-7, 6);
    expect(approaching!.zBack).toBeCloseTo(-35, 6);

    // Wall currently passing player: front clamped at z=0, back approaching 0
    const passing = computeWallZSpan(10.0, 12.0, 11.0, 14);
    expect(passing).not.toBeNull();
    expect(passing!.zFront).toBe(0);
    expect(passing!.zBack).toBeCloseTo(-14, 6);

    // Ended wall returns null
    expect(computeWallZSpan(10.0, 12.0, 12.0, 14)).toBeNull();
  });

  it("clips already-hit head of arcs at z=0 and disappears at tail time", () => {
    const hjd = 1.0;
    expect(isArcVisible(10.0, 12.0, 10.5, hjd)).toBe(true);
    expect(isArcVisible(10.0, 12.0, 12.0, hjd)).toBe(false);

    // Before head hits: full curve u in [0, 1]
    expect(findArcVisibleUStart(-2, -6, -14, -18)).toBe(0);

    // Mid-arc: head already passed (z > 0), tail still ahead (z < 0)
    const p0z = 7;
    const p1z = 1.75;
    const p2z = -8.75;
    const p3z = -14;
    const uStart = findArcVisibleUStart(p0z, p1z, p2z, p3z);
    expect(uStart).not.toBeNull();
    expect(uStart!).toBeGreaterThan(0);
    expect(uStart!).toBeLessThan(1);

    // Evaluate cubic Bezier Z at uStart -> should be ~0
    const u = uStart!;
    const inv = 1 - u;
    const zAtU =
      inv * inv * inv * p0z +
      3 * inv * inv * u * p1z +
      3 * inv * u * u * p2z +
      u * u * u * p3z;
    expect(zAtU).toBeCloseTo(0, 3);

    // Entirely past player -> null
    expect(findArcVisibleUStart(5, 4, 2, 0)).toBeNull();
  });

  it("computes smooth spawn fade-in near half-jump spawn boundary", () => {
    const hjd = 1.0;
    expect(computeSpawnFade(9.0, 10.0, hjd)).toBeCloseTo(0, 6);
    const early = computeSpawnFade(9.09, 10.0, hjd);
    expect(early).toBeGreaterThan(0);
    expect(early).toBeLessThan(1);
    expect(computeSpawnFade(9.5, 10.0, hjd)).toBe(1);
    expect(computeSpawnFade(9.99, 10.0, hjd)).toBe(1);
  });
});
