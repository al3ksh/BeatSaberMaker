export const CUT_EFFECT_DURATION_S = 0.2;

/** Notes & bombs appear during [timeS - hjd, timeS) and vanish at exact hit time b. */
export function isHitObjectVisible(
  targetTimeS: number,
  currentTimeS: number,
  halfJumpDurationSec: number,
): boolean {
  return currentTimeS >= targetTimeS - halfJumpDurationSec && currentTimeS < targetTimeS;
}

/** Normalized cut-effect progress in [0, 1) during [targetTimeS, targetTimeS + durationS), else null. */
export function cutEffectProgress(
  targetTimeS: number,
  currentTimeS: number,
  durationS = CUT_EFFECT_DURATION_S,
): number | null {
  if (durationS <= 0) return null;
  const age = currentTimeS - targetTimeS;
  if (age < 0 || age >= durationS - 1e-9) return null;
  return age / durationS;
}

/** Walls remain active while approaching and passing through the player until endTimeS. */
export function isWallVisible(
  startTimeS: number,
  endTimeS: number,
  currentTimeS: number,
  halfJumpDurationSec: number,
): boolean {
  return currentTimeS >= startTimeS - halfJumpDurationSec && currentTimeS < endTimeS;
}

/**
 * Compute world Z span `[zBack, zFront]` for an active wall, clamping `zFront <= 0`
 * so the wall slides cleanly into the hit plane over its duration without extending behind the camera.
 */
export function computeWallZSpan(
  startTimeS: number,
  endTimeS: number,
  currentTimeS: number,
  njs: number,
): { zBack: number; zFront: number } | null {
  if (currentTimeS >= endTimeS) return null;
  const zFrontRaw = (currentTimeS - startTimeS) * njs;
  const zBack = (currentTimeS - endTimeS) * njs;
  const zFront = Math.min(zFrontRaw, 0);
  if (zFront - zBack <= 1e-4) return null;
  return { zBack, zFront };
}

/** Arcs remain active from head spawn until tailTimeS. */
export function isArcVisible(
  startTimeS: number,
  tailTimeS: number,
  currentTimeS: number,
  halfJumpDurationSec: number,
): boolean {
  return currentTimeS >= startTimeS - halfJumpDurationSec && currentTimeS < tailTimeS;
}

function evalBezierZ(p0z: number, p1z: number, p2z: number, p3z: number, u: number): number {
  const inv = 1 - u;
  const inv2 = inv * inv;
  const u2 = u * u;
  return (
    inv2 * inv * p0z +
    3 * inv2 * u * p1z +
    3 * inv * u2 * p2z +
    u2 * u * p3z
  );
}

/**
 * Return starting parameter `uStart` in [0, 1] where cubic Bezier arc enters `z <= 0`,
 * or `null` if the entire curve has already passed `z >= 0`.
 */
export function findArcVisibleUStart(
  p0z: number,
  p1z: number,
  p2z: number,
  p3z: number,
): number | null {
  if (p3z >= 0 && p0z >= 0) return null;
  if (p0z <= 0) return 0;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 20; i += 1) {
    const mid = (lo + hi) * 0.5;
    if (evalBezierZ(p0z, p1z, p2z, p3z, mid) > 0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return hi;
}

/** Smooth 0 -> 1 spawn fade over the first `fadeFraction` of half-jump flight time. */
export function computeSpawnFade(
  currentTimeS: number,
  targetTimeS: number,
  halfJumpDurationSec: number,
  fadeFraction = 0.18,
): number {
  if (halfJumpDurationSec <= 0) return 1;
  const flightProgress = (currentTimeS - (targetTimeS - halfJumpDurationSec)) / halfJumpDurationSec;
  if (flightProgress <= 0) return 0;
  if (flightProgress >= fadeFraction) return 1;
  const t = flightProgress / fadeFraction;
  // Smoothstep
  return t * t * (3 - 2 * t);
}
