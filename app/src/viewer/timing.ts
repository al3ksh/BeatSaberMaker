import type { BpmChangePoint, JumpMetrics, TempoMap } from "./types";

const DEFAULT_NJS_BY_RANK: Record<string, number> = {
  Easy: 10,
  Normal: 10,
  Hard: 12,
  Expert: 16,
  ExpertPlus: 16,
};

export interface RawBpmEvent {
  b?: number;
  m?: number;
}

/** Build a piecewise-linear TempoMap from Info BPM, songTimeOffset, and v3 bpmEvents. */
export function buildTempoMap(
  baseBpm: number,
  songTimeOffset = 0,
  bpmEvents: RawBpmEvent[] = [],
): TempoMap {
  const safeBaseBpm = Number.isFinite(baseBpm) && baseBpm > 0 ? baseBpm : 120;
  const safeOffset = Number.isFinite(songTimeOffset) ? songTimeOffset : 0;

  const sorted = bpmEvents
    .filter(
      (e) =>
        typeof e.b === "number" &&
        Number.isFinite(e.b) &&
        typeof e.m === "number" &&
        Number.isFinite(e.m) &&
        e.m > 0,
    )
    .map((e) => ({ beat: e.b as number, bpm: e.m as number }))
    .sort((a, b) => a.beat - b.beat);

  const points: BpmChangePoint[] = [{ beat: 0, bpm: safeBaseBpm, timeS: safeOffset }];
  for (const ev of sorted) {
    if (ev.beat <= 0) {
      points[0] = { beat: 0, bpm: ev.bpm, timeS: safeOffset };
      continue;
    }
    const prev = points[points.length - 1];
    if (Math.abs(ev.beat - prev.beat) < 1e-9) {
      points[points.length - 1] = { beat: ev.beat, bpm: ev.bpm, timeS: prev.timeS };
      continue;
    }
    const dt = ((ev.beat - prev.beat) * 60) / prev.bpm;
    points.push({ beat: ev.beat, bpm: ev.bpm, timeS: prev.timeS + dt });
  }

  return {
    baseBpm: safeBaseBpm,
    songTimeOffset: safeOffset,
    points,
  };
}

/** Convert a beat number to song time in seconds. */
export function beatToSeconds(beat: number, tempo: TempoMap): number {
  const pts = tempo.points;
  if (pts.length === 1) {
    return (beat * 60) / pts[0].bpm + pts[0].timeS;
  }
  // Binary search for the last point with point.beat <= beat
  let lo = 0;
  let hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid].beat <= beat) lo = mid;
    else hi = mid - 1;
  }
  const p = pts[lo];
  return p.timeS + ((beat - p.beat) * 60) / p.bpm;
}

/** Convert song time in seconds to a beat number. */
export function secondsToBeat(timeS: number, tempo: TempoMap): number {
  const pts = tempo.points;
  if (pts.length === 1) {
    return ((timeS - pts[0].timeS) * pts[0].bpm) / 60;
  }
  let lo = 0;
  let hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid].timeS <= timeS) lo = mid;
    else hi = mid - 1;
  }
  const p = pts[lo];
  return p.beat + ((timeS - p.timeS) * p.bpm) / 60;
}

/** Compute Beat Saber half-jump duration (beats & seconds) and jump distance (meters). */
export function computeJumpMetrics(
  bpm: number,
  rawNjs: number,
  njsOffset = 0,
  rank = "Expert",
): JumpMetrics {
  const safeBpm = Number.isFinite(bpm) && bpm > 0 ? bpm : 120;
  const njs =
    Number.isFinite(rawNjs) && rawNjs > 0 ? rawNjs : (DEFAULT_NJS_BY_RANK[rank] ?? 14);
  const safeOffset = Number.isFinite(njsOffset) ? njsOffset : 0;

  let hjd = 4;
  const spb = 60 / safeBpm;
  while (njs * spb * hjd > 17.999 && hjd > 0.0625) {
    hjd /= 2;
  }
  const hjdBeats = Math.max(hjd + safeOffset, 0.25);
  const halfJumpDurationSec = hjdBeats * spb;
  const jumpDistance = njs * spb * hjdBeats * 2;

  return {
    njs,
    hjdBeats,
    halfJumpDurationSec,
    jumpDistance,
    halfJumpDistance: jumpDistance / 2,
  };
}

/**
 * Compute Z world coordinate (meters) at `currentTimeS` for an object hitting at `targetTimeS`.
 * Player faces -Z; objects spawn at -halfJumpDistance and travel in +Z toward z=0.
 */
export function objectZAtTime(targetTimeS: number, currentTimeS: number, njs: number): number {
  return (currentTimeS - targetTimeS) * njs;
}
