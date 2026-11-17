import { beatToSeconds, buildTempoMap } from "./timing";
import type {
  ParsedDifficultyData,
  TempoMap,
  TimedObject,
  ViewerArc,
  ViewerBomb,
  ViewerChain,
  ViewerLightEvent,
  ViewerNote,
  ViewerObstacle,
} from "./types";

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function int(v: unknown, fallback = 0): number {
  return Math.round(num(v, fallback));
}

function arr(obj: Record<string, unknown>, key: string): Record<string, unknown>[] {
  const v = obj[key];
  if (!Array.isArray(v)) return [];
  return v.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"));
}

/** Parse v3.3.0 (primary) or v2 Beat Saber difficulty JSON into sorted viewer objects. */
export function parseDifficultyData(
  raw: Record<string, unknown>,
  baseTempo: TempoMap,
): ParsedDifficultyData {
  const isV2 = "_notes" in raw && !("colorNotes" in raw);

  if (isV2) {
    const notes: ViewerNote[] = [];
    const bombs: ViewerBomb[] = [];
    for (const n of arr(raw, "_notes")) {
      const beat = num(n._time);
      const t = int(n._type);
      const x = int(n._lineIndex);
      const y = int(n._lineLayer);
      const timeS = beatToSeconds(beat, baseTempo);
      if (t === 3) {
        bombs.push({ beat, timeS, x, y });
      } else if (t === 0 || t === 1) {
        notes.push({
          beat,
          timeS,
          x,
          y,
          color: t,
          direction: int(n._cutDirection),
          angleOffset: 0,
        });
      }
    }
    const obstacles: ViewerObstacle[] = arr(raw, "_obstacles")
      .map((o) => {
        const beat = num(o._time);
        const durationBeats = Math.max(0.01, num(o._duration, 0.25));
        const typ = int(o._type);
        const timeS = beatToSeconds(beat, baseTempo);
        const endTimeS = beatToSeconds(beat + durationBeats, baseTempo);
        return {
          beat,
          timeS,
          durationBeats,
          durationS: endTimeS - timeS,
          endTimeS,
          x: int(o._lineIndex),
          y: typ === 1 ? 2 : 0,
          w: Math.max(1, int(o._width, 1)),
          h: typ === 1 ? 3 : 5,
        };
      })
      .sort((a, b) => a.timeS - b.timeS);

    const lightEvents: ViewerLightEvent[] = arr(raw, "_events")
      .filter((e) => int(e._type) !== 100)
      .map((e) => {
        const beat = num(e._time);
        return {
          beat,
          timeS: beatToSeconds(beat, baseTempo),
          type: int(e._type),
          value: int(e._value),
          floatValue: "_floatValue" in e ? num(e._floatValue, 1) : 1,
        };
      })
      .sort((a, b) => a.timeS - b.timeS);

    notes.sort((a, b) => a.timeS - b.timeS);
    bombs.sort((a, b) => a.timeS - b.timeS);

    return {
      schema: "v2",
      hasBpmChanges: false,
      notes,
      bombs,
      obstacles,
      arcs: [],
      chains: [],
      lightEvents,
      tempo: baseTempo,
    };
  }

  // v3.3.0
  const rawBpmEvents = arr(raw, "bpmEvents").map((e) => ({
    b: num(e.b),
    m: num(e.m),
  }));
  const tempo =
    rawBpmEvents.length > 0
      ? buildTempoMap(baseTempo.baseBpm, baseTempo.songTimeOffset, rawBpmEvents)
      : baseTempo;

  const notes: ViewerNote[] = arr(raw, "colorNotes")
    .map((n) => {
      const beat = num(n.b);
      return {
        beat,
        timeS: beatToSeconds(beat, tempo),
        x: int(n.x),
        y: int(n.y),
        color: (int(n.c) === 1 ? 1 : 0) as 0 | 1,
        direction: int(n.d),
        angleOffset: num(n.a, 0),
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  const bombs: ViewerBomb[] = arr(raw, "bombNotes")
    .map((b) => {
      const beat = num(b.b);
      return {
        beat,
        timeS: beatToSeconds(beat, tempo),
        x: int(b.x),
        y: int(b.y),
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  const obstacles: ViewerObstacle[] = arr(raw, "obstacles")
    .map((o) => {
      const beat = num(o.b);
      const durationBeats = Math.max(0.01, num(o.d, 0.25));
      const timeS = beatToSeconds(beat, tempo);
      const endTimeS = beatToSeconds(beat + durationBeats, tempo);
      return {
        beat,
        timeS,
        durationBeats,
        durationS: endTimeS - timeS,
        endTimeS,
        x: int(o.x),
        y: int(o.y),
        w: Math.max(1, int(o.w, 1)),
        h: Math.max(1, int(o.h, 1)),
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  const arcs: ViewerArc[] = arr(raw, "sliders")
    .map((a) => {
      const beat = num(a.b);
      const tailBeat = num(a.tb, beat);
      return {
        beat,
        timeS: beatToSeconds(beat, tempo),
        color: (int(a.c) === 1 ? 1 : 0) as 0 | 1,
        x: int(a.x),
        y: int(a.y),
        direction: int(a.d),
        mu: num(a.mu, 1),
        tailBeat,
        tailTimeS: beatToSeconds(tailBeat, tempo),
        tx: int(a.tx),
        ty: int(a.ty),
        tailDirection: int(a.tc),
        tmu: num(a.tmu, 1),
        midAnchorMode: int(a.m, 0),
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  const chains: ViewerChain[] = arr(raw, "burstSliders")
    .map((c) => {
      const beat = num(c.b);
      const tailBeat = num(c.tb, beat);
      return {
        beat,
        timeS: beatToSeconds(beat, tempo),
        color: (int(c.c) === 1 ? 1 : 0) as 0 | 1,
        x: int(c.x),
        y: int(c.y),
        direction: int(c.d),
        tailBeat,
        tailTimeS: beatToSeconds(tailBeat, tempo),
        tx: int(c.tx),
        ty: int(c.ty),
        segments: Math.max(1, int(c.sc, 1)),
        squish: Math.max(0.05, Math.min(1, num(c.s, 1))),
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  const lightEvents: ViewerLightEvent[] = arr(raw, "basicBeatmapEvents")
    .map((e) => {
      const beat = num(e.b);
      return {
        beat,
        timeS: beatToSeconds(beat, tempo),
        type: int(e.et),
        value: int(e.i),
        floatValue: "f" in e ? num(e.f, 1) : 1,
      };
    })
    .sort((a, b) => a.timeS - b.timeS);

  return {
    schema: "v3",
    hasBpmChanges: rawBpmEvents.length > 0,
    notes,
    bombs,
    obstacles,
    arcs,
    chains,
    lightEvents,
    tempo,
  };
}

/** Return sorted items whose `timeS` lies within `[minTimeS, maxTimeS]` via binary search. */
export function queryWindow<T extends TimedObject>(
  items: readonly T[],
  minTimeS: number,
  maxTimeS: number,
): T[] {
  if (items.length === 0 || maxTimeS < minTimeS) return [];
  let lo = 0;
  let hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].timeS < minTimeS) lo = mid + 1;
    else hi = mid;
  }
  const out: T[] = [];
  for (let i = lo; i < items.length && items[i].timeS <= maxTimeS; i += 1) {
    out.push(items[i]);
  }
  return out;
}

/**
 * Return span items (walls, arcs, chains) intersecting `[minTimeS, maxTimeS]`.
 * An item with `[timeS, endTimeS]` overlaps if `endTimeS >= minTimeS && timeS <= maxTimeS`.
 */
export function querySpanWindow<T extends TimedObject>(
  items: readonly T[],
  getEndTimeS: (item: T) => number,
  minTimeS: number,
  maxTimeS: number,
): T[] {
  if (items.length === 0 || maxTimeS < minTimeS) return [];
  const out: T[] = [];
  // Items are sorted by start timeS, so we can stop once timeS > maxTimeS.
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    if (item.timeS > maxTimeS) break;
    if (getEndTimeS(item) >= minTimeS) {
      out.push(item);
    }
  }
  return out;
}

/** Compute local notes-per-second inside a 1-second centered window `[t - 0.5, t + 0.5]`. */
export function computeLocalNps(notes: readonly ViewerNote[], currentTimeS: number): number {
  return queryWindow(notes, currentTimeS - 0.5, currentTimeS + 0.5).length;
}
