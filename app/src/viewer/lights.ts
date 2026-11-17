import type { ViewerLightEvent } from "./types";

export type LightColorKind = "off" | "blue" | "red" | "white";
export type LightGroupId = 0 | 1 | 2 | 3 | 4;

export interface LightChannelState {
  color: LightColorKind;
  intensity: number;
  r: number;
  g: number;
  b: number;
}

export interface LightsState {
  groups: Record<LightGroupId, LightChannelState>;
  ringRotation: number;
  ringSpacing: number;
  leftLaserPhase: number;
  rightLaserPhase: number;
}

interface LaserSpeedKeyframe {
  timeS: number;
  speed: number;
  basePhase: number;
}

interface RingEventKeyframe {
  timeS: number;
  targetValue: number;
  prevValue: number;
}

export interface LightTimeline {
  channels: Record<LightGroupId, ViewerLightEvent[]>;
  ringSpins: RingEventKeyframe[];
  ringZooms: RingEventKeyframe[];
  leftLaserSpeeds: LaserSpeedKeyframe[];
  rightLaserSpeeds: LaserSpeedKeyframe[];
}

const RGB_MAP: Record<LightColorKind, [number, number, number]> = {
  off: [0, 0, 0],
  blue: [0.23, 0.51, 0.96],
  red: [0.94, 0.27, 0.27],
  white: [0.97, 0.98, 1.0],
};

function decodeValue(value: number): {
  color: LightColorKind;
  mode: "off" | "on" | "flash" | "fade" | "transition";
} {
  if (value <= 0) return { color: "off", mode: "off" };
  let color: LightColorKind = "white";
  if (value <= 4) color = "blue";
  else if (value <= 8) color = "red";
  else if (value <= 12) color = "white";
  else return { color: "off", mode: "off" };

  const sub = (value - 1) % 4;
  if (sub === 0) return { color, mode: "on" };
  if (sub === 1) return { color, mode: "flash" };
  if (sub === 2) return { color, mode: "fade" };
  return { color, mode: "transition" };
}

function lastIndexAtOrBefore<T extends { timeS: number }>(arr: readonly T[], timeS: number): number {
  if (arr.length === 0 || timeS < arr[0].timeS) return -1;
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (arr[mid].timeS <= timeS) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function makeState(color: LightColorKind, intensity: number): LightChannelState {
  const [r, g, b] = RGB_MAP[color];
  return { color, intensity, r, g, b };
}

function evalChannelAt(events: readonly ViewerLightEvent[], timeS: number): LightChannelState {
  const idx = lastIndexAtOrBefore(events, timeS);
  if (idx < 0) return makeState("off", 0);

  const ev = events[idx];
  const dt = Math.max(0, timeS - ev.timeS);
  const f = Math.max(0, Math.min(2, ev.floatValue));
  const { color, mode } = decodeValue(ev.value);

  if (mode === "off") return makeState("off", 0);
  if (mode === "on") return makeState(color, f);
  if (mode === "flash") {
    const boost = Math.max(0, 1 - dt / 0.6);
    const intensity = f * (1 + 0.6 * boost * boost);
    return makeState(color, intensity);
  }
  if (mode === "fade") {
    const decay = Math.max(0, 1 - dt / 1.0);
    const intensity = f * 1.45 * decay * decay;
    return decay <= 0 ? makeState("off", 0) : makeState(color, intensity);
  }

  // Transition: smoothly interpolate from settled prior state over 0.5s
  const prev = idx > 0 ? evalChannelAt(events.slice(0, idx), ev.timeS - 1e-4) : makeState("off", 0);
  const alpha = Math.min(1, dt / 0.5);
  const [tr, tg, tb] = RGB_MAP[color];
  return {
    color,
    intensity: prev.intensity + (f - prev.intensity) * alpha,
    r: prev.r + (tr - prev.r) * alpha,
    g: prev.g + (tg - prev.g) * alpha,
    b: prev.b + (tb - prev.b) * alpha,
  };
}

function buildLaserKeyframes(events: readonly ViewerLightEvent[]): LaserSpeedKeyframe[] {
  const out: LaserSpeedKeyframe[] = [];
  let basePhase = 0;
  let lastTime = 0;
  let lastSpeed = 0;
  for (const e of events) {
    basePhase += (e.timeS - lastTime) * lastSpeed;
    const speed = Math.max(0, e.value) * 0.35;
    out.push({ timeS: e.timeS, speed, basePhase });
    lastTime = e.timeS;
    lastSpeed = speed;
  }
  return out;
}

function evalLaserPhase(frames: readonly LaserSpeedKeyframe[], timeS: number): number {
  const idx = lastIndexAtOrBefore(frames, timeS);
  if (idx < 0) return 0;
  const f = frames[idx];
  return f.basePhase + (timeS - f.timeS) * f.speed;
}

/** Pre-index sorted basicBeatmapEvents for O(log N) evaluation at any playback time. */
export function buildLightTimeline(events: readonly ViewerLightEvent[]): LightTimeline {
  const channels: Record<LightGroupId, ViewerLightEvent[]> = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
  };
  const ringSpins: RingEventKeyframe[] = [];
  const ringZooms: RingEventKeyframe[] = [];
  const leftRaw: ViewerLightEvent[] = [];
  const rightRaw: ViewerLightEvent[] = [];

  let currentRot = 0;
  let zoomToggle = false;
  let currentSpacing = 6.0;

  for (const ev of events) {
    if (ev.type >= 0 && ev.type <= 4) {
      channels[ev.type as LightGroupId].push(ev);
    } else if (ev.type === 8) {
      const prev = currentRot;
      currentRot += Math.PI / 6;
      ringSpins.push({ timeS: ev.timeS, prevValue: prev, targetValue: currentRot });
    } else if (ev.type === 9) {
      const prev = currentSpacing;
      zoomToggle = !zoomToggle;
      currentSpacing = zoomToggle ? 3.8 : 6.0;
      ringZooms.push({ timeS: ev.timeS, prevValue: prev, targetValue: currentSpacing });
    } else if (ev.type === 12) {
      leftRaw.push(ev);
    } else if (ev.type === 13) {
      rightRaw.push(ev);
    }
  }

  return {
    channels,
    ringSpins,
    ringZooms,
    leftLaserSpeeds: buildLaserKeyframes(leftRaw),
    rightLaserSpeeds: buildLaserKeyframes(rightRaw),
  };
}

function evalRingValue(
  frames: readonly RingEventKeyframe[],
  timeS: number,
  defaultVal: number,
): number {
  const idx = lastIndexAtOrBefore(frames, timeS);
  if (idx < 0) return defaultVal;
  const f = frames[idx];
  const alpha = Math.min(1, Math.max(0, (timeS - f.timeS) / 0.7));
  const smooth = alpha * alpha * (3 - 2 * alpha);
  return f.prevValue + (f.targetValue - f.prevValue) * smooth;
}

/** Evaluate full DefaultEnvironment lighting state at `timeS` deterministically. */
export function evaluateLightsAtTime(timeline: LightTimeline, timeS: number): LightsState {
  return {
    groups: {
      0: evalChannelAt(timeline.channels[0], timeS),
      1: evalChannelAt(timeline.channels[1], timeS),
      2: evalChannelAt(timeline.channels[2], timeS),
      3: evalChannelAt(timeline.channels[3], timeS),
      4: evalChannelAt(timeline.channels[4], timeS),
    },
    ringRotation: evalRingValue(timeline.ringSpins, timeS, 0),
    ringSpacing: evalRingValue(timeline.ringZooms, timeS, 6.0),
    leftLaserPhase: evalLaserPhase(timeline.leftLaserSpeeds, timeS),
    rightLaserPhase: evalLaserPhase(timeline.rightLaserSpeeds, timeS),
  };
}
