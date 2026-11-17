export interface BpmChangePoint {
  beat: number;
  bpm: number;
  timeS: number;
}

export interface TempoMap {
  baseBpm: number;
  songTimeOffset: number;
  points: BpmChangePoint[];
}

export interface JumpMetrics {
  njs: number;
  hjdBeats: number;
  halfJumpDurationSec: number;
  jumpDistance: number;
  halfJumpDistance: number;
}

export interface TimedObject {
  timeS: number;
}

export interface ViewerNote extends TimedObject {
  beat: number;
  x: number;
  y: number;
  color: 0 | 1;
  direction: number; // 0..8
  angleOffset: number; // degrees CCW
}

export interface ViewerBomb extends TimedObject {
  beat: number;
  x: number;
  y: number;
}

export interface ViewerObstacle extends TimedObject {
  beat: number;
  durationBeats: number;
  durationS: number;
  endTimeS: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ViewerArc extends TimedObject {
  beat: number;
  color: 0 | 1;
  x: number;
  y: number;
  direction: number;
  mu: number;
  tailBeat: number;
  tailTimeS: number;
  tx: number;
  ty: number;
  tailDirection: number;
  tmu: number;
  midAnchorMode: number;
}

export interface ViewerChain extends TimedObject {
  beat: number;
  color: 0 | 1;
  x: number;
  y: number;
  direction: number;
  tailBeat: number;
  tailTimeS: number;
  tx: number;
  ty: number;
  segments: number; // sc (head + sc-1 links)
  squish: number; // s in (0, 1]
}

export interface ViewerLightEvent extends TimedObject {
  beat: number;
  type: number; // et
  value: number; // i
  floatValue: number; // f
}

export interface ParsedDifficultyData {
  schema: "v2" | "v3";
  hasBpmChanges: boolean;
  notes: ViewerNote[];
  bombs: ViewerBomb[];
  obstacles: ViewerObstacle[];
  arcs: ViewerArc[];
  chains: ViewerChain[];
  lightEvents: ViewerLightEvent[];
  tempo: TempoMap;
}
