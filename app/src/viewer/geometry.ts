export const LANE_WIDTH = 0.6;
export const ROW_HEIGHT = 0.55;
export const BASE_Y = 0.8;
export const NOTE_SIZE = 0.42;
export const BOMB_RADIUS = 0.2;
export const PLAYER_MARKER_Z = 1.15;

export type CameraPreset = "player" | "side" | "top";

export interface CameraPresetConfig {
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
}

export const CAMERA_PRESETS: Record<CameraPreset, CameraPresetConfig> = {
  player: { position: [0, 1.72, 2.55], target: [0, 1.35, -10], fov: 68 },
  side: { position: [11.2, 2.15, -5.2], target: [0, 1.35, -5.2], fov: 50 },
  top: { position: [0, 18.0, -3.0], target: [0, 1.0, -5.2], fov: 56 },
};

const DIRECTION_ANGLES_RAD: Record<number, number> = {
  0: 0, // Up
  1: Math.PI, // Down
  2: Math.PI / 2, // Left
  3: -Math.PI / 2, // Right
  4: Math.PI / 4, // Up-Left
  5: -Math.PI / 4, // Up-Right
  6: (3 * Math.PI) / 4, // Down-Left
  7: (-3 * Math.PI) / 4, // Down-Right
  8: 0, // Dot / Any
};

/** Centered horizontal coordinate (meters) for lane index 0..3. */
export function gridX(x: number): number {
  return (x - 1.5) * LANE_WIDTH;
}

/** Vertical coordinate (meters) for layer index 0..2 (~0.8 / 1.35 / 1.9 m). */
export function gridY(y: number): number {
  return BASE_Y + y * ROW_HEIGHT;
}

export function isDotDirection(direction: number): boolean {
  return direction === 8;
}

/** Cut rotation angle around +Z in radians (0 = Up, positive = CCW). */
export function cutDirectionAngleRad(direction: number, angleOffsetDeg = 0): number {
  const base = DIRECTION_ANGLES_RAD[direction] ?? 0;
  const offset = Number.isFinite(angleOffsetDeg) ? (angleOffsetDeg * Math.PI) / 180 : 0;
  return base + offset;
}

/** Unit 2D direction vector in the XY plane for a note's cut direction. */
export function cutDirectionVector(
  direction: number,
  angleOffsetDeg = 0,
): { x: number; y: number } {
  if (isDotDirection(direction)) {
    return { x: 0, y: 0 };
  }
  const rad = cutDirectionAngleRad(direction, angleOffsetDeg);
  return { x: -Math.sin(rad), y: Math.cos(rad) };
}

/** Wall box center X, width, center Y, and height in meters. */
export function obstacleBox(x: number, y: number, w: number, h: number): {
  centerX: number;
  width: number;
  centerY: number;
  height: number;
} {
  const width = Math.max(0.2, w * LANE_WIDTH);
  const centerX = (x - 2 + w / 2) * LANE_WIDTH;
  const bottomY = y <= 0 ? 0.05 : 0.55 + y * ROW_HEIGHT;
  const height = Math.max(0.4, h * ROW_HEIGHT);
  return {
    centerX,
    width,
    centerY: bottomY + height / 2,
    height,
  };
}
