import {
  cutDirectionAngleRad,
  cutDirectionVector,
  gridX,
  gridY,
} from "./geometry";
import { objectZAtTime } from "./timing";
import type { ViewerArc, ViewerChain } from "./types";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface ArcControlPoints {
  p0: Vec3;
  p1: Vec3;
  p2: Vec3;
  p3: Vec3;
}

export interface ChainLinkInstance {
  timeS: number;
  worldX: number;
  worldY: number;
  rotZ: number;
  color: 0 | 1;
}

/**
 * Compute cubic Bezier control points for an arc at `currentTimeS`:
 * P0 = head, P1 = P0 + dir(d) * mu * k, P2 = P3 - dir(tc) * tmu * k, P3 = tail.
 * Control length k scales with head-to-tail distance and is clamped so multi-beat
 * high-NJS arcs curve gracefully within the playfield.
 */
export function computeArcControlPoints(
  arc: ViewerArc,
  currentTimeS: number,
  njs: number,
): ArcControlPoints {
  const p0: Vec3 = {
    x: gridX(arc.x),
    y: gridY(arc.y),
    z: objectZAtTime(arc.timeS, currentTimeS, njs),
  };
  const p3: Vec3 = {
    x: gridX(arc.tx),
    y: gridY(arc.ty),
    z: objectZAtTime(arc.tailTimeS, currentTimeS, njs),
  };

  const dx = p3.x - p0.x;
  const dy = p3.y - p0.y;
  const dz = p3.z - p0.z;
  const dist = Math.hypot(dx, dy, dz);
  const k = Math.max(0.6, Math.min(2.1, 0.45 * dist));

  const headDir = cutDirectionVector(arc.direction);
  const tailDir = cutDirectionVector(arc.tailDirection);

  // Optional lateral bow for midAnchorMode (1=CW, 2=CCW) when head/tail share direction
  const bow = arc.midAnchorMode === 1 ? 0.35 : arc.midAnchorMode === 2 ? -0.35 : 0;

  const p1: Vec3 = {
    x: p0.x + headDir.x * arc.mu * k + bow,
    y: p0.y + headDir.y * arc.mu * k,
    z: p0.z + dz * 0.25,
  };
  const p2: Vec3 = {
    x: p3.x - tailDir.x * arc.tmu * k + bow,
    y: p3.y - tailDir.y * arc.tmu * k,
    z: p0.z + dz * 0.75,
  };

  return { p0, p1, p2, p3 };
}

/** Evaluate cubic Bezier curve at parameter u in [0, 1]. */
export function evaluateCubicBezier(
  p0: Vec3,
  p1: Vec3,
  p2: Vec3,
  p3: Vec3,
  u: number,
): Vec3 {
  const t = Math.max(0, Math.min(1, u));
  const inv = 1 - t;
  const inv2 = inv * inv;
  const t2 = t * t;
  const b0 = inv2 * inv;
  const b1 = 3 * inv2 * t;
  const b2 = 3 * inv * t2;
  const b3 = t2 * t;
  return {
    x: b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x,
    y: b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y,
    z: b0 * p0.z + b1 * p1.z + b2 * p2.z + b3 * p3.z,
  };
}

/**
 * Generate the `sc - 1` chain link instances from head toward tail along a
 * quadratic Bezier spline, respecting `squish` (s in (0, 1]).
 */
export function generateChainLinks(chain: ViewerChain): ChainLinkInstance[] {
  const linkCount = Math.max(0, chain.segments - 1);
  if (linkCount === 0) return [];

  const hx = gridX(chain.x);
  const hy = gridY(chain.y);
  const tx = gridX(chain.tx);
  const ty = gridY(chain.ty);
  const distXY = Math.hypot(tx - hx, ty - hy);
  const dir = cutDirectionVector(chain.direction);
  const cx = hx + dir.x * (distXY * 0.5);
  const cy = hy + dir.y * (distXY * 0.5);
  const fallbackRot = cutDirectionAngleRad(chain.direction);

  const links: ChainLinkInstance[] = [];
  for (let i = 1; i <= linkCount; i += 1) {
    const frac = i / linkCount;
    const u = frac * chain.squish;
    const inv = 1 - u;

    const worldX = inv * inv * hx + 2 * inv * u * cx + u * u * tx;
    const worldY = inv * inv * hy + 2 * inv * u * cy + u * u * ty;

    // Tangent for link orientation
    const tanX = 2 * inv * (cx - hx) + 2 * u * (tx - cx);
    const tanY = 2 * inv * (cy - hy) + 2 * u * (ty - cy);
    const rotZ =
      Math.hypot(tanX, tanY) > 1e-4 ? Math.atan2(-tanX, tanY) : fallbackRot;

    links.push({
      timeS: chain.timeS + (chain.tailTimeS - chain.timeS) * frac,
      worldX,
      worldY,
      rotZ,
      color: chain.color,
    });
  }
  return links;
}
