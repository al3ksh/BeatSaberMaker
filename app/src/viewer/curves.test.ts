import { describe, expect, it } from "vitest";
import {
  computeArcControlPoints,
  evaluateCubicBezier,
  generateChainLinks,
} from "./curves";
import { gridX, gridY } from "./geometry";
import type { ViewerArc, ViewerChain } from "./types";

describe("viewer curves (arcs & chains)", () => {
  it("evaluates cubic Bezier arc with endpoints matching head and tail", () => {
    const arc: ViewerArc = {
      beat: 4,
      timeS: 2.0,
      color: 0,
      x: 1,
      y: 0,
      direction: 1, // Down
      mu: 1,
      tailBeat: 6,
      tailTimeS: 3.0,
      tx: 2,
      ty: 2,
      tailDirection: 0, // Up
      tmu: 1,
      midAnchorMode: 0,
    };

    const { p0, p1, p2, p3 } = computeArcControlPoints(arc, 1.0, 14);
    expect(p0.x).toBeCloseTo(gridX(1), 6);
    expect(p0.y).toBeCloseTo(gridY(0), 6);
    expect(p0.z).toBeCloseTo((1.0 - 2.0) * 14, 6);

    expect(p3.x).toBeCloseTo(gridX(2), 6);
    expect(p3.y).toBeCloseTo(gridY(2), 6);
    expect(p3.z).toBeCloseTo((1.0 - 3.0) * 14, 6);

    const start = evaluateCubicBezier(p0, p1, p2, p3, 0);
    const end = evaluateCubicBezier(p0, p1, p2, p3, 1);
    expect(start).toEqual(p0);
    expect(end).toEqual(p3);

    // Head direction d=1 (Down) pulls P1.y below P0.y
    expect(p1.y).toBeLessThan(p0.y);
    // Tail direction tc=0 (Up) pulls P2.y below P3.y so curve enters P3 going Up
    expect(p2.y).toBeLessThan(p3.y);
  });

  it("generates sc-1 chain links from head to tail respecting squish factor", () => {
    const chain: ViewerChain = {
      beat: 10,
      timeS: 5.0,
      color: 1,
      x: 2,
      y: 0,
      direction: 3, // Right
      tailBeat: 10.25,
      tailTimeS: 5.125,
      tx: 3,
      ty: 0,
      segments: 5,
      squish: 1.0,
    };

    const links = generateChainLinks(chain);
    expect(links).toHaveLength(4); // head + (sc - 1) links = 5 total
    expect(links[links.length - 1].worldX).toBeCloseTo(gridX(3), 5);
    expect(links[links.length - 1].worldY).toBeCloseTo(gridY(0), 5);
    expect(links[links.length - 1].timeS).toBeCloseTo(5.125, 6);

    const squished = generateChainLinks({ ...chain, squish: 0.5 });
    expect(squished).toHaveLength(4);
    expect(squished[squished.length - 1].worldX).toBeLessThan(gridX(3));
  });
});
