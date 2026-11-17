import { describe, expect, it } from "vitest";
import {
  cutDirectionAngleRad,
  cutDirectionVector,
  gridX,
  gridY,
  isDotDirection,
} from "./geometry";

describe("viewer geometry", () => {
  it("centers 4x3 grid at 0.6m columns and ~0.8/1.35/1.9m rows", () => {
    expect(gridX(0)).toBeCloseTo(-0.9, 6);
    expect(gridX(1)).toBeCloseTo(-0.3, 6);
    expect(gridX(2)).toBeCloseTo(0.3, 6);
    expect(gridX(3)).toBeCloseTo(0.9, 6);
    expect((gridX(0) + gridX(3)) / 2).toBeCloseTo(0, 6);

    expect(gridY(0)).toBeCloseTo(0.8, 6);
    expect(gridY(1)).toBeCloseTo(1.35, 6);
    expect(gridY(2)).toBeCloseTo(1.9, 6);
  });

  it("computes cut direction angles and CCW angleOffset for d=0..8", () => {
    expect(cutDirectionAngleRad(0, 0)).toBeCloseTo(0, 6); // Up
    expect(cutDirectionAngleRad(1, 0)).toBeCloseTo(Math.PI, 6); // Down
    expect(cutDirectionAngleRad(2, 0)).toBeCloseTo(Math.PI / 2, 6); // Left
    expect(cutDirectionAngleRad(3, 0)).toBeCloseTo(-Math.PI / 2, 6); // Right
    expect(cutDirectionAngleRad(4, 0)).toBeCloseTo(Math.PI / 4, 6); // Up-Left
    expect(cutDirectionAngleRad(5, 0)).toBeCloseTo(-Math.PI / 4, 6); // Up-Right
    expect(cutDirectionAngleRad(6, 0)).toBeCloseTo((3 * Math.PI) / 4, 6); // Down-Left
    expect(cutDirectionAngleRad(7, 0)).toBeCloseTo((-3 * Math.PI) / 4, 6); // Down-Right
    expect(isDotDirection(8)).toBe(true);
    expect(isDotDirection(0)).toBe(false);

    // +45 deg CCW from Up -> Up-Left
    expect(cutDirectionAngleRad(0, 45)).toBeCloseTo(Math.PI / 4, 6);
    const v = cutDirectionVector(0, 90);
    expect(v.x).toBeCloseTo(-1, 6);
    expect(v.y).toBeCloseTo(0, 6);
  });
});
