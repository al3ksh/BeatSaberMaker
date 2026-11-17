import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  CAMERA_PRESETS,
  PLAYER_MARKER_Z,
  cutDirectionAngleRad,
  cutDirectionVector,
  gridX,
  gridY,
  isDotDirection,
  obstacleBox,
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

  it("computes wall bounding boxes for full-height and crouch walls", () => {
    const full = obstacleBox(0, 0, 1, 5);
    expect(full.centerX).toBeCloseTo(-0.9, 6);
    expect(full.width).toBeCloseTo(0.6, 6);
    expect(full.centerY - full.height / 2).toBeCloseTo(0.05, 6);

    const crouch = obstacleBox(1, 2, 2, 3);
    expect(crouch.centerX).toBeCloseTo(0.0, 6);
    expect(crouch.width).toBeCloseTo(1.2, 6);
    expect(crouch.centerY - crouch.height / 2).toBeCloseTo(1.65, 6);
  });

  it("frames both the player marker and hit plane in Top and Side camera presets", () => {
    const projectNdc = (presetName: "top" | "side", point: [number, number, number]) => {
      const cfg = CAMERA_PRESETS[presetName];
      const cam = new THREE.PerspectiveCamera(cfg.fov, 16 / 9, 0.1, 140);
      cam.position.set(...cfg.position);
      cam.lookAt(...cfg.target);
      cam.updateMatrixWorld();
      cam.updateProjectionMatrix();
      return new THREE.Vector3(...point).project(cam);
    };

    // Top view: player marker near bottom of frame (y < -0.5) and inside [-1, 1], hit plane above it, track forward (-Z) higher
    const topPlayer = projectNdc("top", [0, 0.9, PLAYER_MARKER_Z]);
    const topHit = projectNdc("top", [0, 0.9, 0]);
    const topAhead = projectNdc("top", [0, 0.9, -10]);
    expect(topPlayer.y).toBeGreaterThan(-0.92);
    expect(topPlayer.y).toBeLessThan(-0.4);
    expect(topHit.y).toBeGreaterThan(topPlayer.y);
    expect(topAhead.y).toBeGreaterThan(topHit.y);
    expect(topAhead.y).toBeLessThan(0.95);

    // Side view: player marker and hit plane cleanly inside left side of frame, track extends right
    const sidePlayer = projectNdc("side", [0, 1.2, PLAYER_MARKER_Z]);
    const sideHit = projectNdc("side", [0, 1.2, 0]);
    const sideAhead = projectNdc("side", [0, 1.2, -10]);
    expect(sidePlayer.x).toBeGreaterThan(-0.92);
    expect(sidePlayer.x).toBeLessThan(-0.3);
    expect(sideHit.x).toBeGreaterThan(sidePlayer.x);
    expect(sideAhead.x).toBeGreaterThan(sideHit.x);
    expect(sideAhead.x).toBeLessThan(0.95);
  });
});
