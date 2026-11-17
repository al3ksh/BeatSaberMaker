import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { BASE_Y, LANE_WIDTH, ROW_HEIGHT, gridX, gridY } from "../geometry";

const TRACK_WIDTH = LANE_WIDTH * 4 + 0.36; // 2.76m
const TRACK_LENGTH = 44;

export function TrackEnvironment() {
  const gridLinesGeo = useMemo(() => {
    const pts: number[] = [];
    const minX = gridX(0) - LANE_WIDTH / 2;
    const maxX = gridX(3) + LANE_WIDTH / 2;
    const minY = BASE_Y - ROW_HEIGHT / 2;
    const maxY = gridY(2) + ROW_HEIGHT / 2;

    // 4x3 grid frame at hit plane z = 0
    for (let col = 0; col <= 4; col += 1) {
      const x = minX + col * LANE_WIDTH;
      pts.push(x, minY, 0, x, maxY, 0);
    }
    for (let row = 0; row <= 3; row += 1) {
      const y = minY + row * ROW_HEIGHT;
      pts.push(minX, y, 0, maxX, y, 0);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }, []);

  const laneDividersGeo = useMemo(() => {
    const pts: number[] = [];
    const startX = gridX(0) - LANE_WIDTH / 2;
    for (let col = 0; col <= 4; col += 1) {
      const x = startX + col * LANE_WIDTH;
      pts.push(x, 0.015, 1.5, x, 0.015, -TRACK_LENGTH + 2);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }, []);

  useEffect(() => {
    return () => {
      gridLinesGeo.dispose();
      laneDividersGeo.dispose();
    };
  }, [gridLinesGeo, laneDividersGeo]);

  return (
    <group>
      {/* Main dark floor track */}
      <mesh position={[0, -0.04, -TRACK_LENGTH / 2 + 2]} receiveShadow>
        <boxGeometry args={[TRACK_WIDTH, 0.08, TRACK_LENGTH]} />
        <meshStandardMaterial color="#0b0c12" roughness={0.35} metalness={0.55} />
      </mesh>

      {/* Side glowing track trim rails (left=red, right=blue) */}
      <mesh position={[-TRACK_WIDTH / 2 - 0.03, 0.01, -TRACK_LENGTH / 2 + 2]}>
        <boxGeometry args={[0.04, 0.04, TRACK_LENGTH]} />
        <meshStandardMaterial color="#1e293b" emissive="#ef4444" emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[TRACK_WIDTH / 2 + 0.03, 0.01, -TRACK_LENGTH / 2 + 2]}>
        <boxGeometry args={[0.04, 0.04, TRACK_LENGTH]} />
        <meshStandardMaterial color="#1e293b" emissive="#3b82f6" emissiveIntensity={0.45} />
      </mesh>

      {/* Lane guide lines along the floor */}
      <lineSegments geometry={laneDividersGeo}>
        <lineBasicMaterial color="#27272a" transparent opacity={0.7} />
      </lineSegments>

      {/* Hit-plane baseline at z = 0 */}
      <mesh position={[0, 0.018, 0]}>
        <boxGeometry args={[TRACK_WIDTH, 0.01, 0.04]} />
        <meshBasicMaterial color="#94a3b8" transparent opacity={0.6} />
      </mesh>

      {/* Faint 4x3 grid reference at hit plane */}
      <lineSegments geometry={gridLinesGeo}>
        <lineBasicMaterial color="#475569" transparent opacity={0.28} />
      </lineSegments>
    </group>
  );
}
