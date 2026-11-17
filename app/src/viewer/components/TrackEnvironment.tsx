import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
  BASE_Y,
  LANE_WIDTH,
  PLAYER_MARKER_Z,
  ROW_HEIGHT,
  gridX,
  type CameraPreset,
} from "../geometry";

const TRACK_WIDTH = LANE_WIDTH * 4 + 0.28; // 2.68m
const TRACK_LENGTH = 90;
const TRACK_CENTER_Z = -TRACK_LENGTH / 2 + 3;
const PLAYER_HEAD_POS = new THREE.Vector3(0, 1.62, PLAYER_MARKER_Z);

const HORIZON_MONOLITHS: ReadonlyArray<{
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  d: number;
  rotY: number;
}> = [
  { x: -7.5, y: 4.0, z: -48, w: 2.1, h: 12, d: 2.1, rotY: 0.3 },
  { x: 7.5, y: 4.0, z: -48, w: 2.1, h: 12, d: 2.1, rotY: -0.3 },
  { x: -13.0, y: 6.2, z: -53, w: 3.2, h: 18, d: 3.2, rotY: 0.42 },
  { x: 13.0, y: 6.2, z: -53, w: 3.2, h: 18, d: 3.2, rotY: -0.42 },
  { x: -19.5, y: 8.5, z: -58, w: 4.2, h: 24, d: 4.2, rotY: 0.16 },
  { x: 19.5, y: 8.5, z: -58, w: 4.2, h: 24, d: 4.2, rotY: -0.16 },
];

export interface TrackEnvironmentProps {
  cameraPreset: CameraPreset;
}

export function TrackEnvironment({ cameraPreset }: TrackEnvironmentProps) {
  const { camera } = useThree();
  const playerBodyGroupRef = useRef<THREE.Group>(null);

  const laneDividersGeo = useMemo(() => {
    const pts: number[] = [];
    const startX = gridX(0) - LANE_WIDTH / 2;
    for (let col = 0; col <= 4; col += 1) {
      const x = startX + col * LANE_WIDTH;
      pts.push(x, 0.004, 2.2, x, 0.004, -TRACK_LENGTH + 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }, []);

  // Subtle 4x3 hit-plane grid shown only in Top / Side views so Player view stays clean
  const hitGridGeo = useMemo(() => {
    const pts: number[] = [];
    const left = -2 * LANE_WIDTH;
    const right = 2 * LANE_WIDTH;
    const bottom = BASE_Y - ROW_HEIGHT * 0.5;
    const top = BASE_Y + ROW_HEIGHT * 2.5;
    for (let c = 0; c <= 4; c += 1) {
      const x = left + c * LANE_WIDTH;
      pts.push(x, bottom, 0, x, top, 0);
    }
    for (let r = 0; r <= 3; r += 1) {
      const y = bottom + r * ROW_HEIGHT;
      pts.push(left, y, 0, right, y, 0);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }, []);

  useEffect(() => {
    return () => {
      laneDividersGeo.dispose();
      hitGridGeo.dispose();
    };
  }, [laneDividersGeo, hitGridGeo]);

  useFrame(() => {
    const grp = playerBodyGroupRef.current;
    if (!grp) return;
    // Hide upper figure & sabers only when camera sits right at first-person head height
    const dist = camera.position.distanceTo(PLAYER_HEAD_POS);
    grp.visible = cameraPreset !== "player" || dist > 2.3;
  });

  return (
    <group>
      {/* Abyssal lower void floor (unlit deep obsidian so dark scenes stay genuinely dark) */}
      <mesh position={[0, -2.2, -38]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[180, 180]} />
        <meshBasicMaterial color="#020307" />
      </mesh>

      {/* Main brushed dark obsidian runway */}
      <mesh position={[0, -0.04, TRACK_CENTER_Z]}>
        <boxGeometry args={[TRACK_WIDTH, 0.08, TRACK_LENGTH]} />
        <meshPhysicalMaterial
          color="#05070e"
          roughness={0.42}
          metalness={0.62}
          clearcoat={0.28}
          clearcoatRoughness={0.46}
          envMapIntensity={0.55}
        />
      </mesh>

      {/* Outer dark metallic side rail casings + glowing neon edge strips */}
      <mesh position={[-TRACK_WIDTH / 2 - 0.04, 0.01, TRACK_CENTER_Z]}>
        <boxGeometry args={[0.065, 0.05, TRACK_LENGTH]} />
        <meshStandardMaterial color="#080b13" roughness={0.5} metalness={0.6} envMapIntensity={0.25} />
      </mesh>
      <mesh position={[TRACK_WIDTH / 2 + 0.04, 0.01, TRACK_CENTER_Z]}>
        <boxGeometry args={[0.065, 0.05, TRACK_LENGTH]} />
        <meshStandardMaterial color="#080b13" roughness={0.5} metalness={0.6} envMapIntensity={0.25} />
      </mesh>

      <mesh position={[-TRACK_WIDTH / 2 - 0.032, 0.038, TRACK_CENTER_Z]}>
        <boxGeometry args={[0.02, 0.014, TRACK_LENGTH]} />
        <meshBasicMaterial color={[1.9, 0.22, 0.34]} toneMapped={false} />
      </mesh>
      <mesh position={[TRACK_WIDTH / 2 + 0.032, 0.038, TRACK_CENTER_Z]}>
        <boxGeometry args={[0.02, 0.014, TRACK_LENGTH]} />
        <meshBasicMaterial color={[0.22, 0.85, 2.2]} toneMapped={false} />
      </mesh>

      {/* Subtle lane separator hairlines on runway */}
      <lineSegments geometry={laneDividersGeo}>
        <lineBasicMaterial color="#1d273c" transparent opacity={0.42} />
      </lineSegments>

      {/* Subtle floor hit-plane guide hairline & low edge threshold ticks at z = 0 */}
      <mesh position={[0, 0.004, 0]}>
        <boxGeometry args={[TRACK_WIDTH, 0.005, 0.024]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.45} />
      </mesh>
      <mesh position={[-TRACK_WIDTH / 2 - 0.07, 0.05, 0]}>
        <boxGeometry args={[0.035, 0.1, 0.035]} />
        <meshBasicMaterial color="#f43f5e" transparent opacity={0.75} />
      </mesh>
      <mesh position={[TRACK_WIDTH / 2 + 0.07, 0.05, 0]}>
        <boxGeometry args={[0.035, 0.1, 0.035]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.75} />
      </mesh>

      {/* 4x3 Hit-plane grid lines visible in Side/Top camera presets */}
      <lineSegments geometry={hitGridGeo} visible={cameraPreset !== "player"}>
        <lineBasicMaterial color="#60a5fa" transparent opacity={0.38} />
      </lineSegments>

      {/* Player standing pad on the runway at PLAYER_MARKER_Z */}
      <group position={[0, 0.01, PLAYER_MARKER_Z]}>
        <mesh>
          <cylinderGeometry args={[0.56, 0.6, 0.016, 24]} />
          <meshStandardMaterial color="#0b101d" roughness={0.45} metalness={0.7} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
          <ringGeometry args={[0.51, 0.56, 32]} />
          <meshBasicMaterial color={[0.35, 0.85, 1.75]} toneMapped={false} />
        </mesh>
        {/* Foot stance pads */}
        <mesh position={[-0.18, 0.012, 0]}>
          <boxGeometry args={[0.14, 0.006, 0.28]} />
          <meshBasicMaterial color="#f43f5e" transparent opacity={0.7} />
        </mesh>
        <mesh position={[0.18, 0.012, 0]}>
          <boxGeometry args={[0.14, 0.006, 0.28]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.7} />
        </mesh>
      </group>

      {/* Stylized player silhouette + left/right saber stubs (visible in Top/Side or pulled-back Orbit) */}
      <group ref={playerBodyGroupRef} position={[0, 0, PLAYER_MARKER_Z]} visible={cameraPreset !== "player"}>
        {/* Torso */}
        <mesh position={[0, 1.15, 0]}>
          <capsuleGeometry args={[0.18, 0.36, 6, 12]} />
          <meshStandardMaterial color="#171d2c" roughness={0.4} metalness={0.65} />
        </mesh>
        {/* Head */}
        <mesh position={[0, 1.62, 0]}>
          <sphereGeometry args={[0.14, 16, 12]} />
          <meshStandardMaterial color="#1f273b" roughness={0.35} metalness={0.7} />
        </mesh>
        {/* VR headset visor facing -Z */}
        <mesh position={[0, 1.63, -0.11]}>
          <boxGeometry args={[0.2, 0.08, 0.07]} />
          <meshBasicMaterial color={[0.45, 1.1, 2.0]} toneMapped={false} />
        </mesh>

        {/* Left Red Saber (hilt + glowing blade stub angled toward hit plane) */}
        <group position={[-0.36, 1.08, -0.18]} rotation={[Math.PI * 0.42, 0, 0.14]}>
          <mesh position={[0, -0.09, 0]}>
            <cylinderGeometry args={[0.018, 0.018, 0.18, 10]} />
            <meshStandardMaterial color="#1e293b" roughness={0.3} metalness={0.9} />
          </mesh>
          <mesh position={[0, 0.28, 0]}>
            <cylinderGeometry args={[0.014, 0.016, 0.56, 10]} />
            <meshBasicMaterial color={[2.4, 0.25, 0.4]} toneMapped={false} />
          </mesh>
        </group>

        {/* Right Blue Saber (hilt + glowing blade stub angled toward hit plane) */}
        <group position={[0.36, 1.08, -0.18]} rotation={[Math.PI * 0.42, 0, -0.14]}>
          <mesh position={[0, -0.09, 0]}>
            <cylinderGeometry args={[0.018, 0.018, 0.18, 10]} />
            <meshStandardMaterial color="#1e293b" roughness={0.3} metalness={0.9} />
          </mesh>
          <mesh position={[0, 0.28, 0]}>
            <cylinderGeometry args={[0.014, 0.016, 0.56, 10]} />
            <meshBasicMaterial color={[0.26, 1.05, 2.6]} toneMapped={false} />
          </mesh>
        </group>
      </group>

      {/* Flanking outer dark structural decks for depth */}
      <mesh position={[-3.9, -0.35, TRACK_CENTER_Z]}>
        <boxGeometry args={[1.6, 0.14, TRACK_LENGTH]} />
        <meshStandardMaterial color="#05070e" roughness={0.65} metalness={0.45} envMapIntensity={0.18} />
      </mesh>
      <mesh position={[3.9, -0.35, TRACK_CENTER_Z]}>
        <boxGeometry args={[1.6, 0.14, TRACK_LENGTH]} />
        <meshStandardMaterial color="#05070e" roughness={0.65} metalness={0.45} envMapIntensity={0.18} />
      </mesh>

      {/* Far horizon monolithic skyline silhouettes */}
      {HORIZON_MONOLITHS.map((m, idx) => (
        <mesh key={idx} position={[m.x, m.y, m.z]} rotation={[0, m.rotY, 0]}>
          <boxGeometry args={[m.w, m.h, m.d]} />
          <meshStandardMaterial color="#04060d" roughness={0.55} metalness={0.5} envMapIntensity={0.2} />
        </mesh>
      ))}
    </group>
  );
}
