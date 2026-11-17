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
import { PlayerAvatar } from "./PlayerAvatar";

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

const DECK_PANEL_LENGTH = 3.6;
const DECK_PANEL_GAP = 0.35;
const DECK_PANEL_WIDTH = 1.5;
const DECK_X = 2.65;
const DECK_Y = -0.42;
const DECK_PANELS: ReadonlyArray<{ x: number; z: number; side: -1 | 1 }> = (() => {
  const out: Array<{ x: number; z: number; side: -1 | 1 }> = [];
  const step = DECK_PANEL_LENGTH + DECK_PANEL_GAP;
  for (let z = 1.8; z > -TRACK_LENGTH + 4; z -= step) {
    out.push({ x: -DECK_X, z: z - DECK_PANEL_LENGTH / 2, side: -1 });
    out.push({ x: DECK_X, z: z - DECK_PANEL_LENGTH / 2, side: 1 });
  }
  return out;
})();
const DECK_STRIP_LEFT = new THREE.Color(0.55, 0.06, 0.1);
const DECK_STRIP_RIGHT = new THREE.Color(0.06, 0.22, 0.6);

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

  const deckPanelGeo = useMemo(
    () => new THREE.BoxGeometry(DECK_PANEL_WIDTH, 0.12, DECK_PANEL_LENGTH),
    [],
  );
  const deckPanelMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: "#04050a", roughness: 0.9, metalness: 0.1, envMapIntensity: 0.03 }),
    [],
  );
  const deckStripGeo = useMemo(() => new THREE.BoxGeometry(0.035, 0.02, DECK_PANEL_LENGTH - 0.3), []);
  const deckPanelsRef = useRef<THREE.InstancedMesh>(null);
  const deckStripsRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    const panels = deckPanelsRef.current;
    const strips = deckStripsRef.current;
    if (!panels || !strips) return;
    const m = new THREE.Matrix4();
    DECK_PANELS.forEach((p, i) => {
      m.makeTranslation(p.x, DECK_Y, p.z);
      panels.setMatrixAt(i, m);
      const innerEdgeX = p.x - p.side * (DECK_PANEL_WIDTH / 2 - 0.05);
      m.makeTranslation(innerEdgeX, DECK_Y + 0.07, p.z);
      strips.setMatrixAt(i, m);
      strips.setColorAt(i, p.side < 0 ? DECK_STRIP_LEFT : DECK_STRIP_RIGHT);
    });
    panels.instanceMatrix.needsUpdate = true;
    strips.instanceMatrix.needsUpdate = true;
    if (strips.instanceColor) strips.instanceColor.needsUpdate = true;
  }, []);

  useEffect(() => {
    return () => {
      deckPanelGeo.dispose();
      deckPanelMat.dispose();
      deckStripGeo.dispose();
      laneDividersGeo.dispose();
      hitGridGeo.dispose();
    };
  }, [deckPanelGeo, deckPanelMat, deckStripGeo, laneDividersGeo, hitGridGeo]);

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
          envMapIntensity={0.28}
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

      {/* Floating VR avatar holding both sabers (hidden only at first-person head height) */}
      <group ref={playerBodyGroupRef} position={[0, 0, PLAYER_MARKER_Z]} visible={cameraPreset !== "player"}>
        <PlayerAvatar />
      </group>

      {/* Flanking segmented matte deck panels with dim inner-edge accent strips */}
      <instancedMesh
        ref={deckPanelsRef}
        args={[deckPanelGeo, deckPanelMat, DECK_PANELS.length]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={deckStripsRef}
        args={[deckStripGeo, undefined, DECK_PANELS.length]}
        frustumCulled={false}
      >
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

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
