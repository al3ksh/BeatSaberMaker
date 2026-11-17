import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { queryWindow } from "../parser";
import { objectZAtTime } from "../timing";
import type { ViewerNote } from "../types";
import {
  NOTE_SIZE,
  cutDirectionAngleRad,
  gridX,
  gridY,
  isDotDirection,
} from "../geometry";
import {
  CUT_EFFECT_DURATION_S,
  computeSpawnFade,
  cutEffectProgress,
  isHitObjectVisible,
} from "../visibility";
import { createSoftGlowTexture } from "../textures";

const MAX_NOTE_INSTANCES = 512;
const MAX_CUT_HALVES = 256;
const MAX_CUT_FLASHES = 256;
const MAX_CUT_SPARKS = 512;
const SPARKS_PER_CUT = 4;
const HALF_OFFSET = NOTE_SIZE / 4;
const WHITE = new THREE.Color(1, 1, 1);

const LEFT_COLOR = new THREE.Color("#dc142c");
const RIGHT_COLOR = new THREE.Color("#1864f5");
const LEFT_GLOW = new THREE.Color(2.4, 0.26, 0.42);
const RIGHT_GLOW = new THREE.Color(0.28, 1.25, 2.8);

function createArrowGeometry(): THREE.BufferGeometry {
  // Sleek Beat Saber-style chevron pointing +Y inside the flat front face
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.102);
  shape.lineTo(0.114, -0.026);
  shape.lineTo(0.076, -0.064);
  shape.lineTo(0, 0.02);
  shape.lineTo(-0.076, -0.064);
  shape.lineTo(-0.114, -0.026);
  shape.closePath();
  return new THREE.ShapeGeometry(shape);
}

export interface NotesLayerProps {
  notes: readonly ViewerNote[];
  njs: number;
  halfJumpDurationSec: number;
  getTimeS: () => number;
}

export function NotesLayer({
  notes,
  njs,
  halfJumpDurationSec,
  getTimeS,
}: NotesLayerProps) {
  const bodyMeshRef = useRef<THREE.InstancedMesh>(null);
  const arrowMeshRef = useRef<THREE.InstancedMesh>(null);
  const dotMeshRef = useRef<THREE.InstancedMesh>(null);
  const cutHalfMeshRef = useRef<THREE.InstancedMesh>(null);
  const cutFlashMeshRef = useRef<THREE.InstancedMesh>(null);
  const cutFaceMeshRef = useRef<THREE.InstancedMesh>(null);
  const cutSparkMeshRef = useRef<THREE.InstancedMesh>(null);

  const bodyGeo = useMemo(
    () => new RoundedBoxGeometry(NOTE_SIZE, NOTE_SIZE, NOTE_SIZE, 3, 0.055),
    [],
  );
  const arrowGeo = useMemo(() => createArrowGeometry(), []);
  const dotGeo = useMemo(() => new THREE.CircleGeometry(0.054, 24), []);
  const cutHalfGeo = useMemo(
    () =>
      new RoundedBoxGeometry(NOTE_SIZE * 0.5, NOTE_SIZE, NOTE_SIZE, 2, 0.03),
    [],
  );
  const cutFaceGeo = useMemo(() => new THREE.PlaneGeometry(NOTE_SIZE * 0.84, NOTE_SIZE * 0.84), []);
  const cutFlashGeo = useMemo(() => new THREE.PlaneGeometry(0.62, 0.62), []);
  const cutSparkGeo = useMemo(() => new THREE.BoxGeometry(0.016, 0.09, 0.016), []);
  const glowTex = useMemo(() => createSoftGlowTexture(), []);

  useEffect(() => {
    const dummyColor = new THREE.Color(1, 1, 1);
    for (const ref of [bodyMeshRef, cutHalfMeshRef, cutFaceMeshRef, cutFlashMeshRef, cutSparkMeshRef]) {
      const mesh = ref.current;
      if (mesh && !mesh.instanceColor) {
        mesh.setColorAt(0, dummyColor);
        mesh.count = 0;
      }
    }
    return () => {
      bodyGeo.dispose();
      arrowGeo.dispose();
      dotGeo.dispose();
      cutHalfGeo.dispose();
      cutFaceGeo.dispose();
      cutFlashGeo.dispose();
      cutSparkGeo.dispose();
      glowTex.dispose();
    };
  }, [bodyGeo, arrowGeo, dotGeo, cutHalfGeo, cutFaceGeo, cutFlashGeo, cutSparkGeo, glowTex]);

  const dummy = useMemo(() => {
    const o = new THREE.Object3D();
    // Z then local Y: lets cut halves "open like a book" around the cut line
    o.rotation.order = "ZYX";
    return o;
  }, []);
  const faceLocal = useMemo(() => new THREE.Matrix4(), []);
  const faceWorld = useMemo(() => new THREE.Matrix4(), []);
  const arrowDummy = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  useFrame(() => {
    const bodyMesh = bodyMeshRef.current;
    const arrowMesh = arrowMeshRef.current;
    const dotMesh = dotMeshRef.current;
    const cutHalfMesh = cutHalfMeshRef.current;
    const cutFlashMesh = cutFlashMeshRef.current;
    const cutFaceMesh = cutFaceMeshRef.current;
    const cutSparkMesh = cutSparkMeshRef.current;
    if (!bodyMesh || !arrowMesh || !dotMesh) return;

    const t = getTimeS();
    const visible = queryWindow(notes, t - CUT_EFFECT_DURATION_S, t + halfJumpDurationSec);

    let bodyCount = 0;
    let arrowCount = 0;
    let dotCount = 0;
    let halfCount = 0;
    let flashCount = 0;
    let faceCount = 0;
    let sparkCount = 0;

    const faceOffset = NOTE_SIZE / 2 + 0.0025;

    for (let i = 0; i < visible.length; i += 1) {
      const note = visible[i];
      const wx = gridX(note.x);
      const wy = gridY(note.y);
      const rotZ = cutDirectionAngleRad(note.direction, note.angleOffset);
      const handColor = note.color === 0 ? LEFT_COLOR : RIGHT_COLOR;
      const glowColor = note.color === 0 ? LEFT_GLOW : RIGHT_GLOW;

      if (isHitObjectVisible(note.timeS, t, halfJumpDurationSec)) {
        if (bodyCount >= MAX_NOTE_INSTANCES) continue;
        const wz = objectZAtTime(note.timeS, t, njs);
        const fade = computeSpawnFade(t, note.timeS, halfJumpDurationSec, 0.18);
        const scale = 0.04 + 0.96 * fade;

        dummy.position.set(wx, wy, wz);
        dummy.rotation.set(0, 0, rotZ);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();

        bodyMesh.setMatrixAt(bodyCount, dummy.matrix);
        bodyMesh.setColorAt(bodyCount, handColor);
        bodyCount += 1;

        const zFront = wz + faceOffset * scale;
        if (isDotDirection(note.direction)) {
          arrowDummy.position.set(wx, wy, zFront);
          arrowDummy.rotation.set(0, 0, 0);
          arrowDummy.scale.setScalar(scale);
          arrowDummy.updateMatrix();
          dotMesh.setMatrixAt(dotCount, arrowDummy.matrix);
          dotCount += 1;
        } else {
          arrowDummy.position.set(wx, wy, zFront);
          arrowDummy.rotation.set(0, 0, rotZ);
          arrowDummy.scale.setScalar(scale);
          arrowDummy.updateMatrix();
          arrowMesh.setMatrixAt(arrowCount, arrowDummy.matrix);
          arrowCount += 1;
        }
        continue;
      }

      // Hit cut effect (~0.2 s) strictly at or forward of z = 0 (never behind camera)
      const p = cutEffectProgress(note.timeS, t, CUT_EFFECT_DURATION_S);
      if (p === null) continue;

      const easeOut = 1 - (1 - p) * (1 - p);
      const fadeOut = (1 - p) * (1 - p);
      const perpX = Math.cos(rotZ);
      const perpY = Math.sin(rotZ);
      const cutX = -perpY;
      const cutY = perpX;

      // Two halves of the sliced note: split along the cut line, opening outward and dropping
      if (cutHalfMesh && halfCount + 1 < MAX_CUT_HALVES) {
        const open = 0.01 + easeOut * 0.085;
        const fall = 0.32 * p * p;
        const shrink = Math.max(0.05, 1 - p * 0.35 - p * p * 0.5);
        const zCut = -0.02 - easeOut * 0.12;
        for (let side = -1; side <= 1; side += 2) {
          const off = (HALF_OFFSET + open) * side;
          dummy.position.set(wx + perpX * off, wy + perpY * off - fall, zCut);
          dummy.rotation.set(0, side * 0.55 * easeOut, rotZ - side * 0.22 * easeOut);
          dummy.scale.setScalar(shrink);
          dummy.updateMatrix();
          cutHalfMesh.setMatrixAt(halfCount, dummy.matrix);
          cutHalfMesh.setColorAt(halfCount, handColor);
          halfCount += 1;

          // Glowing exposed cut face on the inner side of each half
          if (cutFaceMesh) {
            faceLocal.makeRotationY((-side * Math.PI) / 2);
            faceLocal.setPosition(-side * (HALF_OFFSET + 0.003), 0, 0);
            faceWorld.multiplyMatrices(dummy.matrix, faceLocal);
            cutFaceMesh.setMatrixAt(faceCount, faceWorld);
            tmpColor.copy(glowColor).lerp(WHITE, 0.25 * (1 - p)).multiplyScalar(0.35 + 0.9 * (1 - p));
            cutFaceMesh.setColorAt(faceCount, tmpColor);
            faceCount += 1;
          }
        }
      }

      if (cutFlashMesh && flashCount + 1 < MAX_CUT_FLASHES) {
        // Thin slash streak along the cut direction
        dummy.position.set(wx, wy, -0.01);
        dummy.rotation.set(0, 0, rotZ);
        dummy.scale.set(0.16 * (1 - p * 0.6), 0.9 + easeOut * 1.5, 1);
        dummy.updateMatrix();
        cutFlashMesh.setMatrixAt(flashCount, dummy.matrix);
        tmpColor.copy(glowColor).lerp(WHITE, 0.35).multiplyScalar(fadeOut * 1.25);
        cutFlashMesh.setColorAt(flashCount, tmpColor);
        flashCount += 1;

        // Small soft bloom at the impact point
        dummy.scale.setScalar(0.55 + easeOut * 0.45);
        dummy.updateMatrix();
        cutFlashMesh.setMatrixAt(flashCount, dummy.matrix);
        tmpColor.copy(glowColor).multiplyScalar(fadeOut * 0.4);
        cutFlashMesh.setColorAt(flashCount, tmpColor);
        flashCount += 1;
      }

      if (cutSparkMesh && sparkCount + SPARKS_PER_CUT <= MAX_CUT_SPARKS) {
        // A few sparks thrown out sideways from the cut line
        for (let k = 0; k < SPARKS_PER_CUT; k += 1) {
          const side = k % 2 === 0 ? 1 : -1;
          const along = k < 2 ? 0.55 : -0.25;
          const dx = perpX * side + cutX * along;
          const dy = perpY * side + cutY * along;
          const len = Math.hypot(dx, dy);
          const dist = 0.1 + easeOut * (0.34 + k * 0.04);
          dummy.position.set(
            wx + (dx / len) * dist,
            wy + (dy / len) * dist - 0.12 * p * p,
            -0.02 - easeOut * 0.08,
          );
          dummy.rotation.set(0, 0, Math.atan2(dy, dx) - Math.PI / 2);
          dummy.scale.set(1, Math.max(0.25, 1.3 - p), 1);
          dummy.updateMatrix();
          cutSparkMesh.setMatrixAt(sparkCount, dummy.matrix);
          tmpColor.copy(glowColor).lerp(WHITE, 0.2).multiplyScalar(fadeOut * 1.5);
          cutSparkMesh.setColorAt(sparkCount, tmpColor);
          sparkCount += 1;
        }
      }
    }

    bodyMesh.count = bodyCount;
    bodyMesh.instanceMatrix.needsUpdate = true;
    if (bodyMesh.instanceColor) bodyMesh.instanceColor.needsUpdate = true;

    arrowMesh.count = arrowCount;
    arrowMesh.instanceMatrix.needsUpdate = true;

    dotMesh.count = dotCount;
    dotMesh.instanceMatrix.needsUpdate = true;

    if (cutHalfMesh) {
      cutHalfMesh.count = halfCount;
      cutHalfMesh.instanceMatrix.needsUpdate = true;
      if (cutHalfMesh.instanceColor) cutHalfMesh.instanceColor.needsUpdate = true;
    }

    if (cutFaceMesh) {
      cutFaceMesh.count = faceCount;
      cutFaceMesh.instanceMatrix.needsUpdate = true;
      if (cutFaceMesh.instanceColor) cutFaceMesh.instanceColor.needsUpdate = true;
    }

    if (cutFlashMesh) {
      cutFlashMesh.count = flashCount;
      cutFlashMesh.instanceMatrix.needsUpdate = true;
      if (cutFlashMesh.instanceColor) cutFlashMesh.instanceColor.needsUpdate = true;
    }

    if (cutSparkMesh) {
      cutSparkMesh.count = sparkCount;
      cutSparkMesh.instanceMatrix.needsUpdate = true;
      if (cutSparkMesh.instanceColor) cutSparkMesh.instanceColor.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh
        ref={bodyMeshRef}
        args={[bodyGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshPhysicalMaterial
          roughness={0.18}
          metalness={0.35}
          clearcoat={0.85}
          clearcoatRoughness={0.12}
          envMapIntensity={1.3}
        />
      </instancedMesh>

      <instancedMesh
        ref={arrowMeshRef}
        args={[arrowGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshBasicMaterial color={[2.35, 2.35, 2.65]} toneMapped={false} />
      </instancedMesh>

      <instancedMesh
        ref={dotMeshRef}
        args={[dotGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshBasicMaterial color={[2.35, 2.35, 2.65]} toneMapped={false} />
      </instancedMesh>

      <instancedMesh
        ref={cutHalfMeshRef}
        args={[cutHalfGeo, undefined, MAX_CUT_HALVES]}
        frustumCulled={false}
      >
        <meshPhysicalMaterial
          roughness={0.2}
          metalness={0.3}
          clearcoat={0.8}
          clearcoatRoughness={0.14}
          envMapIntensity={1.2}
        />
      </instancedMesh>

      <instancedMesh
        ref={cutFaceMeshRef}
        args={[cutFaceGeo, undefined, MAX_CUT_HALVES]}
        frustumCulled={false}
      >
        <meshBasicMaterial color="#ffffff" toneMapped={false} side={THREE.DoubleSide} />
      </instancedMesh>

      <instancedMesh
        ref={cutFlashMeshRef}
        args={[cutFlashGeo, undefined, MAX_CUT_FLASHES]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          color="#ffffff"
          map={glowTex}
          transparent
          opacity={0.9}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
          side={THREE.DoubleSide}
        />
      </instancedMesh>

      <instancedMesh
        ref={cutSparkMeshRef}
        args={[cutSparkGeo, undefined, MAX_CUT_SPARKS]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.95}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </instancedMesh>
    </group>
  );
}
