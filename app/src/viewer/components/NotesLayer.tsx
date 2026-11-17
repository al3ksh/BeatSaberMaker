import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
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

const MAX_NOTE_INSTANCES = 512;
const LEFT_COLOR = new THREE.Color("#dc2626");
const RIGHT_COLOR = new THREE.Color("#2563eb");

function createBeveledCubeGeometry(size: number): THREE.BufferGeometry {
  const bevel = 0.035;
  const inner = size - bevel * 2;
  const half = inner / 2;
  const shape = new THREE.Shape();
  shape.moveTo(-half, -half);
  shape.lineTo(half, -half);
  shape.lineTo(half, half);
  shape.lineTo(-half, half);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: inner,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
  });
  geo.center();
  return geo;
}

function createArrowGeometry(): THREE.BufferGeometry {
  // Crisp chevron arrow pointing +Y, centered near origin
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.125);
  shape.lineTo(0.135, -0.025);
  shape.lineTo(0.085, -0.075);
  shape.lineTo(0, 0.015);
  shape.lineTo(-0.085, -0.075);
  shape.lineTo(-0.135, -0.025);
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

  const bodyGeo = useMemo(() => createBeveledCubeGeometry(NOTE_SIZE), []);
  const arrowGeo = useMemo(() => createArrowGeometry(), []);
  const dotGeo = useMemo(() => new THREE.CircleGeometry(0.068, 20), []);

  useEffect(() => {
    return () => {
      bodyGeo.dispose();
      arrowGeo.dispose();
      dotGeo.dispose();
    };
  }, [bodyGeo, arrowGeo, dotGeo]);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const arrowDummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(() => {
    const bodyMesh = bodyMeshRef.current;
    const arrowMesh = arrowMeshRef.current;
    const dotMesh = dotMeshRef.current;
    if (!bodyMesh || !arrowMesh || !dotMesh) return;

    const t = getTimeS();
    const visible = queryWindow(notes, t - 0.5, t + halfJumpDurationSec);

    let bodyCount = 0;
    let arrowCount = 0;
    let dotCount = 0;

    for (let i = 0; i < visible.length && bodyCount < MAX_NOTE_INSTANCES; i += 1) {
      const note = visible[i];
      const wx = gridX(note.x);
      const wy = gridY(note.y);
      const wz = objectZAtTime(note.timeS, t, njs);
      const rotZ = cutDirectionAngleRad(note.direction, note.angleOffset);

      dummy.position.set(wx, wy, wz);
      dummy.rotation.set(0, 0, rotZ);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();

      bodyMesh.setMatrixAt(bodyCount, dummy.matrix);
      bodyMesh.setColorAt(bodyCount, note.color === 0 ? LEFT_COLOR : RIGHT_COLOR);
      bodyCount += 1;

      const faceOffset = NOTE_SIZE / 2 + 0.004;
      if (isDotDirection(note.direction)) {
        arrowDummy.position.set(wx, wy, wz + faceOffset);
        arrowDummy.rotation.set(0, 0, 0);
        arrowDummy.scale.setScalar(1);
        arrowDummy.updateMatrix();
        dotMesh.setMatrixAt(dotCount, arrowDummy.matrix);
        dotCount += 1;
      } else {
        arrowDummy.position.set(wx, wy, wz + faceOffset);
        arrowDummy.rotation.set(0, 0, rotZ);
        arrowDummy.scale.setScalar(1);
        arrowDummy.updateMatrix();
        arrowMesh.setMatrixAt(arrowCount, arrowDummy.matrix);
        arrowCount += 1;
      }
    }

    bodyMesh.count = bodyCount;
    bodyMesh.instanceMatrix.needsUpdate = true;
    if (bodyMesh.instanceColor) {
      bodyMesh.instanceColor.needsUpdate = true;
    }

    arrowMesh.count = arrowCount;
    arrowMesh.instanceMatrix.needsUpdate = true;

    dotMesh.count = dotCount;
    dotMesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <group>
      <instancedMesh
        ref={bodyMeshRef}
        args={[bodyGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshStandardMaterial roughness={0.28} metalness={0.25} />
      </instancedMesh>

      <instancedMesh
        ref={arrowMeshRef}
        args={[arrowGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshBasicMaterial color="#ffffff" />
      </instancedMesh>

      <instancedMesh
        ref={dotMeshRef}
        args={[dotGeo, undefined, MAX_NOTE_INSTANCES]}
        frustumCulled={false}
      >
        <meshBasicMaterial color="#ffffff" />
      </instancedMesh>
    </group>
  );
}
