import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { BOMB_RADIUS, gridX, gridY, obstacleBox } from "../geometry";
import { querySpanWindow, queryWindow } from "../parser";
import { objectZAtTime } from "../timing";
import type { ViewerBomb, ViewerObstacle } from "../types";

const MAX_BOMBS = 256;
const MAX_WALLS = 128;

export interface WallsAndBombsLayerProps {
  bombs: readonly ViewerBomb[];
  obstacles: readonly ViewerObstacle[];
  njs: number;
  halfJumpDurationSec: number;
  getTimeS: () => number;
}

export function WallsAndBombsLayer({
  bombs,
  obstacles,
  njs,
  halfJumpDurationSec,
  getTimeS,
}: WallsAndBombsLayerProps) {
  const bombMeshRef = useRef<THREE.InstancedMesh>(null);
  const wallMeshRef = useRef<THREE.InstancedMesh>(null);

  const bombGeo = useMemo(() => new THREE.IcosahedronGeometry(BOMB_RADIUS, 1), []);
  const wallGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    return () => {
      bombGeo.dispose();
      wallGeo.dispose();
    };
  }, [bombGeo, wallGeo]);

  useFrame(() => {
    const t = getTimeS();

    const bombMesh = bombMeshRef.current;
    if (bombMesh) {
      const visibleBombs = queryWindow(bombs, t - 0.5, t + halfJumpDurationSec);
      let count = 0;
      for (let i = 0; i < visibleBombs.length && count < MAX_BOMBS; i += 1) {
        const b = visibleBombs[i];
        dummy.position.set(gridX(b.x), gridY(b.y), objectZAtTime(b.timeS, t, njs));
        dummy.rotation.set(t * 1.4, t * 2.1, 0);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        bombMesh.setMatrixAt(count, dummy.matrix);
        count += 1;
      }
      bombMesh.count = count;
      bombMesh.instanceMatrix.needsUpdate = true;
    }

    const wallMesh = wallMeshRef.current;
    if (wallMesh) {
      const visibleWalls = querySpanWindow(
        obstacles,
        (o) => o.endTimeS,
        t - 0.5,
        t + halfJumpDurationSec,
      );
      let count = 0;
      for (let i = 0; i < visibleWalls.length && count < MAX_WALLS; i += 1) {
        const w = visibleWalls[i];
        const box = obstacleBox(w.x, w.y, w.w, w.h);
        const zFront = objectZAtTime(w.timeS, t, njs);
        const zBack = objectZAtTime(w.endTimeS, t, njs);
        const lenZ = Math.max(0.06, zFront - zBack);
        const centerZ = (zFront + zBack) / 2;

        dummy.position.set(box.centerX, box.centerY, centerZ);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(box.width, box.height, lenZ);
        dummy.updateMatrix();
        wallMesh.setMatrixAt(count, dummy.matrix);
        count += 1;
      }
      wallMesh.count = count;
      wallMesh.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh
        ref={bombMeshRef}
        args={[bombGeo, undefined, MAX_BOMBS]}
        frustumCulled={false}
      >
        <meshStandardMaterial
          color="#18181b"
          emissive="#991b1b"
          emissiveIntensity={0.22}
          roughness={0.35}
          metalness={0.75}
          flatShading
        />
      </instancedMesh>

      <instancedMesh
        ref={wallMeshRef}
        args={[wallGeo, undefined, MAX_WALLS]}
        frustumCulled={false}
        renderOrder={2}
      >
        <meshStandardMaterial
          color="#ef4444"
          emissive="#dc2626"
          emissiveIntensity={0.45}
          transparent
          opacity={0.34}
          roughness={0.2}
          depthWrite={false}
        />
      </instancedMesh>
    </group>
  );
}
