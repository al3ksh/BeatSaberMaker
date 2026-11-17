import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  computeArcControlPoints,
  evaluateCubicBezier,
  generateChainLinks,
} from "../curves";
import { querySpanWindow, queryWindow } from "../parser";
import { objectZAtTime } from "../timing";
import type { ViewerArc, ViewerChain } from "../types";

const ARC_SAMPLES = 24;
const MAX_ARC_SEGMENTS = 512;
const MAX_CHAIN_LINKS = 512;

const LEFT_COLOR = new THREE.Color("#f43f5e");
const RIGHT_COLOR = new THREE.Color("#3b82f6");
const UP_AXIS = new THREE.Vector3(0, 1, 0);

export interface ArcsAndChainsLayerProps {
  arcs: readonly ViewerArc[];
  chains: readonly ViewerChain[];
  njs: number;
  halfJumpDurationSec: number;
  getTimeS: () => number;
}

export function ArcsAndChainsLayer({
  arcs,
  chains,
  njs,
  halfJumpDurationSec,
  getTimeS,
}: ArcsAndChainsLayerProps) {
  const arcMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainDotMeshRef = useRef<THREE.InstancedMesh>(null);

  const arcGeo = useMemo(() => new THREE.CylinderGeometry(0.038, 0.038, 1, 8), []);
  const chainGeo = useMemo(() => new THREE.BoxGeometry(0.38, 0.12, 0.34), []);
  const chainDotGeo = useMemo(() => new THREE.CircleGeometry(0.036, 12), []);

  const chainLinks = useMemo(
    () => chains.flatMap(generateChainLinks).sort((a, b) => a.timeS - b.timeS),
    [chains],
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const pA = useMemo(() => new THREE.Vector3(), []);
  const pB = useMemo(() => new THREE.Vector3(), []);
  const dirVec = useMemo(() => new THREE.Vector3(), []);
  const midVec = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => {
    return () => {
      arcGeo.dispose();
      chainGeo.dispose();
      chainDotGeo.dispose();
    };
  }, [arcGeo, chainGeo, chainDotGeo]);

  useFrame(() => {
    const t = getTimeS();

    const arcMesh = arcMeshRef.current;
    if (arcMesh) {
      const visibleArcs = querySpanWindow(
        arcs,
        (a) => a.tailTimeS,
        t - 0.5,
        t + halfJumpDurationSec,
      );
      let segCount = 0;
      for (let aIdx = 0; aIdx < visibleArcs.length; aIdx += 1) {
        const arc = visibleArcs[aIdx];
        const { p0, p1, p2, p3 } = computeArcControlPoints(arc, t, njs);
        const color = arc.color === 0 ? LEFT_COLOR : RIGHT_COLOR;

        let prev = evaluateCubicBezier(p0, p1, p2, p3, 0);
        for (let s = 1; s <= ARC_SAMPLES && segCount < MAX_ARC_SEGMENTS; s += 1) {
          const next = evaluateCubicBezier(p0, p1, p2, p3, s / ARC_SAMPLES);
          pA.set(prev.x, prev.y, prev.z);
          pB.set(next.x, next.y, next.z);
          dirVec.subVectors(pB, pA);
          const len = dirVec.length();
          if (len > 1e-4) {
            midVec.addVectors(pA, pB).multiplyScalar(0.5);
            quat.setFromUnitVectors(UP_AXIS, dirVec.divideScalar(len));
            dummy.position.copy(midVec);
            dummy.quaternion.copy(quat);
            dummy.scale.set(1, len * 1.04, 1);
            dummy.updateMatrix();
            arcMesh.setMatrixAt(segCount, dummy.matrix);
            arcMesh.setColorAt(segCount, color);
            segCount += 1;
          }
          prev = next;
        }
      }
      arcMesh.count = segCount;
      arcMesh.instanceMatrix.needsUpdate = true;
      if (arcMesh.instanceColor) {
        arcMesh.instanceColor.needsUpdate = true;
      }
    }

    const chainMesh = chainMeshRef.current;
    const chainDotMesh = chainDotMeshRef.current;
    if (chainMesh && chainDotMesh) {
      const visibleLinks = queryWindow(chainLinks, t - 0.5, t + halfJumpDurationSec);
      let count = 0;
      for (let i = 0; i < visibleLinks.length && count < MAX_CHAIN_LINKS; i += 1) {
        const link = visibleLinks[i];
        const wz = objectZAtTime(link.timeS, t, njs);
        dummy.position.set(link.worldX, link.worldY, wz);
        dummy.rotation.set(0, 0, link.rotZ);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        chainMesh.setMatrixAt(count, dummy.matrix);
        chainMesh.setColorAt(count, link.color === 0 ? LEFT_COLOR : RIGHT_COLOR);

        dummy.position.set(link.worldX, link.worldY, wz + 0.175);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        chainDotMesh.setMatrixAt(count, dummy.matrix);
        count += 1;
      }
      chainMesh.count = count;
      chainMesh.instanceMatrix.needsUpdate = true;
      if (chainMesh.instanceColor) {
        chainMesh.instanceColor.needsUpdate = true;
      }
      chainDotMesh.count = count;
      chainDotMesh.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh
        ref={arcMeshRef}
        args={[arcGeo, undefined, MAX_ARC_SEGMENTS]}
        frustumCulled={false}
        renderOrder={1}
      >
        <meshBasicMaterial transparent opacity={0.82} />
      </instancedMesh>

      <instancedMesh
        ref={chainMeshRef}
        args={[chainGeo, undefined, MAX_CHAIN_LINKS]}
        frustumCulled={false}
      >
        <meshStandardMaterial roughness={0.28} metalness={0.25} />
      </instancedMesh>

      <instancedMesh
        ref={chainDotMeshRef}
        args={[chainDotGeo, undefined, MAX_CHAIN_LINKS]}
        frustumCulled={false}
      >
        <meshBasicMaterial color="#ffffff" />
      </instancedMesh>
    </group>
  );
}
