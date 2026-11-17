import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import {
  computeArcControlPoints,
  evaluateCubicBezier,
  generateChainLinks,
} from "../curves";
import { querySpanWindow, queryWindow } from "../parser";
import { objectZAtTime } from "../timing";
import type { ViewerArc, ViewerChain } from "../types";
import {
  computeSpawnFade,
  cutEffectProgress,
  findArcVisibleUStart,
  isArcVisible,
  isHitObjectVisible,
} from "../visibility";
import { createSoftGlowTexture } from "../textures";

const ARC_SAMPLES = 30;
const MAX_ARC_SEGMENTS = 768;
const MAX_ARC_FLARES = 64;
const MAX_CHAIN_LINKS = 512;
const MAX_CHAIN_SPARKS = 256;
const CHAIN_SPARK_DURATION_S = 0.14;

const LEFT_BODY_COLOR = new THREE.Color("#dc142c");
const RIGHT_BODY_COLOR = new THREE.Color("#1864f5");
const LEFT_ARC_CORE = new THREE.Color(2.6, 0.52, 0.72);
const RIGHT_ARC_CORE = new THREE.Color(0.45, 1.45, 2.8);
const LEFT_ARC_AURA = new THREE.Color(1.9, 0.18, 0.34);
const RIGHT_ARC_AURA = new THREE.Color(0.18, 0.85, 2.1);
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
  const arcCoreMeshRef = useRef<THREE.InstancedMesh>(null);
  const arcAuraMeshRef = useRef<THREE.InstancedMesh>(null);
  const arcFlareMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainDotMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainSparkMeshRef = useRef<THREE.InstancedMesh>(null);

  const arcCoreGeo = useMemo(() => new THREE.CylinderGeometry(0.026, 0.026, 1, 10), []);
  const arcAuraGeo = useMemo(() => new THREE.CylinderGeometry(0.072, 0.072, 1, 10), []);
  const arcFlareGeo = useMemo(() => new THREE.SphereGeometry(0.068, 12, 12), []);
  const chainGeo = useMemo(() => new RoundedBoxGeometry(0.36, 0.11, 0.36, 2, 0.024), []);
  const chainDotGeo = useMemo(() => new THREE.CircleGeometry(0.036, 16), []);
  const chainSparkGeo = useMemo(() => new THREE.PlaneGeometry(0.36, 0.36), []);
  const sparkTex = useMemo(() => createSoftGlowTexture(), []);

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
  const tmpColor = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    const dummyCol = new THREE.Color(1, 1, 1);
    for (const ref of [
      arcCoreMeshRef,
      arcAuraMeshRef,
      arcFlareMeshRef,
      chainMeshRef,
      chainSparkMeshRef,
    ]) {
      const m = ref.current;
      if (m && !m.instanceColor) {
        m.setColorAt(0, dummyCol);
        m.count = 0;
      }
    }
    return () => {
      arcCoreGeo.dispose();
      arcAuraGeo.dispose();
      arcFlareGeo.dispose();
      chainGeo.dispose();
      chainDotGeo.dispose();
      chainSparkGeo.dispose();
      sparkTex.dispose();
    };
  }, [arcCoreGeo, arcAuraGeo, arcFlareGeo, chainGeo, chainDotGeo, chainSparkGeo, sparkTex]);

  useFrame(() => {
    const t = getTimeS();
    const halfJumpDist = Math.max(1, njs * halfJumpDurationSec);

    const arcCoreMesh = arcCoreMeshRef.current;
    const arcAuraMesh = arcAuraMeshRef.current;
    const arcFlareMesh = arcFlareMeshRef.current;
    if (arcCoreMesh && arcAuraMesh) {
      const visibleArcs = querySpanWindow(
        arcs,
        (a) => a.tailTimeS,
        t + 1e-6,
        t + halfJumpDurationSec,
      );
      let segCount = 0;
      let flareCount = 0;

      for (let aIdx = 0; aIdx < visibleArcs.length; aIdx += 1) {
        const arc = visibleArcs[aIdx];
        if (!isArcVisible(arc.timeS, arc.tailTimeS, t, halfJumpDurationSec)) continue;

        const { p0, p1, p2, p3 } = computeArcControlPoints(arc, t, njs);
        const uStart = findArcVisibleUStart(p0.z, p1.z, p2.z, p3.z);
        if (uStart === null) continue;

        const coreColor = arc.color === 0 ? LEFT_ARC_CORE : RIGHT_ARC_CORE;
        const auraColor = arc.color === 0 ? LEFT_ARC_AURA : RIGHT_ARC_AURA;

        if (uStart > 0 && arcFlareMesh && flareCount < MAX_ARC_FLARES) {
          const hitPt = evaluateCubicBezier(p0, p1, p2, p3, uStart);
          dummy.position.set(hitPt.x, hitPt.y, -0.01);
          dummy.rotation.set(0, 0, 0);
          dummy.scale.setScalar(1);
          dummy.updateMatrix();
          arcFlareMesh.setMatrixAt(flareCount, dummy.matrix);
          arcFlareMesh.setColorAt(flareCount, coreColor);
          flareCount += 1;
        }

        const spanU = 1 - uStart;
        if (spanU <= 1e-4) continue;

        let prev = evaluateCubicBezier(p0, p1, p2, p3, uStart);
        for (let s = 1; s <= ARC_SAMPLES && segCount < MAX_ARC_SEGMENTS; s += 1) {
          const u = uStart + (s / ARC_SAMPLES) * spanU;
          const next = evaluateCubicBezier(p0, p1, p2, p3, u);
          pA.set(prev.x, prev.y, prev.z);
          pB.set(next.x, next.y, next.z);
          dirVec.subVectors(pB, pA);
          const len = dirVec.length();
          if (len > 1e-4) {
            midVec.addVectors(pA, pB).multiplyScalar(0.5);
            // Smooth spawn fade for tail entering at -halfJumpDist
            const zDistFrac = Math.max(0, Math.min(1, (midVec.z + halfJumpDist) / (halfJumpDist * 0.18)));
            const radiusScale = 0.12 + 0.88 * (zDistFrac * zDistFrac * (3 - 2 * zDistFrac));
            quat.setFromUnitVectors(UP_AXIS, dirVec.divideScalar(len));

            dummy.position.copy(midVec);
            dummy.quaternion.copy(quat);
            dummy.scale.set(radiusScale, len * 1.07, radiusScale);
            dummy.updateMatrix();
            arcCoreMesh.setMatrixAt(segCount, dummy.matrix);
            arcCoreMesh.setColorAt(segCount, coreColor);

            // Flattened glowing outer ribbon aura
            dummy.scale.set(radiusScale * 1.4, len * 1.07, radiusScale * 0.65);
            dummy.updateMatrix();
            arcAuraMesh.setMatrixAt(segCount, dummy.matrix);
            arcAuraMesh.setColorAt(segCount, auraColor);
            segCount += 1;
          }
          prev = next;
        }
      }

      arcCoreMesh.count = segCount;
      arcCoreMesh.instanceMatrix.needsUpdate = true;
      if (arcCoreMesh.instanceColor) arcCoreMesh.instanceColor.needsUpdate = true;

      arcAuraMesh.count = segCount;
      arcAuraMesh.instanceMatrix.needsUpdate = true;
      if (arcAuraMesh.instanceColor) arcAuraMesh.instanceColor.needsUpdate = true;

      if (arcFlareMesh) {
        arcFlareMesh.count = flareCount;
        arcFlareMesh.instanceMatrix.needsUpdate = true;
        if (arcFlareMesh.instanceColor) arcFlareMesh.instanceColor.needsUpdate = true;
      }
    }

    const chainMesh = chainMeshRef.current;
    const chainDotMesh = chainDotMeshRef.current;
    const chainSparkMesh = chainSparkMeshRef.current;
    if (chainMesh && chainDotMesh) {
      const visibleLinks = queryWindow(
        chainLinks,
        t - CHAIN_SPARK_DURATION_S,
        t + halfJumpDurationSec,
      );
      let count = 0;
      let sparkCount = 0;

      for (let i = 0; i < visibleLinks.length; i += 1) {
        const link = visibleLinks[i];
        if (isHitObjectVisible(link.timeS, t, halfJumpDurationSec)) {
          if (count >= MAX_CHAIN_LINKS) continue;
          const wz = objectZAtTime(link.timeS, t, njs);
          const fade = computeSpawnFade(t, link.timeS, halfJumpDurationSec, 0.18);
          const scale = 0.05 + 0.95 * fade;

          dummy.position.set(link.worldX, link.worldY, wz);
          dummy.rotation.set(0, 0, link.rotZ);
          dummy.scale.setScalar(scale);
          dummy.updateMatrix();
          chainMesh.setMatrixAt(count, dummy.matrix);
          chainMesh.setColorAt(count, link.color === 0 ? LEFT_BODY_COLOR : RIGHT_BODY_COLOR);

          dummy.position.set(link.worldX, link.worldY, wz + 0.182 * scale);
          dummy.rotation.set(0, 0, 0);
          dummy.scale.setScalar(scale);
          dummy.updateMatrix();
          chainDotMesh.setMatrixAt(count, dummy.matrix);
          count += 1;
        } else if (chainSparkMesh && sparkCount < MAX_CHAIN_SPARKS) {
          const p = cutEffectProgress(link.timeS, t, CHAIN_SPARK_DURATION_S);
          if (p !== null) {
            const ease = 1 - (1 - p) * (1 - p);
            const fadeOut = (1 - p) * (1 - p);
            const base = link.color === 0 ? LEFT_ARC_CORE : RIGHT_ARC_CORE;
            dummy.position.set(link.worldX, link.worldY, -0.02 - ease * 0.08);
            dummy.rotation.set(0, 0, link.rotZ);
            dummy.scale.setScalar(0.6 + ease * 1.1);
            dummy.updateMatrix();
            chainSparkMesh.setMatrixAt(sparkCount, dummy.matrix);
            tmpColor.copy(base).multiplyScalar(fadeOut);
            chainSparkMesh.setColorAt(sparkCount, tmpColor);
            sparkCount += 1;
          }
        }
      }

      chainMesh.count = count;
      chainMesh.instanceMatrix.needsUpdate = true;
      if (chainMesh.instanceColor) chainMesh.instanceColor.needsUpdate = true;

      chainDotMesh.count = count;
      chainDotMesh.instanceMatrix.needsUpdate = true;

      if (chainSparkMesh) {
        chainSparkMesh.count = sparkCount;
        chainSparkMesh.instanceMatrix.needsUpdate = true;
        if (chainSparkMesh.instanceColor) chainSparkMesh.instanceColor.needsUpdate = true;
      }
    }
  });

  return (
    <group>
      <instancedMesh
        ref={arcCoreMeshRef}
        args={[arcCoreGeo, undefined, MAX_ARC_SEGMENTS]}
        frustumCulled={false}
        renderOrder={2}
      >
        <meshBasicMaterial color="#ffffff" toneMapped={false} />
      </instancedMesh>

      <instancedMesh
        ref={arcAuraMeshRef}
        args={[arcAuraGeo, undefined, MAX_ARC_SEGMENTS]}
        frustumCulled={false}
        renderOrder={1}
      >
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.32}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </instancedMesh>

      <instancedMesh
        ref={arcFlareMeshRef}
        args={[arcFlareGeo, undefined, MAX_ARC_FLARES]}
        frustumCulled={false}
        renderOrder={2}
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

      <instancedMesh
        ref={chainMeshRef}
        args={[chainGeo, undefined, MAX_CHAIN_LINKS]}
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
        ref={chainDotMeshRef}
        args={[chainDotGeo, undefined, MAX_CHAIN_LINKS]}
        frustumCulled={false}
      >
        <meshBasicMaterial color={[2.35, 2.35, 2.65]} toneMapped={false} />
      </instancedMesh>

      <instancedMesh
        ref={chainSparkMeshRef}
        args={[chainSparkGeo, undefined, MAX_CHAIN_SPARKS]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          color="#ffffff"
          map={sparkTex}
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
          side={THREE.DoubleSide}
        />
      </instancedMesh>
    </group>
  );
}
