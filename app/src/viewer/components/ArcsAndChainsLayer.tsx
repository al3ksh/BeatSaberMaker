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
import { createRibbonProfileTexture, createSoftGlowTexture } from "../textures";

const ARC_SAMPLES = 48;
const MAX_ARCS = 32;
const ARC_VERTS_PER_ARC = (ARC_SAMPLES + 1) * 2;
const ARC_HALF_WIDTH = 0.085;
const MAX_ARC_FLARES = 64;
const MAX_CHAIN_LINKS = 512;
const MAX_CHAIN_SPARKS = 256;
const CHAIN_SPARK_DURATION_S = 0.14;

const LEFT_BODY_COLOR = new THREE.Color("#dc142c");
const RIGHT_BODY_COLOR = new THREE.Color("#1864f5");
const LEFT_ARC_CORE = new THREE.Color(2.6, 0.52, 0.72);
const RIGHT_ARC_CORE = new THREE.Color(0.45, 1.45, 2.8);

/** Static index/uv layout for MAX_ARCS ribbons of ARC_SAMPLES segments; positions/colors are rewritten per frame. */
function createArcRibbonGeometry(): THREE.BufferGeometry {
  const vertCount = MAX_ARCS * ARC_VERTS_PER_ARC;
  const geo = new THREE.BufferGeometry();
  const pos = new THREE.BufferAttribute(new Float32Array(vertCount * 3), 3);
  pos.setUsage(THREE.DynamicDrawUsage);
  const col = new THREE.BufferAttribute(new Float32Array(vertCount * 3), 3);
  col.setUsage(THREE.DynamicDrawUsage);
  const uv = new Float32Array(vertCount * 2);
  const index: number[] = [];
  for (let a = 0; a < MAX_ARCS; a += 1) {
    const base = a * ARC_VERTS_PER_ARC;
    for (let k = 0; k <= ARC_SAMPLES; k += 1) {
      const v = base + k * 2;
      uv[v * 2] = 0;
      uv[v * 2 + 1] = k / ARC_SAMPLES;
      uv[(v + 1) * 2] = 1;
      uv[(v + 1) * 2 + 1] = k / ARC_SAMPLES;
      if (k < ARC_SAMPLES) {
        index.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);
      }
    }
  }
  geo.setAttribute("position", pos);
  geo.setAttribute("color", col);
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(index);
  geo.setDrawRange(0, 0);
  return geo;
}

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
  const arcFlareMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainDotMeshRef = useRef<THREE.InstancedMesh>(null);
  const chainSparkMeshRef = useRef<THREE.InstancedMesh>(null);

  const arcRibbonGeo = useMemo(() => createArcRibbonGeometry(), []);
  const ribbonTex = useMemo(() => createRibbonProfileTexture(), []);
  const arcFlareGeo = useMemo(() => new THREE.PlaneGeometry(0.42, 0.42), []);
  const chainGeo = useMemo(() => new RoundedBoxGeometry(0.36, 0.11, 0.36, 2, 0.024), []);
  const chainDotGeo = useMemo(() => new THREE.CircleGeometry(0.036, 16), []);
  const chainSparkGeo = useMemo(() => new THREE.PlaneGeometry(0.36, 0.36), []);
  const sparkTex = useMemo(() => createSoftGlowTexture(), []);

  const chainLinks = useMemo(
    () => chains.flatMap(generateChainLinks).sort((a, b) => a.timeS - b.timeS),
    [chains],
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tangent = useMemo(() => new THREE.Vector3(), []);
  const toCam = useMemo(() => new THREE.Vector3(), []);
  const sideVec = useMemo(() => new THREE.Vector3(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);
  const samplePts = useMemo(
    () => Array.from({ length: ARC_SAMPLES + 1 }, () => new THREE.Vector3()),
    [],
  );

  useEffect(() => {
    const dummyCol = new THREE.Color(1, 1, 1);
    for (const ref of [
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
      arcRibbonGeo.dispose();
      ribbonTex.dispose();
      arcFlareGeo.dispose();
      chainGeo.dispose();
      chainDotGeo.dispose();
      chainSparkGeo.dispose();
      sparkTex.dispose();
    };
  }, [arcRibbonGeo, ribbonTex, arcFlareGeo, chainGeo, chainDotGeo, chainSparkGeo, sparkTex]);

  useFrame((state) => {
    const t = getTimeS();
    const halfJumpDist = Math.max(1, njs * halfJumpDurationSec);

    const arcFlareMesh = arcFlareMeshRef.current;
    {
      const camPos = state.camera.position;
      const posAttr = arcRibbonGeo.getAttribute("position") as THREE.BufferAttribute;
      const colAttr = arcRibbonGeo.getAttribute("color") as THREE.BufferAttribute;
      const posArr = posAttr.array as Float32Array;
      const colArr = colAttr.array as Float32Array;
      const visibleArcs = querySpanWindow(
        arcs,
        (a) => a.tailTimeS,
        t + 1e-6,
        t + halfJumpDurationSec,
      );
      let arcCount = 0;
      let flareCount = 0;

      for (let aIdx = 0; aIdx < visibleArcs.length && arcCount < MAX_ARCS; aIdx += 1) {
        const arc = visibleArcs[aIdx];
        if (!isArcVisible(arc.timeS, arc.tailTimeS, t, halfJumpDurationSec)) continue;

        const { p0, p1, p2, p3 } = computeArcControlPoints(arc, t, njs);
        const uStart = findArcVisibleUStart(p0.z, p1.z, p2.z, p3.z);
        if (uStart === null) continue;

        const coreColor = arc.color === 0 ? LEFT_ARC_CORE : RIGHT_ARC_CORE;

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

        for (let k = 0; k <= ARC_SAMPLES; k += 1) {
          const pt = evaluateCubicBezier(p0, p1, p2, p3, uStart + (k / ARC_SAMPLES) * spanU);
          samplePts[k].set(pt.x, pt.y, pt.z);
        }

        const base = arcCount * ARC_VERTS_PER_ARC;
        for (let k = 0; k <= ARC_SAMPLES; k += 1) {
          const pt = samplePts[k];
          tangent.subVectors(samplePts[Math.min(ARC_SAMPLES, k + 1)], samplePts[Math.max(0, k - 1)]);
          toCam.subVectors(camPos, pt);
          sideVec.crossVectors(tangent, toCam);
          const sideLen = sideVec.length();
          if (sideLen > 1e-6) sideVec.divideScalar(sideLen);
          else sideVec.set(1, 0, 0);

          // Smooth spawn fade for the tail entering at -halfJumpDist, plus a soft taper at the far end
          const zDistFrac = Math.max(0, Math.min(1, (pt.z + halfJumpDist) / (halfJumpDist * 0.18)));
          const spawn = zDistFrac * zDistFrac * (3 - 2 * zDistFrac);
          const u = k / ARC_SAMPLES;
          const tailTaper = u > 0.9 ? 0.45 + 0.55 * ((1 - u) / 0.1) : 1;
          const halfW = ARC_HALF_WIDTH * (0.25 + 0.75 * spawn) * tailTaper;
          const bright = 0.12 + 0.88 * spawn;

          const v = base + k * 2;
          posArr[v * 3] = pt.x - sideVec.x * halfW;
          posArr[v * 3 + 1] = pt.y - sideVec.y * halfW;
          posArr[v * 3 + 2] = pt.z - sideVec.z * halfW;
          posArr[v * 3 + 3] = pt.x + sideVec.x * halfW;
          posArr[v * 3 + 4] = pt.y + sideVec.y * halfW;
          posArr[v * 3 + 5] = pt.z + sideVec.z * halfW;
          const r = coreColor.r * bright;
          const g = coreColor.g * bright;
          const bl = coreColor.b * bright;
          colArr[v * 3] = r;
          colArr[v * 3 + 1] = g;
          colArr[v * 3 + 2] = bl;
          colArr[v * 3 + 3] = r;
          colArr[v * 3 + 4] = g;
          colArr[v * 3 + 5] = bl;
        }
        arcCount += 1;
      }

      arcRibbonGeo.setDrawRange(0, arcCount * ARC_SAMPLES * 6);
      posAttr.needsUpdate = true;
      colAttr.needsUpdate = true;

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
      <mesh geometry={arcRibbonGeo} frustumCulled={false} renderOrder={2}>
        <meshBasicMaterial
          map={ribbonTex}
          vertexColors
          transparent
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
          side={THREE.DoubleSide}
        />
      </mesh>

      <instancedMesh
        ref={arcFlareMeshRef}
        args={[arcFlareGeo, undefined, MAX_ARC_FLARES]}
        frustumCulled={false}
        renderOrder={2}
      >
        <meshBasicMaterial
          color="#ffffff"
          map={sparkTex}
          transparent
          opacity={0.95}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
          side={THREE.DoubleSide}
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
