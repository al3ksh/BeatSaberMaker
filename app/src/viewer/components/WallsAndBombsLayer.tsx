import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { BOMB_RADIUS, gridX, gridY, obstacleBox } from "../geometry";
import { querySpanWindow, queryWindow } from "../parser";
import { objectZAtTime } from "../timing";
import type { ViewerBomb, ViewerObstacle } from "../types";
import {
  computeSpawnFade,
  computeWallZSpan,
  cutEffectProgress,
  isHitObjectVisible,
  isWallVisible,
} from "../visibility";
import { createSoftGlowTexture } from "../textures";

const MAX_BOMBS = 256;
const MAX_BOMB_PUFFS = 64;
const MAX_WALLS = 128;
const BOMB_PUFF_DURATION_S = 0.16;

function createSpikyBombGeometry(radius: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const core = new THREE.IcosahedronGeometry(radius * 0.72, 1).toNonIndexed();
  core.deleteAttribute("uv");
  parts.push(core);

  // 12 evenly distributed icosahedral spike directions
  const phi = (1 + Math.sqrt(5)) / 2;
  const rawDirs: [number, number, number][] = [
    [-1, phi, 0],
    [1, phi, 0],
    [-1, -phi, 0],
    [1, -phi, 0],
    [0, -1, phi],
    [0, 1, phi],
    [0, -1, -phi],
    [0, 1, -phi],
    [phi, 0, -1],
    [phi, 0, 1],
    [-phi, 0, -1],
    [-phi, 0, 1],
  ];
  const up = new THREE.Vector3(0, 1, 0);
  const dir = new THREE.Vector3();
  const quat = new THREE.Quaternion();

  for (const [dx, dy, dz] of rawDirs) {
    dir.set(dx, dy, dz).normalize();
    quat.setFromUnitVectors(up, dir);
    const spike = new THREE.ConeGeometry(radius * 0.21, radius * 0.56, 5).toNonIndexed();
    spike.deleteAttribute("uv");
    spike.applyQuaternion(quat);
    spike.translate(dir.x * radius * 0.72, dir.y * radius * 0.72, dir.z * radius * 0.72);
    parts.push(spike);
  }

  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  return merged ?? new THREE.IcosahedronGeometry(radius, 1);
}

const WALL_VERTEX_SHADER = /* glsl */ `
  attribute float aFade;
  varying vec3 vBoxPos;
  varying vec3 vBoxSize;
  varying float vFade;
  varying vec3 vNormalView;
  varying vec3 vViewPos;

  void main() {
    vFade = aFade;
    vBoxSize = vec3(
      length(instanceMatrix[0].xyz),
      length(instanceMatrix[1].xyz),
      length(instanceMatrix[2].xyz)
    );
    vBoxPos = position * vBoxSize;
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vViewPos = -mvPosition.xyz;
    vNormalView = normalize(normalMatrix * mat3(instanceMatrix) * normal);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const WALL_FRAGMENT_SHADER = /* glsl */ `
  uniform float uTime;
  varying vec3 vBoxPos;
  varying vec3 vBoxSize;
  varying float vFade;
  varying vec3 vNormalView;
  varying vec3 vViewPos;

  void main() {
    vec3 halfSize = vBoxSize * 0.5;
    vec3 d = max(vec3(0.0), halfSize - abs(vBoxPos));
    float minD = min(d.x, min(d.y, d.z));
    float maxD = max(d.x, max(d.y, d.z));
    float midD = d.x + d.y + d.z - minD - maxD;

    // Uniform world-space glowing edge border regardless of wall scale
    float outerGlow = 1.0 - smoothstep(0.014, 0.052, midD);
    float innerCore = 1.0 - smoothstep(0.003, 0.016, midD);

    vec3 viewDir = normalize(vViewPos);
    float fresnel = pow(1.0 - abs(dot(normalize(vNormalView), viewDir)), 1.5);
    float shimmer = 0.5 + 0.5 * sin((vBoxPos.z * 2.5 + vBoxPos.y * 2.1) - uTime * 3.2);

    vec3 fillCol = vec3(0.95, 0.09, 0.19);
    vec3 edgeRed = vec3(2.2, 0.25, 0.42);
    vec3 edgeWhite = vec3(2.8, 2.35, 2.5);

    float fillAlpha = 0.09 + 0.09 * fresnel + 0.025 * shimmer;
    vec3 col = mix(fillCol, edgeRed, outerGlow);
    col = mix(col, edgeWhite, innerCore * 0.85);

    float dist = length(vViewPos);
    float fogFade = 1.0 - smoothstep(20.0, 68.0, dist) * 0.9;
    float alpha = clamp(max(fillAlpha, outerGlow * 0.92), 0.0, 0.96) * vFade * fogFade;

    gl_FragColor = vec4(col, alpha);
  }
`;

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
  const bombPuffMeshRef = useRef<THREE.InstancedMesh>(null);
  const wallMeshRef = useRef<THREE.InstancedMesh>(null);

  const bombGeo = useMemo(() => createSpikyBombGeometry(BOMB_RADIUS), []);
  const bombPuffGeo = useMemo(() => new THREE.PlaneGeometry(0.55, 0.55), []);
  const puffTex = useMemo(() => createSoftGlowTexture(), []);
  const wallGeo = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1);
    const fades = new Float32Array(MAX_WALLS).fill(1);
    g.setAttribute("aFade", new THREE.InstancedBufferAttribute(fades, 1));
    return g;
  }, []);

  const wallMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 } },
        vertexShader: WALL_VERTEX_SHADER,
        fragmentShader: WALL_FRAGMENT_SHADER,
        transparent: true,
        depthWrite: false,
        toneMapped: false,
        side: THREE.DoubleSide,
      }),
    [],
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    const puffMesh = bombPuffMeshRef.current;
    if (puffMesh && !puffMesh.instanceColor) {
      puffMesh.setColorAt(0, new THREE.Color(1, 1, 1));
      puffMesh.count = 0;
    }
    return () => {
      bombGeo.dispose();
      bombPuffGeo.dispose();
      puffTex.dispose();
      wallGeo.dispose();
      wallMaterial.dispose();
    };
  }, [bombGeo, bombPuffGeo, puffTex, wallGeo, wallMaterial]);

  useFrame(() => {
    const t = getTimeS();
    wallMaterial.uniforms.uTime.value = t;

    const bombMesh = bombMeshRef.current;
    const puffMesh = bombPuffMeshRef.current;
    if (bombMesh) {
      const visibleBombs = queryWindow(bombs, t - BOMB_PUFF_DURATION_S, t + halfJumpDurationSec);
      let count = 0;
      let puffCount = 0;

      for (let i = 0; i < visibleBombs.length; i += 1) {
        const b = visibleBombs[i];
        if (isHitObjectVisible(b.timeS, t, halfJumpDurationSec)) {
          if (count >= MAX_BOMBS) continue;
          const fade = computeSpawnFade(t, b.timeS, halfJumpDurationSec, 0.18);
          dummy.position.set(gridX(b.x), gridY(b.y), objectZAtTime(b.timeS, t, njs));
          dummy.rotation.set(t * 1.5, t * 2.2, 0);
          dummy.scale.setScalar(0.05 + 0.95 * fade);
          dummy.updateMatrix();
          bombMesh.setMatrixAt(count, dummy.matrix);
          count += 1;
        } else if (puffMesh && puffCount < MAX_BOMB_PUFFS) {
          const p = cutEffectProgress(b.timeS, t, BOMB_PUFF_DURATION_S);
          if (p !== null) {
            const ease = 1 - (1 - p) * (1 - p);
            dummy.position.set(gridX(b.x), gridY(b.y), -0.02 - ease * 0.08);
            dummy.rotation.set(0, 0, t * 2);
            dummy.scale.setScalar(0.65 + ease * 1.15);
            dummy.updateMatrix();
            puffMesh.setMatrixAt(puffCount, dummy.matrix);
            const alphaScale = (1 - p) * (1 - p);
            tmpColor.setRGB(1.8 * alphaScale, 0.22 * alphaScale, 0.35 * alphaScale);
            puffMesh.setColorAt(puffCount, tmpColor);
            puffCount += 1;
          }
        }
      }
      bombMesh.count = count;
      bombMesh.instanceMatrix.needsUpdate = true;

      if (puffMesh) {
        puffMesh.count = puffCount;
        puffMesh.instanceMatrix.needsUpdate = true;
        if (puffMesh.instanceColor) puffMesh.instanceColor.needsUpdate = true;
      }
    }

    const wallMesh = wallMeshRef.current;
    if (wallMesh) {
      const visibleWalls = querySpanWindow(
        obstacles,
        (o) => o.endTimeS,
        t + 1e-6,
        t + halfJumpDurationSec,
      );
      const fadeAttr = wallGeo.getAttribute("aFade") as THREE.InstancedBufferAttribute;
      let count = 0;
      for (let i = 0; i < visibleWalls.length && count < MAX_WALLS; i += 1) {
        const w = visibleWalls[i];
        if (!isWallVisible(w.timeS, w.endTimeS, t, halfJumpDurationSec)) continue;
        const span = computeWallZSpan(w.timeS, w.endTimeS, t, njs);
        if (!span) continue;

        const box = obstacleBox(w.x, w.y, w.w, w.h);
        const lenZ = Math.max(0.06, span.zFront - span.zBack);
        const centerZ = (span.zFront + span.zBack) * 0.5;

        dummy.position.set(box.centerX, box.centerY, centerZ);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(box.width, box.height, lenZ);
        dummy.updateMatrix();
        wallMesh.setMatrixAt(count, dummy.matrix);
        fadeAttr.setX(count, computeSpawnFade(t, w.timeS, halfJumpDurationSec, 0.14));
        count += 1;
      }
      wallMesh.count = count;
      wallMesh.instanceMatrix.needsUpdate = true;
      fadeAttr.needsUpdate = true;
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
          color="#141824"
          emissive="#334155"
          emissiveIntensity={0.22}
          roughness={0.2}
          metalness={0.88}
          envMapIntensity={1.6}
        />
      </instancedMesh>

      <instancedMesh
        ref={bombPuffMeshRef}
        args={[bombPuffGeo, undefined, MAX_BOMB_PUFFS]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          color="#ffffff"
          map={puffTex}
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
          side={THREE.DoubleSide}
        />
      </instancedMesh>

      <instancedMesh
        ref={wallMeshRef}
        args={[wallGeo, wallMaterial, MAX_WALLS]}
        frustumCulled={false}
        renderOrder={3}
      />
    </group>
  );
}
