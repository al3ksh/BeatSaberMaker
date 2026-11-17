import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { buildLightTimeline, evaluateLightsAtTime, type LightChannelState } from "../lights";
import type { ViewerLightEvent } from "../types";

const RING_COUNT = 8;
const RING_BASE_Z = -12;
const RING_APOTHEM = 4.5;
const RING_THICKNESS = 0.42;
const RING_DEPTH = 0.55;
const RING_CENTER_Y = 1.6;

/** Flat octagonal annulus (apothem = distance from center to a side), extruded along Z and centered. */
function octagonAnnulusGeometry(innerApothem: number, outerApothem: number, depth: number): THREE.BufferGeometry {
  const toRadius = (apothem: number) => apothem / Math.cos(Math.PI / 8);
  const outerR = toRadius(outerApothem);
  const innerR = toRadius(innerApothem);
  const shape = new THREE.Shape();
  const hole = new THREE.Path();
  for (let k = 0; k <= 8; k += 1) {
    const a = Math.PI / 8 + (k * Math.PI) / 4;
    const op = [Math.cos(a) * outerR, Math.sin(a) * outerR] as const;
    const ip = [Math.cos(a) * innerR, Math.sin(a) * innerR] as const;
    if (k === 0) {
      shape.moveTo(op[0], op[1]);
      hole.moveTo(ip[0], ip[1]);
    } else {
      shape.lineTo(op[0], op[1]);
      hole.lineTo(ip[0], ip[1]);
    }
  }
  shape.holes.push(hole);
  const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments: 1 });
  geo.translate(0, 0, -depth / 2);
  return geo;
}
const SIDE_LASER_COUNT = 5;
const BACK_LASER_COUNT = 6;

function updateNeonStripMat(
  mat: THREE.MeshBasicMaterial | null | undefined,
  ch: LightChannelState,
  hdrGain: number,
) {
  if (!mat) return;
  const intensity = Math.max(0, Math.min(2.0, ch.intensity));
  if (intensity <= 0.015) {
    mat.color.setRGB(0.02, 0.024, 0.04);
    return;
  }
  const gain = intensity * hdrGain;
  mat.color.setRGB(ch.r * gain, ch.g * gain, ch.b * gain);
}

function updateLaserBeamPair(
  coreMat: THREE.MeshBasicMaterial | null | undefined,
  sheathMat: THREE.MeshBasicMaterial | null | undefined,
  coreMesh: THREE.Mesh | null | undefined,
  sheathMesh: THREE.Mesh | null | undefined,
  ch: LightChannelState,
) {
  const intensity = Math.max(0, Math.min(2.0, ch.intensity));
  const active = intensity > 0.015;
  if (coreMesh) coreMesh.visible = active;
  if (sheathMesh) sheathMesh.visible = active;
  if (!active) return;

  if (coreMat) {
    const coreGain = 0.8 + intensity * 1.9;
    coreMat.color.setRGB(ch.r * coreGain, ch.g * coreGain, ch.b * coreGain);
    coreMat.opacity = Math.min(1, 0.45 + intensity * 0.55);
  }
  if (sheathMat) {
    const sheathGain = 0.45 + intensity * 0.9;
    sheathMat.color.setRGB(ch.r * sheathGain, ch.g * sheathGain, ch.b * sheathGain);
    sheathMat.opacity = Math.min(0.45, intensity * 0.24);
  }
}

export interface DefaultEnvironmentLightsProps {
  lightEvents: readonly ViewerLightEvent[];
  getTimeS: () => number;
}

export function DefaultEnvironmentLights({
  lightEvents,
  getTimeS,
}: DefaultEnvironmentLightsProps) {
  const timeline = useMemo(() => buildLightTimeline(lightEvents), [lightEvents]);

  const ringCasingMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#0a0d16",
        roughness: 0.82,
        metalness: 0.18,
        envMapIntensity: 0.12,
      }),
    [],
  );

  const ringNeonMaterials = useMemo(
    () =>
      Array.from({ length: RING_COUNT }, () =>
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(0.02, 0.024, 0.04),
          toneMapped: false,
        }),
      ),
    [],
  );

  const ringBodyGeo = useMemo(
    () => octagonAnnulusGeometry(RING_APOTHEM, RING_APOTHEM + RING_THICKNESS, RING_DEPTH),
    [],
  );
  const ringNeonGeo = useMemo(
    () => octagonAnnulusGeometry(RING_APOTHEM - 0.015, RING_APOTHEM + 0.07, 0.03),
    [],
  );

  useEffect(() => {
    return () => {
      ringBodyGeo.dispose();
      ringNeonGeo.dispose();
      ringCasingMat.dispose();
      for (const m of ringNeonMaterials) m.dispose();
    };
  }, [ringBodyGeo, ringNeonGeo, ringCasingMat, ringNeonMaterials]);

  const ringGroupRefs = useRef<Array<THREE.Group | null>>([]);

  const backCoreMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const backCoreMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const backSheathMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const backSheathMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);

  const leftPivotRefs = useRef<Array<THREE.Group | null>>([]);
  const leftCoreMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const leftCoreMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const leftSheathMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const leftSheathMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);

  const rightPivotRefs = useRef<Array<THREE.Group | null>>([]);
  const rightCoreMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const rightCoreMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const rightSheathMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const rightSheathMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);

  const centerMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const nearBounceLightRef = useRef<THREE.PointLight>(null);
  const farBounceLightRef = useRef<THREE.PointLight>(null);

  useFrame(() => {
    const t = getTimeS();
    const state = evaluateLightsAtTime(timeline, t);

    // et 1: Concentric architectural rings + et 8/9 rotation & zoom
    for (let i = 0; i < RING_COUNT; i += 1) {
      const grp = ringGroupRefs.current[i];
      if (grp) {
        grp.position.z = RING_BASE_Z - i * state.ringSpacing;
        grp.rotation.z = Math.PI / 8 + state.ringRotation + i * 0.075;
      }
      updateNeonStripMat(ringNeonMaterials[i], state.groups[1], 1.75);
    }

    // et 0: Back lasers
    for (let i = 0; i < BACK_LASER_COUNT; i += 1) {
      updateLaserBeamPair(
        backCoreMatRefs.current[i],
        backSheathMatRefs.current[i],
        backCoreMeshRefs.current[i],
        backSheathMeshRefs.current[i],
        state.groups[0],
      );
    }

    // et 2 + et 12: Rotating left side lasers
    for (let i = 0; i < SIDE_LASER_COUNT; i += 1) {
      const pivot = leftPivotRefs.current[i];
      if (pivot) {
        pivot.rotation.z = -0.48 + Math.sin(state.leftLaserPhase + i * 0.55) * 0.32;
      }
      updateLaserBeamPair(
        leftCoreMatRefs.current[i],
        leftSheathMatRefs.current[i],
        leftCoreMeshRefs.current[i],
        leftSheathMeshRefs.current[i],
        state.groups[2],
      );
    }

    // et 3 + et 13: Rotating right side lasers
    for (let i = 0; i < SIDE_LASER_COUNT; i += 1) {
      const pivot = rightPivotRefs.current[i];
      if (pivot) {
        pivot.rotation.z = 0.48 - Math.sin(state.rightLaserPhase + i * 0.55) * 0.32;
      }
      updateLaserBeamPair(
        rightCoreMatRefs.current[i],
        rightSheathMatRefs.current[i],
        rightCoreMeshRefs.current[i],
        rightSheathMeshRefs.current[i],
        state.groups[3],
      );
    }

    // et 4: Center chevron & horizon core bar
    for (let i = 0; i < centerMatRefs.current.length; i += 1) {
      updateNeonStripMat(centerMatRefs.current[i], state.groups[4], 2.1);
    }

    // Dynamic stage bounce illumination (soft wide wash without blinding floor hotspots)
    const nearLight = nearBounceLightRef.current;
    if (nearLight) {
      const g1 = state.groups[1];
      const g4 = state.groups[4];
      const w1 = g1.intensity;
      const w4 = g4.intensity;
      const total = w1 + w4;
      if (total <= 0.015) {
        nearLight.intensity = 0;
      } else {
        nearLight.color.setRGB(
          (g1.r * w1 + g4.r * w4) / total,
          (g1.g * w1 + g4.g * w4) / total,
          (g1.b * w1 + g4.b * w4) / total,
        );
        nearLight.intensity = Math.min(5.5, total * 2.4);
      }
    }

    const farLight = farBounceLightRef.current;
    if (farLight) {
      const g0 = state.groups[0];
      const g2 = state.groups[2];
      const g3 = state.groups[3];
      const total = g0.intensity + g2.intensity * 0.5 + g3.intensity * 0.5;
      if (total <= 0.015) {
        farLight.intensity = 0;
      } else {
        farLight.color.setRGB(
          (g0.r * g0.intensity + g2.r * g2.intensity * 0.5 + g3.r * g3.intensity * 0.5) / total,
          (g0.g * g0.intensity + g2.g * g2.intensity * 0.5 + g3.g * g3.intensity * 0.5) / total,
          (g0.b * g0.intensity + g2.b * g2.intensity * 0.5 + g3.b * g3.intensity * 0.5) / total,
        );
        farLight.intensity = Math.min(8.5, total * 3.4);
      }
    }
  });

  return (
    <group>
      {/* Dynamic stage bounce point lights (high elevation to avoid sharp runway hotspots) */}
      <pointLight ref={nearBounceLightRef} position={[0, 6.2, -14]} distance={38} decay={1.5} intensity={0} />
      <pointLight ref={farBounceLightRef} position={[0, 7.2, -34]} distance={54} decay={1.4} intensity={0} />

      {/* et 1: Architectural tunnel rings with inner HDR neon strips */}
      {Array.from({ length: RING_COUNT }, (_, idx) => (
        <group
          key={`ring-${idx}`}
          ref={(el) => {
            ringGroupRefs.current[idx] = el;
          }}
          position={[0, RING_CENTER_Y, RING_BASE_Z - idx * 6]}
          rotation={[0, 0, Math.PI / 8]}
        >
          <mesh geometry={ringBodyGeo} material={ringCasingMat} />
          <mesh geometry={ringNeonGeo} material={ringNeonMaterials[idx]} position={[0, 0, RING_DEPTH / 2 + 0.005]} />
        </group>
      ))}

      {/* et 0: Back fan lasers */}
      <group position={[0, -0.8, -54]}>
        {Array.from({ length: BACK_LASER_COUNT }, (_, idx) => {
          const spread = idx - (BACK_LASER_COUNT - 1) / 2;
          const offset = spread * 2.2;
          const tilt = spread * 0.17;
          return (
            <group key={`back-${idx}`} position={[offset, 0, 0]} rotation={[0, 0, tilt]}>
              <mesh
                ref={(el) => {
                  backCoreMeshRefs.current[idx] = el;
                }}
                position={[0, 18, 0]}
                visible={false}
              >
                <cylinderGeometry args={[0.045, 0.09, 36, 8]} />
                <meshBasicMaterial
                  ref={(el) => {
                    backCoreMatRefs.current[idx] = el;
                  }}
                  color="#ffffff"
                  transparent
                  opacity={0.9}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                  toneMapped={false}
                />
              </mesh>
              <mesh
                ref={(el) => {
                  backSheathMeshRefs.current[idx] = el;
                }}
                position={[0, 18, 0]}
                visible={false}
              >
                <cylinderGeometry args={[0.16, 0.34, 36, 8]} />
                <meshBasicMaterial
                  ref={(el) => {
                    backSheathMatRefs.current[idx] = el;
                  }}
                  color="#ffffff"
                  transparent
                  opacity={0.22}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                  toneMapped={false}
                />
              </mesh>
            </group>
          );
        })}
      </group>

      {/* et 2: Left side emitter lasers */}
      {Array.from({ length: SIDE_LASER_COUNT }, (_, idx) => (
        <group
          key={`left-${idx}`}
          ref={(el) => {
            leftPivotRefs.current[idx] = el;
          }}
          position={[-6.2, -0.8, -10 - idx * 8.5]}
          rotation={[0, 0, -0.48]}
        >
          <mesh
            ref={(el) => {
              leftCoreMeshRefs.current[idx] = el;
            }}
            position={[0, 14, 0]}
            visible={false}
          >
            <cylinderGeometry args={[0.03, 0.065, 28, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                leftCoreMatRefs.current[idx] = el;
              }}
              color="#ffffff"
              transparent
              opacity={0.9}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh
            ref={(el) => {
              leftSheathMeshRefs.current[idx] = el;
            }}
            position={[0, 14, 0]}
            visible={false}
          >
            <cylinderGeometry args={[0.12, 0.24, 28, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                leftSheathMatRefs.current[idx] = el;
              }}
              color="#ffffff"
              transparent
              opacity={0.2}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        </group>
      ))}

      {/* et 3: Right side emitter lasers */}
      {Array.from({ length: SIDE_LASER_COUNT }, (_, idx) => (
        <group
          key={`right-${idx}`}
          ref={(el) => {
            rightPivotRefs.current[idx] = el;
          }}
          position={[6.2, -0.8, -10 - idx * 8.5]}
          rotation={[0, 0, 0.48]}
        >
          <mesh
            ref={(el) => {
              rightCoreMeshRefs.current[idx] = el;
            }}
            position={[0, 14, 0]}
            visible={false}
          >
            <cylinderGeometry args={[0.03, 0.065, 28, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                rightCoreMatRefs.current[idx] = el;
              }}
              color="#ffffff"
              transparent
              opacity={0.9}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh
            ref={(el) => {
              rightSheathMeshRefs.current[idx] = el;
            }}
            position={[0, 14, 0]}
            visible={false}
          >
            <cylinderGeometry args={[0.12, 0.24, 28, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                rightSheathMatRefs.current[idx] = el;
              }}
              color="#ffffff"
              transparent
              opacity={0.2}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        </group>
      ))}

      {/* et 4: Center chevron wings & horizon core bar */}
      <group position={[0, 3.2, -50]}>
        <mesh position={[-1.75, 0.75, 0]} rotation={[0, 0, -0.44]} material={ringCasingMat}>
          <boxGeometry args={[3.9, 0.18, 0.18]} />
        </mesh>
        <mesh position={[-1.75, 0.75, 0.08]} rotation={[0, 0, -0.44]}>
          <boxGeometry args={[3.75, 0.075, 0.06]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[0] = el;
            }}
            color={new THREE.Color(0.02, 0.024, 0.04)}
            toneMapped={false}
          />
        </mesh>

        <mesh position={[1.75, 0.75, 0]} rotation={[0, 0, 0.44]} material={ringCasingMat}>
          <boxGeometry args={[3.9, 0.18, 0.18]} />
        </mesh>
        <mesh position={[1.75, 0.75, 0.08]} rotation={[0, 0, 0.44]}>
          <boxGeometry args={[3.75, 0.075, 0.06]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[1] = el;
            }}
            color={new THREE.Color(0.02, 0.024, 0.04)}
            toneMapped={false}
          />
        </mesh>

        <mesh position={[0, -1.35, 0]} material={ringCasingMat}>
          <boxGeometry args={[4.4, 0.16, 0.16]} />
        </mesh>
        <mesh position={[0, -1.35, 0.08]}>
          <boxGeometry args={[4.2, 0.07, 0.06]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[2] = el;
            }}
            color={new THREE.Color(0.02, 0.024, 0.04)}
            toneMapped={false}
          />
        </mesh>
      </group>
    </group>
  );
}
