import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { buildLightTimeline, evaluateLightsAtTime, type LightChannelState } from "../lights";
import type { ViewerLightEvent } from "../types";

const RING_COUNT = 6;
const RING_BASE_Z = -10;
const SIDE_LASER_COUNT = 4;
const BACK_LASER_COUNT = 4;

function applyChannelToMat(
  mat: THREE.MeshBasicMaterial | null | undefined,
  ch: LightChannelState,
  baseOpacity: number,
) {
  if (!mat) return;
  const clamped = Math.max(0, Math.min(1.6, ch.intensity));
  mat.color.setRGB(
    0.06 + ch.r * Math.min(1.2, clamped),
    0.07 + ch.g * Math.min(1.2, clamped),
    0.1 + ch.b * Math.min(1.2, clamped),
  );
  mat.opacity = clamped <= 0.01 ? 0.06 : Math.min(0.95, baseOpacity * Math.min(1.3, clamped));
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

  const ringMaterials = useMemo(
    () =>
      Array.from({ length: RING_COUNT }, () =>
        new THREE.MeshBasicMaterial({ color: "#1e293b", transparent: true, opacity: 0.12 }),
      ),
    [],
  );

  useEffect(() => {
    return () => {
      for (const m of ringMaterials) m.dispose();
    };
  }, [ringMaterials]);

  const ringGroupRefs = useRef<Array<THREE.Group | null>>([]);
  const backMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const leftGroupRef = useRef<THREE.Group>(null);
  const leftMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const rightGroupRef = useRef<THREE.Group>(null);
  const rightMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);
  const centerMatRefs = useRef<Array<THREE.MeshBasicMaterial | null>>([]);

  useFrame(() => {
    const t = getTimeS();
    const state = evaluateLightsAtTime(timeline, t);

    // et 0: Back lasers
    for (let i = 0; i < backMatRefs.current.length; i += 1) {
      applyChannelToMat(backMatRefs.current[i], state.groups[0], 0.55);
    }

    // et 1: Ring lights + et 8/9 rotation & zoom
    for (let i = 0; i < RING_COUNT; i += 1) {
      const grp = ringGroupRefs.current[i];
      if (grp) {
        grp.position.z = RING_BASE_Z - i * state.ringSpacing;
        grp.rotation.z = Math.PI / 4 + state.ringRotation + i * 0.08;
      }
      applyChannelToMat(ringMaterials[i], state.groups[1], 0.72);
    }

    // et 2 + et 12: Left lasers
    if (leftGroupRef.current) {
      leftGroupRef.current.rotation.z = Math.sin(state.leftLaserPhase) * 0.22;
    }
    for (let i = 0; i < leftMatRefs.current.length; i += 1) {
      applyChannelToMat(leftMatRefs.current[i], state.groups[2], 0.65);
    }

    // et 3 + et 13: Right lasers
    if (rightGroupRef.current) {
      rightGroupRef.current.rotation.z = -Math.sin(state.rightLaserPhase) * 0.22;
    }
    for (let i = 0; i < rightMatRefs.current.length; i += 1) {
      applyChannelToMat(rightMatRefs.current[i], state.groups[3], 0.65);
    }

    // et 4: Center lights
    for (let i = 0; i < centerMatRefs.current.length; i += 1) {
      applyChannelToMat(centerMatRefs.current[i], state.groups[4], 0.8);
    }
  });

  return (
    <group>
      {/* et 1: Concentric diamond/square track rings */}
      {Array.from({ length: RING_COUNT }, (_, idx) => (
        <group
          key={`ring-${idx}`}
          ref={(el) => {
            ringGroupRefs.current[idx] = el;
          }}
          position={[0, 1.5, RING_BASE_Z - idx * 6]}
          rotation={[0, 0, Math.PI / 4]}
        >
          {[
            { pos: [0, 3.6, 0] as const, rot: 0 },
            { pos: [0, -3.6, 0] as const, rot: 0 },
            { pos: [3.6, 0, 0] as const, rot: Math.PI / 2 },
            { pos: [-3.6, 0, 0] as const, rot: Math.PI / 2 },
          ].map((bar, bIdx) => (
            <mesh
              key={bIdx}
              position={bar.pos}
              rotation={[0, 0, bar.rot]}
              material={ringMaterials[idx]}
            >
              <boxGeometry args={[7.3, 0.08, 0.08]} />
            </mesh>
          ))}
        </group>
      ))}

      {/* Back lasers (et 0) */}
      <group position={[0, 0, -40]}>
        {Array.from({ length: BACK_LASER_COUNT }, (_, idx) => {
          const offset = (idx - 1.5) * 2.4;
          const tilt = (idx - 1.5) * 0.22;
          return (
            <mesh key={`back-${idx}`} position={[offset, 5.5, 0]} rotation={[0, 0, tilt]}>
              <cylinderGeometry args={[0.04, 0.14, 18, 8]} />
              <meshBasicMaterial
                ref={(el) => {
                  backMatRefs.current[idx] = el;
                }}
                color="#1e293b"
                transparent
                opacity={0.1}
              />
            </mesh>
          );
        })}
      </group>

      {/* Left lasers (et 2) */}
      <group ref={leftGroupRef} position={[-4.5, 0.5, -24]}>
        {Array.from({ length: SIDE_LASER_COUNT }, (_, idx) => (
          <mesh
            key={`left-${idx}`}
            position={[0, 4.5, -idx * 4]}
            rotation={[0, 0, -0.35 - idx * 0.06]}
          >
            <cylinderGeometry args={[0.035, 0.1, 14, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                leftMatRefs.current[idx] = el;
              }}
              color="#1e293b"
              transparent
              opacity={0.1}
            />
          </mesh>
        ))}
      </group>

      {/* Right lasers (et 3) */}
      <group ref={rightGroupRef} position={[4.5, 0.5, -24]}>
        {Array.from({ length: SIDE_LASER_COUNT }, (_, idx) => (
          <mesh
            key={`right-${idx}`}
            position={[0, 4.5, -idx * 4]}
            rotation={[0, 0, 0.35 + idx * 0.06]}
          >
            <cylinderGeometry args={[0.035, 0.1, 14, 8]} />
            <meshBasicMaterial
              ref={(el) => {
                rightMatRefs.current[idx] = el;
              }}
              color="#1e293b"
              transparent
              opacity={0.1}
            />
          </mesh>
        ))}
      </group>

      {/* Center horizon chevron & bar (et 4) */}
      <group position={[0, 2.8, -42]}>
        <mesh position={[-1.4, 0.6, 0]} rotation={[0, 0, -0.45]}>
          <boxGeometry args={[3.2, 0.14, 0.14]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[0] = el;
            }}
            color="#1e293b"
            transparent
            opacity={0.15}
          />
        </mesh>
        <mesh position={[1.4, 0.6, 0]} rotation={[0, 0, 0.45]}>
          <boxGeometry args={[3.2, 0.14, 0.14]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[1] = el;
            }}
            color="#1e293b"
            transparent
            opacity={0.15}
          />
        </mesh>
        <mesh position={[0, -0.9, 0]}>
          <boxGeometry args={[2.2, 0.12, 0.12]} />
          <meshBasicMaterial
            ref={(el) => {
              centerMatRefs.current[2] = el;
            }}
            color="#1e293b"
            transparent
            opacity={0.15}
          />
        </mesh>
      </group>
    </group>
  );
}
