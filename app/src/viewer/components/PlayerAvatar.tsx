import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

/**
 * Stylized floating VR avatar (big rounded head, hoodie torso, no legs, floating gloved hands
 * holding the sabers). Faces -Z, toward incoming notes. Purely decorative, idle-bobs on the clock.
 */

const SKIN = "#f1d9c6";
const HAIR = "#262b4a";
const HOODIE = "#3d4579";
const HOODIE_TRIM = "#5b66a3";
const GLOVE = "#f4f6fb";
const LEFT_ACCENT: [number, number, number] = [2.2, 0.24, 0.4];
const RIGHT_ACCENT: [number, number, number] = [0.26, 1.0, 2.5];

function createTorsoGeometry(): THREE.BufferGeometry {
  // Rounded, slightly tapered hoodie body (lathe profile), flattened front-to-back
  const pts = [
    new THREE.Vector2(0.0, -0.19),
    new THREE.Vector2(0.13, -0.19),
    new THREE.Vector2(0.155, -0.165),
    new THREE.Vector2(0.16, -0.1),
    new THREE.Vector2(0.175, 0.02),
    new THREE.Vector2(0.2, 0.11),
    new THREE.Vector2(0.195, 0.165),
    new THREE.Vector2(0.16, 0.205),
    new THREE.Vector2(0.09, 0.228),
    new THREE.Vector2(0.0, 0.232),
  ];
  const geo = new THREE.LatheGeometry(pts, 28);
  geo.scale(1, 1, 0.68);
  return geo;
}

export function PlayerAvatar() {
  const headRef = useRef<THREE.Group>(null);
  const torsoRef = useRef<THREE.Group>(null);
  const leftHandRef = useRef<THREE.Group>(null);
  const rightHandRef = useRef<THREE.Group>(null);

  const geos = useMemo(
    () => ({
      head: new RoundedBoxGeometry(0.3, 0.28, 0.27, 4, 0.1),
      hairTop: new RoundedBoxGeometry(0.325, 0.15, 0.295, 4, 0.07),
      hairBack: new RoundedBoxGeometry(0.315, 0.22, 0.09, 3, 0.04),
      tuft: new THREE.ConeGeometry(0.045, 0.11, 8),
      eye: new THREE.CapsuleGeometry(0.016, 0.03, 4, 8),
      eyeShine: new THREE.CircleGeometry(0.0065, 8),
      mouth: new THREE.TorusGeometry(0.022, 0.005, 6, 14, Math.PI),
      torso: createTorsoGeometry(),
      hood: new THREE.TorusGeometry(0.115, 0.05, 12, 24, Math.PI * 1.3),
      pocket: new RoundedBoxGeometry(0.2, 0.09, 0.03, 3, 0.012),
      hem: new THREE.TorusGeometry(0.138, 0.02, 8, 28),
      zipper: new THREE.BoxGeometry(0.014, 0.3, 0.01),
      glove: new RoundedBoxGeometry(0.1, 0.115, 0.09, 3, 0.04),
      thumb: new THREE.CapsuleGeometry(0.022, 0.04, 4, 8),
      cuff: new THREE.TorusGeometry(0.048, 0.013, 8, 18),
      hilt: new THREE.CylinderGeometry(0.018, 0.018, 0.2, 10),
      blade: new THREE.CylinderGeometry(0.013, 0.016, 0.6, 10),
    }),
    [],
  );

  const mats = useMemo(
    () => ({
      skin: new THREE.MeshStandardMaterial({ color: SKIN, emissive: SKIN, emissiveIntensity: 0.12, roughness: 0.75, metalness: 0, envMapIntensity: 0.6 }),
      hair: new THREE.MeshStandardMaterial({ color: HAIR, emissive: HAIR, emissiveIntensity: 0.2, roughness: 0.6, metalness: 0.05, envMapIntensity: 0.5 }),
      face: new THREE.MeshBasicMaterial({ color: "#14161f" }),
      shine: new THREE.MeshBasicMaterial({ color: "#ffffff" }),
      hoodie: new THREE.MeshStandardMaterial({ color: HOODIE, emissive: HOODIE, emissiveIntensity: 0.18, roughness: 0.85, metalness: 0.05, envMapIntensity: 0.5 }),
      trim: new THREE.MeshStandardMaterial({ color: HOODIE_TRIM, emissive: HOODIE_TRIM, emissiveIntensity: 0.15, roughness: 0.8, metalness: 0.05, envMapIntensity: 0.5 }),
      glove: new THREE.MeshStandardMaterial({ color: GLOVE, emissive: GLOVE, emissiveIntensity: 0.12, roughness: 0.6, metalness: 0, envMapIntensity: 0.6 }),
      hilt: new THREE.MeshStandardMaterial({ color: "#1e293b", roughness: 0.3, metalness: 0.9 }),
      zipper: new THREE.MeshBasicMaterial({ color: new THREE.Color(0.35, 0.9, 1.8), toneMapped: false }),
      left: new THREE.MeshBasicMaterial({ color: new THREE.Color(...LEFT_ACCENT), toneMapped: false }),
      right: new THREE.MeshBasicMaterial({ color: new THREE.Color(...RIGHT_ACCENT), toneMapped: false }),
    }),
    [],
  );

  useEffect(() => {
    return () => {
      Object.values(geos).forEach((g) => g.dispose());
      Object.values(mats).forEach((m) => m.dispose());
    };
  }, [geos, mats]);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const bob = Math.sin(t * 1.7);
    if (headRef.current) {
      headRef.current.position.y = 1.6 + bob * 0.012;
      headRef.current.rotation.z = Math.sin(t * 0.9) * 0.05;
      headRef.current.rotation.x = Math.sin(t * 1.3 + 0.6) * 0.04;
    }
    if (torsoRef.current) {
      torsoRef.current.position.y = 1.13 + bob * 0.008;
      torsoRef.current.rotation.z = Math.sin(t * 0.9 + 0.4) * 0.025;
    }
    if (leftHandRef.current) {
      leftHandRef.current.position.y = 1.02 + Math.sin(t * 1.7 + 1.2) * 0.018;
      leftHandRef.current.rotation.z = 0.16 + Math.sin(t * 1.1) * 0.05;
    }
    if (rightHandRef.current) {
      rightHandRef.current.position.y = 1.02 + Math.sin(t * 1.7 + 2.4) * 0.018;
      rightHandRef.current.rotation.z = -0.16 - Math.sin(t * 1.1 + 0.8) * 0.05;
    }
  });

  const hand = (side: -1 | 1) => (
    <group
      ref={side < 0 ? leftHandRef : rightHandRef}
      position={[side * 0.31, 1.02, -0.17]}
      rotation={[-Math.PI * 0.36, 0, -side * 0.16]}
    >
      {/* Mitten glove wrapped around the hilt */}
      <mesh geometry={geos.glove} material={mats.glove} />
      <mesh
        geometry={geos.thumb}
        material={mats.glove}
        position={[-side * 0.052, 0.014, -0.03]}
        rotation={[0, 0, -side * 0.5]}
      />
      <mesh
        geometry={geos.cuff}
        material={side < 0 ? mats.left : mats.right}
        position={[0, -0.066, 0]}
        rotation={[Math.PI / 2, 0, 0]}
      />
      {/* Saber: hilt through the fist, glowing blade forward-up */}
      <mesh geometry={geos.hilt} material={mats.hilt} position={[0, -0.01, 0]} />
      <mesh geometry={geos.blade} material={side < 0 ? mats.left : mats.right} position={[0, 0.39, 0]} />
    </group>
  );

  return (
    <group>
      {/* Head (floating, no neck) */}
      <group ref={headRef} position={[0, 1.6, 0]}>
        <mesh geometry={geos.head} material={mats.skin} />
        <mesh geometry={geos.hairTop} material={mats.hair} position={[0, 0.085, 0.012]} />
        <mesh geometry={geos.hairBack} material={mats.hair} position={[0, 0.0, 0.1]} />
        {/* Fringe tufts sweeping over the forehead */}
        <mesh geometry={geos.tuft} material={mats.hair} position={[-0.075, 0.1, -0.12]} rotation={[-1.9, 0, 0.5]} />
        <mesh geometry={geos.tuft} material={mats.hair} position={[0.0, 0.11, -0.125]} rotation={[-2.0, 0, 0.15]} />
        <mesh geometry={geos.tuft} material={mats.hair} position={[0.075, 0.1, -0.12]} rotation={[-1.9, 0, -0.35]} />
        {/* Face on the -Z side */}
        <mesh geometry={geos.eye} material={mats.face} position={[-0.058, 0.0, -0.136]} />
        <mesh geometry={geos.eye} material={mats.face} position={[0.058, 0.0, -0.136]} />
        <mesh geometry={geos.eyeShine} material={mats.shine} position={[-0.052, 0.014, -0.153]} rotation={[0, Math.PI, 0]} />
        <mesh geometry={geos.eyeShine} material={mats.shine} position={[0.064, 0.014, -0.153]} rotation={[0, Math.PI, 0]} />
        <mesh geometry={geos.mouth} material={mats.face} position={[0, -0.055, -0.136]} rotation={[0, 0, Math.PI]} />
      </group>

      {/* Hoodie torso with hood collar, hem band and glowing zipper */}
      <group ref={torsoRef} position={[0, 1.13, 0]}>
        <mesh geometry={geos.torso} material={mats.hoodie} />
        {/* Hood collar: open arc around the neck, bunched at the back */}
        <mesh geometry={geos.hood} material={mats.trim} position={[0, 0.215, 0.02]} rotation={[Math.PI / 2 - 0.25, 0, -Math.PI * 0.15]} />
        <mesh geometry={geos.hem} material={mats.trim} position={[0, -0.18, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[1.12, 0.72, 1]} />
        <mesh geometry={geos.pocket} material={mats.trim} position={[0, -0.09, -0.112]} rotation={[0.08, 0, 0]} />
        <mesh geometry={geos.zipper} material={mats.zipper} position={[0, 0.06, -0.128]} rotation={[-0.12, 0, 0]} scale={[1, 0.6, 1]} />
      </group>

      {hand(-1)}
      {hand(1)}
    </group>
  );
}
