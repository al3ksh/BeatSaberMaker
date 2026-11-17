import { useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, Lightformer, OrbitControls } from "@react-three/drei";
import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { ParsedDifficultyData } from "../types";
import { CAMERA_PRESETS, type CameraPreset } from "../geometry";
import type { ViewerQualityMode } from "../viewerPrefs";
import { TrackEnvironment } from "./TrackEnvironment";
import { NotesLayer } from "./NotesLayer";
import { WallsAndBombsLayer } from "./WallsAndBombsLayer";
import { ArcsAndChainsLayer } from "./ArcsAndChainsLayer";
import { DefaultEnvironmentLights } from "./DefaultEnvironmentLights";
import * as THREE from "three";

/**
 * three ignores material.envMapIntensity when the env comes from scene.environment (it uses
 * scene.environmentIntensity instead). Binding the scene env onto each lit material explicitly makes
 * the per-material envMapIntensity values (dark matte decks vs glossy notes) actually take effect.
 */
function EnvMapIntensityBridge() {
  const frameRef = useRef(0);
  useFrame(({ scene }) => {
    frameRef.current += 1;
    const env = scene.environment;
    if (!env || frameRef.current % 20 !== 1) return;
    scene.traverse((obj) => {
      const mat = (obj as THREE.Mesh).material;
      if (!mat || Array.isArray(mat)) return;
      if (mat instanceof THREE.MeshStandardMaterial && mat.envMap !== env) {
        mat.envMap = env;
        mat.needsUpdate = true;
      }
    });
  });
  return null;
}

function CameraRig({ preset }: { preset: CameraPreset }) {
  const { camera } = useThree();
  const controlsRef = useRef<OrbitControlsImpl>(null);

  useEffect(() => {
    const cfg = CAMERA_PRESETS[preset];
    camera.position.set(...cfg.position);
    if ("fov" in camera && typeof camera.fov === "number") {
      camera.fov = cfg.fov;
      camera.updateProjectionMatrix();
    }
    if (controlsRef.current) {
      controlsRef.current.target.set(...cfg.target);
      controlsRef.current.update();
    } else {
      camera.lookAt(...cfg.target);
    }
  }, [preset, camera]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.12}
      minDistance={1.5}
      maxDistance={32}
    />
  );
}

function FpsProbe() {
  const framesRef = useRef(0);
  const lastSampleRef = useRef(performance.now());

  useFrame(() => {
    framesRef.current += 1;
    const now = performance.now();
    const dt = now - lastSampleRef.current;
    if (dt >= 500) {
      const fps = Math.round((framesRef.current * 1000) / dt);
      (window as unknown as { __bsmViewerFps?: number }).__bsmViewerFps = fps;
      framesRef.current = 0;
      lastSampleRef.current = now;
    }
  });

  return null;
}

export interface ViewerSceneProps {
  difficulty: ParsedDifficultyData;
  njs: number;
  halfJumpDurationSec: number;
  cameraPreset: CameraPreset;
  qualityMode: ViewerQualityMode;
  getTimeS: () => number;
  children?: React.ReactNode;
}

export function ViewerScene({
  difficulty,
  njs,
  halfJumpDurationSec,
  cameraPreset,
  qualityMode,
  getTimeS,
  children,
}: ViewerSceneProps) {
  const initialCam = CAMERA_PRESETS[cameraPreset];

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: initialCam.position, fov: initialCam.fov, near: 0.1, far: 140 }}
      gl={{ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true }}
      className="h-full w-full block bg-[#03040a]"
    >
      <color attach="background" args={["#03040a"]} />
      <fog attach="fog" args={["#03040a", 16, 66]} />

      <ambientLight intensity={0.22} />
      <directionalLight position={[3.5, 9, 5]} intensity={1.15} color="#f1f5f9" />
      <directionalLight position={[-4, 5, -3]} intensity={0.45} color="#93c5fd" />

      {/* Lightweight baked once-at-mount reflection environment for glossy blocks & floor */}
      <Environment resolution={64} frames={1}>
        <Lightformer
          form="rect"
          intensity={1.3}
          color="#e2e8f0"
          position={[0, 9, -4]}
          rotation={[Math.PI / 2, 0, 0]}
          scale={[8, 22, 1]}
        />
        <Lightformer
          form="rect"
          intensity={0.55}
          color="#ef4444"
          position={[-9, 5, -10]}
          rotation={[0, Math.PI / 2, 0]}
          scale={[22, 4, 1]}
        />
        <Lightformer
          form="rect"
          intensity={0.55}
          color="#3b82f6"
          position={[9, 5, -10]}
          rotation={[0, -Math.PI / 2, 0]}
          scale={[22, 4, 1]}
        />
      </Environment>

      <FpsProbe />
      <EnvMapIntensityBridge />
      <CameraRig preset={cameraPreset} />
      <DefaultEnvironmentLights lightEvents={difficulty.lightEvents} getTimeS={getTimeS} />
      <TrackEnvironment cameraPreset={cameraPreset} />
      <NotesLayer
        notes={difficulty.notes}
        njs={njs}
        halfJumpDurationSec={halfJumpDurationSec}
        getTimeS={getTimeS}
      />
      <WallsAndBombsLayer
        bombs={difficulty.bombs}
        obstacles={difficulty.obstacles}
        njs={njs}
        halfJumpDurationSec={halfJumpDurationSec}
        getTimeS={getTimeS}
      />
      <ArcsAndChainsLayer
        arcs={difficulty.arcs}
        chains={difficulty.chains}
        njs={njs}
        halfJumpDurationSec={halfJumpDurationSec}
        getTimeS={getTimeS}
      />

      {qualityMode === "high" && (
        <EffectComposer multisampling={0}>
          <Bloom
            mipmapBlur
            luminanceThreshold={1.0}
            luminanceSmoothing={0.28}
            intensity={0.72}
            radius={0.62}
          />
          <Vignette eskil={false} offset={0.24} darkness={0.44} />
        </EffectComposer>
      )}

      {children}
    </Canvas>
  );
}
