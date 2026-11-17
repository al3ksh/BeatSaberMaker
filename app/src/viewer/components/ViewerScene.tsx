import { useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { ParsedDifficultyData } from "../types";
import { CAMERA_PRESETS, type CameraPreset } from "../geometry";
import { TrackEnvironment } from "./TrackEnvironment";
import { NotesLayer } from "./NotesLayer";

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

export interface ViewerSceneProps {
  difficulty: ParsedDifficultyData;
  njs: number;
  halfJumpDurationSec: number;
  cameraPreset: CameraPreset;
  getTimeS: () => number;
  children?: React.ReactNode;
}

export function ViewerScene({
  difficulty,
  njs,
  halfJumpDurationSec,
  cameraPreset,
  getTimeS,
  children,
}: ViewerSceneProps) {
  const initialCam = CAMERA_PRESETS[cameraPreset];

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: initialCam.position, fov: initialCam.fov, near: 0.1, far: 120 }}
      gl={{ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true }}
      className="h-full w-full block bg-[#05060b]"
    >
      <color attach="background" args={["#05060b"]} />
      <fog attach="fog" args={["#05060b", 22, 65]} />

      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 10, 6]} intensity={1.3} />
      <directionalLight position={[-5, 6, -4]} intensity={0.45} color="#93c5fd" />

      <CameraRig preset={cameraPreset} />
      <TrackEnvironment />
      <NotesLayer
        notes={difficulty.notes}
        njs={njs}
        halfJumpDurationSec={halfJumpDurationSec}
        getTimeS={getTimeS}
      />
      {children}
    </Canvas>
  );
}
