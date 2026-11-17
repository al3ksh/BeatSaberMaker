import { motion } from "framer-motion";

const CYCLE_S = 2.7;

// Continuous physics-sampled trajectory:
// 1. Constant 2:1 isometric velocity along belt (t = 0.00 -> 0.68)
// 2. Smooth parabolic projectile drop off belt lip (t = 0.68 -> 1.00) with zero stall
const TIMES = [0, 0.12, 0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 1];
const X_KEYS = [78, 102, 192, 200.5, 208.5, 215.5, 221, 224.5, 226];
const Y_KEYS = [62, 74, 119, 125.5, 136, 150, 165, 174, 176];
const ROT_KEYS = [0, 0, 0, 8, 18, 30, 42, 50, 54];
const SCALE_KEYS = [0.5, 1, 1, 1, 1, 1, 1, 1, 1];
const OP_KEYS = [0, 1, 1, 1, 0.95, 0.65, 0.2, 0, 0];

type Dir = "down" | "left" | "upright" | "dot";

const PALETTES = {
  red: { top: "#e05561", right: "#b33a46", left: "#7a2530" },
  blue: { top: "#4f8fe0", right: "#3a6db3", left: "#24457a" },
};

// Top face centre and half-edge axes (isometric plane of the top rhombus)
const U = [6.5, 3.25];
const V = [-6.5, 3.25];
const C = [0, -13.5];
const DIRS: Record<Exclude<Dir, "dot">, [number, number]> = {
  down: [1, 1], // screen down  = u + v
  left: [-1, 1], // screen left = -u + v
  upright: [0, -1], // screen up-right = -v
};

function uv(a: number, b: number) {
  const x = C[0] + a * U[0] + b * V[0];
  const y = C[1] + a * U[1] + b * V[1];
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function chevronPoints(dir: Exclude<Dir, "dot">) {
  const [dx0, dy0] = DIRS[dir];
  const n = Math.hypot(dx0, dy0);
  const dx = dx0 / n;
  const dy = dy0 / n;
  const px = -dy;
  const py = dx;
  const tip = [dx * 0.5, dy * 0.5];
  const w1 = [-dx * 0.3 + px * 0.5, -dy * 0.3 + py * 0.5];
  const w2 = [-dx * 0.3 - px * 0.5, -dy * 0.3 - py * 0.5];
  return [w1, tip, w2].map(([a, b]) => uv(a, b)).join(" ");
}

interface IsoCubeProps {
  color: "red" | "blue";
  dir: Dir;
  delay: number;
}

function IsoCube({ color, dir, delay }: IsoCubeProps) {
  const { top, right, left } = PALETTES[color];
  const stroke = "#f4f4f5";

  return (
    <motion.g
      style={{ transformOrigin: "0px -7px" }}
      initial={{ x: X_KEYS[0], y: Y_KEYS[0], rotate: 0, opacity: 0, scale: 0.5 }}
      animate={{
        x: X_KEYS,
        y: Y_KEYS,
        rotate: ROT_KEYS,
        opacity: OP_KEYS,
        scale: SCALE_KEYS,
      }}
      transition={{
        duration: CYCLE_S,
        delay,
        repeat: Infinity,
        ease: "linear",
        times: TIMES,
      }}
    >
      <polygon
        points="-13,-13.5 0,-7 0,6.5 -13,0"
        fill={left}
        stroke={stroke}
        strokeOpacity={0.2}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      <polygon
        points="0,-7 13,-13.5 13,0 0,6.5"
        fill={right}
        stroke={stroke}
        strokeOpacity={0.25}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      <polygon
        points="0,-20 13,-13.5 0,-7 -13,-13.5"
        fill={top}
        stroke={stroke}
        strokeOpacity={0.45}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      {dir === "dot" ? (
        <ellipse
          cx={C[0]}
          cy={C[1]}
          rx={3.2}
          ry={1.6}
          fill="#f4f4f5"
          fillOpacity={0.9}
        />
      ) : (
        <polyline
          points={chevronPoints(dir)}
          fill="none"
          stroke="#f4f4f5"
          strokeOpacity={0.9}
          strokeWidth={1.4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
    </motion.g>
  );
}

const CUBES: { color: "red" | "blue"; dir: Dir }[] = [
  { color: "red", dir: "down" },
  { color: "blue", dir: "left" },
  { color: "red", dir: "upright" },
  { color: "blue", dir: "dot" },
];

// Tunnel: frames shrink toward a vanishing point deep inside the hearth
const HEARTH: [number, number][] = [
  [90, 80],
  [114, 68],
  [114, 44],
  [90, 56],
];
const VP: [number, number] = [96, 57];
const TUNNEL_FRAMES = 4;
const TUNNEL_S = 4.8;
const lerpPt = (p: [number, number], k: number) =>
  [VP[0] + (p[0] - VP[0]) * k, VP[1] + (p[1] - VP[1]) * k] as const;
const hearthAt = (k: number) =>
  HEARTH.map((p) => lerpPt(p, k).map((n) => n.toFixed(2)).join(",")).join(" ");

export function EngineForgeAnimation() {
  return (
    <div className="select-none pointer-events-none" aria-hidden="true">
      <svg
        width="390"
        height="264"
        viewBox="0 0 260 176"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Clips cubes cleanly inside the arch mouth & along the belt + drop zone */}
          <clipPath id="bsm-iso-exit-clip">
            <polygon points="90,56 114,44 252,113 252,200 90,200" />
          </clipPath>
          {/* Clips scrolling belt ribs to top deck surface */}
          <clipPath id="bsm-hearth-clip">
            <polygon points="90,80 114,68 114,44 90,56" />
          </clipPath>
          <clipPath id="bsm-iso-belt-clip">
            <polygon points="88,81 116,67 208,113 180,127" />
          </clipPath>
          {/* Subtle inner hearth floor glow (pure monochrome) */}
          <linearGradient id="bsm-mono-hearth" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#52525b" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#09090b" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Isometric Conveyor Slab */}
        {/* Belt side face (left) */}
        <polygon
          points="88,81 180,127 180,133 88,87"
          fill="#111114"
          stroke="#27272a"
          strokeWidth={0.9}
          strokeLinejoin="round"
        />
        {/* Belt front drop lip */}
        <polygon
          points="180,127 208,113 208,119 180,133"
          fill="#18181b"
          stroke="#3f3f46"
          strokeWidth={0.9}
          strokeLinejoin="round"
        />
        {/* Belt top deck */}
        <polygon
          points="88,81 116,67 208,113 180,127"
          fill="#141418"
          stroke="#27272a"
          strokeWidth={0.9}
          strokeLinejoin="round"
        />

        {/* Animated scrolling belt ribs */}
        <g clipPath="url(#bsm-iso-belt-clip)">
          <motion.g
            initial={{ x: 0, y: 0 }}
            animate={{ x: [0, 16], y: [0, 8] }}
            transition={{ repeat: Infinity, duration: 0.48, ease: "linear" }}
          >
            {Array.from({ length: 10 }).map((_, idx) => {
              const bx = 68 + idx * 16;
              const by = 69 + idx * 8;
              return (
                <line
                  key={idx}
                  x1={bx - 12}
                  y1={by + 6}
                  x2={bx + 12}
                  y2={by - 6}
                  stroke="#27272a"
                  strokeWidth={1.1}
                />
              );
            })}
          </motion.g>
        </g>

        {/* Furnace Body & Full Portal Frame drawn BEFORE exiting cubes for proper 2.5D depth */}
        <polygon
          points="50,34 86,16 120,33 84,51"
          fill="#18181b"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        <polygon
          points="50,34 84,51 84,83 50,66"
          fill="#111114"
          stroke="#27272a"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        {/* Front portal face */}
        <polygon
          points="84,51 120,33 120,65 84,83"
          fill="#141418"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        {/* Inner dark hearth chamber */}
        <polygon
          points="90,80 114,68 114,44 90,56"
          fill="#09090b"
          stroke="#27272a"
          strokeWidth={0.8}
          strokeLinejoin="round"
        />
        <motion.polygon
          points="90,80 114,68 114,44 90,56"
          fill="url(#bsm-mono-hearth)"
          animate={{ opacity: [0.4, 0.9, 0.4] }}
          transition={{ repeat: Infinity, duration: 2.0, ease: "easeInOut" }}
        />

        {/* Tunnel receding into the furnace, clipped to the hearth mouth */}
        <g clipPath="url(#bsm-hearth-clip)">
          {Array.from({ length: TUNNEL_FRAMES }).map((_, i) => (
            <motion.polygon
              key={i}
              points={hearthAt(1)}
              fill="none"
              stroke="#8a8a94"
              strokeWidth={0.7}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              style={{ originX: 0.25, originY: 0.361 }}
              initial={{ scale: 0.2, opacity: 0 }}
              animate={{ scale: [0.2, 0.34, 0.7, 1], opacity: [0, 0.75, 0.75, 0] }}
              transition={{
                repeat: Infinity,
                duration: TUNNEL_S,
                delay: (TUNNEL_S / TUNNEL_FRAMES) * i,
                ease: "linear",
                times: [0, 0.2, 0.8, 1],
              }}
            />
          ))}
        </g>

        {/* Minimal roof vent slot on furnace top (exact 2:1 isometric) */}
        <line
          x1={75}
          y1={38.5}
          x2={95}
          y2={28.5}
          stroke="#52525b"
          strokeWidth={1.2}
          strokeLinecap="round"
        />

        {/* Isometric Cubes emerge from doorway & render IN FRONT OF all exit jambs */}
        <g clipPath="url(#bsm-iso-exit-clip)">
          {CUBES.map((c, idx) => (
            <IsoCube
              key={idx}
              color={c.color}
              dir={c.dir}
              delay={(CYCLE_S / CUBES.length) * idx}
            />
          ))}
        </g>

        {/* Only the left foreground pillar overlaps the leftmost edge cleanly */}
        <polygon
          points="84,51 90,48 90,80 84,83"
          fill="#141418"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}
