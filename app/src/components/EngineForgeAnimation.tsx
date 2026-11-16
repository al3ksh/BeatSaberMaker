import { motion } from "framer-motion";

const CYCLE_S = 2.7;
const CUBE_OFFSETS = [0, 1, 2];

// Continuous physics-sampled trajectory:
// 1. Constant 2:1 isometric velocity along belt (t = 0.00 -> 0.68)
// 2. Smooth parabolic projectile drop off belt lip (t = 0.68 -> 1.00) with zero stall
const TIMES = [0, 0.12, 0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 1];
const X_KEYS = [78, 102, 192, 200.5, 208.5, 215.5, 221, 224.5, 226];
const Y_KEYS = [62, 74, 119, 125.5, 136, 150, 165, 174, 176];
const ROT_KEYS = [0, 0, 0, 8, 18, 30, 42, 50, 54];
const OP_KEYS = [0, 1, 1, 1, 0.95, 0.65, 0.2, 0, 0];

interface IsoCubeProps {
  variant: "light" | "muted";
  delay: number;
}

function IsoCube({ variant, delay }: IsoCubeProps) {
  const isLight = variant === "light";
  const topFill = isLight ? "#e4e4e7" : "#71717a";
  const rightFill = isLight ? "#a1a1aa" : "#52525b";
  const leftFill = isLight ? "#52525b" : "#3f3f46";
  const stroke = isLight ? "#f4f4f5" : "#a1a1aa";
  const arrowStroke = isLight ? "#18181b" : "#e4e4e7";

  return (
    <motion.g
      style={{ transformOrigin: "0px -7px" }}
      initial={{ x: X_KEYS[0], y: Y_KEYS[0], rotate: 0, opacity: 0 }}
      animate={{
        x: X_KEYS,
        y: Y_KEYS,
        rotate: ROT_KEYS,
        opacity: OP_KEYS,
      }}
      transition={{
        duration: CYCLE_S,
        delay,
        repeat: Infinity,
        ease: "linear",
        times: TIMES,
      }}
    >
      {/* Left face */}
      <polygon
        points="-13,-13.5 0,-7 0,6.5 -13,0"
        fill={leftFill}
        stroke={stroke}
        strokeOpacity={0.28}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      {/* Front-right face */}
      <polygon
        points="0,-7 13,-13.5 13,0 0,6.5"
        fill={rightFill}
        stroke={stroke}
        strokeOpacity={0.35}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      {/* Top face */}
      <polygon
        points="0,-20 13,-13.5 0,-7 -13,-13.5"
        fill={topFill}
        stroke={stroke}
        strokeOpacity={0.55}
        strokeWidth={0.8}
        strokeLinejoin="round"
      />
      {/* Minimal isometric chevron mark on top face pointing forward-right (+u) */}
      <polyline
        points="-4,-14.5 3.5,-13.5 -0.5,-10"
        fill="none"
        stroke={arrowStroke}
        strokeOpacity={0.8}
        strokeWidth={1.3}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </motion.g>
  );
}

export function EngineForgeAnimation() {
  return (
    <div className="select-none pointer-events-none" aria-hidden="true">
      <svg
        width="260"
        height="176"
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
          {CUBE_OFFSETS.map((idx) => (
            <IsoCube
              key={idx}
              variant={idx % 2 === 0 ? "light" : "muted"}
              delay={(CYCLE_S / CUBE_OFFSETS.length) * idx}
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
