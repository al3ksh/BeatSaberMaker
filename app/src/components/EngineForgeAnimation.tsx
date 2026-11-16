import { motion } from "framer-motion";

const CYCLE_S = 3.0;
const CUBE_OFFSETS = [0, 1, 2];

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
      initial={{ x: 86, y: 66, rotate: 0, opacity: 0 }}
      animate={{
        x: [86, 102, 192, 203, 207],
        y: [66, 74, 119, 129, 154],
        rotate: [0, 0, 0, 14, 28],
        opacity: [0, 1, 1, 0.85, 0],
      }}
      transition={{
        duration: CYCLE_S,
        delay,
        repeat: Infinity,
        ease: ["linear", "linear", "easeIn", "easeIn"],
        times: [0, 0.1, 0.72, 0.84, 1],
      }}
    >
      {/* Subtle contact shadow on belt */}
      <polygon
        points="0,-6 13,0.5 0,7 -13,0.5"
        fill="#000000"
        fillOpacity={0.35}
      />
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
        height="146"
        viewBox="0 0 260 146"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Clips cubes cleanly inside the furnace exit and belt/drop area */}
          <clipPath id="bsm-iso-exit-clip">
            <polygon points="88,65 122,48 240,107 240,152 170,152 88,108" />
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
            transition={{ repeat: Infinity, duration: 0.55, ease: "linear" }}
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

        {/* Furnace Back/Left Casing (behind emerging cubes) */}
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
        {/* Inner furnace hearth chamber behind exit arch */}
        <polygon
          points="84,51 120,33 120,65 84,83"
          fill="#09090b"
          stroke="#27272a"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        <motion.polygon
          points="90,80 114,68 114,50 90,62"
          fill="url(#bsm-mono-hearth)"
          animate={{ opacity: [0.4, 0.9, 0.4] }}
          transition={{ repeat: Infinity, duration: 2.0, ease: "easeInOut" }}
        />

        {/* Isometric Cubes sliding out and dropping cleanly off the belt end */}
        <g clipPath="url(#bsm-iso-exit-clip)">
          {CUBE_OFFSETS.map((idx) => (
            <IsoCube
              key={idx}
              variant={idx % 2 === 0 ? "light" : "muted"}
              delay={(CYCLE_S / CUBE_OFFSETS.length) * idx}
            />
          ))}
        </g>

        {/* Furnace Front Portal Frame (above & flanking exit so cubes emerge from inside) */}
        {/* Upper lintel */}
        <polygon
          points="84,51 120,33 120,47 84,65"
          fill="#141418"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        {/* Left portal pillar */}
        <polygon
          points="84,65 90,62 90,80 84,83"
          fill="#141418"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
        />
        {/* Right portal pillar */}
        <polygon
          points="114,50 120,47 120,65 114,68"
          fill="#141418"
          stroke="#3f3f46"
          strokeWidth={1}
          strokeLinejoin="round"
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
      </svg>
    </div>
  );
}
