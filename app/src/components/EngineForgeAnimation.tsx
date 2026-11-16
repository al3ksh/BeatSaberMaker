import { motion } from "framer-motion";

interface BlockSpec {
  id: string;
  fill: string;
  rim: string;
  glow: string;
  arrowPoints: string;
  delay: number;
}

const CYCLE_S = 2.8;

const BLOCKS: BlockSpec[] = [
  {
    id: "red",
    fill: "#ef4444",
    rim: "#fca5a5",
    glow: "#ef4444",
    // Down-pointing Beat Saber chevron inside 28x28 cube
    arrowPoints: "8,9 14,15 20,9 20,13.5 14,19.5 8,13.5",
    delay: 0,
  },
  {
    id: "blue",
    fill: "#3b82f6",
    rim: "#93c5fd",
    glow: "#3b82f6",
    // Up-pointing Beat Saber chevron inside 28x28 cube
    arrowPoints: "8,19 14,13 20,19 20,14.5 14,8.5 8,14.5",
    delay: CYCLE_S / 2,
  },
];

function NoteArtwork({ spec, idSuffix }: { spec: BlockSpec; idSuffix: string }) {
  return (
    <g>
      <rect
        x={0}
        y={0}
        width={28}
        height={28}
        rx={4}
        fill={spec.fill}
        stroke={spec.rim}
        strokeWidth={1.1}
        strokeOpacity={0.7}
      />
      {/* Inner recessed bevel */}
      <rect
        x={3}
        y={3}
        width={22}
        height={22}
        rx={2.5}
        fill="#09090b"
        fillOpacity={0.22}
      />
      {/* Directional chevron */}
      <polygon points={spec.arrowPoints} fill="#ffffff" fillOpacity={0.95} />
      {/* Top specular highlight */}
      <rect
        x={3}
        y={1.5}
        width={22}
        height={1.5}
        rx={0.75}
        fill="#ffffff"
        fillOpacity={0.25}
        data-block={idSuffix}
      />
    </g>
  );
}

function ForgedBlockLoop({ spec }: { spec: BlockSpec }) {
  return (
    <g>
      {/* Intact block sliding out of the glowing furnace to the slicer */}
      <motion.g
        initial={{ x: 52, y: 55, opacity: 0 }}
        animate={{
          x: [52, 92, 210, 210, 210],
          y: [55, 55, 55, 55, 55],
          opacity: [0, 1, 1, 0, 0],
        }}
        transition={{
          duration: CYCLE_S,
          delay: spec.delay,
          repeat: Infinity,
          ease: ["linear", "linear", "linear", "linear"],
          times: [0, 0.08, 0.559, 0.56, 1],
        }}
      >
        <NoteArtwork spec={spec} idSuffix="whole" />
        {/* Warm amber cooling glow fading out after leaving furnace */}
        <motion.rect
          x={0}
          y={0}
          width={28}
          height={28}
          rx={4}
          fill="#fb923c"
          initial={{ opacity: 0.7 }}
          animate={{ opacity: [0.75, 0.35, 0, 0] }}
          transition={{
            duration: CYCLE_S,
            delay: spec.delay,
            repeat: Infinity,
            times: [0, 0.14, 0.28, 1],
          }}
        />
      </motion.g>

      {/* Upper-left sliced half */}
      <motion.g
        style={{ transformOrigin: "14px 14px" }}
        initial={{ x: 210, y: 55, rotate: 0, opacity: 0 }}
        animate={{
          x: [210, 210, 210, 272, 318],
          y: [55, 55, 55, 48, 53],
          rotate: [0, 0, 0, -14, -18],
          opacity: [0, 0, 1, 0.95, 0],
        }}
        transition={{
          duration: CYCLE_S,
          delay: spec.delay,
          repeat: Infinity,
          ease: "easeOut",
          times: [0, 0.559, 0.56, 0.8, 0.96],
        }}
      >
        <g clipPath="url(#bsm-half-tl)">
          <NoteArtwork spec={spec} idSuffix="tl" />
          {/* Glowing saber cut seam */}
          <line x1={27} y1={1} x2={1} y2={27} stroke="#ffffff" strokeWidth={1.8} />
        </g>
      </motion.g>

      {/* Lower-right sliced half */}
      <motion.g
        style={{ transformOrigin: "14px 14px" }}
        initial={{ x: 210, y: 55, rotate: 0, opacity: 0 }}
        animate={{
          x: [210, 210, 210, 266, 310],
          y: [55, 55, 55, 61, 63],
          rotate: [0, 0, 0, 11, 15],
          opacity: [0, 0, 1, 0.95, 0],
        }}
        transition={{
          duration: CYCLE_S,
          delay: spec.delay,
          repeat: Infinity,
          ease: "easeOut",
          times: [0, 0.559, 0.56, 0.8, 0.96],
        }}
      >
        <g clipPath="url(#bsm-half-br)">
          <NoteArtwork spec={spec} idSuffix="br" />
          <line x1={27} y1={1} x2={1} y2={27} stroke="#ffffff" strokeWidth={1.8} />
        </g>
      </motion.g>

      {/* Saber slash beam flash synced at t = 0.56 */}
      <motion.g
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0, 1, 0.9, 0, 0] }}
        transition={{
          duration: CYCLE_S,
          delay: spec.delay,
          repeat: Infinity,
          times: [0, 0.53, 0.555, 0.59, 0.65, 1],
        }}
      >
        <line
          x1={242}
          y1={30}
          x2={206}
          y2={84}
          stroke={spec.glow}
          strokeWidth={6}
          strokeLinecap="round"
          strokeOpacity={0.45}
        />
        <line
          x1={242}
          y1={30}
          x2={206}
          y2={84}
          stroke="#ffffff"
          strokeWidth={2}
          strokeLinecap="round"
        />
      </motion.g>

      {/* Tiny crisp sparks flying from the cut */}
      {[
        { dx: 18, dy: -16 },
        { dx: 24, dy: -4 },
        { dx: -12, dy: -18 },
        { dx: 14, dy: 10 },
      ].map((spark, idx) => (
        <motion.rect
          key={`${spec.id}-spark-${idx}`}
          x={224}
          y={67}
          width={2.5}
          height={2.5}
          rx={0.5}
          fill={idx % 2 === 0 ? "#ffffff" : spec.rim}
          initial={{ x: 0, y: 0, opacity: 0 }}
          animate={{
            x: [0, 0, spark.dx, spark.dx * 1.4],
            y: [0, 0, spark.dy, spark.dy * 1.3],
            opacity: [0, 0, 1, 0],
          }}
          transition={{
            duration: CYCLE_S,
            delay: spec.delay,
            repeat: Infinity,
            ease: "easeOut",
            times: [0, 0.555, 0.62, 0.74],
          }}
        />
      ))}
    </g>
  );
}

export function EngineForgeAnimation() {
  return (
    <div
      className="select-none pointer-events-none"
      aria-hidden="true"
    >
      <svg
        width="360"
        height="116"
        viewBox="0 0 360 116"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="overflow-visible"
      >
        <defs>
          {/* Clips belt & blocks cleanly inside the furnace exit through right fade */}
          <clipPath id="bsm-conveyor-zone">
            <rect x={76} y={16} width={270} height={82} />
          </clipPath>
          <clipPath id="bsm-belt-clip">
            <rect x={80} y={85} width={258} height={8} rx={4} />
          </clipPath>
          <clipPath id="bsm-half-tl">
            <polygon points="0,0 28,0 0,28" />
          </clipPath>
          <clipPath id="bsm-half-br">
            <polygon points="28,0 28,28 0,28" />
          </clipPath>
          <linearGradient id="bsm-hearth" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#9a3412" />
            <stop offset="55%" stopColor="#ea580c" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
          <linearGradient id="bsm-hearth-spill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#f97316" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="bsm-right-fade" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#09090b" stopOpacity="0" />
            <stop offset="100%" stopColor="#09090b" stopOpacity="1" />
          </linearGradient>
        </defs>

        {/* Conveyor track housing */}
        <rect
          x={78}
          y={84}
          width={262}
          height={10}
          rx={5}
          fill="#141418"
          stroke="#27272a"
          strokeWidth={1}
        />

        {/* Animated conveyor belt rollers */}
        <g clipPath="url(#bsm-belt-clip)">
          <motion.g
            initial={{ x: 0 }}
            animate={{ x: [0, 20] }}
            transition={{ repeat: Infinity, duration: 0.45, ease: "linear" }}
          >
            {Array.from({ length: 15 }).map((_, i) => (
              <rect
                key={i}
                x={64 + i * 20}
                y={87.5}
                width={8}
                height={3}
                rx={1.5}
                fill="#3f3f46"
              />
            ))}
          </motion.g>
        </g>

        {/* Subtle warm furnace spill light onto conveyor */}
        <rect x={84} y={50} width={54} height={34} fill="url(#bsm-hearth-spill)" />

        {/* Overhead saber slicer gantry */}
        <path
          d="M236 24 L236 16 L248 16"
          stroke="#27272a"
          strokeWidth={2}
          strokeLinecap="round"
        />
        <rect
          x={231}
          y={22}
          width={14}
          height={9}
          rx={2.5}
          fill="#18181b"
          stroke="#3f3f46"
          strokeWidth={1}
        />
        <motion.circle
          cx={238}
          cy={26.5}
          r={1.6}
          fill="#f4f4f5"
          animate={{ opacity: [0.35, 1, 0.35] }}
          transition={{ repeat: Infinity, duration: 1.4, ease: "easeInOut" }}
        />

        {/* Emerging + sliced blocks (clipped so they emerge from inside furnace) */}
        <g clipPath="url(#bsm-conveyor-zone)">
          {BLOCKS.map((spec) => (
            <ForgedBlockLoop key={spec.id} spec={spec} />
          ))}
          {/* Smooth edge fade-out at right end of belt */}
          <rect x={312} y={24} width={34} height={64} fill="url(#bsm-right-fade)" />
        </g>

        {/* Furnace ("Piec") drawn on the left edge over block spawn point */}
        {/* Chimney + rising heat wisps */}
        <motion.circle
          cx={38}
          cy={9}
          r={2.2}
          fill="#fb923c"
          animate={{ y: [0, -7], opacity: [0.45, 0], scale: [0.9, 1.3] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeOut" }}
        />
        <motion.circle
          cx={42}
          cy={7}
          r={1.6}
          fill="#fbbf24"
          animate={{ y: [0, -8], opacity: [0.35, 0], scale: [0.8, 1.2] }}
          transition={{ repeat: Infinity, duration: 1.8, delay: 0.9, ease: "easeOut" }}
        />
        <path
          d="M31 24 L31 13 L45 13 L45 24"
          fill="#18181b"
          stroke="#27272a"
          strokeWidth={1.2}
        />

        {/* Furnace main casing */}
        <path
          d="M18 30 C18 26 21 24 25 24 L76 24 L86 35 L86 94 L18 94 Z"
          fill="#18181b"
          stroke="#27272a"
          strokeWidth={1.2}
        />
        {/* Vents / industrial detail */}
        <line x1={27} y1={35} x2={46} y2={35} stroke="#27272a" strokeWidth={1.5} strokeLinecap="round" />
        <line x1={27} y1={40} x2={40} y2={40} stroke="#27272a" strokeWidth={1.5} strokeLinecap="round" />
        <circle cx={29} cy={78} r={2} fill="#27272a" />

        {/* Glowing oven hearth opening */}
        <path
          d="M54 84 L54 53 C54 46 60 42 68 42 L84 42 L84 84 Z"
          fill="#270e07"
          stroke="#3f3f46"
          strokeWidth={1}
        />
        <motion.path
          d="M56 83 L56 54 C56 48 61 44 68 44 L82 44 L82 83 Z"
          fill="url(#bsm-hearth)"
          animate={{ opacity: [0.68, 0.98, 0.68] }}
          transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
        />
        {/* Heat-shield baffle strips at mouth */}
        <rect x={78} y={43} width={2.5} height={14} rx={1} fill="#18181b" fillOpacity={0.85} />
        <rect x={81.5} y={43} width={2.5} height={16} rx={1} fill="#18181b" fillOpacity={0.75} />
      </svg>
    </div>
  );
}
