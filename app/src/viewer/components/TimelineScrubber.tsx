import { useRef, useState } from "react";
import { cn } from "../../components/ui/primitives";

function formatTooltipTime(sec: number): string {
  const safe = Math.max(0, Number.isFinite(sec) ? sec : 0);
  const mins = Math.floor(safe / 60);
  const whole = Math.floor(safe - mins * 60);
  return `${mins}:${String(whole).padStart(2, "0")}`;
}

export interface TimelineScrubberProps {
  currentTime: number;
  duration: number;
  disabled?: boolean;
  onSeek: (timeSec: number) => void;
}

export function TimelineScrubber({
  currentTime,
  duration,
  disabled = false,
  onSeek,
}: TimelineScrubberProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [hoverFrac, setHoverFrac] = useState<number | null>(null);

  const safeDuration = Math.max(1, duration);
  const progressFrac = Math.max(0, Math.min(1, currentTime / safeDuration));
  const progressPct = progressFrac * 100;

  function updateHover(clientX: number) {
    const el = trackRef.current;
    if (!el || disabled) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    setHoverFrac(frac);
  }

  const hoverTime = hoverFrac !== null ? hoverFrac * safeDuration : null;

  return (
    <div
      ref={trackRef}
      onMouseMove={(e) => updateHover(e.clientX)}
      onMouseLeave={() => setHoverFrac(null)}
      className={cn(
        "group relative flex-1 min-w-44 h-7 flex items-center select-none",
        disabled && "opacity-40 pointer-events-none",
      )}
    >
      {/* Hover time tooltip */}
      {hoverTime !== null && hoverFrac !== null && (
        <div
          data-testid="viewer-seek-tooltip"
          style={{ left: `${hoverFrac * 100}%` }}
          className="pointer-events-none absolute -top-7 -translate-x-1/2 rounded-md bg-zinc-900/95 border border-zinc-700/80 px-1.5 py-0.5 text-[10px] font-mono text-zinc-100 shadow-lg whitespace-nowrap"
        >
          {formatTooltipTime(hoverTime)}
        </div>
      )}

      {/* Track background */}
      <div className="relative w-full h-1.5 group-hover:h-2 transition-all rounded-full bg-zinc-800/95 overflow-visible">
        {/* Hover preview fill */}
        {hoverFrac !== null && (
          <div
            style={{ width: `${hoverFrac * 100}%` }}
            className="absolute inset-y-0 left-0 rounded-full bg-white/15 pointer-events-none"
          />
        )}

        {/* Filled progress */}
        <div
          style={{ width: `${progressPct}%` }}
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-red-500 via-indigo-400 to-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.55)] pointer-events-none"
        />

        {/* Thumb */}
        <div
          style={{ left: `${progressPct}%` }}
          className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2 h-3.5 w-3.5 rounded-full bg-white shadow-[0_0_10px_rgba(96,165,250,0.9)] ring-2 ring-blue-400/50 transition-transform group-hover:scale-115"
        />
      </div>

      {/* Native range input overlay for drag, keyboard, a11y, and E2E control */}
      <input
        data-testid="viewer-seek"
        type="range"
        aria-label="Seek timeline"
        min={0}
        max={safeDuration}
        step={0.01}
        value={Math.min(currentTime, safeDuration)}
        disabled={disabled}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer m-0"
      />
    </div>
  );
}
