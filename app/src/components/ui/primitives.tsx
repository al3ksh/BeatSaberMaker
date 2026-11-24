import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, X } from "lucide-react";

export function cn(...inputs: Array<string | undefined | null | false>): string {
  return twMerge(clsx(inputs));
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
}

export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  const variants = {
    primary:
      "bg-zinc-100 text-zinc-950 border border-zinc-100 hover:bg-white hover:border-white active:bg-zinc-200 active:border-zinc-200",
    secondary:
      "bg-transparent text-zinc-200 border border-zinc-700 hover:bg-zinc-800/70 hover:border-zinc-600 hover:text-zinc-50 active:bg-zinc-800",
    danger:
      "bg-transparent text-red-300 border border-red-500/50 hover:bg-red-500/10 hover:border-red-400/70 hover:text-red-200 active:bg-red-500/15",
    ghost:
      "bg-transparent text-zinc-400 border border-transparent hover:bg-zinc-800/60 hover:text-zinc-100 active:bg-zinc-800",
  };
  const sizes = {
    sm: "h-8 px-3 text-xs rounded-md [&_svg]:h-3.5 [&_svg]:w-3.5",
    md: "h-9 px-3.5 text-sm rounded-lg [&_svg]:h-4 [&_svg]:w-4",
    lg: "h-10 px-4 text-sm rounded-lg [&_svg]:h-4 [&_svg]:w-4",
  };

  return (
    <button
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-medium whitespace-nowrap transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-900 disabled:opacity-40 disabled:pointer-events-none cursor-pointer",
        variants[variant],
        sizes[size],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Card({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl bg-zinc-900/80 border border-zinc-800/90 shadow-[0_10px_30px_-14px_rgba(0,0,0,0.65)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export type BadgeVariant =
  | "neutral"
  | "blue"
  | "sky"
  | "red"
  | "emerald"
  | "amber"
  | "purple";

export function Badge({
  variant = "neutral",
  className,
  children,
}: {
  variant?: BadgeVariant;
  className?: string;
  children: React.ReactNode;
}) {
  const styles: Record<BadgeVariant, string> = {
    neutral: "bg-zinc-800/90 text-zinc-300 border-zinc-700/80",
    blue: "bg-blue-500/15 text-blue-300 border-blue-500/30",
    sky: "bg-sky-500/15 text-sky-300 border-sky-500/30",
    red: "bg-red-500/15 text-red-300 border-red-500/30",
    emerald: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    amber: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    purple: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        styles[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
  accent?: "blue" | "red" | "gradient";
}

function accentIndicatorClass(accent?: "blue" | "red" | "gradient"): string {
  if (accent === "red") {
    return "bg-gradient-to-r from-red-600 to-rose-500 border-red-400/35 shadow-md shadow-red-950/50";
  }
  if (accent === "gradient") {
    return "bg-gradient-to-r from-red-500/90 via-purple-500/90 to-blue-600/95 border-white/20 shadow-md shadow-blue-950/40";
  }
  return "bg-gradient-to-r from-blue-600 to-indigo-500 border-blue-400/35 shadow-md shadow-blue-950/50";
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  equalWidth = false,
  size = "md",
  className,
}: {
  options: Array<SegmentedOption<T>>;
  value: T;
  onChange: (next: T) => void;
  equalWidth?: boolean;
  size?: "sm" | "md";
  className?: string;
}) {
  const activeIdx = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const activeOpt = options[activeIdx];
  const btnRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [indicator, setIndicator] = useState<{ left: number; width: number }>({
    left: 0,
    width: 0,
  });

  useLayoutEffect(() => {
    if (equalWidth) return;
    function measure() {
      const btn = btnRefs.current[activeIdx];
      if (!btn) return;
      const nextLeft = btn.offsetLeft;
      const nextWidth = btn.offsetWidth;
      setIndicator((prev) =>
        prev.left === nextLeft && prev.width === nextWidth
          ? prev
          : { left: nextLeft, width: nextWidth },
      );
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [activeIdx, equalWidth, options]);

  const padClass = size === "sm" ? "px-3 py-1 text-xs" : "px-4 py-1.5 text-xs sm:text-sm";
  const radiusContainer = size === "sm" ? "rounded-xl" : "rounded-2xl";
  const radiusPill = size === "sm" ? "rounded-lg" : "rounded-xl";

  return (
    <div
      className={cn(
        "relative bg-zinc-900/90 border border-zinc-800/90 p-1 shadow-inner shadow-black/40",
        radiusContainer,
        equalWidth ? "grid w-full" : "inline-flex items-center gap-1",
        className,
      )}
      style={
        equalWidth
          ? { gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }
          : undefined
      }
    >
      {equalWidth ? (
        <motion.div
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute top-1 bottom-1 border",
            radiusPill,
            accentIndicatorClass(activeOpt?.accent),
          )}
          style={{ width: `calc(${100 / options.length}% - 8px)` }}
          initial={false}
          animate={{ left: `calc(${(activeIdx * 100) / options.length}% + 4px)` }}
          transition={{ type: "spring", stiffness: 480, damping: 36 }}
        />
      ) : (
        indicator.width > 0 && (
          <motion.div
            aria-hidden="true"
            className={cn(
              "pointer-events-none absolute top-1 bottom-1 left-0 border",
              radiusPill,
              accentIndicatorClass(activeOpt?.accent),
            )}
            initial={false}
            animate={{ x: indicator.left, width: indicator.width }}
            transition={{ type: "spring", stiffness: 480, damping: 36 }}
          />
        )
      )}

      {options.map((opt, idx) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            ref={(el) => {
              btnRefs.current[idx] = el;
            }}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "relative z-10 inline-flex items-center justify-center gap-1.5 font-medium transition-colors duration-150 cursor-pointer select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
              radiusPill,
              padClass,
              active
                ? "text-white"
                : "text-zinc-400 hover:text-zinc-200",
              !equalWidth && active && indicator.width === 0
                ? accentIndicatorClass(opt.accent)
                : "",
            )}
          >
            {opt.icon}
            <span>{opt.label}</span>
            {opt.badge}
          </button>
        );
      })}
    </div>
  );
}

export interface SelectOption {
  value: string;
  label: string;
}

export function Select({
  value,
  options,
  onChange,
  ariaLabel,
  className,
}: {
  value: string;
  options: SelectOption[];
  onChange: (next: string) => void;
  ariaLabel?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((o) => o.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    function onPointerDown(ev: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(ev.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "w-full flex items-center justify-between gap-2 rounded-xl bg-zinc-950/90 border px-3.5 py-2 text-sm text-zinc-100 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
          open ? "border-blue-500/80 ring-2 ring-blue-500/20" : "border-zinc-800 hover:border-zinc-700",
        )}
      >
        <span className="truncate">{selected?.label ?? value}</span>
        <ChevronDown
          className={cn(
            "h-4 w-4 text-zinc-400 shrink-0 transition-transform duration-150",
            open && "rotate-180 text-blue-400",
          )}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            aria-label={ariaLabel}
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.12 }}
            className="absolute z-40 left-0 right-0 bottom-full mb-1.5 max-h-52 overflow-y-auto rounded-xl bg-zinc-900 border border-zinc-700/80 p-1 shadow-2xl shadow-black/70"
          >
            {options.map((opt) => {
              const isSelected = opt.value === value;
              return (
                <li key={opt.value} role="option" aria-selected={isSelected}>
                  <button
                    type="button"
                    onClick={() => {
                      onChange(opt.value);
                      setOpen(false);
                    }}
                    className={cn(
                      "w-full flex items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer",
                      isSelected
                        ? "bg-blue-600/20 text-blue-200 font-medium"
                        : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100",
                    )}
                  >
                    <span className="truncate">{opt.label}</span>
                    {isSelected && <Check className="h-3.5 w-3.5 text-blue-400 shrink-0" />}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function Input({ className, ...rest }, ref) {
  return (
    <input
      ref={ref}
      className={cn(
        "w-full rounded-xl bg-zinc-950/90 border border-zinc-800 px-3.5 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 transition-colors focus:outline-none focus:border-blue-500/80 focus:ring-2 focus:ring-blue-500/20",
        className,
      )}
      {...rest}
    />
  );
});

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.5,
  displayValue,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  displayValue?: string;
  onChange: (next: number) => void;
}) {
  const id = React.useId();
  const pct = max > min ? Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100)) : 0;

  return (
    <div className="space-y-2 rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3.5 py-3">
      <div className="flex items-center justify-between text-xs">
        <label htmlFor={id} className="text-zinc-300 font-medium">
          {label}
        </label>
        <span className="font-mono font-semibold text-blue-300 bg-blue-500/15 border border-blue-500/30 rounded-lg px-2 py-0.5">
          {displayValue ?? value}
        </span>
      </div>

      <div className="relative h-6 flex items-center group">
        <div className="w-full h-2 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-red-500 via-purple-500 to-blue-500 transition-[width] duration-75"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div
          className="pointer-events-none absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-4 rounded-full bg-white border-2 border-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.65)] transition-transform duration-100 group-hover:scale-110"
          style={{ left: `${pct}%` }}
        />
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
      </div>

      <div className="flex justify-between text-[10px] font-mono text-zinc-500">
        <span>{min}</span>
        <span>{((min + max) / 2).toFixed(1)}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export function Checkbox({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
}) {
  const id = React.useId();
  return (
    <label htmlFor={id} className="flex items-start gap-3 cursor-pointer select-none group">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="sr-only"
      />
      <span
        aria-hidden="true"
        className={cn(
          "mt-0.5 h-5 w-5 rounded-md border flex items-center justify-center shrink-0 transition-all duration-150",
          checked
            ? "bg-gradient-to-br from-blue-500 to-indigo-500 border-blue-400/60 shadow-sm shadow-blue-950/60"
            : "bg-zinc-950 border-zinc-700 group-hover:border-zinc-500",
        )}
      >
        <Check
          className={cn(
            "h-3.5 w-3.5 text-white transition-transform duration-150",
            checked ? "scale-100 opacity-100" : "scale-50 opacity-0",
          )}
          strokeWidth={3}
        />
      </span>
      <span className="text-sm text-zinc-200 leading-tight">
        {label}
        {hint && <span className="block text-xs text-zinc-400 mt-0.5">{hint}</span>}
      </span>
    </label>
  );
}

export function ProgressBar({
  pct,
  className,
}: {
  pct: number;
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div
      className={cn(
        "h-3 w-full overflow-hidden rounded-full bg-zinc-950 border border-zinc-800 p-0.5",
        className,
      )}
      role="progressbar"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <motion.div
        className="h-full rounded-full bg-gradient-to-r from-red-500 via-purple-500 to-blue-500"
        initial={false}
        animate={{ width: `${clamped}%` }}
        transition={{ duration: 0.22, ease: "easeOut" }}
      />
    </div>
  );
}

export function Modal({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.key === "Escape") {
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) {
              onClose();
            }
          }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          role="dialog"
          aria-modal="true"
          aria-label={title}
        >
          <motion.div
            className="w-full max-w-lg rounded-2xl bg-zinc-900 border border-zinc-800 shadow-2xl overflow-hidden"
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.16 }}
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800 bg-zinc-950/50">
              <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="p-5 max-h-[80vh] overflow-y-auto">{children}</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
