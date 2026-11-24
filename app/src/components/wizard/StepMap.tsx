import { useState } from "react";
import {
  ArrowLeft,
  Check,
  ChevronDown,
  ChevronRight,
  Dices,
  Layers,
  Sliders,
  Sparkles,
} from "lucide-react";
import {
  ENVIRONMENT_LABELS_EN,
  EN,
  RANK_LABELS_EN,
  STYLE_LABELS_EN,
} from "../../i18n/en";
import type { MapGenerationConfig } from "../../types";
import {
  Badge,
  type BadgeVariant,
  Button,
  Card,
  Checkbox,
  cn,
  Input,
  SegmentedControl,
  Select,
  Slider,
} from "../ui/primitives";

const ALL_RANKS = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"] as const;
const ALL_STYLES = [
  "tech",
  "dance",
  "speed",
  "accuracy",
  "balanced",
  "challenge",
  "fitness",
] as const;

export const PACK_DEFAULT_CONFIG: MapGenerationConfig = {
  mode: "pack",
  levels: ["Hard", "Expert", "ExpertPlus"],
  nps: null,
  styles: [],
  ornaments: null,
  lights: 2,
  environment: "DefaultEnvironment",
  seed: 0,
  lightsRules: false,
};

const RANK_THEME: Record<string, { dot: string; active: string }> = {
  Easy: {
    dot: "bg-emerald-400",
    active: "border-emerald-500/60 bg-emerald-500/12 text-emerald-100",
  },
  Normal: {
    dot: "bg-sky-400",
    active: "border-sky-500/60 bg-sky-500/12 text-sky-100",
  },
  Hard: {
    dot: "bg-amber-400",
    active: "border-amber-500/60 bg-amber-500/12 text-amber-100",
  },
  Expert: {
    dot: "bg-red-400",
    active: "border-red-500/60 bg-red-500/12 text-red-100",
  },
  ExpertPlus: {
    dot: "bg-purple-400",
    active: "border-purple-500/60 bg-purple-500/12 text-purple-100",
  },
};

export function rankToBadgeVariant(rank: string): BadgeVariant {
  switch (rank) {
    case "Easy":
      return "emerald";
    case "Normal":
      return "sky";
    case "Hard":
      return "amber";
    case "Expert":
      return "red";
    case "ExpertPlus":
      return "purple";
    default:
      return "blue";
  }
}

export interface StepMapProps {
  config: MapGenerationConfig;
  isRegenerate: boolean;
  isSubmitting?: boolean;
  onChange: (next: MapGenerationConfig) => void;
  onBack: () => void;
  onGenerate: () => void;
}

const ORNAMENT_OPTIONS = [
  { value: "auto", label: EN.wizard.map.ornamentsAuto },
  { value: "0", label: "0" },
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
];

const LIGHT_OPTIONS = [
  { value: "0", label: "0" },
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
];

const ENVIRONMENT_OPTIONS = Object.entries(ENVIRONMENT_LABELS_EN).map(
  ([value, label]) => ({ value, label }),
);

export function StepMap({
  config,
  isRegenerate,
  isSubmitting = false,
  onChange,
  onBack,
  onGenerate,
}: StepMapProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const selectedLevels = config.levels ?? ["Hard", "Expert", "ExpertPlus"];
  const isSingleNps = config.mode === "custom" && config.nps !== null;

  function toggleLevel(rank: string) {
    const has = selectedLevels.includes(rank);
    if (has && selectedLevels.length <= 1) return;
    const nextLevels = has
      ? selectedLevels.filter((r) => r !== rank)
      : ALL_RANKS.filter((r) => r === rank || selectedLevels.includes(r));
    onChange({ ...config, levels: nextLevels, nps: null });
  }

  function toggleStyle(style: string) {
    const has = config.styles.includes(style);
    const nextStyles = has
      ? config.styles.filter((s) => s !== style)
      : [...config.styles, style];
    onChange({ ...config, styles: nextStyles });
  }

  function randomizeSeed() {
    const nextSeed = Math.floor(Math.random() * 2_000_000_000) + 1;
    onChange({ ...config, seed: nextSeed });
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-3.5">
      <div className="shrink-0 flex justify-center sm:justify-start">
        <SegmentedControl
          value={config.mode}
          onChange={(nextMode) => {
            if (nextMode === "pack") {
              onChange({
                ...PACK_DEFAULT_CONFIG,
                seed: config.seed,
                lightsRules: config.lightsRules,
              });
            } else {
              onChange({
                ...config,
                mode: "custom",
                levels: config.levels ?? ["Hard", "Expert", "ExpertPlus"],
              });
            }
          }}
          options={[
            {
              value: "pack",
              label: EN.wizard.map.modePack,
              icon: <Sparkles className="h-4 w-4" />,
              accent: "blue",
            },
            {
              value: "custom",
              label: EN.wizard.map.modeCustom,
              icon: <Sliders className="h-4 w-4" />,
              accent: "gradient",
            },
          ]}
        />
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto pr-0.5 space-y-3">
        {config.mode === "pack" ? (
          <Card className="p-5 space-y-4">
            <div className="flex items-start gap-3.5">
              <div className="h-10 w-10 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center justify-center shrink-0">
                <Sparkles className="h-5 w-5 text-blue-400" />
              </div>
              <div className="space-y-1">
                <div className="text-base font-semibold text-zinc-100">
                  {EN.wizard.map.packSummaryTitle}
                </div>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  {EN.wizard.map.packSummaryDesc}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Badge variant="amber">Hard</Badge>
              <Badge variant="red">Expert</Badge>
              <Badge variant="purple">Expert+</Badge>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3.5 py-2.5">
                <div className="text-[11px] text-zinc-500">
                  {EN.wizard.map.packSpecs.lights}
                </div>
                <div className="text-xs font-medium text-zinc-200 mt-0.5">
                  {EN.wizard.map.packSpecs.lightsValue}
                </div>
              </div>
              <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3.5 py-2.5">
                <div className="text-[11px] text-zinc-500">
                  {EN.wizard.map.packSpecs.ornaments}
                </div>
                <div className="text-xs font-medium text-zinc-200 mt-0.5">
                  {EN.wizard.map.packSpecs.ornamentsValue}
                </div>
              </div>
              <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3.5 py-2.5">
                <div className="text-[11px] text-zinc-500">
                  {EN.wizard.map.packSpecs.environment}
                </div>
                <div className="text-xs font-medium text-zinc-200 mt-0.5">
                  {EN.wizard.map.packSpecs.environmentValue}
                </div>
              </div>
            </div>
          </Card>
        ) : (
          <Card className="p-4 space-y-3.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-medium text-zinc-400">
                {EN.wizard.map.customModeLabel}
              </span>
              <SegmentedControl
                size="sm"
                value={isSingleNps ? "nps" : "levels"}
                onChange={(next) => {
                  if (next === "levels") {
                    onChange({
                      ...config,
                      nps: null,
                      levels: config.levels ?? ["Hard", "Expert", "ExpertPlus"],
                    });
                  } else {
                    onChange({
                      ...config,
                      nps: config.nps ?? 4.0,
                      levels: null,
                    });
                  }
                }}
                options={[
                  {
                    value: "levels",
                    label: EN.wizard.map.modeLevels,
                    icon: <Layers className="h-3.5 w-3.5" />,
                  },
                  {
                    value: "nps",
                    label: EN.wizard.map.modeSingleNps,
                  },
                ]}
              />
            </div>

            {isSingleNps ? (
              <Slider
                label={EN.wizard.map.npsLabel}
                value={config.nps ?? 4.0}
                min={0.5}
                max={12}
                step={0.5}
                displayValue={`${(config.nps ?? 4.0).toFixed(1)} NPS`}
                onChange={(nps) => onChange({ ...config, nps, levels: null })}
              />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {ALL_RANKS.map((rank) => {
                  const active = selectedLevels.includes(rank);
                  const theme = RANK_THEME[rank];
                  return (
                    <button
                      key={rank}
                      type="button"
                      aria-pressed={active}
                      onClick={() => toggleLevel(rank)}
                      className={cn(
                        "flex items-center justify-between gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition-all cursor-pointer",
                        active
                          ? theme.active
                          : "border-zinc-800 bg-zinc-950/70 text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200",
                      )}
                    >
                      <span className="flex items-center gap-2">
                        <span className={cn("h-2 w-2 rounded-full shrink-0", theme.dot)} />
                        <span>{RANK_LABELS_EN[rank]}</span>
                      </span>
                      {active && <Check className="h-3.5 w-3.5 shrink-0 opacity-90" />}
                    </button>
                  );
                })}
              </div>
            )}

            <div className="space-y-1.5">
              <span className="block text-xs font-medium text-zinc-400">
                {EN.wizard.map.stylesLabel}
              </span>
              <div className="flex flex-wrap gap-1.5">
                {ALL_STYLES.map((st) => {
                  const active = config.styles.includes(st);
                  return (
                    <button
                      key={st}
                      type="button"
                      aria-pressed={active}
                      onClick={() => toggleStyle(st)}
                      className={cn(
                        "inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-all cursor-pointer",
                        active
                          ? "border-red-500/70 bg-red-500/15 text-red-100 shadow-sm shadow-red-950/40"
                          : "border-zinc-800 bg-zinc-950/70 text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200",
                      )}
                    >
                      <span
                        className={cn(
                          "h-1.5 w-1.5 rounded-full transition-colors",
                          active ? "bg-red-400" : "bg-zinc-600",
                        )}
                      />
                      <span>{STYLE_LABELS_EN[st]}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div className="space-y-1">
                <span className="block text-xs font-medium text-zinc-400">
                  {EN.wizard.map.ornamentsLabel}
                </span>
                <SegmentedControl
                  equalWidth
                  size="sm"
                  value={config.ornaments === null ? "auto" : String(config.ornaments)}
                  onChange={(val) =>
                    onChange({
                      ...config,
                      ornaments: val === "auto" ? null : (Number(val) as 0 | 1 | 2 | 3),
                    })
                  }
                  options={ORNAMENT_OPTIONS}
                />
              </div>

              <div className="space-y-1">
                <span className="block text-xs font-medium text-zinc-400">
                  {EN.wizard.map.lightsLabel}
                </span>
                <SegmentedControl
                  equalWidth
                  size="sm"
                  value={String(config.lights)}
                  onChange={(val) =>
                    onChange({ ...config, lights: Number(val) as 0 | 1 | 2 | 3 })
                  }
                  options={LIGHT_OPTIONS}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div className="space-y-1 text-xs">
                <span className="block text-zinc-400 font-medium">
                  {EN.wizard.map.environmentLabel}
                </span>
                <Select
                  ariaLabel={EN.wizard.map.environmentLabel}
                  value={config.environment}
                  options={ENVIRONMENT_OPTIONS}
                  onChange={(environment) => onChange({ ...config, environment })}
                />
              </div>

              <div className="space-y-1 text-xs">
                <label htmlFor="bsm-seed-input" className="block text-zinc-400 font-medium">
                  {EN.wizard.map.seedLabel}
                </label>
                <div className="flex gap-2">
                  <Input
                    id="bsm-seed-input"
                    type="number"
                    value={config.seed}
                    onChange={(e) =>
                      onChange({ ...config, seed: Number(e.target.value) || 0 })
                    }
                    className="py-2"
                  />
                  <Button variant="secondary" size="sm" onClick={randomizeSeed} className="shrink-0">
                    <Dices className="h-4 w-4" />
                    {EN.wizard.map.randomSeedButton}
                  </Button>
                </div>
              </div>
            </div>

            <div className="border-t border-zinc-800/80 pt-2.5">
              <button
                type="button"
                onClick={() => setShowAdvanced((v) => !v)}
                className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors cursor-pointer"
              >
                {showAdvanced ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
                {EN.wizard.map.advanced}
              </button>
              {showAdvanced && (
                <div className="mt-2.5 rounded-xl bg-zinc-950/60 border border-zinc-800/80 p-3">
                  <Checkbox
                    label={EN.wizard.map.lightsRules}
                    hint={EN.wizard.map.lightsRulesHint}
                    checked={config.lightsRules}
                    onChange={(lightsRules) => onChange({ ...config, lightsRules })}
                  />
                </div>
              )}
            </div>
          </Card>
        )}
      </div>

      {isRegenerate && (
        <div className="shrink-0 rounded-xl bg-amber-500/10 border border-amber-500/30 px-3.5 py-2 text-xs text-amber-200">
          {EN.wizard.map.regenerateNotice}
        </div>
      )}

      <div className="shrink-0 flex items-center justify-between">
        <Button variant="secondary" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" />
          {EN.common.back}
        </Button>
        <Button size="lg" onClick={onGenerate} disabled={isSubmitting}>
          <Sparkles className="h-4 w-4" />
          {isRegenerate ? EN.wizard.map.queueRegenerate : EN.wizard.map.addToQueue}
        </Button>
      </div>
    </div>
  );
}
