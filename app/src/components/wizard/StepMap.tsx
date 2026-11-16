import { useState } from "react";
import { ChevronDown, ChevronRight, Dices, Sparkles } from "lucide-react";
import {
  ENVIRONMENT_LABELS_PL,
  PL,
  RANK_LABELS_PL,
  STYLE_LABELS_PL,
} from "../../i18n/pl";
import type { MapGenerationConfig } from "../../types";
import { Badge, Button, Card, Checkbox, Input, Slider } from "../ui/primitives";

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

export interface StepMapProps {
  config: MapGenerationConfig;
  isRegenerate: boolean;
  onChange: (next: MapGenerationConfig) => void;
  onBack: () => void;
  onGenerate: () => void;
}

export function StepMap({
  config,
  isRegenerate,
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
    <div className="space-y-5">
      <div className="inline-flex rounded-xl bg-zinc-900 border border-zinc-800 p-1">
        <button
          type="button"
          onClick={() =>
            onChange({
              ...PACK_DEFAULT_CONFIG,
              seed: config.seed,
              lightsRules: config.lightsRules,
            })
          }
          className={
            config.mode === "pack"
              ? "px-4 py-1.5 rounded-lg text-sm font-medium bg-blue-600 text-white"
              : "px-4 py-1.5 rounded-lg text-sm font-medium text-zinc-400 hover:text-zinc-200"
          }
        >
          {PL.wizard.map.modePack}
        </button>
        <button
          type="button"
          onClick={() =>
            onChange({
              ...config,
              mode: "custom",
              levels: config.levels ?? ["Hard", "Expert", "ExpertPlus"],
            })
          }
          className={
            config.mode === "custom"
              ? "px-4 py-1.5 rounded-lg text-sm font-medium bg-blue-600 text-white"
              : "px-4 py-1.5 rounded-lg text-sm font-medium text-zinc-400 hover:text-zinc-200"
          }
        >
          {PL.wizard.map.modeCustom}
        </button>
      </div>

      {config.mode === "pack" ? (
        <Card className="p-5 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            <Sparkles className="h-4 w-4 text-blue-400" />
            <span>{PL.wizard.map.packSummaryTitle}</span>
          </div>
          <p className="text-sm text-zinc-400">{PL.wizard.map.packSummaryDesc}</p>
          <div className="flex flex-wrap gap-2 pt-1">
            <Badge variant="blue">Hard</Badge>
            <Badge variant="blue">Expert</Badge>
            <Badge variant="red">Expert+</Badge>
          </div>
        </Card>
      ) : (
        <Card className="p-5 space-y-5">
          <div className="space-y-2">
            <span className="block text-xs font-medium text-zinc-400">
              {PL.wizard.map.customModeLabel}
            </span>
            <div className="flex gap-2">
              <Button
                variant={!isSingleNps ? "primary" : "secondary"}
                size="sm"
                onClick={() =>
                  onChange({
                    ...config,
                    nps: null,
                    levels: config.levels ?? ["Hard", "Expert", "ExpertPlus"],
                  })
                }
              >
                {PL.wizard.map.modeLevels}
              </Button>
              <Button
                variant={isSingleNps ? "primary" : "secondary"}
                size="sm"
                onClick={() =>
                  onChange({
                    ...config,
                    nps: config.nps ?? 4.0,
                    levels: null,
                  })
                }
              >
                {PL.wizard.map.modeSingleNps}
              </Button>
            </div>
          </div>

          {isSingleNps ? (
            <Slider
              label={PL.wizard.map.npsLabel}
              value={config.nps ?? 4.0}
              min={0.5}
              max={12}
              step={0.5}
              displayValue={`${(config.nps ?? 4.0).toFixed(1)} NPS`}
              onChange={(nps) => onChange({ ...config, nps, levels: null })}
            />
          ) : (
            <div className="flex flex-wrap gap-2">
              {ALL_RANKS.map((rank) => {
                const active = selectedLevels.includes(rank);
                return (
                  <button
                    key={rank}
                    type="button"
                    onClick={() => toggleLevel(rank)}
                    className={
                      active
                        ? "rounded-lg border border-blue-500 bg-blue-500/15 px-3 py-1.5 text-xs font-medium text-blue-300"
                        : "rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800"
                    }
                  >
                    {RANK_LABELS_PL[rank]}
                  </button>
                );
              })}
            </div>
          )}

          <div className="space-y-2">
            <span className="block text-xs font-medium text-zinc-400">
              {PL.wizard.map.stylesLabel}
            </span>
            <div className="flex flex-wrap gap-2">
              {ALL_STYLES.map((st) => {
                const active = config.styles.includes(st);
                return (
                  <button
                    key={st}
                    type="button"
                    onClick={() => toggleStyle(st)}
                    className={
                      active
                        ? "rounded-full border border-red-500 bg-red-500/15 px-3 py-1 text-xs font-medium text-red-300"
                        : "rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800"
                    }
                  >
                    {STYLE_LABELS_PL[st]}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <span className="block text-xs font-medium text-zinc-400">
                {PL.wizard.map.ornamentsLabel}
              </span>
              <div className="flex gap-1.5">
                {([null, 0, 1, 2, 3] as const).map((val) => {
                  const active = config.ornaments === val;
                  return (
                    <button
                      key={String(val)}
                      type="button"
                      onClick={() => onChange({ ...config, ornaments: val })}
                      className={
                        active
                          ? "flex-1 rounded-lg border border-blue-500 bg-blue-500/15 py-1.5 text-xs font-medium text-blue-300"
                          : "flex-1 rounded-lg border border-zinc-800 bg-zinc-950 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800"
                      }
                    >
                      {val === null ? PL.wizard.map.ornamentsAuto : val}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-1.5">
              <span className="block text-xs font-medium text-zinc-400">
                {PL.wizard.map.lightsLabel}
              </span>
              <div className="flex gap-1.5">
                {([0, 1, 2, 3] as const).map((val) => {
                  const active = config.lights === val;
                  return (
                    <button
                      key={val}
                      type="button"
                      onClick={() => onChange({ ...config, lights: val })}
                      className={
                        active
                          ? "flex-1 rounded-lg border border-blue-500 bg-blue-500/15 py-1.5 text-xs font-medium text-blue-300"
                          : "flex-1 rounded-lg border border-zinc-800 bg-zinc-950 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800"
                      }
                    >
                      {val}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <label className="space-y-1.5 text-xs">
              <span className="block text-zinc-400 font-medium">
                {PL.wizard.map.environmentLabel}
              </span>
              <select
                value={config.environment}
                onChange={(e) => onChange({ ...config, environment: e.target.value })}
                className="w-full rounded-lg bg-zinc-950 border border-zinc-800 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-blue-500"
              >
                {Object.entries(ENVIRONMENT_LABELS_PL).map(([envKey, label]) => (
                  <option key={envKey} value={envKey}>
                    {label}
                  </option>
                ))}
              </select>
            </label>

            <div className="space-y-1.5 text-xs">
              <label htmlFor="bsm-seed-input" className="block text-zinc-400 font-medium">
                {PL.wizard.map.seedLabel}
              </label>
              <div className="flex gap-2">
                <Input
                  id="bsm-seed-input"
                  type="number"
                  value={config.seed}
                  onChange={(e) =>
                    onChange({ ...config, seed: Number(e.target.value) || 0 })
                  }
                />
                <Button variant="secondary" size="sm" onClick={randomizeSeed}>
                  <Dices className="h-4 w-4" />
                  {PL.wizard.map.randomSeedButton}
                </Button>
              </div>
            </div>
          </div>

          <div className="border-t border-zinc-800 pt-3">
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200"
            >
              {showAdvanced ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              {PL.wizard.map.advanced}
            </button>
            {showAdvanced && (
              <div className="mt-3 pl-2">
                <Checkbox
                  label={PL.wizard.map.lightsRules}
                  hint={PL.wizard.map.lightsRulesHint}
                  checked={config.lightsRules}
                  onChange={(lightsRules) => onChange({ ...config, lightsRules })}
                />
              </div>
            )}
          </div>
        </Card>
      )}

      {isRegenerate && (
        <div className="rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-3 text-xs text-amber-200">
          {PL.wizard.map.regenerateNotice}
        </div>
      )}

      <div className="flex items-center justify-between">
        <Button variant="secondary" onClick={onBack}>
          {PL.common.back}
        </Button>
        <Button size="lg" onClick={onGenerate}>
          {PL.wizard.map.generate}
        </Button>
      </div>
    </div>
  );
}
