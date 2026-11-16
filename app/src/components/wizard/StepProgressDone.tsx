import { useState } from "react";
import {
  AlertTriangle,
  Archive,
  Check,
  CheckCircle2,
  FileText,
  FolderOpen,
  Info,
  Loader2,
  RotateCcw,
} from "lucide-react";
import { ENVIRONMENT_LABELS_EN, formatErrorEn, EN, RANK_LABELS_EN } from "../../i18n/en";
import type { GenerateOutput, ProgressEvent, StageName } from "../../types";
import { Badge, Button, Card, cn, ProgressBar } from "../ui/primitives";
import { rankToBadgeVariant } from "./StepMap";

export interface StepProgressDoneProps {
  hasDownloadStage: boolean;
  status: "running" | "done" | "error";
  progress: ProgressEvent | null;
  result: GenerateOutput | null;
  errorCode: string | null;
  onReset: () => void;
  onGoToLibrary?: () => void;
}

export function StepProgressDone({
  hasDownloadStage,
  status,
  progress,
  result,
  errorCode,
  onReset,
  onGoToLibrary,
}: StepProgressDoneProps) {
  const [cancelling, setCancelling] = useState(false);
  const [exportNote, setExportNote] = useState<string | null>(null);

  const stages: StageName[] = hasDownloadStage
    ? ["downloading", "analyzing", "notes", "layers", "writing"]
    : ["analyzing", "notes", "layers", "writing"];

  const currentStageIndex = progress ? stages.indexOf(progress.stage) : 0;

  async function handleCancel() {
    setCancelling(true);
    try {
      await window.bsm.cancelActive();
    } finally {
      setCancelling(false);
    }
  }

  async function handleExportZip() {
    if (!result) return;
    setExportNote(null);
    try {
      await window.bsm.exportMapZip(result.folder, result.title);
      setExportNote(EN.library.exportedSuccess);
    } catch {
      // Cancelled save dialog or error handled silently.
    }
  }

  if (status === "running") {
    const pct = progress?.pct ?? 0;
    return (
      <Card className="p-6 space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
              <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
            </div>
            <h2 className="text-lg font-semibold text-zinc-100">
              {EN.wizard.progress.title}
            </h2>
          </div>
          <span className="font-mono text-sm font-semibold text-blue-300 bg-blue-500/10 border border-blue-500/25 rounded-lg px-2.5 py-1">
            {pct}%
          </span>
        </div>

        <ProgressBar pct={pct} />

        <div className="space-y-2 rounded-xl bg-zinc-950/60 border border-zinc-800/80 p-4">
          {stages.map((st, idx) => {
            const done = idx < currentStageIndex;
            const active = idx === currentStageIndex;
            return (
              <div
                key={st}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-blue-500/10 text-zinc-100 font-medium"
                    : done
                      ? "text-emerald-400"
                      : "text-zinc-500",
                )}
              >
                {done ? (
                  <Check className="h-4 w-4 text-emerald-400 shrink-0" />
                ) : active ? (
                  <span className="h-2.5 w-2.5 rounded-full bg-blue-400 shadow-[0_0_8px_#60a5fa] animate-pulse shrink-0 mx-0.5" />
                ) : (
                  <span className="h-2 w-2 rounded-full bg-zinc-700 shrink-0 mx-1" />
                )}
                <span>{EN.wizard.progress.stages[st]}</span>
              </div>
            );
          })}
        </div>

        <div className="flex justify-end">
          <Button variant="danger" onClick={handleCancel} disabled={cancelling}>
            {cancelling ? EN.wizard.progress.cancelling : EN.common.cancel}
          </Button>
        </div>
      </Card>
    );
  }

  if (status === "error") {
    const code = errorCode ?? "internal";
    return (
      <Card className="p-6 space-y-5 border-red-500/25">
        <div className="flex items-start gap-3.5">
          <div className="h-10 w-10 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-center shrink-0">
            <AlertTriangle className="h-5 w-5 text-red-400" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium text-red-200 leading-relaxed">
              {formatErrorEn(code)}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap justify-end gap-2">
          {code === "internal" && (
            <Button variant="secondary" onClick={() => window.bsm.openLogs()}>
              <FileText className="h-4 w-4" />
              {EN.settings.openLogs}
            </Button>
          )}
          <Button onClick={onReset}>
            <RotateCcw className="h-4 w-4" />
            {EN.wizard.done.createAnother}
          </Button>
        </div>
      </Card>
    );
  }

  if (!result) {
    return null;
  }

  const infoNotices: string[] = [];
  const hasLayerMissingWarning = result.warnings.some(
    (w) => /checkpoint|layer_stats/i.test(w),
  );
  if (hasLayerMissingWarning) {
    infoNotices.push(EN.wizard.done.warnings.missingLayerModels);
  }
  if (result.used_rule_lights) {
    infoNotices.push(EN.wizard.done.warnings.ruleLights);
  }
  if (!result.grid_steady) {
    infoNotices.push(EN.wizard.done.warnings.unsteadyGrid);
  }

  return (
    <Card className="p-6 space-y-5 overflow-y-auto">
      <div className="flex items-start gap-3.5">
        <div className="h-11 w-11 rounded-xl bg-emerald-500/15 border border-emerald-500/30 shadow-[0_0_20px_-6px_rgba(16,185,129,0.6)] flex items-center justify-center shrink-0">
          <CheckCircle2 className="h-6 w-6 text-emerald-400" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">
            {EN.wizard.done.successTitle}
          </h2>
          <p className="text-sm text-zinc-400 mt-0.5">
            {result.title} — {result.artist}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {result.levels.map((lv) => (
          <Badge key={lv.rank} variant={rankToBadgeVariant(lv.rank)}>
            {RANK_LABELS_EN[lv.rank] ?? lv.rank}: {lv.n_notes} ({lv.nps.toFixed(1)} NPS)
          </Badge>
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">{EN.wizard.done.bpmLabel}</div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {Math.round(result.bpm)}
          </div>
        </div>
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">{EN.wizard.done.wallsLabel}</div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {result.n_walls}
          </div>
        </div>
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">{EN.wizard.done.bombsLabel}</div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {result.n_bombs}
          </div>
        </div>
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">{EN.wizard.done.lightsLabel}</div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {result.n_lights}
          </div>
        </div>
        <div className="col-span-2 sm:col-span-1 rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">{EN.wizard.map.environmentLabel}</div>
          <div className="text-xs font-medium text-zinc-200 truncate mt-0.5">
            {ENVIRONMENT_LABELS_EN[result.environment] ?? result.environment}
          </div>
        </div>
      </div>

      {infoNotices.length > 0 && (
        <div className="rounded-xl bg-blue-500/10 border border-blue-500/25 p-4 space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-300">
            <Info className="h-3.5 w-3.5" />
            <span>{EN.wizard.done.warningsTitle}</span>
          </div>
          {infoNotices.map((notice) => (
            <p key={notice} className="text-xs text-blue-200/90">
              {notice}
            </p>
          ))}
        </div>
      )}

      {exportNote && <p className="text-xs text-emerald-400">{exportNote}</p>}

      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-zinc-800/80">
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            onClick={() => window.bsm.openPath(result.folder)}
          >
            <FolderOpen className="h-4 w-4" />
            {EN.wizard.done.openFolder}
          </Button>
          <Button variant="secondary" onClick={handleExportZip}>
            <Archive className="h-4 w-4" />
            {EN.wizard.done.exportZip}
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          {onGoToLibrary && (
            <Button variant="ghost" onClick={onGoToLibrary}>
              {EN.wizard.done.goToLibrary}
            </Button>
          )}
          <Button onClick={onReset}>
            <RotateCcw className="h-4 w-4" />
            {EN.wizard.done.createAnother}
          </Button>
        </div>
      </div>
    </Card>
  );
}
