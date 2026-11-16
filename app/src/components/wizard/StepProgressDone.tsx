import { useState } from "react";
import {
  AlertTriangle,
  Archive,
  CheckCircle2,
  FileText,
  FolderOpen,
  Info,
  Loader2,
  RotateCcw,
} from "lucide-react";
import { ENVIRONMENT_LABELS_PL, formatErrorPl, PL, RANK_LABELS_PL } from "../../i18n/pl";
import type { GenerateOutput, ProgressEvent, StageName } from "../../types";
import { Badge, Button, Card, ProgressBar } from "../ui/primitives";

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
      setExportNote(PL.library.exportedSuccess);
    } catch {
      // Cancelled save dialog or error handled silently.
    }
  }

  if (status === "running") {
    const pct = progress?.pct ?? 0;
    return (
      <Card className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
            <h2 className="text-lg font-semibold">{PL.wizard.progress.title}</h2>
          </div>
          <span className="font-mono text-sm text-blue-400">{pct}%</span>
        </div>

        <ProgressBar pct={pct} />

        <div className="space-y-2">
          {stages.map((st, idx) => {
            const done = idx < currentStageIndex;
            const active = idx === currentStageIndex;
            return (
              <div
                key={st}
                className={
                  active
                    ? "flex items-center gap-2.5 text-sm text-zinc-100 font-medium"
                    : done
                      ? "flex items-center gap-2.5 text-sm text-emerald-400"
                      : "flex items-center gap-2.5 text-sm text-zinc-500"
                }
              >
                <span
                  className={
                    active
                      ? "h-2 w-2 rounded-full bg-blue-400"
                      : done
                        ? "h-2 w-2 rounded-full bg-emerald-400"
                        : "h-2 w-2 rounded-full bg-zinc-700"
                  }
                />
                <span>{PL.wizard.progress.stages[st]}</span>
              </div>
            );
          })}
        </div>

        <div className="flex justify-end">
          <Button variant="danger" onClick={handleCancel} disabled={cancelling}>
            {cancelling ? PL.wizard.progress.cancelling : PL.common.cancel}
          </Button>
        </div>
      </Card>
    );
  }

  if (status === "error") {
    const code = errorCode ?? "internal";
    return (
      <Card className="p-6 space-y-5">
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-center shrink-0">
            <AlertTriangle className="h-5 w-5 text-red-400" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium text-red-200">{formatErrorPl(code)}</p>
          </div>
        </div>

        <div className="flex flex-wrap justify-end gap-2">
          {code === "internal" && (
            <Button variant="secondary" onClick={() => window.bsm.openLogs()}>
              <FileText className="h-4 w-4" />
              {PL.settings.openLogs}
            </Button>
          )}
          <Button onClick={onReset}>
            <RotateCcw className="h-4 w-4" />
            {PL.wizard.done.createAnother}
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
    infoNotices.push(PL.wizard.done.warnings.missingLayerModels);
  }
  if (result.used_rule_lights) {
    infoNotices.push(PL.wizard.done.warnings.ruleLights);
  }
  if (!result.grid_steady) {
    infoNotices.push(PL.wizard.done.warnings.unsteadyGrid);
  }

  return (
    <Card className="p-6 space-y-6">
      <div className="flex items-start gap-3">
        <div className="h-10 w-10 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center shrink-0">
          <CheckCircle2 className="h-5 w-5 text-emerald-400" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">
            {PL.wizard.done.successTitle}
          </h2>
          <p className="text-sm text-zinc-400 mt-0.5">
            {result.title} — {result.artist}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {result.levels.map((lv) => (
          <Badge key={lv.rank} variant="blue">
            {RANK_LABELS_PL[lv.rank] ?? lv.rank}: {lv.n_notes} ({lv.nps.toFixed(1)} NPS)
          </Badge>
        ))}
        <Badge>
          {PL.wizard.done.bpmLabel}: {Math.round(result.bpm)}
        </Badge>
        <Badge>
          {PL.wizard.done.wallsLabel}: {result.n_walls}
        </Badge>
        <Badge>
          {PL.wizard.done.bombsLabel}: {result.n_bombs}
        </Badge>
        <Badge>
          {PL.wizard.done.lightsLabel}: {result.n_lights}
        </Badge>
        <Badge>
          {ENVIRONMENT_LABELS_PL[result.environment] ?? result.environment}
        </Badge>
      </div>

      {infoNotices.length > 0 && (
        <div className="rounded-xl bg-blue-500/10 border border-blue-500/25 p-4 space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-300">
            <Info className="h-3.5 w-3.5" />
            <span>{PL.wizard.done.warningsTitle}</span>
          </div>
          {infoNotices.map((notice) => (
            <p key={notice} className="text-xs text-blue-200/90">
              {notice}
            </p>
          ))}
        </div>
      )}

      {exportNote && <p className="text-xs text-emerald-400">{exportNote}</p>}

      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            onClick={() => window.bsm.openPath(result.folder)}
          >
            <FolderOpen className="h-4 w-4" />
            {PL.wizard.done.openFolder}
          </Button>
          <Button variant="secondary" onClick={handleExportZip}>
            <Archive className="h-4 w-4" />
            {PL.wizard.done.exportZip}
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          {onGoToLibrary && (
            <Button variant="ghost" onClick={onGoToLibrary}>
              {PL.wizard.done.goToLibrary}
            </Button>
          )}
          <Button onClick={onReset}>
            <RotateCcw className="h-4 w-4" />
            {PL.wizard.done.createAnother}
          </Button>
        </div>
      </div>
    </Card>
  );
}
