import { ListOrdered, Loader2, X } from "lucide-react";
import { EN } from "../i18n/en";
import type { QueueItem, StageName } from "../types";
import { Button, ProgressBar } from "./ui/primitives";

export interface QueueStatusBarProps {
  items: QueueItem[];
  onOpenQueue: () => void;
  onCancelActive: (id: string) => void;
}

/**
 * Slim bottom bar summarizing active progress and queued item counts outside the Queue tab.
 */
export function QueueStatusBar({
  items,
  onOpenQueue,
  onCancelActive,
}: QueueStatusBarProps) {
  if (items.length === 0) {
    return null;
  }

  const activeItem =
    items.find(
      (item) => item.status === "downloading" || item.status === "running",
    ) ?? null;
  const waitingCount = items.filter((item) => item.status === "waiting").length;
  const doneCount = items.filter((item) => item.status === "done").length;
  const failedCount = items.filter((item) => item.status === "failed").length;
  const cancelledCount = items.filter((item) => item.status === "cancelled").length;

  const activeStage: StageName | null = activeItem
    ? activeItem.progress?.stage ??
      (activeItem.status === "downloading" ? "downloading" : "analyzing")
    : null;
  const activePct = activeItem?.progress?.pct ?? 0;

  return (
    <div
      data-testid="queue-status-bar"
      className="shrink-0 border-t border-zinc-800/90 bg-zinc-950/95 backdrop-blur px-6 py-2.5 flex items-center justify-between gap-4"
    >
      <div className="min-w-0 flex-1 flex items-center gap-3">
        {activeItem && activeStage ? (
          <>
            <Loader2 className="h-4 w-4 text-blue-400 animate-spin shrink-0" />
            <div className="min-w-0 flex items-center gap-2">
              <span className="text-xs font-semibold text-zinc-100 truncate max-w-[180px] sm:max-w-[240px]">
                {activeItem.request.title}
              </span>
              <span className="text-xs text-zinc-400 truncate hidden sm:inline">
                — {EN.queue.stages[activeStage]}
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <ProgressBar pct={activePct} className="h-2 w-24 sm:w-36" />
              <span className="text-xs font-mono tabular-nums text-zinc-300 w-9 text-right">
                {activePct}%
              </span>
            </div>
          </>
        ) : (
          <div className="flex items-center gap-2 text-xs text-zinc-300">
            <ListOrdered className="h-4 w-4 text-zinc-400 shrink-0" />
            <span className="font-medium">{EN.queue.noActiveTitle}</span>
          </div>
        )}

        <div className="hidden md:flex items-center gap-2 text-xs text-zinc-400 truncate">
          {waitingCount > 0 && (
            <span>· {EN.queue.waitingSummary(waitingCount)}</span>
          )}
          {doneCount > 0 && <span>· {EN.queue.doneSummary(doneCount)}</span>}
          {failedCount > 0 && (
            <span className="text-red-400">
              · {EN.queue.failedSummary(failedCount)}
            </span>
          )}
          {cancelledCount > 0 && (
            <span>· {EN.queue.cancelledSummary(cancelledCount)}</span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {activeItem && (
          <Button
            variant="ghost"
            size="sm"
            disabled={Boolean(activeItem.cancelling)}
            onClick={() => onCancelActive(activeItem.id)}
          >
            <X className="h-3.5 w-3.5" />
            {activeItem.cancelling ? EN.queue.cancelling : EN.common.cancel}
          </Button>
        )}
        <Button variant="secondary" size="sm" onClick={onOpenQueue}>
          {EN.queue.openQueue}
        </Button>
      </div>
    </div>
  );
}
