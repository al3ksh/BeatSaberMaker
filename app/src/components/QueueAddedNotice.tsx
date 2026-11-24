import { CheckCircle2, X } from "lucide-react";
import { EN } from "../i18n/en";
import { Button } from "./ui/primitives";

export interface QueueAddedNoticeProps {
  title: string;
  onViewQueue?: () => void;
  onDismiss: () => void;
  className?: string;
}

/**
 * Non-blocking confirmation banner shown after enqueuing a map generation.
 */
export function QueueAddedNotice({
  title,
  onViewQueue,
  onDismiss,
  className = "",
}: QueueAddedNoticeProps) {
  return (
    <div
      role="status"
      className={`shrink-0 flex items-center justify-between gap-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 px-3.5 py-2 text-xs text-emerald-200 ${className}`}
    >
      <div className="flex items-center gap-2 min-w-0">
        <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
        <span className="truncate">{EN.queue.addedNotice(title)}</span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        {onViewQueue && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-emerald-200 hover:text-white hover:bg-emerald-500/20"
            onClick={onViewQueue}
          >
            {EN.queue.viewQueue}
          </Button>
        )}
        <button
          type="button"
          aria-label={EN.common.close}
          onClick={onDismiss}
          className="rounded p-1 text-emerald-300/80 hover:text-emerald-100 hover:bg-emerald-500/20 transition-colors cursor-pointer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
