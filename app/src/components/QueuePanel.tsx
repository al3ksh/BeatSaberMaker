import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Archive,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Disc3,
  Eye,
  FileText,
  FolderOpen,
  Info,
  ListOrdered,
  Loader2,
  PlusCircle,
  RotateCcw,
  Trash2,
  X,
  XCircle,
} from "lucide-react";
import {
  EN,
  ENVIRONMENT_LABELS_EN,
  RANK_LABELS_EN,
  formatErrorEn,
} from "../i18n/en";
import type {
  GenerateOutput,
  MapGenerationConfig,
  QueueItem,
  StageName,
  StartGenerationRequest,
} from "../types";
import { Badge, Button, Card, ProgressBar, cn } from "./ui/primitives";
import { rankToBadgeVariant } from "./wizard/StepMap";

const DOWNLOAD_STAGES: StageName[] = [
  "downloading",
  "analyzing",
  "notes",
  "layers",
  "writing",
];
const LOCAL_STAGES: StageName[] = ["analyzing", "notes", "layers", "writing"];
const DEFAULT_PACK_LEVELS = ["Hard", "Expert", "ExpertPlus"];

export interface QueuePanelProps {
  items: QueueItem[];
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onMove: (id: string, direction: "up" | "down") => void;
  onClearFinished: () => void;
  onPreview?: (folder: string) => void;
  onGoToCreate?: () => void;
}

function formatSourceLabel(request: StartGenerationRequest): string {
  if (request.regenerateId || request.sourceType === "cached") {
    return EN.queue.source.regenerate;
  }
  if (request.sourceType === "youtube") {
    return EN.queue.source.youtube;
  }
  return EN.queue.source.file;
}

function collectInfoNotices(result: GenerateOutput): string[] {
  const notices: string[] = [];
  if (result.warnings.some((w) => /checkpoint|layer_stats/i.test(w))) {
    notices.push(EN.wizard.done.warnings.missingLayerModels);
  }
  if (result.used_rule_lights) {
    notices.push(EN.wizard.done.warnings.ruleLights);
  }
  if (!result.grid_steady) {
    notices.push(EN.wizard.done.warnings.unsteadyGrid);
  }
  return notices;
}

function sortFinishedNewestFirst(items: QueueItem[]): QueueItem[] {
  return [...items]
    .map((item, idx) => ({ item, idx }))
    .sort((a, b) => {
      const aTime = a.item.finishedAt ? Date.parse(a.item.finishedAt) : 0;
      const bTime = b.item.finishedAt ? Date.parse(b.item.finishedAt) : 0;
      if (bTime !== aTime) {
        return bTime - aTime;
      }
      return b.idx - a.idx;
    })
    .map(({ item }) => item);
}

function QueueThumbnail({
  src,
  alt,
  spinning = false,
}: {
  src: string | null;
  alt: string;
  spinning?: boolean;
}) {
  if (src) {
    return (
      <img
        src={src}
        alt={alt}
        className="h-11 w-11 rounded-xl object-cover border border-zinc-800 shrink-0"
      />
    );
  }
  return (
    <div className="h-11 w-11 rounded-xl bg-zinc-950 border border-zinc-800 flex items-center justify-center shrink-0">
      <Disc3
        className={cn(
          "h-5 w-5",
          spinning
            ? "text-blue-400/80 animate-spin [animation-duration:6s]"
            : "text-zinc-500",
        )}
      />
    </div>
  );
}

function ConfigSummaryBadges({ config }: { config: MapGenerationConfig }) {
  const levels =
    config.mode === "pack"
      ? (config.levels ?? DEFAULT_PACK_LEVELS)
      : config.nps !== null
        ? []
        : (config.levels ?? []);
  const nps = config.mode === "pack" ? null : config.nps;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge variant="neutral">
        {config.mode === "pack" ? EN.queue.mode.pack : EN.queue.mode.custom}
      </Badge>
      {nps !== null ? (
        <Badge variant="blue">{nps.toFixed(1)} NPS</Badge>
      ) : (
        levels.map((rank) => (
          <Badge key={rank} variant={rankToBadgeVariant(rank)}>
            {RANK_LABELS_EN[rank] ?? rank}
          </Badge>
        ))
      )}
    </div>
  );
}

function QueueDoneDetails({ result }: { result: GenerateOutput }) {
  // Walls/bombs differ per difficulty; older engines only report the densest level's counts.
  const perLevelDecor =
    result.levels.length > 1 &&
    result.levels.every(
      (lv) => typeof lv.n_walls === "number" && typeof lv.n_bombs === "number",
    );
  const infoNotices = collectInfoNotices(result);

  return (
    <div className="pt-3 mt-3 border-t border-zinc-800/80 space-y-3">
      {perLevelDecor ? (
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 overflow-hidden">
          <table className="w-full text-sm" data-testid="done-levels-table">
            <thead>
              <tr className="text-[11px] text-zinc-500 border-b border-zinc-800/80">
                <th className="text-left font-normal px-3 py-1.5">
                  {EN.wizard.done.levelLabel}
                </th>
                <th className="text-right font-normal px-3 py-1.5">
                  {EN.wizard.done.notesLabel}
                </th>
                <th className="text-right font-normal px-3 py-1.5">
                  {EN.wizard.done.npsLabel}
                </th>
                <th className="text-right font-normal px-3 py-1.5">
                  {EN.wizard.done.wallsLabel}
                </th>
                <th className="text-right font-normal px-3 py-1.5">
                  {EN.wizard.done.bombsLabel}
                </th>
              </tr>
            </thead>
            <tbody className="font-mono text-zinc-100">
              {result.levels.map((lv) => (
                <tr
                  key={lv.rank}
                  className="border-t border-zinc-800/50 first:border-t-0"
                >
                  <td className="px-3 py-1.5 font-sans">
                    <Badge variant={rankToBadgeVariant(lv.rank)}>
                      {RANK_LABELS_EN[lv.rank] ?? lv.rank}
                    </Badge>
                  </td>
                  <td className="text-right px-3 py-1.5">{lv.n_notes}</td>
                  <td className="text-right px-3 py-1.5">{lv.nps.toFixed(1)}</td>
                  <td className="text-right px-3 py-1.5">{lv.n_walls}</td>
                  <td className="text-right px-3 py-1.5">{lv.n_bombs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {result.levels.map((lv) => (
            <Badge key={lv.rank} variant={rankToBadgeVariant(lv.rank)}>
              {RANK_LABELS_EN[lv.rank] ?? lv.rank}: {lv.n_notes} (
              {lv.nps.toFixed(1)} NPS)
            </Badge>
          ))}
        </div>
      )}

      <div
        className={
          perLevelDecor
            ? "grid grid-cols-3 gap-2.5"
            : "grid grid-cols-2 sm:grid-cols-5 gap-2.5"
        }
      >
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">
            {EN.wizard.done.bpmLabel}
          </div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {Math.round(result.bpm)}
          </div>
        </div>
        {!perLevelDecor && (
          <>
            <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
              <div className="text-[11px] text-zinc-500">
                {EN.wizard.done.wallsLabel}
              </div>
              <div className="text-sm font-semibold font-mono text-zinc-100">
                {result.n_walls}
              </div>
            </div>
            <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
              <div className="text-[11px] text-zinc-500">
                {EN.wizard.done.bombsLabel}
              </div>
              <div className="text-sm font-semibold font-mono text-zinc-100">
                {result.n_bombs}
              </div>
            </div>
          </>
        )}
        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2">
          <div className="text-[11px] text-zinc-500">
            {EN.wizard.done.lightsLabel}
          </div>
          <div className="text-sm font-semibold font-mono text-zinc-100">
            {result.n_lights}
          </div>
        </div>
        <div
          className={cn(
            "rounded-xl bg-zinc-950/70 border border-zinc-800/80 px-3 py-2 min-w-0",
            !perLevelDecor && "col-span-2 sm:col-span-1",
          )}
        >
          <div className="text-[11px] text-zinc-500">
            {EN.wizard.map.environmentLabel}
          </div>
          <div className="text-xs font-medium text-zinc-200 truncate mt-0.5">
            {ENVIRONMENT_LABELS_EN[result.environment] ?? result.environment}
          </div>
        </div>
      </div>

      {infoNotices.length > 0 && (
        <div className="rounded-xl bg-blue-500/10 border border-blue-500/25 p-3.5 space-y-1">
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
    </div>
  );
}

/**
 * Full queue management view grouped into Now Running, Up Next, and Finished.
 */
export function QueuePanel({
  items,
  onCancel,
  onRetry,
  onMove,
  onClearFinished,
  onPreview,
  onGoToCreate,
}: QueuePanelProps) {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});
  const [exportNotes, setExportNotes] = useState<Record<string, string>>({});
  const [coverUrls, setCoverUrls] = useState<Record<string, string>>({});
  const mountedRef = useRef(true);
  const attemptedCoverKeysRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const readCover = window.bsm?.readCoverDataUrl;
    if (!readCover) return;

    const toFetch: Array<{ id: string; path: string }> = [];
    for (const item of items) {
      if (item.thumbnailUrl) continue;
      const coverPath = item.entry?.coverCachePath ?? item.request.coverPath;
      if (!coverPath) continue;
      const key = `${item.id}:${coverPath}`;
      if (attemptedCoverKeysRef.current.has(key)) continue;
      attemptedCoverKeysRef.current.add(key);
      toFetch.push({ id: item.id, path: coverPath });
    }
    if (toFetch.length === 0) return;

    void Promise.all(
      toFetch.map(async ({ id, path }) => {
        try {
          const url = await readCover(path);
          return url ? { id, url } : null;
        } catch {
          return null;
        }
      }),
    ).then((loaded) => {
      if (!mountedRef.current) return;
      const updates: Record<string, string> = {};
      for (const entry of loaded) {
        if (entry) updates[entry.id] = entry.url;
      }
      if (Object.keys(updates).length > 0) {
        setCoverUrls((prev) => ({ ...prev, ...updates }));
      }
    });
  }, [items]);

  const activeItem = useMemo(
    () =>
      items.find(
        (item) => item.status === "downloading" || item.status === "running",
      ) ?? null,
    [items],
  );

  const waitingItems = useMemo(
    () => items.filter((item) => item.status === "waiting"),
    [items],
  );

  const finishedItems = useMemo(
    () =>
      sortFinishedNewestFirst(
        items.filter(
          (item) =>
            item.status === "done" ||
            item.status === "failed" ||
            item.status === "cancelled",
        ),
      ),
    [items],
  );

  function toggleExpanded(id: string) {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  async function handleExportZip(id: string, result: GenerateOutput) {
    setExportNotes((prev) => {
      if (!prev[id]) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      await window.bsm.exportMapZip(result.folder, result.title);
      setExportNotes((prev) => ({ ...prev, [id]: EN.library.exportedSuccess }));
    } catch {
      // Cancelled save dialog or error handled silently.
    }
  }

  if (items.length === 0) {
    return (
      <div className="flex-1 min-h-0 flex flex-col justify-center">
        <Card className="w-full max-w-lg mx-auto p-8 text-center space-y-4">
          <div className="mx-auto h-12 w-12 rounded-2xl bg-zinc-950 border border-zinc-800 flex items-center justify-center">
            <ListOrdered className="h-6 w-6 text-zinc-400" />
          </div>
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-zinc-100">
              {EN.queue.emptyTitle}
            </h2>
            <p className="text-sm text-zinc-400">{EN.queue.emptyDesc}</p>
          </div>
          {onGoToCreate && (
            <div className="pt-1">
              <Button variant="primary" onClick={onGoToCreate}>
                <PlusCircle className="h-4 w-4" />
                {EN.queue.newMap}
              </Button>
            </div>
          )}
        </Card>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-6">
      {activeItem && (
        <section data-testid="queue-section-now" className="space-y-2.5">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
            {EN.queue.sections.nowRunning}
          </h2>
          {(() => {
            const stages =
              activeItem.request.sourceType === "youtube"
                ? DOWNLOAD_STAGES
                : LOCAL_STAGES;
            const currentStage: StageName =
              activeItem.progress?.stage ??
              (activeItem.status === "downloading" ? "downloading" : "analyzing");
            const stageIdx = Math.max(0, stages.indexOf(currentStage));
            const pct = activeItem.progress?.pct ?? 0;
            const thumb = activeItem.thumbnailUrl ?? coverUrls[activeItem.id] ?? null;

            return (
              <Card className="p-4 sm:p-5 space-y-4 border-blue-500/30">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-3.5 min-w-0">
                    <QueueThumbnail
                      src={thumb}
                      alt={activeItem.request.title}
                      spinning
                    />
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-base font-semibold text-zinc-100 truncate">
                          {activeItem.request.title}
                        </h3>
                        <Badge variant="blue">
                          {activeItem.status === "downloading"
                            ? EN.queue.status.downloading
                            : EN.queue.status.running}
                        </Badge>
                      </div>
                      <p className="text-xs text-zinc-400 truncate">
                        {activeItem.request.artist || EN.common.unknownArtist}
                      </p>
                      <ConfigSummaryBadges config={activeItem.request.config} />
                    </div>
                  </div>

                  <div className="text-right shrink-0 space-y-0.5">
                    <div className="font-mono text-lg font-semibold tabular-nums text-zinc-100">
                      {pct}%
                    </div>
                    <div className="text-xs text-zinc-400">
                      {EN.queue.stages[currentStage]}
                    </div>
                  </div>
                </div>

                <ProgressBar pct={pct} />

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-3">
                    {stages.map((st, idx) => {
                      const done = idx < stageIdx;
                      const active = idx === stageIdx;
                      return (
                        <div
                          key={st}
                          data-testid={`stage-step-${st}`}
                          className={cn(
                            "flex items-center gap-1.5 text-xs",
                            active
                              ? "text-blue-300 font-medium"
                              : done
                                ? "text-emerald-400"
                                : "text-zinc-500",
                          )}
                        >
                          {done ? (
                            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                          ) : active ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
                          ) : (
                            <span className="h-2 w-2 rounded-full bg-zinc-700 mx-0.5 shrink-0" />
                          )}
                          <span>{EN.queue.stageShort[st]}</span>
                        </div>
                      );
                    })}
                  </div>

                  <Button
                    variant="danger"
                    size="sm"
                    disabled={Boolean(activeItem.cancelling)}
                    onClick={() => onCancel(activeItem.id)}
                  >
                    <X className="h-3.5 w-3.5" />
                    {activeItem.cancelling
                      ? EN.queue.cancelling
                      : EN.common.cancel}
                  </Button>
                </div>
              </Card>
            );
          })()}
        </section>
      )}

      {waitingItems.length > 0 && (
        <section data-testid="queue-section-next" className="space-y-2.5">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
            {EN.queue.sections.upNext} ({waitingItems.length})
          </h2>
          <Card className="divide-y divide-zinc-800/80 overflow-hidden">
            {waitingItems.map((item, idx) => {
              const thumb = item.thumbnailUrl ?? coverUrls[item.id] ?? null;
              return (
                <div
                  key={item.id}
                  className="px-4 py-3 flex items-center justify-between gap-3"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-xs font-mono text-zinc-500 w-6 shrink-0">
                      #{idx + 1}
                    </span>
                    <QueueThumbnail src={thumb} alt={item.request.title} />
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-zinc-100 truncate">
                          {item.request.title}
                        </span>
                        <Badge variant="sky">
                          {formatSourceLabel(item.request)}
                        </Badge>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
                        <span className="truncate">
                          {item.request.artist || EN.common.unknownArtist}
                        </span>
                        <ConfigSummaryBadges config={item.request.config} />
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={EN.queue.moveUp}
                      disabled={idx === 0}
                      onClick={() => onMove(item.id, "up")}
                    >
                      <ArrowUp className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={EN.queue.moveDown}
                      disabled={idx === waitingItems.length - 1}
                      onClick={() => onMove(item.id, "down")}
                    >
                      <ArrowDown className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={EN.common.cancel}
                      onClick={() => onCancel(item.id)}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </Card>
        </section>
      )}

      {finishedItems.length > 0 && (
        <section data-testid="queue-section-finished" className="space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              {EN.queue.sections.finished} ({finishedItems.length})
            </h2>
            <Button variant="ghost" size="sm" onClick={onClearFinished}>
              <Trash2 className="h-3.5 w-3.5" />
              {EN.queue.clearFinished}
            </Button>
          </div>

          <Card className="divide-y divide-zinc-800/80 overflow-hidden">
            {finishedItems.map((item) => {
              const title = item.result?.title || item.request.title;
              const artist =
                item.result?.artist ||
                item.request.artist ||
                EN.common.unknownArtist;
              const thumb = item.thumbnailUrl ?? coverUrls[item.id] ?? null;
              const expanded = Boolean(expandedIds[item.id]);
              const errorCode = item.errorCode ?? "internal";

              return (
                <div key={item.id} className="px-4 py-3.5 space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <QueueThumbnail src={thumb} alt={title} />
                      <div className="min-w-0 space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            data-testid="queue-finished-title"
                            className="text-sm font-medium text-zinc-100 truncate"
                          >
                            {title}
                          </span>
                          {item.status === "done" && (
                            <Badge variant="emerald">
                              <CheckCircle2 className="h-3 w-3" />
                              {EN.queue.status.done}
                            </Badge>
                          )}
                          {item.status === "failed" && (
                            <Badge variant="red">
                              <AlertTriangle className="h-3 w-3" />
                              {EN.queue.status.failed}
                            </Badge>
                          )}
                          {item.status === "cancelled" && (
                            <Badge variant="neutral">
                              <XCircle className="h-3 w-3" />
                              {EN.queue.status.cancelled}
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-zinc-400 truncate">
                          {artist}
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                      {item.status === "done" && item.result && (
                        <>
                          {onPreview && (
                            <Button
                              variant="primary"
                              size="sm"
                              onClick={() => onPreview(item.result!.folder)}
                            >
                              <Eye className="h-3.5 w-3.5" />
                              {EN.wizard.done.preview}
                            </Button>
                          )}
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() =>
                              window.bsm.openPath(item.result!.folder)
                            }
                          >
                            <FolderOpen className="h-3.5 w-3.5" />
                            {EN.wizard.done.openFolder}
                          </Button>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() =>
                              void handleExportZip(item.id, item.result!)
                            }
                          >
                            <Archive className="h-3.5 w-3.5" />
                            {EN.wizard.done.exportZip}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            aria-expanded={expanded}
                            onClick={() => toggleExpanded(item.id)}
                          >
                            {expanded ? (
                              <ChevronUp className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronDown className="h-3.5 w-3.5" />
                            )}
                            {expanded
                              ? EN.queue.hideDetails
                              : EN.queue.details}
                          </Button>
                        </>
                      )}

                      {item.status === "failed" && (
                        <>
                          {errorCode === "internal" && (
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() => window.bsm.openLogs()}
                            >
                              <FileText className="h-3.5 w-3.5" />
                              {EN.settings.openLogs}
                            </Button>
                          )}
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => onRetry(item.id)}
                          >
                            <RotateCcw className="h-3.5 w-3.5" />
                            {EN.queue.retry}
                          </Button>
                        </>
                      )}

                      {item.status === "cancelled" && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => onRetry(item.id)}
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                          {EN.queue.retry}
                        </Button>
                      )}
                    </div>
                  </div>

                  {item.status === "failed" && (
                    <p className="text-xs text-red-300 pl-14">
                      {formatErrorEn(errorCode)}
                    </p>
                  )}

                  {exportNotes[item.id] && (
                    <p className="text-xs text-emerald-400 pl-14">
                      {exportNotes[item.id]}
                    </p>
                  )}

                  {item.status === "done" && item.result && expanded && (
                    <QueueDoneDetails result={item.result} />
                  )}
                </div>
              );
            })}
          </Card>
        </section>
      )}
    </div>
  );
}
