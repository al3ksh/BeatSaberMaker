import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  ChevronLeft,
  ChevronRight,
  Disc3,
  FolderHeart,
  FolderOpen,
  PlusCircle,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { EN, RANK_LABELS_EN } from "../i18n/en";
import type { LibraryEntry } from "../types";
import { Badge, Button, Card, cn, Input } from "./ui/primitives";
import { rankToBadgeVariant } from "./wizard/StepMap";

const PAGE_SIZE = 6;

function visiblePageNumbers(current: number, total: number): Array<number | "ellipsis"> {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const pages: Array<number | "ellipsis"> = [1];
  const left = Math.max(2, current - 1);
  const right = Math.min(total - 1, current + 1);
  if (left > 2) pages.push("ellipsis");
  for (let p = left; p <= right; p += 1) {
    pages.push(p);
  }
  if (right < total - 1) pages.push("ellipsis");
  pages.push(total);
  return pages;
}

export interface LibraryViewProps {
  onRegenerate: (entry: LibraryEntry) => void;
  onGoToCreate?: () => void;
}

export function LibraryView({ onRegenerate, onGoToCreate }: LibraryViewProps) {
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [covers, setCovers] = useState<Record<string, string>>({});
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const gridScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    window.bsm
      .listLibrary()
      .then(async (list) => {
        if (!active) return;
        setEntries(list);
        const loadedCovers: Record<string, string> = {};
        for (const item of list) {
          if (item.coverCachePath) {
            const url = await window.bsm.readCoverDataUrl(item.coverCachePath);
            if (url) loadedCovers[item.id] = url;
          }
        }
        if (active) setCovers(loadedCovers);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const filteredEntries = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter((entry) => {
      if (entry.title.toLowerCase().includes(q)) return true;
      if (entry.artist.toLowerCase().includes(q)) return true;
      return entry.levels.some((lv) =>
        (RANK_LABELS_EN[lv.rank] ?? lv.rank).toLowerCase().includes(q),
      );
    });
  }, [entries, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filteredEntries.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);

  useEffect(() => {
    if (page !== safePage) {
      setPage(safePage);
    }
  }, [page, safePage]);

  const pagedEntries = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return filteredEntries.slice(start, start + PAGE_SIZE);
  }, [filteredEntries, safePage]);

  function goToPage(next: number) {
    const clamped = Math.max(1, Math.min(totalPages, next));
    setPage(clamped);
    if (gridScrollRef.current) {
      gridScrollRef.current.scrollTop = 0;
    }
  }

  async function handleConfirmDelete(id: string) {
    const next = await window.bsm.deleteLibraryEntry(id);
    setEntries(next);
    setConfirmDeleteId(null);
  }

  async function handleExport(entry: LibraryEntry) {
    setExportNotice(null);
    try {
      await window.bsm.exportMapZip(entry.folder, entry.title);
      setExportNotice(`${EN.library.exportedSuccess} (${entry.title})`);
    } catch {
      // Cancelled save dialog.
    }
  }

  if (loading) {
    return <p className="text-sm text-zinc-400">{EN.common.loading}</p>;
  }

  if (entries.length === 0) {
    return (
      <Card className="flex-1 flex flex-col items-center justify-center p-10 text-center gap-4">
        <div className="h-14 w-14 rounded-2xl bg-zinc-950 border border-zinc-800 flex items-center justify-center">
          <FolderHeart className="h-7 w-7 text-zinc-500" />
        </div>
        <p className="text-sm text-zinc-400 max-w-md">{EN.library.empty}</p>
        {onGoToCreate && (
          <Button onClick={onGoToCreate}>
            <PlusCircle className="h-4 w-4" />
            {EN.library.createFirst}
          </Button>
        )}
      </Card>
    );
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-3.5">
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <h2 className="text-lg font-semibold text-zinc-100">{EN.library.title}</h2>
          <Badge variant="neutral">
            {searchQuery.trim()
              ? `${filteredEntries.length} / ${entries.length}`
              : entries.length}
          </Badge>
          {exportNotice && (
            <span className="text-xs text-emerald-400">{exportNotice}</span>
          )}
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <Input
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            placeholder={EN.library.searchPlaceholder}
            aria-label={EN.library.searchPlaceholder}
            className="pl-9 pr-8 py-1.5 text-xs"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery("");
                setPage(1);
              }}
              aria-label={EN.library.clearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/80 transition-colors cursor-pointer"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {filteredEntries.length === 0 ? (
        <Card className="flex-1 flex flex-col items-center justify-center p-8 text-center gap-3">
          <p className="text-sm text-zinc-400">{EN.library.noMatches}</p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setSearchQuery("");
              setPage(1);
            }}
          >
            {EN.library.clearSearch}
          </Button>
        </Card>
      ) : (
        <div ref={gridScrollRef} className="flex-1 min-h-0 overflow-y-auto pr-1">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 pb-1">
            {pagedEntries.map((entry) => {
              const coverUrl = covers[entry.id];
              const isConfirming = confirmDeleteId === entry.id;
              return (
                <Card
                  key={entry.id}
                  className="p-3.5 flex flex-col justify-between gap-3 hover:border-zinc-700/90 transition-colors"
                >
                  <div className="flex gap-3">
                    {coverUrl ? (
                      <img
                        src={coverUrl}
                        alt={entry.title}
                        className="h-14 w-14 rounded-xl object-cover border border-zinc-800 shrink-0 shadow-md"
                      />
                    ) : (
                      <div className="h-14 w-14 rounded-xl bg-zinc-950 border border-zinc-800 flex items-center justify-center shrink-0">
                        <Disc3 className="h-6 w-6 text-zinc-600" />
                      </div>
                    )}

                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="font-semibold text-sm text-zinc-100 truncate">
                        {entry.title}
                      </div>
                      <div className="text-xs text-zinc-400 truncate">{entry.artist}</div>
                      <div className="text-[11px] font-mono text-zinc-500">
                        {new Date(entry.createdAt).toLocaleDateString("en-US")} ·{" "}
                        {Math.round(entry.bpm)} BPM
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5">
                    {entry.levels.map((lv) => (
                      <Badge key={lv.rank} variant={rankToBadgeVariant(lv.rank)}>
                        {RANK_LABELS_EN[lv.rank] ?? lv.rank}
                      </Badge>
                    ))}
                  </div>

                  {isConfirming ? (
                    <div className="flex items-center justify-end gap-2 pt-2 border-t border-zinc-800/80">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setConfirmDeleteId(null)}
                      >
                        {EN.library.cancelDelete}
                      </Button>
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => handleConfirmDelete(entry.id)}
                      >
                        {EN.library.confirmDelete}
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-zinc-800/80">
                      <div className="flex flex-wrap gap-1.5">
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => onRegenerate(entry)}
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                          {EN.library.regenerate}
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => handleExport(entry)}
                        >
                          <Archive className="h-3.5 w-3.5" />
                          {EN.library.exportZip}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => window.bsm.openPath(entry.folder)}
                          aria-label={EN.library.openFolder}
                        >
                          <FolderOpen className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setConfirmDeleteId(entry.id)}
                        className="text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        {EN.library.delete}
                      </Button>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {totalPages > 1 && (
        <div className="shrink-0 flex items-center justify-between pt-1 border-t border-zinc-800/70">
          <span className="text-xs text-zinc-500 font-mono">
            {EN.library.pageLabel} {safePage} {EN.library.ofLabel} {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="secondary"
              size="sm"
              disabled={safePage <= 1}
              onClick={() => goToPage(safePage - 1)}
              aria-label={EN.library.prevPage}
              className="px-2"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            {visiblePageNumbers(safePage, totalPages).map((item, idx) =>
              item === "ellipsis" ? (
                <span key={`e-${idx}`} className="px-1.5 text-xs text-zinc-600 select-none">
                  …
                </span>
              ) : (
                <button
                  key={item}
                  type="button"
                  onClick={() => goToPage(item)}
                  aria-current={item === safePage ? "page" : undefined}
                  className={cn(
                    "h-7 min-w-7 px-2 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer",
                    item === safePage
                      ? "bg-blue-600 text-white"
                      : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800",
                  )}
                >
                  {item}
                </button>
              ),
            )}
            <Button
              variant="secondary"
              size="sm"
              disabled={safePage >= totalPages}
              onClick={() => goToPage(safePage + 1)}
              aria-label={EN.library.nextPage}
              className="px-2"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
