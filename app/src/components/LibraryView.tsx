import { useEffect, useState } from "react";
import { Archive, Disc3, FolderOpen, RefreshCw, Trash2 } from "lucide-react";
import { PL, RANK_LABELS_PL } from "../i18n/pl";
import type { LibraryEntry } from "../types";
import { Badge, Button, Card } from "./ui/primitives";

export interface LibraryViewProps {
  onRegenerate: (entry: LibraryEntry) => void;
}

export function LibraryView({ onRegenerate }: LibraryViewProps) {
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [covers, setCovers] = useState<Record<string, string>>({});
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

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

  async function handleConfirmDelete(id: string) {
    const next = await window.bsm.deleteLibraryEntry(id);
    setEntries(next);
    setConfirmDeleteId(null);
  }

  async function handleExport(entry: LibraryEntry) {
    setExportNotice(null);
    try {
      await window.bsm.exportMapZip(entry.folder, entry.title);
      setExportNotice(`${PL.library.exportedSuccess} (${entry.title})`);
    } catch {
      // Cancelled save dialog.
    }
  }

  if (loading) {
    return <p className="text-sm text-zinc-400">{PL.common.loading}</p>;
  }

  if (entries.length === 0) {
    return (
      <Card className="p-8 text-center text-sm text-zinc-400">
        {PL.library.empty}
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-100">{PL.library.title}</h2>
        {exportNotice && (
          <span className="text-xs text-emerald-400">{exportNotice}</span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {entries.map((entry) => {
          const coverUrl = covers[entry.id];
          const isConfirming = confirmDeleteId === entry.id;
          return (
            <Card key={entry.id} className="p-4 flex flex-col justify-between gap-4">
              <div className="flex gap-3.5">
                {coverUrl ? (
                  <img
                    src={coverUrl}
                    alt={entry.title}
                    className="h-16 w-16 rounded-xl object-cover border border-zinc-800 shrink-0"
                  />
                ) : (
                  <div className="h-16 w-16 rounded-xl bg-zinc-950 border border-zinc-800 flex items-center justify-center shrink-0">
                    <Disc3 className="h-7 w-7 text-zinc-600" />
                  </div>
                )}

                <div className="min-w-0 flex-1 space-y-1">
                  <div className="font-semibold text-sm text-zinc-100 truncate">
                    {entry.title}
                  </div>
                  <div className="text-xs text-zinc-400 truncate">{entry.artist}</div>
                  <div className="text-[11px] text-zinc-500">
                    {new Date(entry.createdAt).toLocaleDateString("pl-PL")} ·{" "}
                    {Math.round(entry.bpm)} BPM
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {entry.levels.map((lv) => (
                  <Badge key={lv.rank} variant="blue">
                    {RANK_LABELS_PL[lv.rank] ?? lv.rank}
                  </Badge>
                ))}
              </div>

              {isConfirming ? (
                <div className="flex items-center justify-end gap-2 pt-1 border-t border-zinc-800">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setConfirmDeleteId(null)}
                  >
                    {PL.library.cancelDelete}
                  </Button>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => handleConfirmDelete(entry.id)}
                  >
                    {PL.library.confirmDelete}
                  </Button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-zinc-800">
                  <div className="flex flex-wrap gap-1.5">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => onRegenerate(entry)}
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                      {PL.library.regenerate}
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleExport(entry)}
                    >
                      <Archive className="h-3.5 w-3.5" />
                      {PL.library.exportZip}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => window.bsm.openPath(entry.folder)}
                      aria-label={PL.library.openFolder}
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
                    {PL.library.delete}
                  </Button>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
