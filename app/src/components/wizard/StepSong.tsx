import React, { useRef, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Disc3,
  FileAudio,
  ImagePlus,
  Loader2,
  Music,
  Search,
  UploadCloud,
  X,
  Youtube,
} from "lucide-react";
import { formatErrorEn, EN } from "../../i18n/en";
import type { YouTubeItem } from "../../types";
import { Badge, Button, Card, cn, Input, SegmentedControl } from "../ui/primitives";

export interface SongDraft {
  sourceType: "youtube" | "file" | "cached";
  sourcePathOrUrl: string;
  title: string;
  artist: string;
  coverPath: string | null;
  thumbnailUrl: string | null;
}

export interface StepSongProps {
  song: SongDraft;
  onChange: (next: SongDraft) => void;
  onNext: () => void;
}

const QUICK_SEARCH_HINTS = ["Camellia", "Synthwave", "Drum & Bass", "Rock"];

function isLikelyYouTubeUrl(text: string): boolean {
  return /^https?:\/\/(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)\//i.test(text.trim());
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function extractErrorCode(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const m = /BSM_CODE:([a-z0-9_]+)/i.exec(msg);
  return m ? m[1] : "internal";
}

function stemFromPath(filePath: string): string {
  const base = filePath.split(/[/\\]/).pop() ?? "";
  return base.replace(/\.[^.]+$/, "") || "untitled";
}

export function StepSong({ song, onChange, onNext }: StepSongProps) {
  const [tab, setTab] = useState<"youtube" | "file">(
    song.sourceType === "file" ? "file" : "youtube",
  );
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<YouTubeItem[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [draggingFile, setDraggingFile] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  async function runSearch(rawQuery: string) {
    const q = rawQuery.trim();
    if (!q) return;
    setSearching(true);
    setErrorMsg(null);
    try {
      if (isLikelyYouTubeUrl(q)) {
        const item = await window.bsm.resolveYouTube(q);
        setResults([item]);
        setHasSearched(true);
        onChange({
          sourceType: "youtube",
          sourcePathOrUrl: item.url,
          title: item.track_title || item.title,
          artist: item.artist || "Unknown",
          coverPath: null,
          thumbnailUrl: item.thumbnail_url,
        });
      } else {
        const items = await window.bsm.searchYouTube(q);
        setResults(items);
        setHasSearched(true);
      }
    } catch (err) {
      setErrorMsg(formatErrorEn(extractErrorCode(err)));
    } finally {
      setSearching(false);
    }
  }

  function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault();
    void runSearch(query);
  }

  function handleSelectItem(item: YouTubeItem) {
    onChange({
      sourceType: "youtube",
      sourcePathOrUrl: item.url,
      title: item.track_title || item.title,
      artist: item.artist || "Unknown",
      coverPath: null,
      thumbnailUrl: item.thumbnail_url,
    });
  }

  async function handlePickAudioFile() {
    setErrorMsg(null);
    const picked = await window.bsm.pickAudioFile();
    if (!picked) return;
    onChange({
      sourceType: "file",
      sourcePathOrUrl: picked,
      title: stemFromPath(picked),
      artist: "Unknown",
      coverPath: null,
      thumbnailUrl: null,
    });
  }

  async function handlePickCoverImage() {
    const picked = await window.bsm.pickCoverImage?.();
    if (!picked) return;
    onChange({
      ...song,
      coverPath: picked.path,
      thumbnailUrl: picked.dataUrl,
    });
  }

  function handleDropAudioFile(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDraggingFile(false);
    const file = e.dataTransfer.files?.[0] as (File & { path?: string }) | undefined;
    const droppedPath = file?.path;
    if (!droppedPath) return;
    if (!/\.(mp3|wav|ogg|flac|m4a|aac|opus|webm)$/i.test(droppedPath)) return;
    setErrorMsg(null);
    onChange({
      sourceType: "file",
      sourcePathOrUrl: droppedPath,
      title: stemFromPath(droppedPath),
      artist: "Unknown",
      coverPath: null,
      thumbnailUrl: null,
    });
  }

  const canProceed = Boolean(song.sourcePathOrUrl && song.title.trim());

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-3.5">
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl
          value={tab}
          onChange={setTab}
          options={[
            {
              value: "youtube",
              label: EN.wizard.song.tabs.youtube,
              icon: <Youtube className="h-4 w-4" />,
              accent: "red",
            },
            {
              value: "file",
              label: EN.wizard.song.tabs.file,
              icon: <FileAudio className="h-4 w-4" />,
              accent: "blue",
            },
          ]}
        />
        <span className="text-xs text-zinc-500 hidden sm:inline">
          {tab === "youtube" ? EN.wizard.song.pasteTip : EN.wizard.song.dropzoneHint}
        </span>
      </div>

      {tab === "youtube" ? (
        <Card className="flex-1 min-h-0 flex flex-col p-4 gap-3">
          <form role="search" onSubmit={handleSearchSubmit} className="shrink-0 flex gap-2">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
              <Input
                ref={searchInputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={EN.wizard.song.searchPlaceholder}
                aria-label={EN.wizard.song.searchPlaceholder}
                className="pl-10 pr-9"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => {
                    setQuery("");
                    searchInputRef.current?.focus();
                  }}
                  aria-label="Clear"
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/80 transition-colors"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            <Button
              type="submit"
              disabled={searching || !query.trim()}
              className="px-5 shrink-0"
            >
              {searching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              {searching ? EN.wizard.song.searching : EN.wizard.song.searchButton}
            </Button>
          </form>

          {errorMsg && (
            <div className="shrink-0 rounded-xl bg-red-500/10 border border-red-500/25 px-3.5 py-2 text-xs text-red-300">
              {errorMsg}
            </div>
          )}

          {results.length > 0 ? (
            <div className="flex-1 min-h-0 overflow-y-auto space-y-2 pr-1">
              {results.map((item) => {
                const selected = song.sourcePathOrUrl === item.url;
                return (
                  <button
                    key={item.id || item.url}
                    type="button"
                    onClick={() => handleSelectItem(item)}
                    className={cn(
                      "w-full text-left flex items-center gap-3.5 rounded-xl border p-2 transition-colors cursor-pointer",
                      selected
                        ? "border-blue-500/80 bg-blue-500/10"
                        : "border-zinc-800/80 bg-zinc-950/60 hover:bg-zinc-800/60 hover:border-zinc-700",
                    )}
                  >
                    <div className="relative h-12 w-20 rounded-lg overflow-hidden shrink-0 bg-zinc-900 border border-zinc-800/80">
                      {item.thumbnail_url ? (
                        <img
                          src={item.thumbnail_url}
                          alt={item.title}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="h-full w-full flex items-center justify-center">
                          <Music className="h-4 w-4 text-zinc-500" />
                        </div>
                      )}
                      {item.duration_s !== null && (
                        <span className="absolute bottom-1 right-1 rounded bg-black/80 px-1 py-0.5 font-mono text-[10px] text-zinc-200">
                          {formatDuration(item.duration_s)}
                        </span>
                      )}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-zinc-100 truncate">
                        {item.title}
                      </div>
                      <div className="text-xs text-zinc-400 truncate mt-0.5">
                        {item.channel}
                      </div>
                    </div>

                    {selected && (
                      <CheckCircle2 className="h-4 w-4 text-blue-400 shrink-0 mr-1" />
                    )}
                  </button>
                );
              })}
            </div>
          ) : hasSearched ? (
            <div className="flex-1 min-h-0 flex items-center justify-center text-xs text-zinc-500">
              {EN.wizard.song.noResults}
            </div>
          ) : (
            <div className="flex-1 min-h-0 flex flex-col items-center justify-center text-center gap-2.5 px-4">
              <div className="h-11 w-11 rounded-2xl bg-zinc-950 border border-zinc-800 flex items-center justify-center">
                <Youtube className="h-5 w-5 text-zinc-400" />
              </div>
              <p className="text-sm text-zinc-300">{EN.wizard.song.emptyTitle}</p>
              <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-zinc-500">
                <span>{EN.wizard.song.quickHintsLabel}</span>
                {QUICK_SEARCH_HINTS.map((hint) => (
                  <button
                    key={hint}
                    type="button"
                    disabled={searching}
                    onClick={() => {
                      setQuery(hint);
                      void runSearch(hint);
                    }}
                    className="text-zinc-300 hover:text-blue-400 underline underline-offset-4 decoration-zinc-700 hover:decoration-blue-400 transition-colors cursor-pointer"
                  >
                    {hint}
                  </button>
                ))}
              </div>
            </div>
          )}
        </Card>
      ) : (
        <Card className="flex-1 min-h-0 flex flex-col p-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDraggingFile(true);
            }}
            onDragLeave={() => setDraggingFile(false)}
            onDrop={handleDropAudioFile}
            className={cn(
              "flex-1 min-h-0 rounded-xl border-2 border-dashed p-6 flex flex-col items-center justify-center text-center transition-colors gap-3",
              draggingFile
                ? "border-blue-400 bg-blue-500/10"
                : "border-zinc-800 bg-zinc-950/50 hover:border-zinc-700",
            )}
          >
            <div className="h-11 w-11 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center">
              <UploadCloud className="h-5 w-5 text-blue-400" />
            </div>
            <div className="space-y-0.5">
              <p className="text-sm font-medium text-zinc-200">
                {EN.wizard.song.dropzoneTitle}
              </p>
              <p className="text-xs text-zinc-500">{EN.wizard.song.dropzoneHint}</p>
            </div>
            <Button variant="secondary" onClick={handlePickAudioFile}>
              <FileAudio className="h-4 w-4" />
              {EN.wizard.song.pickFileButton}
            </Button>
            {song.sourceType === "file" && song.sourcePathOrUrl && (
              <p className="text-xs font-mono text-blue-300 bg-blue-500/10 border border-blue-500/25 rounded-lg px-3 py-1.5 break-all max-w-full">
                {EN.wizard.song.selectedFileLabel} {song.sourcePathOrUrl}
              </p>
            )}
          </div>
        </Card>
      )}

      {song.sourcePathOrUrl && (
        <Card className="shrink-0 p-3 flex flex-col sm:flex-row gap-3 items-center border-blue-500/30">
          <div className="relative shrink-0">
            <button
              type="button"
              onClick={handlePickCoverImage}
              title={EN.wizard.song.pickCoverHint}
              aria-label={EN.wizard.song.coverLabel}
              className="group relative h-14 w-14 rounded-lg overflow-hidden bg-zinc-950 border border-zinc-800 hover:border-blue-500/70 flex items-center justify-center transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {song.thumbnailUrl ? (
                <>
                  <img
                    src={song.thumbnailUrl}
                    alt={song.title}
                    className="h-full w-full object-cover"
                  />
                  <span className="absolute inset-0 bg-black/55 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <ImagePlus className="h-4 w-4 text-white" />
                  </span>
                </>
              ) : (
                <>
                  <Disc3 className="h-6 w-6 text-blue-400/70 group-hover:opacity-0 transition-opacity" />
                  <ImagePlus className="absolute h-5 w-5 text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                </>
              )}
            </button>
            {song.coverPath && (
              <button
                type="button"
                onClick={() => onChange({ ...song, coverPath: null, thumbnailUrl: null })}
                aria-label={EN.wizard.song.clearCoverLabel}
                title={EN.wizard.song.clearCoverLabel}
                className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-zinc-900 border border-zinc-700 hover:border-red-500/70 hover:text-red-300 text-zinc-400 flex items-center justify-center shadow cursor-pointer"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>

          <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full">
            <label className="space-y-1 text-xs">
              <span className="text-zinc-400 font-medium flex items-center gap-1.5">
                {EN.wizard.song.titleLabel}
                <Badge variant="blue" className="py-0 px-1.5 text-[10px]">
                  {EN.wizard.song.selectedBadge}
                </Badge>
              </span>
              <Input
                value={song.title}
                onChange={(e) => onChange({ ...song, title: e.target.value })}
                className="py-1.5"
              />
            </label>
            <label className="space-y-1 text-xs">
              <span className="text-zinc-400 font-medium">{EN.wizard.song.artistLabel}</span>
              <Input
                value={song.artist}
                onChange={(e) => onChange({ ...song, artist: e.target.value })}
                className="py-1.5"
              />
            </label>
          </div>
        </Card>
      )}

      <div className="shrink-0 flex justify-end">
        <Button size="lg" disabled={!canProceed} onClick={onNext}>
          <span>{EN.common.next}</span>
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
