import React, { useState } from "react";
import { Disc3, FileAudio, Music, Search, Youtube } from "lucide-react";
import { formatErrorPl, PL } from "../../i18n/pl";
import type { YouTubeItem } from "../../types";
import { Button, Card, Input } from "../ui/primitives";

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

  async function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
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
      setErrorMsg(formatErrorPl(extractErrorCode(err)));
    } finally {
      setSearching(false);
    }
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

  const canProceed = Boolean(song.sourcePathOrUrl && song.title.trim());

  return (
    <div className="space-y-5">
      <div className="flex gap-2">
        <Button
          variant={tab === "youtube" ? "primary" : "secondary"}
          size="sm"
          onClick={() => setTab("youtube")}
        >
          <Youtube className="h-4 w-4" />
          {PL.wizard.song.tabs.youtube}
        </Button>
        <Button
          variant={tab === "file" ? "primary" : "secondary"}
          size="sm"
          onClick={() => setTab("file")}
        >
          <FileAudio className="h-4 w-4" />
          {PL.wizard.song.tabs.file}
        </Button>
      </div>

      {tab === "youtube" ? (
        <Card className="p-4 space-y-4">
          <form role="search" onSubmit={handleSearchSubmit} className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={PL.wizard.song.searchPlaceholder}
              aria-label={PL.wizard.song.searchPlaceholder}
            />
            <Button type="submit" disabled={searching || !query.trim()}>
              <Search className="h-4 w-4" />
              {searching ? PL.wizard.song.searching : PL.wizard.song.searchButton}
            </Button>
          </form>

          {errorMsg && <p className="text-xs text-red-400">{errorMsg}</p>}

          {results.length > 0 ? (
            <div className="max-h-64 overflow-y-auto space-y-2 pr-1">
              {results.map((item) => {
                const selected = song.sourcePathOrUrl === item.url;
                return (
                  <button
                    key={item.id || item.url}
                    type="button"
                    onClick={() => handleSelectItem(item)}
                    className={
                      selected
                        ? "w-full text-left flex items-center gap-3 rounded-xl border border-blue-500 bg-blue-500/10 p-2.5 transition"
                        : "w-full text-left flex items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-950/70 hover:bg-zinc-800/70 p-2.5 transition"
                    }
                  >
                    {item.thumbnail_url ? (
                      <img
                        src={item.thumbnail_url}
                        alt={item.title}
                        className="h-12 w-20 rounded-lg object-cover shrink-0 bg-zinc-900"
                      />
                    ) : (
                      <div className="h-12 w-20 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center shrink-0">
                        <Music className="h-5 w-5 text-zinc-500" />
                      </div>
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-zinc-100 truncate">
                        {item.title}
                      </div>
                      <div className="text-xs text-zinc-400 flex items-center gap-2 mt-0.5">
                        <span className="truncate">{item.channel}</span>
                        {item.duration_s !== null && (
                          <span className="font-mono text-zinc-500">
                            {formatDuration(item.duration_s)}
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            hasSearched && (
              <p className="text-xs text-zinc-500">{PL.wizard.song.noResults}</p>
            )
          )}
        </Card>
      ) : (
        <Card className="p-4 space-y-3">
          <Button variant="secondary" onClick={handlePickAudioFile}>
            <FileAudio className="h-4 w-4" />
            {PL.wizard.song.pickFileButton}
          </Button>
          {song.sourceType === "file" && song.sourcePathOrUrl && (
            <p className="text-xs font-mono text-zinc-400 break-all">
              {PL.wizard.song.selectedFileLabel} {song.sourcePathOrUrl}
            </p>
          )}
        </Card>
      )}

      {song.sourcePathOrUrl && (
        <Card className="p-4 flex flex-col sm:flex-row gap-4 items-start">
          {song.thumbnailUrl ? (
            <img
              src={song.thumbnailUrl}
              alt={song.title}
              className="h-20 w-20 rounded-xl object-cover border border-zinc-800 shrink-0"
            />
          ) : (
            <div className="h-20 w-20 rounded-xl bg-zinc-950 border border-zinc-800 flex items-center justify-center shrink-0">
              <Disc3 className="h-8 w-8 text-zinc-600" />
            </div>
          )}

          <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
            <label className="space-y-1 text-xs">
              <span className="text-zinc-400 font-medium">{PL.wizard.song.titleLabel}</span>
              <Input
                value={song.title}
                onChange={(e) => onChange({ ...song, title: e.target.value })}
              />
            </label>
            <label className="space-y-1 text-xs">
              <span className="text-zinc-400 font-medium">{PL.wizard.song.artistLabel}</span>
              <Input
                value={song.artist}
                onChange={(e) => onChange({ ...song, artist: e.target.value })}
              />
            </label>
          </div>
        </Card>
      )}

      <div className="flex justify-end">
        <Button size="lg" disabled={!canProceed} onClick={onNext}>
          {PL.common.next}
        </Button>
      </div>
    </div>
  );
}
