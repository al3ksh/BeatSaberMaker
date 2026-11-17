import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Camera,
  Disc3,
  FolderOpen,
  Pause,
  Play,
  RotateCcw,
  RotateCw,
  Volume2,
  VolumeX,
} from "lucide-react";
import { EN, RANK_LABELS_EN } from "../../i18n/en";
import type { ViewerMapPackage } from "../../types";
import { Badge, Button, SegmentedControl, cn } from "../../components/ui/primitives";
import { rankToBadgeVariant } from "../../components/wizard/StepMap";
import {
  DEFAULT_MUSIC_VOLUME,
  PLAYBACK_RATES,
  ViewerAudioPlayer,
  type PlaybackRate,
} from "../audioPlayer";
import type { CameraPreset } from "../geometry";
import { computeLocalNps, parseDifficultyData } from "../parser";
import { buildTempoMap, computeJumpMetrics, secondsToBeat } from "../timing";
import type { JumpMetrics, ParsedDifficultyData } from "../types";
import { ViewerScene } from "./ViewerScene";

const MemoViewerScene = React.memo(ViewerScene);

interface ParsedDiffEntry {
  key: string;
  characteristic: string;
  rank: string;
  data: ParsedDifficultyData;
  jump: JumpMetrics;
}

function formatTimeSec(sec: number, showDecimal = false): string {
  const safe = Math.max(0, Number.isFinite(sec) ? sec : 0);
  const mins = Math.floor(safe / 60);
  const rem = safe - mins * 60;
  if (showDecimal) {
    const whole = Math.floor(rem);
    const tenths = Math.floor((rem - whole) * 10);
    return `${mins}:${String(whole).padStart(2, "0")}.${tenths}`;
  }
  const whole = Math.floor(rem);
  return `${mins}:${String(whole).padStart(2, "0")}`;
}

function computeFallbackDuration(data: ParsedDifficultyData): number {
  let maxT = 10;
  for (const n of data.notes) if (n.timeS > maxT) maxT = n.timeS;
  for (const b of data.bombs) if (b.timeS > maxT) maxT = b.timeS;
  for (const o of data.obstacles) if (o.endTimeS > maxT) maxT = o.endTimeS;
  for (const a of data.arcs) if (a.tailTimeS > maxT) maxT = a.tailTimeS;
  for (const c of data.chains) if (c.tailTimeS > maxT) maxT = c.tailTimeS;
  for (const e of data.lightEvents) if (e.timeS > maxT) maxT = e.timeS;
  return maxT + 3;
}

class SceneErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: boolean }
> {
  state = { error: false };
  static getDerivedStateFromError() {
    return { error: true };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="h-full w-full flex items-center justify-center text-sm text-zinc-400">
          WebGL 3D context could not be initialized.
        </div>
      );
    }
    return this.props.children;
  }
}

export interface MapViewerProps {
  folder: string;
  onClose: () => void;
  onSwitchFolder: (nextFolder: string) => void;
}

export function MapViewer({ folder, onClose, onSwitchFolder }: MapViewerProps) {
  const [pkg, setPkg] = useState<ViewerMapPackage | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedDiffKey, setSelectedDiffKey] = useState<string>("");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("player");

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(1);
  const [audioDecoded, setAudioDecoded] = useState(true);
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1);
  const [volume, setVolume] = useState(DEFAULT_MUSIC_VOLUME);
  const [muted, setMuted] = useState(false);

  const [hudTime, setHudTime] = useState(0);
  const playerRef = useRef<ViewerAudioPlayer | null>(null);

  if (!playerRef.current) {
    playerRef.current = new ViewerAudioPlayer();
  }

  const parsedDiffs = useMemo<ParsedDiffEntry[]>(() => {
    if (!pkg) return [];
    const baseTempo = buildTempoMap(pkg.bpm, pkg.songTimeOffset, []);
    return pkg.difficulties.map((d, idx) => {
      const data = parseDifficultyData(d.raw, baseTempo);
      const jump = computeJumpMetrics(pkg.bpm, d.njs, d.njsOffset, d.rank);
      return {
        key: `${d.characteristic}:${d.rank}:${idx}`,
        characteristic: d.characteristic,
        rank: d.rank,
        data,
        jump,
      };
    });
  }, [pkg]);

  const activeDiff = useMemo(() => {
    if (parsedDiffs.length === 0) return null;
    return parsedDiffs.find((d) => d.key === selectedDiffKey) ?? parsedDiffs[0];
  }, [parsedDiffs, selectedDiffKey]);

  useEffect(() => {
    let active = true;
    const player = playerRef.current!;
    setLoading(true);
    setLoadError(null);
    setIsPlaying(false);
    setHudTime(0);

    window.bsm
      .readMapForViewer(folder)
      .then(async (loaded) => {
        if (!active) return;
        setPkg(loaded);

        const baseTempo = buildTempoMap(loaded.bpm, loaded.songTimeOffset, []);
        const firstRaw = loaded.difficulties[0];
        const fallbackDur = firstRaw
          ? computeFallbackDuration(parseDifficultyData(firstRaw.raw, baseTempo))
          : 60;

        if (loaded.difficulties.length > 0) {
          // Pick highest-rank Standard difficulty by default or index 0
          let pickIdx = 0;
          let bestScore = -1;
          loaded.difficulties.forEach((d, idx) => {
            const score = (d.characteristic === "Standard" ? 100 : 0) + d.difficultyRank;
            if (score > bestScore) {
              bestScore = score;
              pickIdx = idx;
            }
          });
          const picked = loaded.difficulties[pickIdx];
          setSelectedDiffKey(`${picked.characteristic}:${picked.rank}:${pickIdx}`);
        }

        const ok = await player.load(loaded.audioBytes, fallbackDur);
        if (!active) return;
        setAudioDecoded(ok);
        setDuration(player.duration);
        setHudTime(0);
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setLoadError(EN.viewer.loadError);
        setLoading(false);
      });

    return () => {
      active = false;
      player.pause();
    };
  }, [folder]);

  useEffect(() => {
    const player = playerRef.current!;
    player.onEnded(() => {
      setIsPlaying(false);
      setHudTime(player.duration);
    });
    return () => {
      void player.dispose();
    };
  }, []);

  useEffect(() => {
    let rafId = 0;
    let lastUiUpdate = 0;
    const tick = (now: number) => {
      const player = playerRef.current;
      if (player) {
        if (now - lastUiUpdate >= 45) {
          lastUiUpdate = now;
          const t = player.getCurrentTime();
          setHudTime(t);
          if (player.isPlaying !== isPlaying) {
            setIsPlaying(player.isPlaying);
          }
        }
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  const getTimeS = useCallback(() => playerRef.current?.getCurrentTime() ?? 0, []);

  const togglePlay = useCallback(async () => {
    const player = playerRef.current;
    if (!player || loading || loadError) return;
    if (player.isPlaying) {
      player.pause();
      setIsPlaying(false);
      setHudTime(player.getCurrentTime());
    } else {
      await player.play();
      setIsPlaying(player.isPlaying);
    }
  }, [loading, loadError]);

  const seekTo = useCallback(async (nextSec: number) => {
    const player = playerRef.current;
    if (!player) return;
    await player.seek(nextSec);
    setHudTime(player.getCurrentTime());
  }, []);

  const changeRate = useCallback((rate: PlaybackRate) => {
    setPlaybackRate(rate);
    playerRef.current?.setPlaybackRate(rate);
  }, []);

  const changeVolume = useCallback(
    (vol: number) => {
      setVolume(vol);
      setMuted(vol === 0);
      playerRef.current?.setVolume(vol);
    },
    [],
  );

  const toggleMute = useCallback(() => {
    const nextMuted = !muted;
    setMuted(nextMuted);
    playerRef.current?.setVolume(nextMuted ? 0 : volume);
  }, [muted, volume]);

  useEffect(() => {
    function onKeyDown(ev: KeyboardEvent) {
      const tag = (ev.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" && (ev.target as HTMLInputElement).type !== "range") return;
      if (tag === "TEXTAREA" || tag === "SELECT") return;

      if (ev.code === "Escape") {
        ev.preventDefault();
        onClose();
      } else if (ev.code === "Space") {
        ev.preventDefault();
        void togglePlay();
      } else if (ev.code === "ArrowLeft") {
        ev.preventDefault();
        const cur = playerRef.current?.getCurrentTime() ?? 0;
        void seekTo(cur - 5);
      } else if (ev.code === "ArrowRight") {
        ev.preventDefault();
        const cur = playerRef.current?.getCurrentTime() ?? 0;
        void seekTo(cur + 5);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, togglePlay, seekTo]);

  async function handlePickFolder() {
    const picked = await window.bsm.pickFolder(EN.viewer.openMapFolder);
    if (picked) {
      onSwitchFolder(picked);
    }
  }

  const currentBeat = activeDiff ? secondsToBeat(hudTime, activeDiff.data.tempo) : 0;
  const localNps = activeDiff ? computeLocalNps(activeDiff.data.notes, hudTime) : 0;

  return (
    <div
      data-testid="map-viewer"
      className="fixed inset-0 z-50 bg-zinc-950 text-zinc-100 flex flex-col select-none"
    >
      {/* Top Bar */}
      <header className="shrink-0 border-b border-zinc-800/80 bg-zinc-950/95 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <Button variant="secondary" size="sm" onClick={onClose}>
            <ArrowLeft className="h-3.5 w-3.5" />
            {EN.viewer.back}
          </Button>

          {pkg?.coverDataUrl ? (
            <img
              src={pkg.coverDataUrl}
              alt={pkg.title}
              className="h-9 w-9 rounded-lg object-cover border border-zinc-800 shrink-0"
            />
          ) : (
            <div className="h-9 w-9 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center shrink-0">
              <Disc3 className="h-4 w-4 text-zinc-500" />
            </div>
          )}

          <div className="min-w-0">
            <div className="text-sm font-semibold text-zinc-100 truncate">
              {pkg?.title ?? EN.viewer.title}
            </div>
            <div className="text-xs text-zinc-400 truncate flex items-center gap-2">
              <span>{pkg?.artist ?? ""}</span>
              {pkg && (
                <span className="font-mono text-[11px] text-zinc-500">
                  {Math.round(pkg.bpm)} BPM
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {parsedDiffs.length > 0 && (
            <div className="flex items-center gap-1 bg-zinc-900/90 border border-zinc-800 rounded-xl p-1">
              {parsedDiffs.map((d) => {
                const active = d.key === activeDiff?.key;
                const label =
                  d.characteristic !== "Standard"
                    ? `${d.characteristic} ${RANK_LABELS_EN[d.rank] ?? d.rank}`
                    : (RANK_LABELS_EN[d.rank] ?? d.rank);
                return (
                  <button
                    key={d.key}
                    type="button"
                    onClick={() => setSelectedDiffKey(d.key)}
                    className={cn(
                      "px-2.5 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer",
                      active
                        ? "bg-blue-600 text-white shadow-sm"
                        : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800",
                    )}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          )}

          <SegmentedControl
            size="sm"
            value={cameraPreset}
            onChange={setCameraPreset}
            options={[
              {
                value: "player",
                label: EN.viewer.cameras.player,
                icon: <Camera className="h-3 w-3" />,
              },
              { value: "side", label: EN.viewer.cameras.side },
              { value: "top", label: EN.viewer.cameras.top },
            ]}
          />

          <Button variant="secondary" size="sm" onClick={handlePickFolder}>
            <FolderOpen className="h-3.5 w-3.5" />
            {EN.viewer.openMapFolder}
          </Button>
        </div>
      </header>

      {/* Main 3D Canvas + HUD */}
      <div className="relative flex-1 min-h-0 overflow-hidden bg-[#05060b]">
        {loading ? (
          <div className="h-full w-full flex flex-col items-center justify-center gap-3 text-sm text-zinc-400">
            <Disc3 className="h-8 w-8 text-blue-400 animate-spin" />
            <span>{EN.viewer.loadingMap}</span>
          </div>
        ) : loadError ? (
          <div className="h-full w-full flex flex-col items-center justify-center gap-3 text-sm text-red-300 px-6 text-center">
            <p>{loadError}</p>
            <Button variant="secondary" onClick={handlePickFolder}>
              <FolderOpen className="h-4 w-4" />
              {EN.viewer.openMapFolder}
            </Button>
          </div>
        ) : !activeDiff ? (
          <div className="h-full w-full flex items-center justify-center text-sm text-zinc-400">
            {EN.viewer.noDifficulties}
          </div>
        ) : (
          <>
            <SceneErrorBoundary>
              <MemoViewerScene
                difficulty={activeDiff.data}
                njs={activeDiff.jump.njs}
                halfJumpDurationSec={activeDiff.jump.halfJumpDurationSec}
                cameraPreset={cameraPreset}
                getTimeS={getTimeS}
              />
            </SceneErrorBoundary>

            {/* Top-left HUD: time, beat, local NPS */}
            <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap items-center gap-2 rounded-xl bg-zinc-950/75 border border-zinc-800/80 backdrop-blur-md px-3 py-2 text-xs">
              <div data-testid="viewer-time" className="font-mono font-semibold text-zinc-100 tabular-nums">
                {formatTimeSec(hudTime, true)} / {formatTimeSec(duration, false)}
              </div>
              <span className="text-zinc-600">·</span>
              <div data-testid="viewer-beat" className="font-mono text-blue-300 tabular-nums">
                {EN.viewer.stats.beat} {currentBeat.toFixed(1)}
              </div>
              <span className="text-zinc-600">·</span>
              <div data-testid="viewer-nps" className="font-mono text-rose-300 tabular-nums">
                {localNps} {EN.viewer.stats.nps}
              </div>
            </div>

            {/* Top-right HUD: object counts & NJS/HJD */}
            <div className="pointer-events-none absolute top-3 right-3 flex flex-wrap items-center justify-end gap-1.5 max-w-md">
              <Badge variant={rankToBadgeVariant(activeDiff.rank)}>
                {RANK_LABELS_EN[activeDiff.rank] ?? activeDiff.rank} · NJS{" "}
                {activeDiff.jump.njs.toFixed(1)} · HJD {activeDiff.jump.hjdBeats.toFixed(2)}b
              </Badge>
              <Badge variant="neutral">
                {EN.viewer.stats.notes}: {activeDiff.data.notes.length}
              </Badge>
              {activeDiff.data.arcs.length > 0 && (
                <Badge variant="blue">
                  {EN.viewer.stats.arcs}: {activeDiff.data.arcs.length}
                </Badge>
              )}
              {activeDiff.data.chains.length > 0 && (
                <Badge variant="sky">
                  {EN.viewer.stats.chains}: {activeDiff.data.chains.length}
                </Badge>
              )}
              {activeDiff.data.obstacles.length > 0 && (
                <Badge variant="red">
                  {EN.viewer.stats.walls}: {activeDiff.data.obstacles.length}
                </Badge>
              )}
              {activeDiff.data.bombs.length > 0 && (
                <Badge variant="amber">
                  {EN.viewer.stats.bombs}: {activeDiff.data.bombs.length}
                </Badge>
              )}
              {activeDiff.data.lightEvents.length > 0 && (
                <Badge variant="purple">
                  {EN.viewer.stats.lights}: {activeDiff.data.lightEvents.length}
                </Badge>
              )}
            </div>

            {/* Notices (no audio / variable BPM) */}
            {(!audioDecoded || activeDiff.data.hasBpmChanges) && (
              <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col gap-1 text-[11px] text-amber-300/90 bg-zinc-950/75 border border-amber-500/30 rounded-lg px-2.5 py-1">
                {!audioDecoded && <span>{EN.viewer.noAudioNotice}</span>}
                {activeDiff.data.hasBpmChanges && <span>{EN.viewer.bpmChangesNotice}</span>}
              </div>
            )}
          </>
        )}
      </div>

      {/* Bottom Transport Controls */}
      <footer className="shrink-0 border-t border-zinc-800/80 bg-zinc-950/95 px-4 py-2.5 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void seekTo((playerRef.current?.getCurrentTime() ?? 0) - 5)}
            aria-label={EN.viewer.seekBack}
            title={EN.viewer.seekBack}
            disabled={loading || Boolean(loadError)}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            -5s
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => void togglePlay()}
            aria-label={isPlaying ? EN.viewer.pause : EN.viewer.play}
            disabled={loading || Boolean(loadError)}
            className="min-w-20"
          >
            {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {isPlaying ? EN.viewer.pause : EN.viewer.play}
          </Button>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => void seekTo((playerRef.current?.getCurrentTime() ?? 0) + 5)}
            aria-label={EN.viewer.seekForward}
            title={EN.viewer.seekForward}
            disabled={loading || Boolean(loadError)}
          >
            +5s
            <RotateCw className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Timeline Scrubber */}
        <div className="flex-1 min-w-44 flex items-center gap-2.5">
          <input
            data-testid="viewer-seek"
            type="range"
            aria-label="Seek timeline"
            min={0}
            max={Math.max(1, duration)}
            step={0.01}
            value={Math.min(hudTime, duration)}
            disabled={loading || Boolean(loadError)}
            onChange={(e) => {
              void seekTo(Number(e.target.value));
            }}
            className="w-full accent-blue-500 cursor-pointer"
          />
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-0.5">
          {PLAYBACK_RATES.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => changeRate(r)}
              className={cn(
                "px-2 py-1 rounded-md text-[11px] font-mono transition-colors cursor-pointer",
                playbackRate === r
                  ? "bg-blue-600 text-white font-semibold"
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800",
              )}
            >
              {r}x
            </button>
          ))}
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2 w-28">
          <button
            type="button"
            onClick={toggleMute}
            aria-label={EN.viewer.volumeLabel}
            className="p-1 text-zinc-400 hover:text-zinc-100 cursor-pointer"
          >
            {muted || volume === 0 ? (
              <VolumeX className="h-4 w-4 text-zinc-500" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </button>
          <input
            type="range"
            aria-label={EN.viewer.volumeLabel}
            min={0}
            max={1}
            step={0.02}
            value={muted ? 0 : volume}
            onChange={(e) => changeVolume(Number(e.target.value))}
            className="w-full accent-blue-500 cursor-pointer"
          />
        </div>
      </footer>
    </div>
  );
}
