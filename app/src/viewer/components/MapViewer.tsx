import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  Camera,
  Crosshair,
  Disc3,
  FolderOpen,
  Pause,
  Play,
  Sparkles,
  Volume2,
  VolumeX,
} from "lucide-react";
import { EN, RANK_LABELS_EN } from "../../i18n/en";
import type { ViewerMapPackage } from "../../types";
import { Button, SegmentedControl, cn } from "../../components/ui/primitives";
import {
  PLAYBACK_RATES,
  ViewerAudioPlayer,
  type PlaybackRate,
} from "../audioPlayer";
import {
  clampVolumePercent,
  sliderToHitsoundGain,
  sliderToMusicGain,
} from "../audioVolume";
import type { CameraPreset } from "../geometry";
import { extractHitsoundTimes } from "../hitsound";
import { computeLocalNps, parseDifficultyData } from "../parser";
import {
  SEEK_OVERLAY_DURATION_MS,
  SEEK_STEP_SEC,
  accumulateSeekOverlay,
  formatSeekOverlayText,
  type SeekOverlayState,
} from "../seekOverlay";
import { buildTempoMap, computeJumpMetrics, secondsToBeat } from "../timing";
import type { JumpMetrics, ParsedDifficultyData } from "../types";
import {
  loadViewerPrefs,
  saveHitsoundEnabled,
  saveHitsoundVolumePercent,
  saveMusicVolumePercent,
  saveQualityMode,
  type ViewerQualityMode,
} from "../viewerPrefs";
import { TimelineScrubber } from "./TimelineScrubber";
import { ViewerScene } from "./ViewerScene";

const MemoViewerScene = React.memo(ViewerScene);

interface ParsedDiffEntry {
  key: string;
  characteristic: string;
  rank: string;
  data: ParsedDifficultyData;
  jump: JumpMetrics;
}

interface PlayFlashState {
  icon: "play" | "pause";
  seq: number;
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

function rankAccentTextClass(rank: string): string {
  switch (rank) {
    case "Easy":
      return "text-emerald-300";
    case "Normal":
      return "text-sky-300";
    case "Hard":
      return "text-amber-300";
    case "Expert":
      return "text-rose-300";
    case "ExpertPlus":
      return "text-purple-300";
    default:
      return "text-zinc-200";
  }
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
  const initialPrefs = useMemo(() => loadViewerPrefs(), []);

  const [pkg, setPkg] = useState<ViewerMapPackage | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedDiffKey, setSelectedDiffKey] = useState<string>("");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("player");
  const [qualityMode, setQualityMode] = useState<ViewerQualityMode>(initialPrefs.qualityMode);

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(1);
  const [audioDecoded, setAudioDecoded] = useState(true);
  const [playbackRate, setPlaybackRate] = useState<PlaybackRate>(1);
  const [musicVolumePercent, setMusicVolumePercent] = useState(initialPrefs.musicVolumePercent);
  const [muted, setMuted] = useState(false);
  const [hitsoundEnabled, setHitsoundEnabled] = useState(initialPrefs.hitsoundEnabled);
  const [hitsoundVolumePercent, setHitsoundVolumePercent] = useState(
    initialPrefs.hitsoundVolumePercent,
  );

  const [hudTime, setHudTime] = useState(0);
  const [seekOverlay, setSeekOverlay] = useState<SeekOverlayState | null>(null);
  const [playFlash, setPlayFlash] = useState<PlayFlashState | null>(null);

  const playerRef = useRef<ViewerAudioPlayer | null>(null);
  const seekTimerRef = useRef<number | null>(null);
  const playFlashTimerRef = useRef<number | null>(null);

  if (!playerRef.current) {
    playerRef.current = new ViewerAudioPlayer();
    playerRef.current.setVolume(sliderToMusicGain(initialPrefs.musicVolumePercent));
    playerRef.current.setHitsoundEnabled(initialPrefs.hitsoundEnabled);
    playerRef.current.setHitsoundVolume(sliderToHitsoundGain(initialPrefs.hitsoundVolumePercent));
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
    const player = playerRef.current;
    if (!player) return;
    const hitTimes = activeDiff ? extractHitsoundTimes(activeDiff.data.notes) : [];
    player.setNoteHitTimes(hitTimes);
  }, [activeDiff]);

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
      if (seekTimerRef.current !== null) window.clearTimeout(seekTimerRef.current);
      if (playFlashTimerRef.current !== null) window.clearTimeout(playFlashTimerRef.current);
      void player.dispose();
    };
  }, []);

  useEffect(() => {
    let rafId = 0;
    let lastUiUpdate = 0;
    const tick = (now: number) => {
      const player = playerRef.current;
      if (player) {
        player.tickHitsounds();
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

  const triggerPlayFlash = useCallback((nextPlaying: boolean) => {
    setPlayFlash((prev) => ({
      icon: nextPlaying ? "play" : "pause",
      seq: (prev?.seq ?? 0) + 1,
    }));
    if (playFlashTimerRef.current !== null) {
      window.clearTimeout(playFlashTimerRef.current);
    }
    playFlashTimerRef.current = window.setTimeout(() => {
      setPlayFlash(null);
    }, 480);
  }, []);

  const togglePlay = useCallback(
    async (withFlash = false) => {
      const player = playerRef.current;
      if (!player || loading || loadError) return;
      if (player.isPlaying) {
        player.pause();
        setIsPlaying(false);
        setHudTime(player.getCurrentTime());
        if (withFlash) triggerPlayFlash(false);
      } else {
        await player.play();
        setIsPlaying(player.isPlaying);
        if (withFlash) triggerPlayFlash(true);
      }
    },
    [loading, loadError, triggerPlayFlash],
  );

  const seekTo = useCallback(async (nextSec: number) => {
    const player = playerRef.current;
    if (!player) return;
    await player.seek(nextSec);
    setHudTime(player.getCurrentTime());
  }, []);

  const stepSeekWithOverlay = useCallback(
    (direction: -1 | 1) => {
      const player = playerRef.current;
      if (!player || loading || loadError) return;
      const cur = player.getCurrentTime();
      void seekTo(cur + direction * SEEK_STEP_SEC);

      const now = performance.now();
      setSeekOverlay((prev) => accumulateSeekOverlay(prev, direction, now));
      if (seekTimerRef.current !== null) {
        window.clearTimeout(seekTimerRef.current);
      }
      seekTimerRef.current = window.setTimeout(() => {
        setSeekOverlay(null);
      }, SEEK_OVERLAY_DURATION_MS);
    },
    [loading, loadError, seekTo],
  );

  const changeRate = useCallback((rate: PlaybackRate) => {
    setPlaybackRate(rate);
    playerRef.current?.setPlaybackRate(rate);
  }, []);

  const changeVolume = useCallback((percent: number) => {
    const clamped = clampVolumePercent(percent);
    setMusicVolumePercent(clamped);
    setMuted(clamped === 0);
    saveMusicVolumePercent(clamped);
    playerRef.current?.setVolume(sliderToMusicGain(clamped));
  }, []);

  const toggleMute = useCallback(() => {
    const nextMuted = !muted;
    setMuted(nextMuted);
    playerRef.current?.setVolume(nextMuted ? 0 : sliderToMusicGain(musicVolumePercent));
  }, [muted, musicVolumePercent]);

  const toggleHitsound = useCallback(() => {
    const next = !hitsoundEnabled;
    setHitsoundEnabled(next);
    saveHitsoundEnabled(next);
    playerRef.current?.setHitsoundEnabled(next);
  }, [hitsoundEnabled]);

  const changeHitsoundVolume = useCallback(
    (percent: number) => {
      const clamped = clampVolumePercent(percent);
      setHitsoundVolumePercent(clamped);
      saveHitsoundVolumePercent(clamped);
      if (clamped > 0 && !hitsoundEnabled) {
        setHitsoundEnabled(true);
        saveHitsoundEnabled(true);
        playerRef.current?.setHitsoundEnabled(true);
      }
      playerRef.current?.setHitsoundVolume(sliderToHitsoundGain(clamped));
    },
    [hitsoundEnabled],
  );

  const toggleQualityMode = useCallback(() => {
    setQualityMode((prev) => {
      const next: ViewerQualityMode = prev === "high" ? "fast" : "high";
      saveQualityMode(next);
      return next;
    });
  }, []);

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
        if (!ev.repeat) {
          void togglePlay(true);
        }
      } else if (ev.code === "ArrowLeft") {
        ev.preventDefault();
        stepSeekWithOverlay(-1);
      } else if (ev.code === "ArrowRight") {
        ev.preventDefault();
        stepSeekWithOverlay(1);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, togglePlay, stepSeekWithOverlay]);

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
      <header className="shrink-0 border-b border-zinc-800/80 bg-zinc-950/95 px-4 py-2 flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-3 min-w-0">
          <Button variant="secondary" size="sm" onClick={onClose}>
            <ArrowLeft className="h-3.5 w-3.5" />
            {EN.viewer.back}
          </Button>

          {pkg?.coverDataUrl ? (
            <img
              src={pkg.coverDataUrl}
              alt={pkg.title}
              className="h-8 w-8 rounded-lg object-cover border border-zinc-800 shrink-0"
            />
          ) : (
            <div className="h-8 w-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center shrink-0">
              <Disc3 className="h-4 w-4 text-zinc-500" />
            </div>
          )}

          <div className="min-w-0">
            <div className="text-sm font-semibold text-zinc-100 truncate">
              {pkg?.title ?? EN.viewer.title}
            </div>
            <div className="text-[11px] text-zinc-400 truncate flex items-center gap-2">
              <span>{pkg?.artist ?? ""}</span>
              {pkg && (
                <span className="font-mono text-zinc-500">{Math.round(pkg.bpm)} BPM</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
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

          <button
            type="button"
            data-testid="viewer-quality-toggle"
            onClick={toggleQualityMode}
            title={`${EN.viewer.qualityLabel}: ${qualityMode === "high" ? EN.viewer.qualityHigh : EN.viewer.qualityFast}`}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer",
              qualityMode === "high"
                ? "bg-indigo-500/15 border-indigo-400/35 text-indigo-200 hover:bg-indigo-500/25"
                : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200",
            )}
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>
              {EN.viewer.qualityLabel}:{" "}
              {qualityMode === "high" ? EN.viewer.qualityHigh : EN.viewer.qualityFast}
            </span>
          </button>

          <Button variant="secondary" size="sm" onClick={handlePickFolder}>
            <FolderOpen className="h-3.5 w-3.5" />
            {EN.viewer.openMapFolder}
          </Button>
        </div>
      </header>

      {/* Main 3D Canvas + HUD */}
      <div className="relative flex-1 min-h-0 overflow-hidden bg-[#03040a]">
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
                qualityMode={qualityMode}
                getTimeS={getTimeS}
              />
            </SceneErrorBoundary>

            {/* Top-left HUD: time, beat, local NPS */}
            <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded-xl bg-zinc-950/70 border border-zinc-800/75 backdrop-blur-md px-2.5 py-1.5 text-[11px]">
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

            {/* Top-right HUD: compact grouped stats strip */}
            <div className="pointer-events-none absolute top-3 right-3 flex flex-wrap items-center gap-2 rounded-xl bg-zinc-950/70 border border-zinc-800/75 backdrop-blur-md px-2.5 py-1.5 text-[11px] font-mono tabular-nums text-zinc-300">
              <span className={cn("font-sans font-semibold", rankAccentTextClass(activeDiff.rank))}>
                {RANK_LABELS_EN[activeDiff.rank] ?? activeDiff.rank}
              </span>
              <span className="text-zinc-500">
                NJS {activeDiff.jump.njs.toFixed(1)} · HJD {activeDiff.jump.hjdBeats.toFixed(2)}b
              </span>
              <span className="text-zinc-700">|</span>
              <span className="text-zinc-100">
                {EN.viewer.stats.notes}: {activeDiff.data.notes.length}
              </span>
              {activeDiff.data.arcs.length > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <span className="text-blue-300/90">
                    {EN.viewer.stats.arcs}: {activeDiff.data.arcs.length}
                  </span>
                </>
              )}
              {activeDiff.data.chains.length > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <span className="text-sky-300/90">
                    {EN.viewer.stats.chains}: {activeDiff.data.chains.length}
                  </span>
                </>
              )}
              {activeDiff.data.obstacles.length > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <span className="text-rose-300/90">
                    {EN.viewer.stats.walls}: {activeDiff.data.obstacles.length}
                  </span>
                </>
              )}
              {activeDiff.data.bombs.length > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <span className="text-amber-300/90">
                    {EN.viewer.stats.bombs}: {activeDiff.data.bombs.length}
                  </span>
                </>
              )}
              {activeDiff.data.lightEvents.length > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <span className="text-purple-300/90">
                    {EN.viewer.stats.lights}: {activeDiff.data.lightEvents.length}
                  </span>
                </>
              )}
            </div>

            {/* YouTube-style ±5s Seek Ripple Overlay */}
            {seekOverlay && (
              <div
                data-testid="viewer-seek-overlay"
                className={cn(
                  "pointer-events-none absolute top-1/2 -translate-y-1/2 h-28 w-28 flex items-center justify-center",
                  seekOverlay.direction < 0 ? "left-10" : "right-10",
                )}
              >
                <motion.div
                  key={seekOverlay.seq}
                  initial={{ opacity: 0.96, scale: 0.92 }}
                  animate={{ opacity: 0, scale: 1.05 }}
                  transition={{ duration: SEEK_OVERLAY_DURATION_MS / 1000, ease: "easeOut" }}
                  className="relative h-full w-full rounded-full bg-black/55 border border-white/15 backdrop-blur-md flex items-center justify-center shadow-2xl"
                >
                  <motion.div
                    initial={{ opacity: 0.5, scale: 0.72 }}
                    animate={{ opacity: 0, scale: 1.45 }}
                    transition={{ duration: SEEK_OVERLAY_DURATION_MS / 1000, ease: "easeOut" }}
                    className="absolute inset-0 rounded-full border border-white/30"
                  />
                  <span className="relative text-xs font-semibold text-white tracking-wide text-center px-2">
                    {formatSeekOverlayText(seekOverlay.direction, seekOverlay.amountSec)}
                  </span>
                </motion.div>
              </div>
            )}

            {/* YouTube-style Space Play/Pause Centered Flash */}
            <AnimatePresence>
              {playFlash && (
                // Centering lives on the wrapper: framer-motion's `scale` writes `transform` and
                // would wipe out Tailwind's translate-based centering on the scaled element.
                <motion.div
                  key={playFlash.seq}
                  exit={{ opacity: 0 }}
                  className="pointer-events-none absolute inset-0 flex items-center justify-center"
                >
                  <motion.div
                    data-testid="viewer-play-flash"
                    initial={{ opacity: 0.95, scale: 0.85 }}
                    animate={{ opacity: 0, scale: 1.3 }}
                    transition={{ duration: 0.45, ease: "easeOut" }}
                    className="h-16 w-16 rounded-full bg-black/60 border border-white/20 backdrop-blur-md flex items-center justify-center shadow-2xl text-white"
                  >
                    {playFlash.icon === "play" ? (
                      <Play className="h-7 w-7 fill-white translate-x-0.5" />
                    ) : (
                      <Pause className="h-7 w-7 fill-white" />
                    )}
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

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
      <footer className="shrink-0 border-t border-zinc-800/80 bg-zinc-950/95 px-4 py-2 flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void togglePlay(false)}
          aria-label={isPlaying ? EN.viewer.pause : EN.viewer.play}
          disabled={loading || Boolean(loadError)}
          className="min-w-20"
        >
          {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {isPlaying ? EN.viewer.pause : EN.viewer.play}
        </Button>

        {/* Custom Progress Bar with filled gradient, thumb, and hover tooltip */}
        <TimelineScrubber
          currentTime={hudTime}
          duration={duration}
          disabled={loading || Boolean(loadError)}
          onSeek={(sec) => void seekTo(sec)}
        />

        {/* Speed Selector */}
        <div className="flex items-center gap-0.5 bg-zinc-900 border border-zinc-800 rounded-lg p-0.5">
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

        {/* Music Volume */}
        <div className="flex items-center gap-1.5 w-36 shrink-0" title={EN.viewer.volumeLabel}>
          <button
            type="button"
            onClick={toggleMute}
            aria-label={EN.viewer.volumeLabel}
            className="p-1 text-zinc-400 hover:text-zinc-100 cursor-pointer shrink-0"
          >
            {muted || musicVolumePercent === 0 ? (
              <VolumeX className="h-4 w-4 text-zinc-500" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </button>
          <input
            data-testid="viewer-volume-slider"
            type="range"
            aria-label={EN.viewer.volumeLabel}
            min={0}
            max={100}
            step={1}
            value={muted ? 0 : musicVolumePercent}
            onInput={(e) => changeVolume(Number((e.target as HTMLInputElement).value))}
            onChange={(e) => changeVolume(Number(e.target.value))}
            className="w-20 accent-blue-500 cursor-pointer"
          />
          <span
            data-testid="viewer-volume-percent"
            className={cn(
              "w-9 text-right font-mono text-[11px] tabular-nums shrink-0",
              muted || musicVolumePercent === 0 ? "text-zinc-500" : "text-zinc-300",
            )}
          >
            {muted ? 0 : musicVolumePercent}%
          </span>
        </div>

        {/* Hitsound Toggle + Separate Volume Slider */}
        <div
          className="flex items-center gap-1.5 pl-2 border-l border-zinc-800/80 w-44 shrink-0"
          title={EN.viewer.hitsoundLabel}
        >
          <button
            type="button"
            data-testid="viewer-hitsound-toggle"
            onClick={toggleHitsound}
            aria-pressed={hitsoundEnabled}
            aria-label={EN.viewer.hitsoundLabel}
            className={cn(
              "inline-flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-medium transition-colors cursor-pointer shrink-0",
              hitsoundEnabled
                ? "bg-emerald-500/15 border-emerald-500/35 text-emerald-300 hover:bg-emerald-500/25"
                : "bg-zinc-900 border-zinc-800 text-zinc-500 hover:text-zinc-300",
            )}
          >
            <Crosshair className="h-3 w-3" />
            <span>Hit</span>
          </button>
          <input
            data-testid="viewer-hitsound-volume"
            type="range"
            aria-label={EN.viewer.hitsoundVolumeLabel}
            min={0}
            max={100}
            step={1}
            value={hitsoundEnabled ? hitsoundVolumePercent : 0}
            onInput={(e) => changeHitsoundVolume(Number((e.target as HTMLInputElement).value))}
            onChange={(e) => changeHitsoundVolume(Number(e.target.value))}
            className="w-20 accent-emerald-500 cursor-pointer"
          />
          <span
            data-testid="viewer-hitsound-percent"
            className={cn(
              "w-9 text-right font-mono text-[11px] tabular-nums shrink-0",
              !hitsoundEnabled || hitsoundVolumePercent === 0 ? "text-zinc-500" : "text-emerald-300/90",
            )}
          >
            {hitsoundEnabled ? hitsoundVolumePercent : 0}%
          </span>
        </div>
      </footer>
    </div>
  );
}
