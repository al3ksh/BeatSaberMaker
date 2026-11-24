import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ArrowRight, Check } from "lucide-react";
import { EN, extractErrorCode, formatErrorEn } from "../../i18n/en";
import type { LibraryEntry, MapGenerationConfig } from "../../types";
import { QueueAddedNotice } from "../QueueAddedNotice";
import { cn } from "../ui/primitives";
import { PACK_DEFAULT_CONFIG, StepMap } from "./StepMap";
import { SongDraft, StepSong } from "./StepSong";

const EMPTY_SONG: SongDraft = {
  sourceType: "youtube",
  sourcePathOrUrl: "",
  title: "",
  artist: "",
  coverPath: null,
  thumbnailUrl: null,
};

const NOTICE_TIMEOUT_MS = 4000;
const PACK_SEED_MAX = 2_000_000_000;

export interface WizardProps {
  regenerateTarget: LibraryEntry | null;
  onClearRegenerate: () => void;
  onGoToQueue?: () => void;
}

export function Wizard({
  regenerateTarget,
  onClearRegenerate,
  onGoToQueue,
}: WizardProps) {
  const [step, setStep] = useState<1 | 2>(regenerateTarget ? 2 : 1);
  const [song, setSong] = useState<SongDraft>(EMPTY_SONG);
  const [config, setConfig] = useState<MapGenerationConfig>(PACK_DEFAULT_CONFIG);
  const [submitting, setSubmitting] = useState(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);
  const [addedNoticeTitle, setAddedNoticeTitle] = useState<string | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!regenerateTarget) {
      return;
    }
    setSong({
      sourceType: "cached",
      sourcePathOrUrl: regenerateTarget.audioCachePath,
      title: regenerateTarget.title,
      artist: regenerateTarget.artist,
      coverPath: regenerateTarget.coverCachePath,
      thumbnailUrl: null,
    });
    setConfig({
      ...regenerateTarget.config,
      mode: "custom",
    });
    setEnqueueError(null);
    setStep(2);
  }, [regenerateTarget]);

  function showAddedNotice(title: string) {
    setAddedNoticeTitle(title);
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = setTimeout(() => {
      setAddedNoticeTitle(null);
      noticeTimerRef.current = null;
    }, NOTICE_TIMEOUT_MS);
  }

  function dismissAddedNotice() {
    if (noticeTimerRef.current) {
      clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setAddedNoticeTitle(null);
  }

  async function handleEnqueue() {
    if (submitting) return;
    const effectiveConfig: MapGenerationConfig =
      config.mode === "pack"
        ? {
            ...config,
            seed:
              config.seed > 0
                ? config.seed
                : Math.floor(Math.random() * PACK_SEED_MAX) + 1,
          }
        : config;

    setSubmitting(true);
    setEnqueueError(null);

    try {
      await window.bsm.enqueueGeneration({
        sourceType: song.sourceType,
        sourcePathOrUrl: song.sourcePathOrUrl,
        title: song.title,
        artist: song.artist || "Unknown",
        coverPath: song.coverPath,
        thumbnailUrl: song.thumbnailUrl,
        config: effectiveConfig,
        regenerateId: regenerateTarget?.id,
      });
      const queuedTitle = song.title;
      onClearRegenerate();
      setSong(EMPTY_SONG);
      setConfig(PACK_DEFAULT_CONFIG);
      setStep(1);
      showAddedNotice(queuedTitle);
    } catch (err) {
      setEnqueueError(formatErrorEn(extractErrorCode(err)));
    } finally {
      setSubmitting(false);
    }
  }

  const steps: Array<{ num: 1 | 2; label: string }> = [
    { num: 1, label: EN.wizard.steps.song },
    { num: 2, label: EN.wizard.steps.map },
  ];

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-4">
      {/* Step Pills with Connecting Arrow */}
      <div className="shrink-0 flex items-center justify-center max-w-sm mx-auto w-full">
        {steps.map((s, idx) => {
          const active = step === s.num;
          const done = step > s.num;
          return (
            <div
              key={s.num}
              className={cn(
                "flex items-center",
                idx < steps.length - 1 ? "flex-1" : "",
              )}
            >
              <div
                className={cn(
                  "flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors select-none",
                  active
                    ? "bg-blue-500/15 border-blue-500/50 text-blue-100"
                    : done
                      ? "bg-emerald-500/10 border-emerald-500/35 text-emerald-200"
                      : "bg-zinc-900 border-zinc-800 text-zinc-500",
                )}
              >
                <span
                  className={cn(
                    "h-5 w-5 rounded-full flex items-center justify-center text-[11px] font-semibold shrink-0",
                    active
                      ? "bg-blue-600 text-white"
                      : done
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        : "bg-zinc-800 text-zinc-400",
                  )}
                >
                  {done ? <Check className="h-3 w-3" /> : s.num}
                </span>
                <span className={active ? "font-semibold" : ""}>{s.label}</span>
              </div>

              {idx < steps.length - 1 && (
                <div className="flex-1 flex items-center mx-2">
                  <div
                    className={cn(
                      "h-[2px] flex-1 rounded-full transition-colors duration-200",
                      step > s.num ? "bg-emerald-500/60" : "bg-zinc-800",
                    )}
                  />
                  <ArrowRight
                    className={cn(
                      "h-3.5 w-3.5 -ml-1 shrink-0 transition-colors duration-200",
                      step > s.num ? "text-emerald-400" : "text-zinc-600",
                    )}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {addedNoticeTitle && step === 1 && (
        <QueueAddedNotice
          title={addedNoticeTitle}
          onViewQueue={
            onGoToQueue
              ? () => {
                  dismissAddedNotice();
                  onGoToQueue();
                }
              : undefined
          }
          onDismiss={dismissAddedNotice}
        />
      )}

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div
            key="step-1"
            className="flex-1 min-h-0 flex flex-col"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.14 }}
          >
            <StepSong
              song={song}
              onChange={(next) => {
                setSong(next);
                if (addedNoticeTitle) dismissAddedNotice();
              }}
              onNext={() => {
                setEnqueueError(null);
                setStep(2);
              }}
            />
          </motion.div>
        )}

        {step === 2 && (
          <motion.div
            key="step-2"
            className="flex-1 min-h-0 flex flex-col gap-3"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.14 }}
          >
            {enqueueError && (
              <div
                role="alert"
                className="shrink-0 flex items-center gap-2 rounded-xl bg-red-500/10 border border-red-500/30 px-3.5 py-2 text-xs text-red-200"
              >
                <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                <span>{enqueueError}</span>
              </div>
            )}
            <StepMap
              config={config}
              isRegenerate={Boolean(regenerateTarget)}
              isSubmitting={submitting}
              onChange={setConfig}
              onBack={() => {
                if (regenerateTarget) {
                  onClearRegenerate();
                }
                setEnqueueError(null);
                setStep(1);
              }}
              onGenerate={() => {
                void handleEnqueue();
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
