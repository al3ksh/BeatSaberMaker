import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Check } from "lucide-react";
import { EN } from "../../i18n/en";
import type {
  GenerateOutput,
  LibraryEntry,
  MapGenerationConfig,
  ProgressEvent,
} from "../../types";
import { cn } from "../ui/primitives";
import { PACK_DEFAULT_CONFIG, StepMap } from "./StepMap";
import { StepProgressDone } from "./StepProgressDone";
import { SongDraft, StepSong } from "./StepSong";

const EMPTY_SONG: SongDraft = {
  sourceType: "youtube",
  sourcePathOrUrl: "",
  title: "",
  artist: "",
  coverPath: null,
  thumbnailUrl: null,
};

function extractErrorCode(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const m = /BSM_CODE:([a-z0-9_]+)/i.exec(msg);
  return m ? m[1] : "internal";
}

export interface WizardProps {
  regenerateTarget: LibraryEntry | null;
  onClearRegenerate: () => void;
  onGoToLibrary: () => void;
}

export function Wizard({
  regenerateTarget,
  onClearRegenerate,
  onGoToLibrary,
}: WizardProps) {
  const [step, setStep] = useState<1 | 2 | 3>(regenerateTarget ? 2 : 1);
  const [song, setSong] = useState<SongDraft>(EMPTY_SONG);
  const [config, setConfig] = useState<MapGenerationConfig>(PACK_DEFAULT_CONFIG);
  const [status, setStatus] = useState<"running" | "done" | "error">("running");
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [result, setResult] = useState<GenerateOutput | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);

  useEffect(() => {
    if (!regenerateTarget) return;
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
    setStep(2);
    setResult(null);
    setErrorCode(null);
  }, [regenerateTarget]);

  function handleReset() {
    onClearRegenerate();
    setSong(EMPTY_SONG);
    setConfig(PACK_DEFAULT_CONFIG);
    setProgress(null);
    setResult(null);
    setErrorCode(null);
    setStep(1);
  }

  async function handleGenerate() {
    const effectiveConfig: MapGenerationConfig =
      config.mode === "pack"
        ? {
            ...config,
            seed:
              config.seed > 0
                ? config.seed
                : Math.floor(Math.random() * 2_000_000_000) + 1,
          }
        : config;

    setConfig(effectiveConfig);
    setStep(3);
    setStatus("running");
    setProgress({
      stage: song.sourceType === "youtube" ? "downloading" : "analyzing",
      pct: 0,
    });
    setResult(null);
    setErrorCode(null);

    const unsub = window.bsm.onProgress?.((ev) => {
      setProgress(ev);
    });

    try {
      const { result: genRes } = await window.bsm.startGeneration({
        sourceType: song.sourceType,
        sourcePathOrUrl: song.sourcePathOrUrl,
        title: song.title,
        artist: song.artist || "Unknown",
        coverPath: song.coverPath,
        config: effectiveConfig,
        regenerateId: regenerateTarget?.id,
      });
      setResult(genRes);
      setStatus("done");
      onClearRegenerate();
    } catch (err) {
      const code = extractErrorCode(err);
      if (code === "cancelled") {
        setStep(2);
        return;
      }
      setErrorCode(code);
      setStatus("error");
    } finally {
      unsub?.();
    }
  }

  const steps: Array<{ num: 1 | 2 | 3; label: string }> = [
    { num: 1, label: EN.wizard.steps.song },
    { num: 2, label: EN.wizard.steps.map },
    { num: 3, label: EN.wizard.steps.done },
  ];

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-4">
      {/* Step Pills with Connecting Arrows */}
      <div className="shrink-0 flex items-center justify-center max-w-lg mx-auto w-full">
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
              onChange={setSong}
              onNext={() => setStep(2)}
            />
          </motion.div>
        )}

        {step === 2 && (
          <motion.div
            key="step-2"
            className="flex-1 min-h-0 flex flex-col"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.14 }}
          >
            <StepMap
              config={config}
              isRegenerate={Boolean(regenerateTarget)}
              onChange={setConfig}
              onBack={() => {
                if (regenerateTarget) {
                  onClearRegenerate();
                }
                setStep(1);
              }}
              onGenerate={handleGenerate}
            />
          </motion.div>
        )}

        {step === 3 && (
          <motion.div
            key="step-3"
            className="flex-1 min-h-0 flex flex-col"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.14 }}
          >
            <StepProgressDone
              hasDownloadStage={song.sourceType === "youtube"}
              songTitle={song.title}
              songArtist={song.artist}
              thumbnailUrl={song.thumbnailUrl}
              status={status}
              progress={progress}
              result={result}
              errorCode={errorCode}
              onReset={handleReset}
              onGoToLibrary={onGoToLibrary}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
