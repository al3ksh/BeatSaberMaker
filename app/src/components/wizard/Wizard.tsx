import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PL } from "../../i18n/pl";
import type {
  GenerateOutput,
  LibraryEntry,
  MapGenerationConfig,
  ProgressEvent,
} from "../../types";
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
    { num: 1, label: PL.wizard.steps.song },
    { num: 2, label: PL.wizard.steps.map },
    { num: 3, label: PL.wizard.steps.done },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        {steps.map((s, idx) => {
          const active = step === s.num;
          const done = step > s.num;
          return (
            <div key={s.num} className="flex items-center gap-2">
              <div
                className={
                  active
                    ? "flex items-center gap-2 rounded-full bg-blue-500/15 border border-blue-500/40 px-3 py-1 text-xs font-semibold text-blue-300"
                    : done
                      ? "flex items-center gap-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-medium text-emerald-300"
                      : "flex items-center gap-2 rounded-full bg-zinc-900 border border-zinc-800 px-3 py-1 text-xs font-medium text-zinc-500"
                }
              >
                <span>{s.num}.</span>
                <span>{s.label}</span>
              </div>
              {idx < steps.length - 1 && (
                <div className="h-px w-6 bg-zinc-800" />
              )}
            </div>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div
            key="step-1"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.15 }}
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
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.15 }}
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
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.15 }}
          >
            <StepProgressDone
              hasDownloadStage={song.sourceType === "youtube"}
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
