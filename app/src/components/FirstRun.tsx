import { useEffect, useState } from "react";
import { CheckCircle2, Download, FolderOpen, Sparkles } from "lucide-react";
import { PL } from "../i18n/pl";
import type { EngineInfo, EnginePackProgress, Settings, StartupState } from "../types";
import { Badge, Button, Card, ProgressBar } from "./ui/primitives";

export interface FirstRunProps {
  startup: StartupState;
  onComplete: (settings: Settings, info: EngineInfo | null) => void;
}

export function FirstRun({ startup, onComplete }: FirstRunProps) {
  const [engineReady, setEngineReady] = useState(!startup.needsEnginePack);
  const [downloading, setDownloading] = useState(false);
  const [packProgress, setPackProgress] = useState<EnginePackProgress | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const initialFolder =
    startup.settings.customLevels || startup.info?.detected_custom_levels || "";
  const [folder, setFolder] = useState(initialFolder);
  const [folderValid, setFolderValid] = useState(Boolean(initialFolder));
  const [folderError, setFolderError] = useState<string | null>(null);
  const [info, setInfo] = useState<EngineInfo | null>(startup.info);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const unsub = window.bsm.onEnginePackProgress?.((ev) => {
      setPackProgress(ev);
    });
    return () => {
      unsub?.();
    };
  }, []);

  async function handleDownloadEngine() {
    setDownloading(true);
    setDownloadError(null);
    try {
      await window.bsm.installEnginePack();
      const fetchedInfo = await window.bsm.getInfo();
      setInfo(fetchedInfo);
      setEngineReady(true);
      if (!folder && fetchedInfo.detected_custom_levels) {
        setFolder(fetchedInfo.detected_custom_levels);
        setFolderValid(true);
        setFolderError(null);
      }
    } catch {
      setDownloadError(PL.firstRun.errorDownload);
    } finally {
      setDownloading(false);
    }
  }

  async function handlePickFolder() {
    setFolderError(null);
    const picked = await window.bsm.pickFolder(PL.firstRun.folderLabel);
    if (!picked) return;
    const res = await window.bsm.validateCustomLevels(picked);
    if (res.valid && res.resolvedPath) {
      setFolder(res.resolvedPath);
      setFolderValid(true);
      setFolderError(null);
    } else {
      setFolder(picked);
      setFolderValid(false);
      setFolderError(PL.firstRun.invalidFolder);
    }
  }

  async function handleContinue() {
    if (!engineReady || !folderValid || !folder) return;
    setSaving(true);
    try {
      const nextSettings = await window.bsm.saveSettings({ customLevels: folder });
      onComplete(nextSettings, info);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center p-6">
      <Card className="w-full max-w-xl p-6 space-y-6">
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-red-500/20 to-blue-500/20 border border-zinc-700 flex items-center justify-center shrink-0">
            <Sparkles className="h-5 w-5 text-blue-400" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{PL.firstRun.title}</h1>
            <p className="text-sm text-zinc-400 mt-1">{PL.firstRun.subtitle}</p>
          </div>
        </div>

        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-200">1. Silnik AI</span>
            {engineReady ? (
              <Badge variant="emerald">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {PL.firstRun.engineReady}
              </Badge>
            ) : (
              <Button
                onClick={handleDownloadEngine}
                disabled={downloading}
                size="sm"
              >
                <Download className="h-4 w-4" />
                {PL.firstRun.downloadEngine}
              </Button>
            )}
          </div>

          {downloading && packProgress && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-zinc-400">
                <span>{PL.firstRun.stages[packProgress.stage]}</span>
                <span className="font-mono">{packProgress.pct}%</span>
              </div>
              <ProgressBar pct={packProgress.pct} />
            </div>
          )}

          {downloadError && (
            <p className="text-xs text-red-400">{downloadError}</p>
          )}
        </div>

        <div className="rounded-xl bg-zinc-950/70 border border-zinc-800 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium text-zinc-200">
              2. {PL.firstRun.folderLabel}
            </span>
            <Button variant="secondary" size="sm" onClick={handlePickFolder}>
              <FolderOpen className="h-4 w-4" />
              {PL.firstRun.pickFolder}
            </Button>
          </div>

          <div className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-xs font-mono text-zinc-300 break-all">
            {folder || PL.firstRun.folderPlaceholder}
          </div>

          {info?.detected_custom_levels && folder === info.detected_custom_levels && (
            <p className="text-xs text-emerald-400">{PL.firstRun.detectedHint}</p>
          )}
          {folderError && <p className="text-xs text-red-400">{folderError}</p>}
        </div>

        <div className="flex justify-end">
          <Button
            size="lg"
            disabled={!engineReady || !folderValid || saving}
            onClick={handleContinue}
          >
            {PL.common.next}
          </Button>
        </div>
      </Card>
    </div>
  );
}
