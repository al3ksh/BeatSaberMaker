import { useState } from "react";
import { Cpu, FileText, FolderOpen, LocateFixed } from "lucide-react";
import { EN } from "../i18n/en";
import type { DevicePref, EngineInfo, Settings, UpdateStatus } from "../types";
import { UpdateStatusCard } from "./UpdateStatusCard";
import { Button, Modal, SegmentedControl } from "./ui/primitives";

export interface SettingsModalProps {
  open: boolean;
  settings: Settings;
  info: EngineInfo | null;
  appVersion: string;
  updateStatus: UpdateStatus;
  onClose: () => void;
  onSettingsSaved: (next: Settings) => void;
}

export function SettingsModal({
  open,
  settings,
  info,
  appVersion,
  updateStatus,
  onClose,
  onSettingsSaved,
}: SettingsModalProps) {
  const [folderError, setFolderError] = useState<string | null>(null);

  async function handlePickCustomLevels() {
    setFolderError(null);
    const picked = await window.bsm.pickFolder(EN.settings.customLevelsLabel);
    if (!picked) return;
    const res = await window.bsm.validateCustomLevels(picked);
    if (res.valid && res.resolvedPath) {
      const next = await window.bsm.saveSettings({ customLevels: res.resolvedPath });
      onSettingsSaved(next);
    } else {
      setFolderError(EN.settings.customLevelsInvalid);
    }
  }

  async function handleAutoDetect() {
    setFolderError(null);
    const fresh = await window.bsm.getInfo();
    if (fresh.detected_custom_levels) {
      const next = await window.bsm.saveSettings({
        customLevels: fresh.detected_custom_levels,
      });
      onSettingsSaved(next);
    } else {
      setFolderError(EN.settings.customLevelsInvalid);
    }
  }

  async function handlePickExportDir() {
    const picked = await window.bsm.pickFolder(EN.settings.exportDirLabel);
    if (!picked) return;
    const next = await window.bsm.saveSettings({ exportDir: picked });
    onSettingsSaved(next);
  }

  async function handleDeviceChange(device: DevicePref) {
    const next = await window.bsm.saveSettings({ device });
    onSettingsSaved(next);
  }

  const deviceOptions: Array<{ value: DevicePref; label: string }> = [
    { value: "auto", label: EN.settings.devices.auto },
    { value: "cuda", label: EN.settings.devices.cuda },
    { value: "cpu", label: EN.settings.devices.cpu },
  ];

  return (
    <Modal open={open} title={EN.settings.title} onClose={onClose}>
      <div className="space-y-4">
        <div className="rounded-xl bg-zinc-950/60 border border-zinc-800/80 p-3.5 space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-zinc-300">
              {EN.settings.customLevelsLabel}
            </span>
            <div className="flex items-center gap-1.5">
              <Button variant="secondary" size="sm" onClick={handleAutoDetect}>
                <LocateFixed className="h-3.5 w-3.5" />
                {EN.settings.detectAuto}
              </Button>
              <Button variant="secondary" size="sm" onClick={handlePickCustomLevels}>
                <FolderOpen className="h-3.5 w-3.5" />
                {EN.settings.changeFolder}
              </Button>
            </div>
          </div>
          <div
            title={settings.customLevels || "-"}
            className="rounded-lg bg-zinc-950 border border-zinc-800/90 px-3 py-2 text-xs font-mono text-zinc-300 break-all"
          >
            {settings.customLevels || "-"}
          </div>
          {folderError && <p className="text-xs text-red-400">{folderError}</p>}
        </div>

        <div className="rounded-xl bg-zinc-950/60 border border-zinc-800/80 p-3.5 space-y-2.5">
          <label className="block text-xs font-medium text-zinc-300">
            {EN.settings.deviceLabel}
          </label>
          <SegmentedControl
            equalWidth
            size="sm"
            value={settings.device}
            onChange={handleDeviceChange}
            options={deviceOptions}
          />
          {info?.cuda_available ? (
            <div className="flex items-center gap-2 text-xs text-emerald-400 pt-0.5">
              <Cpu className="h-3.5 w-3.5 shrink-0" />
              <span>
                {EN.settings.gpuDetectedPrefix} {info.gpu_name ?? "NVIDIA CUDA"}
              </span>
            </div>
          ) : (
            <p className="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
              {EN.settings.cpuNotice}
            </p>
          )}
        </div>

        <div className="rounded-xl bg-zinc-950/60 border border-zinc-800/80 p-3.5 space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-zinc-300">
              {EN.settings.exportDirLabel}
            </span>
            <Button variant="secondary" size="sm" onClick={handlePickExportDir}>
              <FolderOpen className="h-3.5 w-3.5" />
              {EN.settings.changeFolder}
            </Button>
          </div>
          <div
            title={settings.exportDir}
            className="rounded-lg bg-zinc-950 border border-zinc-800/90 px-3 py-2 text-xs font-mono text-zinc-300 truncate"
          >
            {settings.exportDir}
          </div>
        </div>

        <UpdateStatusCard status={updateStatus} appVersion={appVersion} />

        <div className="border-t border-zinc-800/80 pt-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs font-mono text-zinc-500">
            v{appVersion} · Engine {info?.version ?? "-"}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" onClick={() => window.bsm.openLogs()}>
              <FileText className="h-3.5 w-3.5" />
              {EN.settings.openLogs}
            </Button>
          </div>
        </div>

      </div>
    </Modal>
  );
}
