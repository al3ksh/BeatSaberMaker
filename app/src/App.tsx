import { useEffect, useState } from "react";
import { Cpu, FolderHeart, PlusCircle, Settings as SettingsIcon } from "lucide-react";
import { PL } from "./i18n/pl";
import { FirstRun } from "./components/FirstRun";
import { LibraryView } from "./components/LibraryView";
import { SettingsModal } from "./components/SettingsModal";
import { Badge, Button } from "./components/ui/primitives";
import { Wizard } from "./components/wizard/Wizard";
import type { EngineInfo, LibraryEntry, Settings, StartupState, UpdateStatus } from "./types";

type Tab = "create" | "library";

export function App() {
  const [startup, setStartup] = useState<StartupState | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [needsFirstRun, setNeedsFirstRun] = useState(false);
  const [tab, setTab] = useState<Tab>("create");
  const [regenerateTarget, setRegenerateTarget] = useState<LibraryEntry | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>({ state: "idle" });

  useEffect(() => {
    let active = true;
    window.bsm?.getStartupState().then((state) => {
      if (!active) return;
      setStartup(state);
      setSettings(state.settings);
      setInfo(state.info);
      setNeedsFirstRun(state.needsEnginePack || !state.settings.customLevels);
    });
    const unsubUpdate = window.bsm?.onUpdateStatus?.((st) => {
      setUpdateStatus(st);
    });
    return () => {
      active = false;
      unsubUpdate?.();
    };
  }, []);

  if (!startup || !settings) {
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-400 flex items-center justify-center text-sm">
        {PL.common.loading}
      </div>
    );
  }

  if (needsFirstRun) {
    return (
      <FirstRun
        startup={startup}
        onComplete={(nextSettings, nextInfo) => {
          setSettings(nextSettings);
          if (nextInfo) setInfo(nextInfo);
          setNeedsFirstRun(false);
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <header className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur px-6 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-red-500 to-blue-500 flex items-center justify-center text-xs font-bold text-white shadow">
            BS
          </div>
          <span className="font-semibold tracking-tight">{PL.appName}</span>
          {info && !info.cuda_available && (
            <Badge variant="amber">
              <Cpu className="h-3 w-3" />
              {PL.settings.devices.cpu}
            </Badge>
          )}
        </div>

        <nav className="flex items-center gap-1.5">
          <Button
            variant={tab === "create" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setTab("create")}
          >
            <PlusCircle className="h-4 w-4" />
            {PL.nav.create}
          </Button>
          <Button
            variant={tab === "library" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setTab("library")}
          >
            <FolderHeart className="h-4 w-4" />
            {PL.nav.library}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSettingsOpen(true)}
            aria-label={PL.nav.settings}
          >
            <SettingsIcon className="h-4 w-4" />
            {PL.nav.settings}
          </Button>
        </nav>
      </header>

      {updateStatus.state === "downloaded" && (
        <div className="bg-blue-600/20 border-b border-blue-500/30 px-6 py-2 flex items-center justify-between text-xs text-blue-200">
          <span>{PL.updateBanner.ready}</span>
          <Button size="sm" onClick={() => window.bsm.installUpdate()}>
            {PL.updateBanner.action}
          </Button>
        </div>
      )}

      <main className="flex-1 max-w-4xl w-full mx-auto p-6">
        {tab === "create" ? (
          <Wizard
            regenerateTarget={regenerateTarget}
            onClearRegenerate={() => setRegenerateTarget(null)}
            onGoToLibrary={() => setTab("library")}
          />
        ) : (
          <LibraryView
            onRegenerate={(entry) => {
              setRegenerateTarget(entry);
              setTab("create");
            }}
          />
        )}
      </main>

      <SettingsModal
        open={settingsOpen}
        settings={settings}
        info={info}
        appVersion={startup.appVersion}
        updateStatus={updateStatus}
        onClose={() => setSettingsOpen(false)}
        onSettingsSaved={(next) => setSettings(next)}
      />
    </div>
  );
}

export default App;
