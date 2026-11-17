import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Cpu, FolderHeart, PlusCircle, Settings as SettingsIcon } from "lucide-react";
import appIcon from "./assets/icon.png";
import { EN } from "./i18n/en";
import { EngineForgeAnimation } from "./components/EngineForgeAnimation";
import { FirstRun } from "./components/FirstRun";
import { LibraryView } from "./components/LibraryView";
import { SettingsModal } from "./components/SettingsModal";
import { Badge, Button, SegmentedControl } from "./components/ui/primitives";
import { Wizard } from "./components/wizard/Wizard";
import { MapViewer } from "./viewer/components/MapViewer";
import type { EngineInfo, LibraryEntry, Settings, StartupState, UpdateStatus } from "./types";

type Tab = "create" | "library";

const GITHUB_REPO_URL = "https://github.com/al3ksh/BeatSaberMaker";

export function App() {
  const [startup, setStartup] = useState<StartupState | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [needsFirstRun, setNeedsFirstRun] = useState(false);
  const [tab, setTab] = useState<Tab>("create");
  const [regenerateTarget, setRegenerateTarget] = useState<LibraryEntry | null>(null);
  const [regenerateImmediate, setRegenerateImmediate] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>({ state: "idle" });
  const [libraryEntries, setLibraryEntries] = useState<LibraryEntry[]>([]);
  const [libraryCovers, setLibraryCovers] = useState<Record<string, string>>({});
  const [libraryLoaded, setLibraryLoaded] = useState(false);
  const [previewFolder, setPreviewFolder] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    window.bsm?.getStartupState().then((state) => {
      if (!active) return;
      setStartup(state);
      setSettings(state.settings);
      setInfo(state.info);
      setNeedsFirstRun(state.needsEnginePack || !state.settings.customLevels);
    });
    window.bsm?.listLibrary?.().then(async (list) => {
      if (!active) return;
      setLibraryEntries(list);
      const nextCovers: Record<string, string> = {};
      for (const item of list) {
        if (item.coverCachePath) {
          const url = await window.bsm.readCoverDataUrl?.(item.coverCachePath);
          if (url) nextCovers[item.id] = url;
        }
      }
      if (active) {
        setLibraryCovers(nextCovers);
        setLibraryLoaded(true);
      }
    }).catch(() => {});
    const unsubUpdate = window.bsm?.onUpdateStatus?.((st) => {
      setUpdateStatus(st);
    });
    return () => {
      active = false;
      unsubUpdate?.();
    };
  }, []);

  const viewKey = !startup || !settings ? "splash" : needsFirstRun ? "first-run" : "main";

  return (
    <div className="h-screen overflow-hidden bg-zinc-950 bg-[radial-gradient(ellipse_80%_45%_at_50%_0%,rgba(39,39,42,0.28),transparent)] text-zinc-100 flex flex-col">
      <AnimatePresence mode="wait">
        {viewKey === "splash" && (
          <motion.div
            key="splash"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.985 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="h-full w-full flex flex-col items-center justify-center gap-7 px-6"
          >
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              className="flex items-center gap-4"
            >
              <img
                src={appIcon}
                alt={EN.appName}
                className="h-16 w-16 rounded-2xl shadow-xl shadow-black/60"
              />
              <div>
                <div className="text-2xl font-semibold tracking-tight text-zinc-100">
                  {EN.appName}
                </div>
                <div className="text-sm text-zinc-400">{EN.common.startingEngine}</div>
              </div>
            </motion.div>
            <EngineForgeAnimation />
          </motion.div>
        )}

        {viewKey === "first-run" && startup && (
          <motion.div
            key="first-run"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="h-full w-full"
          >
            <FirstRun
              startup={startup}
              onComplete={(nextSettings, nextInfo) => {
                setSettings(nextSettings);
                if (nextInfo) setInfo(nextInfo);
                setNeedsFirstRun(false);
              }}
            />
          </motion.div>
        )}

        {viewKey === "main" && startup && settings && (
          <motion.div
            key="main"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="h-full w-full flex flex-col"
          >
            <header className="shrink-0 border-b border-zinc-800/80 bg-zinc-950/90 px-6 py-2.5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-2.5">
                <button
                  type="button"
                  onClick={() => window.bsm?.openExternalUrl?.(GITHUB_REPO_URL)}
                  title={GITHUB_REPO_URL}
                  className="flex items-center gap-2.5 rounded-xl py-0.5 pr-1.5 hover:opacity-85 transition-opacity cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                >
                  <img
                    src={appIcon}
                    alt={EN.appName}
                    className="h-8 w-8 rounded-lg shadow-sm"
                  />
                  <div className="flex items-baseline gap-2">
                    <span className="font-semibold tracking-tight text-zinc-100">
                      {EN.appName}
                    </span>
                    <span className="text-[11px] font-mono text-zinc-500 hidden sm:inline">
                      v{startup.appVersion}
                    </span>
                  </div>
                </button>
                {info && !info.cuda_available && (
                  <Badge variant="amber">
                    <Cpu className="h-3 w-3" />
                    {EN.settings.devices.cpu}
                  </Badge>
                )}
              </div>

              <nav className="flex items-center gap-2">
                <SegmentedControl
                  size="sm"
                  value={tab}
                  onChange={setTab}
                  options={[
                    {
                      value: "create",
                      label: EN.nav.create,
                      icon: <PlusCircle className="h-3.5 w-3.5" />,
                    },
                    {
                      value: "library",
                      label: EN.nav.library,
                      icon: <FolderHeart className="h-3.5 w-3.5" />,
                    },
                  ]}
                />

                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setSettingsOpen(true)}
                  aria-label={EN.nav.settings}
                  title={EN.nav.settings}
                  className="rounded-xl px-2.5 h-8 w-8 p-0"
                >
                  <SettingsIcon className="h-4 w-4" />
                </Button>
              </nav>
            </header>

            {updateStatus.state === "downloaded" && (
              <div className="shrink-0 bg-blue-600/20 border-b border-blue-500/30 px-6 py-2 flex items-center justify-between text-xs text-blue-200">
                <span>{EN.updateBanner.ready}</span>
                <Button size="sm" onClick={() => window.bsm.installUpdate()}>
                  {EN.updateBanner.action}
                </Button>
              </div>
            )}

            <main className="flex-1 min-h-0 flex flex-col max-w-4xl w-full mx-auto px-6 py-5 overflow-hidden">
              {tab === "create" ? (
                <Wizard
                  regenerateTarget={regenerateTarget}
                  regenerateImmediate={regenerateImmediate}
                  onClearRegenerate={() => {
                    setRegenerateTarget(null);
                    setRegenerateImmediate(false);
                  }}
                  onGoToLibrary={() => setTab("library")}
                  onPreview={(folder) => setPreviewFolder(folder)}
                />
              ) : (
                <LibraryView
                  initialEntries={libraryEntries}
                  initialCovers={libraryCovers}
                  initialLoaded={libraryLoaded}
                  onEntriesChanged={(next) => {
                    setLibraryEntries(next);
                    setLibraryLoaded(true);
                  }}
                  onCoversChanged={setLibraryCovers}
                  onPreview={(folder) => setPreviewFolder(folder)}
                  onRegenerate={(entry, immediate) => {
                    setRegenerateTarget(entry);
                    setRegenerateImmediate(Boolean(immediate));
                    setTab("create");
                  }}
                  onGoToCreate={() => setTab("create")}
                />
              )}
            </main>

            {previewFolder && (
              <MapViewer
                folder={previewFolder}
                onClose={() => setPreviewFolder(null)}
                onSwitchFolder={(nextFolder) => setPreviewFolder(nextFolder)}
              />
            )}

            <SettingsModal
              open={settingsOpen}
              settings={settings}
              info={info}
              appVersion={startup.appVersion}
              updateStatus={updateStatus}
              onClose={() => setSettingsOpen(false)}
              onSettingsSaved={(next) => setSettings(next)}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
