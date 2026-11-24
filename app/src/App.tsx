import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Cpu,
  FolderHeart,
  ListOrdered,
  PlusCircle,
  X,
} from "lucide-react";
import appIcon from "./assets/icon.png";
import { EN, extractErrorCode, formatErrorEn } from "./i18n/en";
import { EngineForgeAnimation } from "./components/EngineForgeAnimation";
import { FirstRun } from "./components/FirstRun";
import { LibraryView } from "./components/LibraryView";
import { QueueAddedNotice } from "./components/QueueAddedNotice";
import { QueuePanel } from "./components/QueuePanel";
import { QueueStatusBar } from "./components/QueueStatusBar";
import { SettingsButton } from "./components/SettingsButton";
import { SettingsModal } from "./components/SettingsModal";
import { Badge, Button, SegmentedControl, cn } from "./components/ui/primitives";
import { Wizard } from "./components/wizard/Wizard";
import { useQueue } from "./hooks/useQueue";
import { MapViewer } from "./viewer/components/MapViewer";
import { mergeUpdateStatus } from "./lib/updateStatus";
import type { EngineInfo, LibraryEntry, Settings, StartupState, UpdateStatus } from "./types";

type Tab = "create" | "queue" | "library";

const GITHUB_REPO_URL = "https://github.com/al3ksh/BeatSaberMaker";
const NOTICE_TIMEOUT_MS = 4000;

export function App() {
  const [startup, setStartup] = useState<StartupState | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [needsFirstRun, setNeedsFirstRun] = useState(false);
  const [tab, setTab] = useState<Tab>("create");
  const [regenerateTarget, setRegenerateTarget] = useState<LibraryEntry | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>({ state: "idle" });
  const [libraryEntries, setLibraryEntries] = useState<LibraryEntry[]>([]);
  const [libraryCovers, setLibraryCovers] = useState<Record<string, string>>({});
  const [libraryLoaded, setLibraryLoaded] = useState(false);
  const [previewFolder, setPreviewFolder] = useState<string | null>(null);
  const [libraryNoticeTitle, setLibraryNoticeTitle] = useState<string | null>(null);
  const [libraryNoticeError, setLibraryNoticeError] = useState<string | null>(null);

  const loadedCoverPathsRef = useRef<Map<string, string>>(new Map());
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const queue = useQueue();
  const activeQueueCount = (queue.activeItem ? 1 : 0) + queue.waitingCount;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    };
  }, []);

  useEffect(() => {
    let active = true;
    let receivedLibraryUpdate = false;
    let coverSyncSeq = 0;

    async function syncMissingCovers(list: LibraryEntry[], replaceAll = false) {
      const seq = ++coverSyncSeq;
      const validIds = new Set(list.map((e) => e.id));
      const entryCoverById = new Map(list.map((e) => [e.id, e.coverCachePath]));
      const nextLoadedPaths = new Map<string, string>();
      if (!replaceAll) {
        for (const [id, coverPath] of loadedCoverPathsRef.current.entries()) {
          if (validIds.has(id)) {
            nextLoadedPaths.set(id, coverPath);
          }
        }
      }

      const updates: Record<string, string> = {};
      for (const item of list) {
        if (!active || seq !== coverSyncSeq) return;
        if (!item.coverCachePath) {
          nextLoadedPaths.delete(item.id);
          continue;
        }
        if (!replaceAll && nextLoadedPaths.get(item.id) === item.coverCachePath) {
          continue;
        }
        try {
          const url = await window.bsm?.readCoverDataUrl?.(item.coverCachePath);
          if (!active || seq !== coverSyncSeq) return;
          if (url) {
            nextLoadedPaths.set(item.id, item.coverCachePath);
            updates[item.id] = url;
          }
        } catch {
          // Ignore unreadable cover files.
        }
      }

      if (!active || seq !== coverSyncSeq) return;
      loadedCoverPathsRef.current = nextLoadedPaths;
      setLibraryCovers((prev) => {
        const next: Record<string, string> = {};
        for (const [id, url] of Object.entries(prev)) {
          if (validIds.has(id) && entryCoverById.get(id)) {
            next[id] = url;
          }
        }
        return { ...next, ...updates };
      });
    }

    window.bsm?.getStartupState().then((state) => {
      if (!active) return;
      setStartup(state);
      setSettings(state.settings);
      setInfo(state.info);
      setNeedsFirstRun(state.needsEnginePack || !state.settings.customLevels);
    });

    window.bsm
      ?.listLibrary?.()
      .then(async (list) => {
        if (!active || receivedLibraryUpdate) return;
        setLibraryEntries(list);
        await syncMissingCovers(list, true);
        if (active && !receivedLibraryUpdate) {
          setLibraryLoaded(true);
        }
      })
      .catch(() => {});

    const unsubLibrary = window.bsm?.onLibraryChanged?.((list) => {
      if (!active) return;
      receivedLibraryUpdate = true;
      setLibraryEntries(list);
      setLibraryLoaded(true);
      void syncMissingCovers(list);
    });

    const unsubUpdate = window.bsm?.onUpdateStatus?.((st) => {
      setUpdateStatus((prev) => mergeUpdateStatus(prev, st));
    });

    return () => {
      active = false;
      unsubLibrary?.();
      unsubUpdate?.();
    };
  }, []);

  function dismissLibraryNotice() {
    if (noticeTimerRef.current) {
      clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setLibraryNoticeTitle(null);
    setLibraryNoticeError(null);
  }

  function showLibraryQueuedNotice(title: string) {
    setLibraryNoticeError(null);
    setLibraryNoticeTitle(title);
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = setTimeout(() => {
      if (!mountedRef.current) return;
      setLibraryNoticeTitle(null);
      noticeTimerRef.current = null;
    }, NOTICE_TIMEOUT_MS);
  }

  async function handleImmediateRegenerate(entry: LibraryEntry) {
    setLibraryNoticeError(null);
    try {
      await queue.actions.enqueue({
        sourceType: "cached",
        sourcePathOrUrl: entry.audioCachePath,
        title: entry.title,
        artist: entry.artist || "Unknown",
        coverPath: entry.coverCachePath,
        thumbnailUrl: libraryCovers[entry.id] ?? null,
        config: {
          ...entry.config,
          mode: "custom",
        },
        regenerateId: entry.id,
      });
      if (mountedRef.current) {
        showLibraryQueuedNotice(entry.title);
      }
    } catch (err) {
      if (mountedRef.current) {
        setLibraryNoticeTitle(null);
        setLibraryNoticeError(formatErrorEn(extractErrorCode(err)));
      }
    }
  }

  const viewKey = !startup || !settings ? "splash" : needsFirstRun ? "first-run" : "main";

  const queueTabBadge =
    activeQueueCount > 0 || queue.failedCount > 0 ? (
      <span className="inline-flex items-center gap-1 ml-0.5">
        {activeQueueCount > 0 && (
          <span
            data-testid="queue-tab-count"
            className={cn(
              "rounded-full px-1.5 py-0 text-[10px] font-mono font-semibold tabular-nums leading-4",
              tab === "queue"
                ? "bg-white/20 text-white"
                : "bg-blue-500/20 text-blue-300 border border-blue-500/35",
            )}
          >
            {activeQueueCount}
          </span>
        )}
        {queue.failedCount > 0 && (
          <span
            data-testid="queue-tab-failed-dot"
            aria-hidden="true"
            className="h-2 w-2 rounded-full bg-red-400"
          />
        )}
      </span>
    ) : undefined;

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
                      value: "queue",
                      label: EN.nav.queue,
                      icon: <ListOrdered className="h-3.5 w-3.5" />,
                      badge: queueTabBadge,
                    },
                    {
                      value: "library",
                      label: EN.nav.library,
                      icon: <FolderHeart className="h-3.5 w-3.5" />,
                    },
                  ]}
                />

                <SettingsButton
                  updateState={updateStatus.state}
                  onClick={() => setSettingsOpen(true)}
                />
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
              {tab === "library" && libraryNoticeTitle && (
                <QueueAddedNotice
                  title={libraryNoticeTitle}
                  className="mb-3"
                  onViewQueue={() => {
                    dismissLibraryNotice();
                    setTab("queue");
                  }}
                  onDismiss={dismissLibraryNotice}
                />
              )}

              {tab === "library" && libraryNoticeError && (
                <div
                  role="alert"
                  className="shrink-0 mb-3 flex items-center justify-between gap-3 rounded-xl bg-red-500/10 border border-red-500/30 px-3.5 py-2 text-xs text-red-200"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                    <span className="truncate">{libraryNoticeError}</span>
                  </div>
                  <button
                    type="button"
                    aria-label={EN.common.close}
                    onClick={dismissLibraryNotice}
                    className="rounded p-1 text-red-300/80 hover:text-red-100 hover:bg-red-500/20 transition-colors cursor-pointer"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}

              {tab === "create" && (
                <Wizard
                  regenerateTarget={regenerateTarget}
                  onClearRegenerate={() => setRegenerateTarget(null)}
                  onGoToQueue={() => setTab("queue")}
                />
              )}

              {tab === "queue" && (
                <QueuePanel
                  items={queue.items}
                  onCancel={(id) => {
                    void queue.actions.cancel(id);
                  }}
                  onRetry={(id) => {
                    void queue.actions.retry(id);
                  }}
                  onMove={(id, dir) => {
                    void queue.actions.move(id, dir);
                  }}
                  onClearFinished={() => {
                    void queue.actions.clearFinished();
                  }}
                  onPreview={(folder) => setPreviewFolder(folder)}
                  onGoToCreate={() => setTab("create")}
                />
              )}

              {tab === "library" && (
                <LibraryView
                  initialEntries={libraryEntries}
                  initialCovers={libraryCovers}
                  initialLoaded={libraryLoaded}
                  onEntriesChanged={(next) => {
                    setLibraryEntries(next);
                    setLibraryLoaded(true);
                  }}
                  onCoversChanged={(next) => {
                    setLibraryCovers(next);
                  }}
                  onPreview={(folder) => setPreviewFolder(folder)}
                  onRegenerate={(entry, immediate) => {
                    if (immediate) {
                      void handleImmediateRegenerate(entry);
                    } else {
                      setRegenerateTarget(entry);
                      setTab("create");
                    }
                  }}
                  onGoToCreate={() => setTab("create")}
                />
              )}
            </main>

            {tab !== "queue" && (
              <QueueStatusBar
                items={queue.items}
                onOpenQueue={() => setTab("queue")}
                onCancelActive={(id) => {
                  void queue.actions.cancel(id);
                }}
              />
            )}

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
