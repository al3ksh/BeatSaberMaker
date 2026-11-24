import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EngineError } from "./engineManager";
import type { IpcDeps } from "./ipc";
import { LibraryStore } from "./library";
import { QueueManager } from "./queue";
import { SettingsStore } from "./store";
import type { EngineInfo, GenerateOutput, QueueItem, Settings } from "./types";

const handlers = new Map<string, (...args: unknown[]) => unknown>();
const exposeInMainWorld = vi.fn();
const ipcRendererInvoke = vi.fn();
const ipcRendererOn = vi.fn();
const ipcRendererRemoveListener = vi.fn();

vi.mock("electron", () => ({
  BrowserWindow: class {},
  dialog: {
    showOpenDialog: vi.fn(),
    showSaveDialog: vi.fn(),
  },
  shell: {
    openPath: vi.fn(),
    openExternal: vi.fn(),
    trashItem: vi.fn(),
  },
  ipcMain: {
    handle: vi.fn((channel: string, fn: (...args: unknown[]) => unknown) => {
      handlers.set(channel, fn);
    }),
  },
  contextBridge: {
    exposeInMainWorld,
  },
  ipcRenderer: {
    invoke: ipcRendererInvoke,
    on: ipcRendererOn,
    removeListener: ipcRendererRemoveListener,
  },
}));

function makeQueueItem(overrides: Partial<QueueItem> = {}): QueueItem {
  return {
    id: "q-1",
    status: "waiting",
    request: {
      sourceType: "file",
      sourcePathOrUrl: "C:\\songs\\test.mp3",
      title: "Test Song",
      artist: "Test Artist",
      coverPath: null,
      config: {
        mode: "pack",
        levels: ["Hard", "Expert"],
        nps: null,
        styles: [],
        ornaments: null,
        lights: 2,
        environment: "DefaultEnvironment",
        seed: 42,
        lightsRules: false,
      },
    },
    thumbnailUrl: null,
    enqueuedAt: "2026-11-24T12:00:00.000Z",
    startedAt: null,
    finishedAt: null,
    progress: null,
    errorCode: null,
    result: null,
    entry: null,
    ...overrides,
  };
}

function makeDeps(overrides: Partial<IpcDeps> = {}): {
  deps: IpcDeps;
  mocks: {
    queueManager: {
      getItems: ReturnType<typeof vi.fn>;
      enqueue: ReturnType<typeof vi.fn>;
      cancel: ReturnType<typeof vi.fn>;
      retry: ReturnType<typeof vi.fn>;
      move: ReturnType<typeof vi.fn>;
      clearFinished: ReturnType<typeof vi.fn>;
      kick: ReturnType<typeof vi.fn>;
      pause: ReturnType<typeof vi.fn>;
      resume: ReturnType<typeof vi.fn>;
    };
    engine: {
      stop: ReturnType<typeof vi.fn>;
      request: ReturnType<typeof vi.fn>;
      cancelActive: ReturnType<typeof vi.fn>;
    };
    enginePack: {
      needsInstall: ReturnType<typeof vi.fn>;
      install: ReturnType<typeof vi.fn>;
    };
    settingsStore: {
      load: ReturnType<typeof vi.fn>;
      save: ReturnType<typeof vi.fn>;
    };
  };
} {
  const settings: Settings = {
    customLevels: "C:\\Beat Saber\\CustomLevels",
    device: "auto",
    exportDir: "C:\\Downloads",
    engineVersionInstalled: "4",
  };

  const queueManager = {
    getItems: vi.fn(() => [makeQueueItem()]),
    enqueue: vi.fn(async () => makeQueueItem()),
    cancel: vi.fn(async () => [makeQueueItem({ status: "cancelled" })]),
    retry: vi.fn(async () => [makeQueueItem({ status: "waiting" })]),
    move: vi.fn(async () => [makeQueueItem()]),
    clearFinished: vi.fn(async () => []),
    kick: vi.fn(),
    pause: vi.fn(async () => {}),
    resume: vi.fn(),
  };

  const engine = {
    stop: vi.fn(async () => {}),
    request: vi.fn(async () => ({})),
    cancelActive: vi.fn(async () => {}),
  };

  const enginePack = {
    needsInstall: vi.fn(async () => false),
    install: vi.fn(async () => {}),
  };

  const settingsStore = {
    load: vi.fn(async () => ({ ...settings })),
    save: vi.fn(async (partial: Partial<Settings>) => ({ ...settings, ...partial })),
  };

  const deps: IpcDeps = {
    appVersion: "0.1.4",
    appPath: "C:\\app",
    userDataDir: "C:\\userData",
    mediaDir: path.join("C:\\userData", "media"),
    engine: engine as unknown as IpcDeps["engine"],
    enginePack: enginePack as unknown as IpcDeps["enginePack"],
    expectedEngineVersion: "4",
    settingsStore: settingsStore as unknown as IpcDeps["settingsStore"],
    libraryStore: {
      list: vi.fn(async () => []),
      deleteEntry: vi.fn(async () => []),
    } as unknown as IpcDeps["libraryStore"],
    queueManager: queueManager as unknown as IpcDeps["queueManager"],
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      engineStderr: vi.fn(),
      getLogsDir: vi.fn(() => "C:\\userData\\logs"),
    } as unknown as IpcDeps["logger"],
    getWindow: () => null,
    ...overrides,
  };

  return { deps, mocks: { queueManager, engine, enginePack, settingsStore } };
}

describe("IPC queue handlers", () => {
  beforeEach(() => {
    handlers.clear();
    exposeInMainWorld.mockClear();
    ipcRendererInvoke.mockClear();
    ipcRendererOn.mockClear();
    ipcRendererRemoveListener.mockClear();
  });

  it("registers queue channels and omits removed legacy generation channels", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps } = makeDeps();
    registerIpcHandlers(deps);

    expect(handlers.has("bsm:getQueue")).toBe(true);
    expect(handlers.has("bsm:enqueueGeneration")).toBe(true);
    expect(handlers.has("bsm:cancelQueueItem")).toBe(true);
    expect(handlers.has("bsm:retryQueueItem")).toBe(true);
    expect(handlers.has("bsm:moveQueueItem")).toBe(true);
    expect(handlers.has("bsm:clearFinishedQueue")).toBe(true);

    expect(handlers.has("bsm:startGeneration")).toBe(false);
    expect(handlers.has("bsm:cancelActive")).toBe(false);
  });

  it("delegates queue operations and wraps EngineError codes on enqueue", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps, mocks } = makeDeps();
    registerIpcHandlers(deps);

    const list = await handlers.get("bsm:getQueue")!({});
    expect(mocks.queueManager.getItems).toHaveBeenCalledTimes(1);
    expect(list).toHaveLength(1);

    const req = makeQueueItem().request;
    await handlers.get("bsm:enqueueGeneration")!({}, req);
    expect(mocks.queueManager.enqueue).toHaveBeenCalledWith(req);

    mocks.queueManager.enqueue.mockRejectedValueOnce(new EngineError("not_found", "no folder"));
    await expect(handlers.get("bsm:enqueueGeneration")!({}, req)).rejects.toThrow("BSM_CODE:not_found");

    mocks.queueManager.enqueue.mockRejectedValueOnce(new Error("boom"));
    await expect(handlers.get("bsm:enqueueGeneration")!({}, req)).rejects.toThrow("BSM_CODE:internal");

    await handlers.get("bsm:cancelQueueItem")!({}, "q-1");
    expect(mocks.queueManager.cancel).toHaveBeenCalledWith("q-1");

    await handlers.get("bsm:retryQueueItem")!({}, "q-1");
    expect(mocks.queueManager.retry).toHaveBeenCalledWith("q-1");

    await handlers.get("bsm:moveQueueItem")!({}, "q-1", "up");
    expect(mocks.queueManager.move).toHaveBeenCalledWith("q-1", "up");

    await handlers.get("bsm:clearFinishedQueue")!({});
    expect(mocks.queueManager.clearFinished).toHaveBeenCalledTimes(1);
  });

  it("kicks the queue on saveSettings only when customLevels is set and engine pack is installed", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps, mocks } = makeDeps();
    registerIpcHandlers(deps);

    mocks.settingsStore.load.mockResolvedValue({
      customLevels: "",
      device: "auto",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    mocks.settingsStore.save.mockResolvedValue({
      customLevels: "C:\\Beat Saber\\CustomLevels",
      device: "auto",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    mocks.enginePack.needsInstall.mockResolvedValue(false);

    await handlers.get("bsm:saveSettings")!({}, { customLevels: "C:\\Beat Saber\\CustomLevels" });
    expect(mocks.queueManager.kick).toHaveBeenCalledTimes(1);

    mocks.queueManager.kick.mockClear();
    mocks.enginePack.needsInstall.mockResolvedValue(true);
    await handlers.get("bsm:saveSettings")!({}, { customLevels: "C:\\Beat Saber\\CustomLevels" });
    expect(mocks.queueManager.kick).not.toHaveBeenCalled();

    mocks.enginePack.needsInstall.mockResolvedValue(false);
    mocks.settingsStore.save.mockResolvedValue({
      customLevels: "",
      device: "auto",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    await handlers.get("bsm:saveSettings")!({}, { device: "cpu" });
    expect(mocks.queueManager.kick).not.toHaveBeenCalled();
  });

  it("pauses queue, stops engine before installEnginePack, caches info before kick, and resumes on failure", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps, mocks } = makeDeps();
    const callOrder: string[] = [];

    mocks.queueManager.pause.mockImplementation(async () => {
      callOrder.push("pause");
    });
    mocks.engine.stop.mockImplementation(async () => {
      callOrder.push("stop");
    });
    mocks.enginePack.install.mockImplementation(async () => {
      callOrder.push("install");
    });
    mocks.engine.request.mockImplementation(async (cmd: string) => {
      callOrder.push(`engine:${cmd}`);
      return {
        version: "4",
        cuda_available: true,
        gpu_name: "RTX 4080",
        active_device: "cuda",
        detected_custom_levels: "C:\\Beat Saber\\CustomLevels",
        has_layer_models: true,
        missing_layer_models: [],
        environments: [],
        styles: [],
      } satisfies EngineInfo;
    });
    mocks.queueManager.resume.mockImplementation(() => {
      callOrder.push("resume");
    });
    mocks.queueManager.kick.mockImplementation(() => {
      callOrder.push("kick");
    });

    registerIpcHandlers(deps);
    await handlers.get("bsm:installEnginePack")!({});

    expect(callOrder).toEqual(["pause", "stop", "install", "engine:info", "resume", "kick"]);

    callOrder.length = 0;
    mocks.enginePack.install.mockImplementationOnce(async () => {
      callOrder.push("install");
      throw new Error("network error");
    });
    await expect(handlers.get("bsm:installEnginePack")!({})).rejects.toThrow("network error");
    expect(callOrder).toEqual(["pause", "stop", "install", "resume"]);
  });

  it("kicks queue after info on getStartupState (including auto-detected customLevels) and avoids blocking on active item reloads", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps, mocks } = makeDeps();
    const callOrder: string[] = [];
    const sampleInfo: EngineInfo = {
      version: "4",
      cuda_available: true,
      gpu_name: "RTX 4080",
      active_device: "cuda",
      detected_custom_levels: "C:\\Detected\\CustomLevels",
      has_layer_models: true,
      missing_layer_models: [],
      environments: ["DefaultEnvironment"],
      styles: [],
    };

    mocks.settingsStore.load.mockResolvedValue({
      customLevels: "",
      device: "auto",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    mocks.queueManager.getItems.mockReturnValue([makeQueueItem({ status: "waiting" })]);
    mocks.engine.request.mockImplementation(async (cmd: string) => {
      callOrder.push(`engine:${cmd}`);
      return sampleInfo;
    });
    mocks.queueManager.kick.mockImplementation(() => {
      callOrder.push("kick");
    });

    registerIpcHandlers(deps);
    const startup1 = await handlers.get("bsm:getStartupState")!({});
    expect(mocks.settingsStore.save).toHaveBeenCalledWith({
      customLevels: "C:\\Detected\\CustomLevels",
    });
    expect(callOrder).toEqual(["engine:info", "kick"]);
    expect((startup1 as { info: EngineInfo | null }).info).toEqual(sampleInfo);

    // Reload while an active item is running: return cached info immediately without calling engine.request("info").
    mocks.engine.request.mockClear();
    mocks.queueManager.kick.mockClear();
    mocks.settingsStore.load.mockResolvedValue({
      customLevels: "C:\\Detected\\CustomLevels",
      device: "auto",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    mocks.queueManager.getItems.mockReturnValue([makeQueueItem({ status: "running" })]);

    const startup2 = await handlers.get("bsm:getStartupState")!({});
    expect(mocks.engine.request).not.toHaveBeenCalled();
    expect((startup2 as { info: EngineInfo | null }).info).toEqual(sampleInfo);

    // Changing device invalidates cached info; reload while active still does not block on engine.request("info").
    mocks.settingsStore.save.mockResolvedValueOnce({
      customLevels: "C:\\Detected\\CustomLevels",
      device: "cpu",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    await handlers.get("bsm:saveSettings")!({}, { device: "cpu" });
    const startup3 = await handlers.get("bsm:getStartupState")!({});
    expect(mocks.engine.request).not.toHaveBeenCalled();
    expect((startup3 as { info: EngineInfo | null }).info).toBeNull();

    // Once queue has no active item, getStartupState fetches fresh info for the new device.
    mocks.queueManager.getItems.mockReturnValue([]);
    mocks.settingsStore.load.mockResolvedValue({
      customLevels: "C:\\Detected\\CustomLevels",
      device: "cpu",
      exportDir: "C:\\Downloads",
      engineVersionInstalled: "4",
    });
    await handlers.get("bsm:getStartupState")!({});
    expect(mocks.engine.request).toHaveBeenCalledWith("info", { device: "cpu" });
  });

  it("installEnginePack pauses real QueueManager so waiting item does not start until install completes and FirstRun getInfo does not block", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-ipc-install-"));
    const customLevels = path.join(tmp, "CustomLevels");
    await fs.mkdir(customLevels, { recursive: true });
    const audioFile = path.join(tmp, "track.wav");
    await fs.writeFile(audioFile, "fake-wav");

    const settingsStore = new SettingsStore(path.join(tmp, "settings.json"), path.join(tmp, "export"));
    await settingsStore.save({ customLevels, engineVersionInstalled: "3" });
    const libraryStore = new LibraryStore(path.join(tmp, "library.json"), path.join(tmp, "media"));
    const logger = {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      engineStderr: vi.fn(),
      getLogsDir: () => path.join(tmp, "logs"),
    };

    let item1Started!: () => void;
    const item1StartedP = new Promise<void>((r) => {
      item1Started = r;
    });
    let rejectItem1!: (err: unknown) => void;
    let item2Started!: () => void;
    const item2StartedP = new Promise<void>((r) => {
      item2Started = r;
    });
    let resolveItem2!: (out: GenerateOutput) => void;
    const events: string[] = [];

    const engineInfo: EngineInfo = {
      version: "4",
      cuda_available: true,
      gpu_name: "RTX 4080",
      active_device: "cuda",
      detected_custom_levels: customLevels,
      has_layer_models: true,
      missing_layer_models: [],
      environments: ["DefaultEnvironment"],
      styles: [],
    };

    const engine = {
      request: vi.fn(async (cmd: string, params?: Record<string, unknown>) => {
        if (cmd === "info") {
          events.push("engine:info");
          return engineInfo;
        }
        if (cmd === "generate") {
          const title = String(params?.title);
          events.push(`engine:generate:${title}`);
          if (title === "ActiveItem") {
            item1Started();
            return await new Promise<GenerateOutput>((_resolve, reject) => {
              rejectItem1 = reject;
            });
          }
          if (title === "WaitingItem") {
            item2Started();
            return await new Promise<GenerateOutput>((resolve) => {
              resolveItem2 = resolve;
            });
          }
        }
        return {};
      }),
      cancelActive: vi.fn(async () => {
        events.push("engine:cancelActive");
        rejectItem1(new EngineError("cancelled", "cancelled"));
      }),
      stop: vi.fn(async () => {
        events.push("engine:stop");
      }),
    };

    const qm = new QueueManager({
      filePath: path.join(tmp, "queue.json"),
      tmpDownloadsRoot: path.join(tmp, "tmp-downloads"),
      engine: engine as unknown as ConstructorParameters<typeof QueueManager>[0]["engine"],
      settingsStore,
      libraryStore,
      trashItem: async () => {},
      logger,
    });
    await qm.init({ canAutoStart: async () => false });

    const reqBase = makeQueueItem().request;
    const active = await qm.enqueue({ ...reqBase, sourcePathOrUrl: audioFile, title: "ActiveItem" });
    await item1StartedP;
    const waiting = await qm.enqueue({ ...reqBase, sourcePathOrUrl: audioFile, title: "WaitingItem" });

    let installObservedStatuses: Record<string, string> = {};
    const enginePack = {
      needsInstall: vi.fn(async () => false),
      install: vi.fn(async () => {
        events.push("enginePack:install:start");
        const items = qm.getItems();
        installObservedStatuses = Object.fromEntries(items.map((it) => [it.id, it.status]));
        await new Promise((r) => setTimeout(r, 30));
        events.push("enginePack:install:end");
      }),
    };

    registerIpcHandlers({
      appVersion: "0.1.4",
      appPath: tmp,
      userDataDir: tmp,
      mediaDir: path.join(tmp, "media"),
      engine: engine as unknown as IpcDeps["engine"],
      enginePack: enginePack as unknown as IpcDeps["enginePack"],
      expectedEngineVersion: "4",
      settingsStore,
      libraryStore,
      queueManager: qm,
      logger: logger as unknown as IpcDeps["logger"],
      getWindow: () => null,
    });

    await handlers.get("bsm:installEnginePack")!({});

    // During install, item 1 must already be cancelled and item 2 must still be waiting.
    expect(installObservedStatuses[active.id]).toBe("cancelled");
    expect(installObservedStatuses[waiting.id]).toBe("waiting");

    // Item 2 starts only after install and post-install info complete.
    await item2StartedP;
    expect(events.indexOf("enginePack:install:end")).toBeLessThan(events.indexOf("engine:info"));
    expect(events.indexOf("engine:info")).toBeLessThan(events.indexOf("engine:generate:WaitingItem"));

    // FirstRun's immediate getInfo() returns cached info while item 2 is running without calling engine.request("info") again.
    const infoCallCountBefore = engine.request.mock.calls.filter(([c]) => c === "info").length;
    const infoFromFirstRun = await handlers.get("bsm:getInfo")!({});
    expect(infoFromFirstRun).toEqual(engineInfo);
    const infoCallCountAfter = engine.request.mock.calls.filter(([c]) => c === "info").length;
    expect(infoCallCountAfter).toBe(infoCallCountBefore);

    const out2Folder = path.join(customLevels, "BSM - WaitingItem");
    await fs.mkdir(out2Folder, { recursive: true });
    resolveItem2({
      folder: out2Folder,
      title: "WaitingItem",
      artist: "Test Artist",
      bpm: 128,
      grid_steady: true,
      n_notes: 100,
      nps: 3,
      rank: "Hard",
      levels: [{ rank: "Hard", target_nps: 3, n_notes: 100, nps: 3, n_arcs: 0, n_chains: 0 }],
      n_walls: 0,
      n_bombs: 0,
      n_lights: 0,
      environment: "DefaultEnvironment",
      warnings: [],
      used_rule_lights: false,
      device_used: "cuda",
    });

    const deadline = Date.now() + 3000;
    while (qm.getItems().find((i) => i.id === waiting.id)?.status !== "done") {
      if (Date.now() > deadline) throw new Error("Timed out waiting for WaitingItem");
      await new Promise((r) => setTimeout(r, 15));
    }
    expect(qm.getItems().find((i) => i.id === waiting.id)?.status).toBe("done");
  });

  it("exposes queue API and omits removed legacy methods in preload", async () => {
    vi.resetModules();
    await import("./preload");

    expect(exposeInMainWorld).toHaveBeenCalledTimes(1);
    const [apiKey, api] = exposeInMainWorld.mock.calls[0] as [string, Record<string, (...args: unknown[]) => unknown>];
    expect(apiKey).toBe("bsm");

    expect(api.startGeneration).toBeUndefined();
    expect(api.cancelActive).toBeUndefined();
    expect(api.onProgress).toBeUndefined();

    const req = makeQueueItem().request;
    api.getQueue();
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:getQueue");

    api.enqueueGeneration(req);
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:enqueueGeneration", req);

    api.cancelQueueItem("q-1");
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:cancelQueueItem", "q-1");

    api.retryQueueItem("q-1");
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:retryQueueItem", "q-1");

    api.moveQueueItem("q-1", "down");
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:moveQueueItem", "q-1", "down");

    api.clearFinishedQueue();
    expect(ipcRendererInvoke).toHaveBeenCalledWith("bsm:clearFinishedQueue");

    const cbChanged = vi.fn();
    const unsubChanged = api.onQueueChanged(cbChanged) as () => void;
    expect(ipcRendererOn).toHaveBeenCalledWith("bsm:queueChanged", expect.any(Function));
    const listenerChanged = ipcRendererOn.mock.calls[0][1];
    listenerChanged({}, [makeQueueItem()]);
    expect(cbChanged).toHaveBeenCalledWith([makeQueueItem()]);
    unsubChanged();
    expect(ipcRendererRemoveListener).toHaveBeenCalledWith("bsm:queueChanged", listenerChanged);

    const cbProgress = vi.fn();
    const unsubProgress = api.onQueueProgress(cbProgress) as () => void;
    expect(ipcRendererOn).toHaveBeenCalledWith("bsm:queueProgress", expect.any(Function));
    const listenerProgress = ipcRendererOn.mock.calls[1][1];
    listenerProgress({}, { id: "q-1", stage: "notes", pct: 42 });
    expect(cbProgress).toHaveBeenCalledWith({ id: "q-1", stage: "notes", pct: 42 });
    unsubProgress();
    expect(ipcRendererRemoveListener).toHaveBeenCalledWith("bsm:queueProgress", listenerProgress);

    const cbLibrary = vi.fn();
    const unsubLibrary = api.onLibraryChanged(cbLibrary) as () => void;
    expect(ipcRendererOn).toHaveBeenCalledWith("bsm:libraryChanged", expect.any(Function));
    const listenerLibrary = ipcRendererOn.mock.calls[2][1];
    listenerLibrary({}, [{ id: "lib-1" }]);
    expect(cbLibrary).toHaveBeenCalledWith([{ id: "lib-1" }]);
    unsubLibrary();
    expect(ipcRendererRemoveListener).toHaveBeenCalledWith("bsm:libraryChanged", listenerLibrary);
  });
});
