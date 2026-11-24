import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EngineError } from "./engineManager";
import type { IpcDeps } from "./ipc";
import type { QueueItem, Settings } from "./types";

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

  it("cancels active item, stops engine before installEnginePack, and kicks queue if configured", async () => {
    const { registerIpcHandlers } = await import("./ipc");
    const { deps, mocks } = makeDeps();
    const callOrder: string[] = [];

    mocks.queueManager.getItems.mockReturnValue([
      makeQueueItem({ id: "active-1", status: "running" }),
    ]);
    mocks.queueManager.cancel.mockImplementation(async () => {
      callOrder.push("cancel");
      return [];
    });
    mocks.engine.stop.mockImplementation(async () => {
      callOrder.push("stop");
    });
    mocks.enginePack.install.mockImplementation(async () => {
      callOrder.push("install");
    });
    mocks.queueManager.kick.mockImplementation(() => {
      callOrder.push("kick");
    });

    registerIpcHandlers(deps);
    await handlers.get("bsm:installEnginePack")!({});

    expect(mocks.queueManager.cancel).toHaveBeenCalledWith("active-1");
    expect(callOrder).toEqual(["cancel", "stop", "install", "kick"]);
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
    unsubProgress();

    const cbLibrary = vi.fn();
    const unsubLibrary = api.onLibraryChanged(cbLibrary) as () => void;
    expect(ipcRendererOn).toHaveBeenCalledWith("bsm:libraryChanged", expect.any(Function));
    unsubLibrary();
  });
});
