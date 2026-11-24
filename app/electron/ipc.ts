import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { BrowserWindow, dialog, ipcMain, shell } from "electron";
import { EngineError, EngineManager } from "./engineManager";
import type { EnginePackInstaller } from "./enginePack";
import type { LibraryStore } from "./library";
import type { AppLogger } from "./logger";
import { readMapFolderForViewer } from "./mapReader";
import type { QueueManager } from "./queue";
import type { SettingsStore } from "./store";
import type {
  EngineInfo,
  LibraryEntry,
  QueueEnqueueRequest,
  QueueItem,
  StartupState,
  YouTubeItem,
} from "./types";

export async function validateCustomLevelsFolder(
  pickedPath: string,
): Promise<{ valid: boolean; resolvedPath: string | null }> {
  if (!pickedPath || !pickedPath.trim()) {
    return { valid: false, resolvedPath: null };
  }
  const resolved = path.resolve(pickedPath.trim());
  try {
    const stat = await fsp.stat(resolved);
    if (!stat.isDirectory()) {
      return { valid: false, resolvedPath: null };
    }
  } catch {
    return { valid: false, resolvedPath: null };
  }

  const base = path.basename(resolved).toLowerCase();
  if (base === "customlevels") {
    return { valid: true, resolvedPath: resolved };
  }

  if (base === "beat saber_data") {
    const cl = path.join(resolved, "CustomLevels");
    await fsp.mkdir(cl, { recursive: true });
    return { valid: true, resolvedPath: cl };
  }

  const bsData = path.join(resolved, "Beat Saber_Data");
  try {
    const dataStat = await fsp.stat(bsData);
    if (dataStat.isDirectory()) {
      const cl = path.join(bsData, "CustomLevels");
      await fsp.mkdir(cl, { recursive: true });
      return { valid: true, resolvedPath: cl };
    }
  } catch {
    // Not a Beat Saber root directory.
  }

  return { valid: false, resolvedPath: null };
}

function wrapIpcError(err: unknown, logger: AppLogger): never {
  if (err instanceof EngineError) {
    throw new Error(`BSM_CODE:${err.code}`);
  }
  logger.error(err instanceof Error ? err.stack || err.message : String(err));
  throw new Error("BSM_CODE:internal");
}

function mimeForImageExt(ext: string): string | null {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".webp":
      return "image/webp";
    default:
      return null;
  }
}

export interface IpcDeps {
  appVersion: string;
  appPath: string;
  userDataDir: string;
  mediaDir: string;
  engine: EngineManager;
  enginePack: EnginePackInstaller;
  expectedEngineVersion: string;
  settingsStore: SettingsStore;
  libraryStore: LibraryStore;
  queueManager: QueueManager;
  logger: AppLogger;
  getWindow: () => BrowserWindow | null;
  checkForUpdates?: () => Promise<void>;
  installUpdate?: () => void;
}

export function registerIpcHandlers(deps: IpcDeps): void {
  const {
    appVersion,
    mediaDir,
    engine,
    enginePack,
    expectedEngineVersion,
    settingsStore,
    libraryStore,
    queueManager,
    logger,
    getWindow,
  } = deps;

  ipcMain.handle("bsm:getStartupState", async (): Promise<StartupState> => {
    const settings = await settingsStore.load();
    const needsEnginePack = await enginePack.needsInstall();
    let info: EngineInfo | null = null;
    if (!needsEnginePack) {
      try {
        info = await engine.request<EngineInfo>("info", { device: settings.device });
        if (!settings.customLevels && info.detected_custom_levels) {
          await settingsStore.save({ customLevels: info.detected_custom_levels });
          settings.customLevels = info.detected_custom_levels;
        }
      } catch (err) {
        logger.warn(`Startup info failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    return {
      needsEnginePack,
      engineVersion: expectedEngineVersion,
      appVersion,
      settings,
      info,
    };
  });

  ipcMain.handle("bsm:installEnginePack", async (): Promise<void> => {
    const active = queueManager
      .getItems()
      .find((item) => item.status === "downloading" || item.status === "running");
    if (active) {
      await queueManager.cancel(active.id);
    }
    await engine.stop();
    await enginePack.install((progress) => {
      getWindow()?.webContents.send("bsm:enginePackProgress", progress);
    });
    const nextSettings = await settingsStore.save({
      engineVersionInstalled: expectedEngineVersion,
    });
    if (nextSettings.customLevels) {
      queueManager.kick();
    }
  });

  ipcMain.handle("bsm:getInfo", async (): Promise<EngineInfo> => {
    try {
      const settings = await settingsStore.load();
      return await engine.request<EngineInfo>("info", { device: settings.device });
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle("bsm:getSettings", async () => settingsStore.load());

  ipcMain.handle("bsm:saveSettings", async (_ev, partial) => {
    const next = await settingsStore.save(partial);
    if (next.customLevels && !(await enginePack.needsInstall())) {
      queueManager.kick();
    }
    return next;
  });

  ipcMain.handle("bsm:validateCustomLevels", async (_ev, folder: string) =>
    validateCustomLevelsFolder(folder),
  );

  ipcMain.handle("bsm:pickFolder", async (_ev, title?: string): Promise<string | null> => {
    const win = getWindow();
    const opts = {
      title: title || "Select folder",
      properties: ["openDirectory" as const],
    };
    const res = win ? await dialog.showOpenDialog(win, opts) : await dialog.showOpenDialog(opts);
    if (res.canceled || res.filePaths.length === 0) {
      return null;
    }
    return res.filePaths[0];
  });

  ipcMain.handle("bsm:pickAudioFile", async (): Promise<string | null> => {
    const win = getWindow();
    const opts = {
      title: "Select audio file",
      properties: ["openFile" as const],
      filters: [
        {
          name: "Audio",
          extensions: ["mp3", "wav", "ogg", "egg", "flac", "m4a", "aac", "opus", "webm"],
        },
      ],
    };
    const res = win ? await dialog.showOpenDialog(win, opts) : await dialog.showOpenDialog(opts);
    if (res.canceled || res.filePaths.length === 0) {
      return null;
    }
    return res.filePaths[0];
  });

  ipcMain.handle(
    "bsm:pickCoverImage",
    async (): Promise<{ path: string; dataUrl: string } | null> => {
      const win = getWindow();
      const opts = {
        title: "Select cover image",
        properties: ["openFile" as const],
        filters: [
          {
            name: "Images",
            extensions: ["jpg", "jpeg", "png", "webp"],
          },
        ],
      };
      const res = win ? await dialog.showOpenDialog(win, opts) : await dialog.showOpenDialog(opts);
      if (res.canceled || res.filePaths.length === 0) {
        return null;
      }
      const chosen = res.filePaths[0];
      const mime = mimeForImageExt(path.extname(chosen));
      if (!mime) return null;
      try {
        const buf = await fsp.readFile(chosen);
        return {
          path: chosen,
          dataUrl: `data:${mime};base64,${buf.toString("base64")}`,
        };
      } catch {
        return null;
      }
    },
  );

  ipcMain.handle("bsm:openPath", async (_ev, targetPath: string): Promise<void> => {
    if (!targetPath) return;
    await shell.openPath(path.resolve(targetPath));
  });

  ipcMain.handle("bsm:openExternalUrl", async (_ev, rawUrl: string): Promise<void> => {
    if (!rawUrl) return;
    try {
      const parsed = new URL(rawUrl);
      if (parsed.protocol === "https:") {
        await shell.openExternal(parsed.toString());
      }
    } catch {
      // Ignore malformed URLs.
    }
  });

  ipcMain.handle("bsm:openLogs", async (): Promise<void> => {
    const logsDir = logger.getLogsDir();
    await fsp.mkdir(logsDir, { recursive: true });
    await shell.openPath(logsDir);
  });

  ipcMain.handle("bsm:readCoverDataUrl", async (_ev, filePath: string): Promise<string | null> => {
    if (!filePath) return null;
    const resolvedFile = path.resolve(filePath);
    const resolvedMedia = path.resolve(mediaDir);
    const normFile = process.platform === "win32" ? resolvedFile.toLowerCase() : resolvedFile;
    const normMedia = process.platform === "win32" ? resolvedMedia.toLowerCase() : resolvedMedia;
    if (!normFile.startsWith(normMedia + path.sep)) {
      return null;
    }
    const mime = mimeForImageExt(path.extname(resolvedFile));
    if (!mime) return null;
    try {
      const buf = await fsp.readFile(resolvedFile);
      return `data:${mime};base64,${buf.toString("base64")}`;
    } catch {
      return null;
    }
  });

  ipcMain.handle("bsm:searchYouTube", async (_ev, query: string): Promise<YouTubeItem[]> => {
    try {
      const res = await engine.request<{ items: YouTubeItem[] }>("search", { query });
      return res.items;
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle("bsm:resolveYouTube", async (_ev, url: string): Promise<YouTubeItem> => {
    try {
      return await engine.request<YouTubeItem>("resolve", { url });
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle("bsm:getQueue", async (): Promise<QueueItem[]> => queueManager.getItems());

  ipcMain.handle(
    "bsm:enqueueGeneration",
    async (_ev, req: QueueEnqueueRequest): Promise<QueueItem> => {
      try {
        return await queueManager.enqueue(req);
      } catch (err) {
        wrapIpcError(err, logger);
      }
    },
  );

  ipcMain.handle("bsm:cancelQueueItem", async (_ev, id: string): Promise<QueueItem[]> => {
    try {
      return await queueManager.cancel(id);
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle("bsm:retryQueueItem", async (_ev, id: string): Promise<QueueItem[]> => {
    try {
      return await queueManager.retry(id);
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle(
    "bsm:moveQueueItem",
    async (_ev, id: string, direction: "up" | "down"): Promise<QueueItem[]> => {
      try {
        return await queueManager.move(id, direction);
      } catch (err) {
        wrapIpcError(err, logger);
      }
    },
  );

  ipcMain.handle("bsm:clearFinishedQueue", async (): Promise<QueueItem[]> => {
    try {
      return await queueManager.clearFinished();
    } catch (err) {
      wrapIpcError(err, logger);
    }
  });

  ipcMain.handle("bsm:listLibrary", async (): Promise<LibraryEntry[]> => libraryStore.list());

  ipcMain.handle("bsm:deleteLibraryEntry", async (_ev, id: string): Promise<LibraryEntry[]> => {
    const settings = await settingsStore.load();
    return libraryStore.deleteEntry(id, settings.customLevels, (folderPath) =>
      shell.trashItem(folderPath),
    );
  });

  ipcMain.handle(
    "bsm:exportMapZip",
    async (_ev, folder: string, suggestedName?: string): Promise<string> => {
      const settings = await settingsStore.load();
      const baseName = suggestedName || path.basename(folder) || "map";
      const safeName = baseName.replace(/[<>:"/\\|?*]/g, " ").trim() || "map";
      let outZip = process.env.BSM_TEST_EXPORT_PATH || "";

      if (!outZip) {
        const win = getWindow();
        const opts = {
          title: "Export map as ZIP",
          defaultPath: path.join(settings.exportDir, `${safeName}.zip`),
          filters: [{ name: "ZIP", extensions: ["zip"] }],
        };
        const res = win ? await dialog.showSaveDialog(win, opts) : await dialog.showSaveDialog(opts);
        if (res.canceled || !res.filePath) {
          throw new Error("BSM_CODE:cancelled");
        }
        outZip = res.filePath;
      }

      try {
        const out = await engine.request<{ out_zip: string }>("export_zip", {
          folder,
          out_zip: outZip,
        });
        return out.out_zip;
      } catch (err) {
        wrapIpcError(err, logger);
      }
    },
  );

  ipcMain.handle("bsm:checkForUpdates", async (): Promise<void> => {
    if (deps.checkForUpdates) {
      await deps.checkForUpdates();
    }
  });

  ipcMain.handle("bsm:installUpdate", async (): Promise<void> => {
    if (deps.installUpdate) {
      deps.installUpdate();
    }
  });

  ipcMain.handle("bsm:readMapForViewer", async (_ev, folder: string) => {
    try {
      return await readMapFolderForViewer(folder);
    } catch (err) {
      logger.warn(`readMapForViewer failed: ${err instanceof Error ? err.message : String(err)}`);
      throw new Error("BSM_CODE:not_found");
    }
  });
}
