import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { BrowserWindow, dialog, ipcMain, shell } from "electron";
import { EngineError, EngineManager } from "./engineManager";
import type { EnginePackInstaller } from "./enginePack";
import type { LibraryStore } from "./library";
import type { AppLogger } from "./logger";
import type { SettingsStore } from "./store";
import type {
  EngineInfo,
  GenerateOutput,
  LibraryEntry,
  StartGenerationRequest,
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
  logger: AppLogger;
  getWindow: () => BrowserWindow | null;
  checkForUpdates?: () => Promise<void>;
  installUpdate?: () => void;
}

export function registerIpcHandlers(deps: IpcDeps): void {
  const {
    appVersion,
    userDataDir,
    mediaDir,
    engine,
    enginePack,
    expectedEngineVersion,
    settingsStore,
    libraryStore,
    logger,
    getWindow,
  } = deps;

  const tmpDownloadsRoot = path.join(userDataDir, "tmp-downloads");

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
    await engine.stop();
    await enginePack.install((progress) => {
      getWindow()?.webContents.send("bsm:enginePackProgress", progress);
    });
    await settingsStore.save({ engineVersionInstalled: expectedEngineVersion });
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

  ipcMain.handle("bsm:saveSettings", async (_ev, partial) => settingsStore.save(partial));

  ipcMain.handle("bsm:validateCustomLevels", async (_ev, folder: string) =>
    validateCustomLevelsFolder(folder),
  );

  ipcMain.handle("bsm:pickFolder", async (_ev, title?: string): Promise<string | null> => {
    const win = getWindow();
    const opts = {
      title: title || "Wybierz folder",
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
      title: "Wybierz plik audio",
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

  ipcMain.handle("bsm:openPath", async (_ev, targetPath: string): Promise<void> => {
    if (!targetPath) return;
    await shell.openPath(path.resolve(targetPath));
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

  ipcMain.handle(
    "bsm:startGeneration",
    async (
      _ev,
      req: StartGenerationRequest,
    ): Promise<{ result: GenerateOutput; entry: LibraryEntry }> => {
      const settings = await settingsStore.load();
      if (!settings.customLevels) {
        throw new Error("BSM_CODE:not_found");
      }

      const id = req.regenerateId || crypto.randomUUID();
      const tmpDownloadDir = path.join(tmpDownloadsRoot, `${id}-${Date.now()}`);
      try {
        let audioSrc = req.sourcePathOrUrl;
        let coverSrc = req.coverPath;

        if (req.sourceType === "youtube") {
          const dl = await engine.request<{
            audio_path: string;
            cover_path: string | null;
          }>("download", { url: req.sourcePathOrUrl, dest_dir: tmpDownloadDir });
          audioSrc = dl.audio_path;
          if (!coverSrc && dl.cover_path) {
            coverSrc = dl.cover_path;
          }
        }

        const cached = await libraryStore.cacheMedia(id, audioSrc, coverSrc);
        const cfg = req.config;
        const levels =
          cfg.mode === "pack"
            ? cfg.levels ?? ["Hard", "Expert", "ExpertPlus"]
            : cfg.nps !== null
              ? null
              : cfg.levels;
        const nps = cfg.mode === "pack" ? null : cfg.nps;

        const result = await engine.request<GenerateOutput>("generate", {
          audio_path: cached.audioCachePath,
          out_dir: settings.customLevels,
          levels,
          nps,
          styles: cfg.styles,
          ornaments: cfg.ornaments,
          lights: cfg.lights,
          environment: cfg.environment,
          seed: cfg.seed,
          lights_rules: cfg.lightsRules,
          title: req.title,
          artist: req.artist,
          cover_src: cached.coverCachePath,
          unique_folder: true,
          device: settings.device,
        });

        const entry: LibraryEntry = {
          id,
          title: result.title,
          artist: result.artist,
          folder: result.folder,
          createdAt: new Date().toISOString(),
          bpm: result.bpm,
          levels: result.levels.map((lv) => ({
            rank: lv.rank,
            nps: lv.nps,
            nNotes: lv.n_notes,
            nArcs: lv.n_arcs,
            nChains: lv.n_chains,
          })),
          nWalls: result.n_walls,
          nBombs: result.n_bombs,
          nLights: result.n_lights,
          environment: result.environment,
          audioCachePath: cached.audioCachePath,
          coverCachePath: cached.coverCachePath,
          config: cfg,
        };

        if (req.regenerateId) {
          await libraryStore.replaceAfterRegenerate(
            req.regenerateId,
            entry,
            settings.customLevels,
            (folderPath) => shell.trashItem(folderPath),
          );
        } else {
          await libraryStore.saveEntry(entry);
        }

        return { result, entry };
      } catch (err) {
        wrapIpcError(err, logger);
      } finally {
        await fsp.rm(tmpDownloadDir, { recursive: true, force: true }).catch(() => {});
      }
    },
  );

  ipcMain.handle("bsm:cancelActive", async (): Promise<void> => {
    await engine.cancelActive();
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
          title: "Eksportuj mapę jako ZIP",
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
}
