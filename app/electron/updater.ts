import { app, BrowserWindow } from "electron";
import { autoUpdater } from "electron-updater";
import type { AppLogger } from "./logger";
import type { UpdateStatus } from "./types";

export interface UpdaterController {
  checkForUpdates: () => Promise<void>;
  installUpdate: () => void;
}

export function setupUpdater(
  getWindow: () => BrowserWindow | null,
  logger: AppLogger,
): UpdaterController {
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  const send = (status: UpdateStatus) => {
    try {
      getWindow()?.webContents.send("bsm:updateStatus", status);
    } catch {
      // Window may be closed.
    }
  };

  autoUpdater.on("checking-for-update", () => {
    send({ state: "checking" });
  });
  autoUpdater.on("update-available", (info) => {
    send({ state: "available", version: info?.version });
  });
  autoUpdater.on("update-not-available", (info) => {
    send({ state: "not-available", version: info?.version });
  });
  autoUpdater.on("download-progress", (progress) => {
    send({ state: "downloading", pct: Math.round(progress.percent) });
  });
  autoUpdater.on("update-downloaded", (info) => {
    send({ state: "downloaded", version: info?.version });
  });
  autoUpdater.on("error", (err) => {
    logger.warn(`Auto-updater error: ${err instanceof Error ? err.message : String(err)}`);
    send({ state: "error" });
  });

  if (app.isPackaged && process.env.BSM_DISABLE_UPDATER !== "1") {
    autoUpdater.checkForUpdatesAndNotify().catch((err) => {
      logger.warn(`Startup update check failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }

  return {
    checkForUpdates: async () => {
      if (!app.isPackaged || process.env.BSM_DISABLE_UPDATER === "1") {
        send({ state: "not-available" });
        return;
      }
      await autoUpdater.checkForUpdates();
    },
    installUpdate: () => {
      autoUpdater.quitAndInstall();
    },
  };
}
