import fs from "node:fs";
import path from "node:path";
import { app, BrowserWindow, Menu, nativeTheme, shell } from "electron";
import { createDefaultResolveLaunch, EngineManager } from "./engineManager";
import { defaultManifestUrl, EnginePackInstaller, readEngineVersionInfo } from "./enginePack";
import { registerIpcHandlers } from "./ipc";
import { LibraryStore } from "./library";
import { createLogger } from "./logger";
import { QueueManager } from "./queue";
import { SettingsStore } from "./store";
import { setupUpdater } from "./updater";

app.setName("BeatSaberMaker");
// Dark title bar on Windows even when the system uses the light theme.
nativeTheme.themeSource = "dark";
if (process.env.BSM_USER_DATA_DIR) {
  app.setPath("userData", path.resolve(process.env.BSM_USER_DATA_DIR));
}
if (process.env.BSM_MUTE_AUDIO === "1") {
  app.commandLine.appendSwitch("mute-audio");
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;
let engineManager: EngineManager | null = null;
let queueManager: QueueManager | null = null;

function sendToWindow(channel: string, payload: unknown): void {
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(channel, payload);
    }
  } catch {
    // Window may be closed or tearing down.
  }
}

function createWindow(): BrowserWindow {
  const iconPath = path.resolve(__dirname, "..", "build", "icon.ico");
  const win = new BrowserWindow({
    width: 1080,
    height: 740,
    minWidth: 880,
    minHeight: 640,
    backgroundColor: "#09090b",
    // Shown on ready-to-show so the window never paints a blank (white) frame first.
    show: false,
    title: "BeatSaberMaker",
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  Menu.setApplicationMenu(null);

  const reveal = () => {
    if (!win.isDestroyed() && !win.isVisible()) win.show();
  };
  win.once("ready-to-show", reveal);
  setTimeout(reveal, 3000); // Fallback if the first paint never comes.

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    win.loadURL(devUrl).catch(() => {});
  } else {
    win.loadFile(path.resolve(__dirname, "..", "dist", "index.html")).catch(() => {});
  }

  win.on("closed", () => {
    if (mainWindow === win) {
      mainWindow = null;
    }
  });

  return win;
}

if (gotLock) {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore();
      }
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    const userDataDir = app.getPath("userData");
    const logsDir = path.join(userDataDir, "logs");
    const mediaDir = path.join(userDataDir, "media");
    const logger = createLogger(logsDir);
    const settingsStore = new SettingsStore(
      path.join(userDataDir, "settings.json"),
      app.getPath("downloads"),
    );
    const libraryStore = new LibraryStore(path.join(userDataDir, "library.json"), mediaDir);

    engineManager = new EngineManager({
      resolveLaunch: createDefaultResolveLaunch(userDataDir, logsDir),
      onProgress: (ev) => {
        queueManager?.handleProgress(ev);
      },
      onStderr: (line) => {
        logger.engineStderr(line);
      },
    });

    const engineInfo = readEngineVersionInfo(app.getAppPath());
    const expectedEngineVersion = engineInfo.engineVersion;
    const enginePack = new EnginePackInstaller({
      engineDir: path.join(userDataDir, "engine"),
      downloadDir: path.join(userDataDir, "engine-downloads"),
      expectedVersion: expectedEngineVersion,
      manifestUrl: defaultManifestUrl(app.getVersion(), engineInfo.releaseTag),
    });

    queueManager = new QueueManager({
      filePath: path.join(userDataDir, "queue.json"),
      tmpDownloadsRoot: path.join(userDataDir, "tmp-downloads"),
      engine: engineManager,
      settingsStore,
      libraryStore,
      trashItem: (folderPath) => shell.trashItem(folderPath),
      logger,
      onChange: (items) => {
        sendToWindow("bsm:queueChanged", items);
      },
      onProgress: (ev) => {
        sendToWindow("bsm:queueProgress", ev);
      },
      onLibraryChanged: (entries) => {
        sendToWindow("bsm:libraryChanged", entries);
      },
    });

    const updater = setupUpdater(() => mainWindow, logger);

    registerIpcHandlers({
      appVersion: app.getVersion(),
      appPath: app.getAppPath(),
      userDataDir,
      mediaDir,
      engine: engineManager,
      enginePack,
      expectedEngineVersion,
      settingsStore,
      libraryStore,
      queueManager,
      logger,
      getWindow: () => mainWindow,
      checkForUpdates: updater.checkForUpdates,
      installUpdate: updater.installUpdate,
    });

    try {
      await queueManager.init({
        canAutoStart: async () => {
          if (await enginePack.needsInstall()) {
            return false;
          }
          const settings = await settingsStore.load();
          return Boolean(settings.customLevels);
        },
      });
    } catch (err) {
      logger.error(`Queue initialization failed: ${err instanceof Error ? err.stack || err.message : String(err)}`);
    }

    mainWindow = createWindow();
  });

  let quitting = false;
  app.on("before-quit", (ev) => {
    if (quitting || (!queueManager && !engineManager)) {
      return;
    }
    ev.preventDefault();
    quitting = true;
    (async () => {
      try {
        await queueManager?.shutdown();
      } catch {
        // Best-effort queue flush on quit.
      }
      try {
        await engineManager?.stop();
      } catch {
        // Best-effort engine stop on quit.
      }
    })().finally(() => {
      app.quit();
    });
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}
