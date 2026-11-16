import path from "node:path";
import { app, BrowserWindow, Menu } from "electron";
import { createDefaultResolveLaunch, EngineManager } from "./engineManager";
import { registerIpcHandlers } from "./ipc";
import { LibraryStore } from "./library";
import { createLogger } from "./logger";
import { SettingsStore } from "./store";

app.setName("BeatSaberMaker");
if (process.env.BSM_USER_DATA_DIR) {
  app.setPath("userData", path.resolve(process.env.BSM_USER_DATA_DIR));
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;
let engineManager: EngineManager | null = null;

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1120,
    height: 760,
    minWidth: 920,
    minHeight: 640,
    backgroundColor: "#09090b",
    title: "BeatSaberMaker",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  Menu.setApplicationMenu(null);

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

  app.whenReady().then(() => {
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
        mainWindow?.webContents.send("bsm:progress", ev);
      },
      onStderr: (line) => {
        logger.engineStderr(line);
      },
    });

    registerIpcHandlers({
      appVersion: app.getVersion(),
      appPath: app.getAppPath(),
      userDataDir,
      mediaDir,
      engine: engineManager,
      settingsStore,
      libraryStore,
      logger,
      getWindow: () => mainWindow,
    });

    mainWindow = createWindow();
  });

  let quitting = false;
  app.on("before-quit", (ev) => {
    if (quitting || !engineManager) {
      return;
    }
    ev.preventDefault();
    quitting = true;
    engineManager
      .stop()
      .catch(() => {})
      .finally(() => {
        app.quit();
      });
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}
