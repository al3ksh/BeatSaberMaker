import { contextBridge, ipcRenderer } from "electron";
import type {
  EnginePackProgress,
  ProgressEvent,
  Settings,
  StartGenerationRequest,
  UpdateStatus,
} from "./types";

function subscribe<T>(channel: string, cb: (payload: T) => void): () => void {
  const listener = (_event: Electron.IpcRendererEvent, payload: T) => cb(payload);
  ipcRenderer.on(channel, listener);
  return () => {
    ipcRenderer.removeListener(channel, listener);
  };
}

contextBridge.exposeInMainWorld("bsm", {
  getStartupState: () => ipcRenderer.invoke("bsm:getStartupState"),
  installEnginePack: () => ipcRenderer.invoke("bsm:installEnginePack"),
  getInfo: () => ipcRenderer.invoke("bsm:getInfo"),
  getSettings: () => ipcRenderer.invoke("bsm:getSettings"),
  saveSettings: (partial: Partial<Settings>) => ipcRenderer.invoke("bsm:saveSettings", partial),
  validateCustomLevels: (folder: string) => ipcRenderer.invoke("bsm:validateCustomLevels", folder),
  pickFolder: (title?: string) => ipcRenderer.invoke("bsm:pickFolder", title),
  pickAudioFile: () => ipcRenderer.invoke("bsm:pickAudioFile"),
  pickCoverImage: () => ipcRenderer.invoke("bsm:pickCoverImage"),
  openPath: (targetPath: string) => ipcRenderer.invoke("bsm:openPath", targetPath),
  openExternalUrl: (url: string) => ipcRenderer.invoke("bsm:openExternalUrl", url),
  openLogs: () => ipcRenderer.invoke("bsm:openLogs"),
  readCoverDataUrl: (filePath: string) => ipcRenderer.invoke("bsm:readCoverDataUrl", filePath),
  searchYouTube: (query: string) => ipcRenderer.invoke("bsm:searchYouTube", query),
  resolveYouTube: (url: string) => ipcRenderer.invoke("bsm:resolveYouTube", url),
  startGeneration: (req: StartGenerationRequest) => ipcRenderer.invoke("bsm:startGeneration", req),
  cancelActive: () => ipcRenderer.invoke("bsm:cancelActive"),
  listLibrary: () => ipcRenderer.invoke("bsm:listLibrary"),
  deleteLibraryEntry: (id: string) => ipcRenderer.invoke("bsm:deleteLibraryEntry", id),
  exportMapZip: (folder: string, suggestedName?: string) =>
    ipcRenderer.invoke("bsm:exportMapZip", folder, suggestedName),
  checkForUpdates: () => ipcRenderer.invoke("bsm:checkForUpdates"),
  installUpdate: () => ipcRenderer.invoke("bsm:installUpdate"),
  onProgress: (cb: (ev: ProgressEvent) => void) => subscribe<ProgressEvent>("bsm:progress", cb),
  onEnginePackProgress: (cb: (ev: EnginePackProgress) => void) =>
    subscribe<EnginePackProgress>("bsm:enginePackProgress", cb),
  onUpdateStatus: (cb: (ev: UpdateStatus) => void) =>
    subscribe<UpdateStatus>("bsm:updateStatus", cb),
});
