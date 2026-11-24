import { contextBridge, ipcRenderer } from "electron";
import type {
  EnginePackProgress,
  LibraryEntry,
  QueueEnqueueRequest,
  QueueItem,
  QueueProgressEvent,
  Settings,
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
  getQueue: () => ipcRenderer.invoke("bsm:getQueue"),
  enqueueGeneration: (req: QueueEnqueueRequest) => ipcRenderer.invoke("bsm:enqueueGeneration", req),
  cancelQueueItem: (id: string) => ipcRenderer.invoke("bsm:cancelQueueItem", id),
  retryQueueItem: (id: string) => ipcRenderer.invoke("bsm:retryQueueItem", id),
  moveQueueItem: (id: string, direction: "up" | "down") =>
    ipcRenderer.invoke("bsm:moveQueueItem", id, direction),
  clearFinishedQueue: () => ipcRenderer.invoke("bsm:clearFinishedQueue"),
  listLibrary: () => ipcRenderer.invoke("bsm:listLibrary"),
  deleteLibraryEntry: (id: string) => ipcRenderer.invoke("bsm:deleteLibraryEntry", id),
  exportMapZip: (folder: string, suggestedName?: string) =>
    ipcRenderer.invoke("bsm:exportMapZip", folder, suggestedName),
  checkForUpdates: () => ipcRenderer.invoke("bsm:checkForUpdates"),
  installUpdate: () => ipcRenderer.invoke("bsm:installUpdate"),
  readMapForViewer: (folder: string) => ipcRenderer.invoke("bsm:readMapForViewer", folder),
  onQueueChanged: (cb: (items: QueueItem[]) => void) =>
    subscribe<QueueItem[]>("bsm:queueChanged", cb),
  onQueueProgress: (cb: (ev: QueueProgressEvent) => void) =>
    subscribe<QueueProgressEvent>("bsm:queueProgress", cb),
  onLibraryChanged: (cb: (entries: LibraryEntry[]) => void) =>
    subscribe<LibraryEntry[]>("bsm:libraryChanged", cb),
  onEnginePackProgress: (cb: (ev: EnginePackProgress) => void) =>
    subscribe<EnginePackProgress>("bsm:enginePackProgress", cb),
  onUpdateStatus: (cb: (ev: UpdateStatus) => void) =>
    subscribe<UpdateStatus>("bsm:updateStatus", cb),
});
