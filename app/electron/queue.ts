import crypto from "node:crypto";
import fsp from "node:fs/promises";
import path from "node:path";
import { EngineError, type EngineManager } from "./engineManager";
import type { LibraryStore } from "./library";
import type { AppLogger } from "./logger";
import { atomicWriteJson, type SettingsStore } from "./store";
import type {
  GenerateOutput,
  LibraryEntry,
  MapGenerationConfig,
  ProgressEvent,
  QueueEnqueueRequest,
  QueueItem,
  QueueItemStatus,
  QueueProgressEvent,
  StartGenerationRequest,
} from "./types";

const DEFAULT_MAX_FINISHED_HISTORY = 50;
const PACK_SEED_MAX = 2_000_000_000;
const DEFAULT_PACK_LEVELS = ["Hard", "Expert", "ExpertPlus"];
const FINISHED_STATUSES = new Set<QueueItemStatus>(["done", "failed", "cancelled"]);

export interface QueueManagerOptions {
  filePath: string;
  tmpDownloadsRoot: string;
  engine: Pick<EngineManager, "request" | "cancelActive" | "stop">;
  settingsStore: Pick<SettingsStore, "load">;
  libraryStore: LibraryStore;
  trashItem: (folderPath: string) => Promise<void>;
  logger: Pick<AppLogger, "info" | "warn" | "error">;
  onChange?: (items: QueueItem[]) => void;
  onProgress?: (ev: QueueProgressEvent) => void;
  onLibraryChanged?: (entries: LibraryEntry[]) => void;
  maxFinishedHistory?: number;
}

function isFinishedStatus(status: QueueItemStatus): boolean {
  return FINISHED_STATUSES.has(status);
}

function isEnoent(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    (err as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function resolveSeed(config: MapGenerationConfig): number {
  if (config.mode === "pack" && config.seed <= 0) {
    return Math.floor(Math.random() * PACK_SEED_MAX) + 1;
  }
  return config.seed;
}

function cloneItem(item: QueueItem): QueueItem {
  return {
    ...item,
    progress: item.progress ? { ...item.progress } : null,
  };
}

function sanitizeForDisk(item: QueueItem): QueueItem {
  const { cancelling: _cancelling, ...rest } = item;
  return {
    ...rest,
    thumbnailUrl:
      rest.thumbnailUrl && rest.thumbnailUrl.startsWith("data:") ? null : rest.thumbnailUrl,
    progress: rest.progress ? { ...rest.progress } : null,
  };
}

function buildLibraryEntry(
  id: string,
  result: GenerateOutput,
  cached: { audioCachePath: string; coverCachePath: string | null },
  config: MapGenerationConfig,
): LibraryEntry {
  return {
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
      nWalls: lv.n_walls,
      nBombs: lv.n_bombs,
    })),
    nWalls: result.n_walls,
    nBombs: result.n_bombs,
    nLights: result.n_lights,
    environment: result.environment,
    audioCachePath: cached.audioCachePath,
    coverCachePath: cached.coverCachePath,
    config,
  };
}

export class QueueManager {
  private items: QueueItem[] = [];
  private activeItemId: string | null = null;
  private activeCancelRequested = false;
  private draining = false;
  private drainPromise: Promise<void> | null = null;
  private pauseCount = 0;
  private shuttingDown = false;
  private shutdownRequested = false;
  private persistChain: Promise<void> = Promise.resolve();
  private readonly maxFinishedHistory: number;

  constructor(private readonly options: QueueManagerOptions) {
    this.maxFinishedHistory = Math.max(
      0,
      options.maxFinishedHistory ?? DEFAULT_MAX_FINISHED_HISTORY,
    );
  }

  private emitChange(): void {
    try {
      this.options.onChange?.(this.getItems());
    } catch (err) {
      const msg = err instanceof Error ? err.stack || err.message : String(err);
      this.options.logger.error(`Queue onChange callback failed: ${msg}`);
    }
  }

  private persist(): Promise<void> {
    const run = async () => {
      const snapshot = this.items.map(sanitizeForDisk);
      await atomicWriteJson(this.options.filePath, snapshot);
    };
    const next = this.persistChain.then(run, run);
    this.persistChain = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async cleanupTmpDownloadsRoot(): Promise<void> {
    try {
      const entries = await fsp.readdir(this.options.tmpDownloadsRoot);
      await Promise.all(
        entries.map((name) =>
          fsp
            .rm(path.join(this.options.tmpDownloadsRoot, name), {
              recursive: true,
              force: true,
            })
            .catch(() => {}),
        ),
      );
    } catch {
      // tmpDownloadsRoot may not exist yet.
    }
  }

  private async readPersistedItems(): Promise<QueueItem[]> {
    try {
      const raw = await fsp.readFile(this.options.filePath, "utf8");
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as QueueItem[]) : [];
    } catch (err) {
      if (!isEnoent(err)) {
        const backup = `${this.options.filePath}.corrupt-${Date.now()}`;
        await fsp.copyFile(this.options.filePath, backup).catch(() => {});
      }
      return [];
    }
  }

  async init(opts?: { canAutoStart?: () => Promise<boolean> }): Promise<void> {
    await this.cleanupTmpDownloadsRoot();
    const loaded = await this.readPersistedItems();
    const now = new Date().toISOString();
    let changed = false;

    this.items = loaded.map((raw) => {
      const { cancelling: _cancelling, ...item } = raw;
      if (item.status === "downloading" || item.status === "running") {
        changed = true;
        return {
          ...item,
          status: "failed" as const,
          errorCode: "interrupted",
          progress: null,
          finishedAt: now,
        };
      }
      return item;
    });

    if (changed) {
      await this.persist();
    }
    this.emitChange();

    const canAutoStart = opts?.canAutoStart ? await opts.canAutoStart() : true;
    if (canAutoStart) {
      this.kick();
    }
  }

  getItems(): QueueItem[] {
    return this.items.map(cloneItem);
  }

  async enqueue(req: QueueEnqueueRequest): Promise<QueueItem> {
    const settings = await this.options.settingsStore.load();
    if (!settings.customLevels) {
      throw new EngineError("not_found", "CustomLevels folder is not configured");
    }

    const { thumbnailUrl = null, ...startReq } = req;
    const request: StartGenerationRequest = {
      ...startReq,
      config: {
        ...startReq.config,
        styles: [...startReq.config.styles],
        levels: startReq.config.levels ? [...startReq.config.levels] : null,
        seed: resolveSeed(startReq.config),
      },
    };

    const item: QueueItem = {
      id: crypto.randomUUID(),
      status: "waiting",
      request,
      thumbnailUrl,
      enqueuedAt: new Date().toISOString(),
      startedAt: null,
      finishedAt: null,
      progress: null,
      errorCode: null,
      result: null,
      entry: null,
    };

    this.items.push(item);
    await this.persist();
    this.emitChange();
    const snapshot = cloneItem(item);
    this.kick();
    return snapshot;
  }

  async cancel(id: string): Promise<QueueItem[]> {
    const item = this.items.find((entry) => entry.id === id);
    if (!item) {
      return this.getItems();
    }

    if (item.id === this.activeItemId) {
      if (
        item.status === "downloading" ||
        item.status === "running" ||
        item.status === "waiting"
      ) {
        this.activeCancelRequested = true;
        item.cancelling = true;
        this.emitChange();
        await this.options.engine.cancelActive();
      }
      return this.getItems();
    }

    if (item.status === "waiting") {
      item.status = "cancelled";
      item.finishedAt = new Date().toISOString();
      item.progress = null;
      item.cancelling = false;
      await this.persist();
      this.emitChange();
      return this.getItems();
    }

    return this.getItems();
  }

  async retry(id: string): Promise<QueueItem[]> {
    const idx = this.items.findIndex((entry) => entry.id === id);
    if (idx === -1) {
      return this.getItems();
    }
    const item = this.items[idx];
    if (item.status !== "failed" && item.status !== "cancelled") {
      return this.getItems();
    }

    this.items.splice(idx, 1);
    item.status = "waiting";
    item.progress = null;
    item.errorCode = null;
    item.startedAt = null;
    item.finishedAt = null;
    item.result = null;
    item.entry = null;
    item.cancelling = false;
    this.items.push(item);

    await this.persist();
    this.emitChange();
    this.kick();
    return this.getItems();
  }

  async move(id: string, direction: "up" | "down"): Promise<QueueItem[]> {
    const waitingIndices: number[] = [];
    for (let i = 0; i < this.items.length; i++) {
      if (this.items[i].status === "waiting") {
        waitingIndices.push(i);
      }
    }

    const pos = waitingIndices.findIndex((idx) => this.items[idx].id === id);
    if (pos === -1) {
      return this.getItems();
    }

    const targetPos = direction === "up" ? pos - 1 : pos + 1;
    if (targetPos < 0 || targetPos >= waitingIndices.length) {
      return this.getItems();
    }

    const fromIdx = waitingIndices[pos];
    const toIdx = waitingIndices[targetPos];
    [this.items[fromIdx], this.items[toIdx]] = [this.items[toIdx], this.items[fromIdx]];

    await this.persist();
    this.emitChange();
    return this.getItems();
  }

  async clearFinished(): Promise<QueueItem[]> {
    const next = this.items.filter((item) => !isFinishedStatus(item.status));
    if (next.length !== this.items.length) {
      this.items = next;
      await this.persist();
      this.emitChange();
    }
    return this.getItems();
  }

  handleProgress(ev: ProgressEvent): void {
    if (!this.activeItemId) {
      return;
    }
    const item = this.items.find((entry) => entry.id === this.activeItemId);
    if (!item || (item.status !== "downloading" && item.status !== "running")) {
      return;
    }
    item.progress = { stage: ev.stage, pct: ev.pct };
    try {
      this.options.onProgress?.({ id: item.id, stage: ev.stage, pct: ev.pct });
    } catch (err) {
      const msg = err instanceof Error ? err.stack || err.message : String(err);
      this.options.logger.warn(`Queue onProgress callback failed: ${msg}`);
    }
  }

  kick(): void {
    if (this.draining || this.shuttingDown || this.pauseCount > 0) {
      return;
    }
    if (!this.items.some((item) => item.status === "waiting")) {
      return;
    }
    this.draining = true;
    this.drainPromise = this.drain();
  }

  async pause(): Promise<void> {
    this.pauseCount += 1;
    if (this.activeItemId !== null) {
      await this.cancel(this.activeItemId);
    }
    if (this.drainPromise) {
      await this.drainPromise;
    }
  }

  resume(): void {
    if (this.pauseCount > 0) {
      this.pauseCount -= 1;
    }
  }

  async shutdown(): Promise<void> {
    this.shuttingDown = true;
    if (this.activeItemId !== null) {
      this.shutdownRequested = true;
      await this.options.engine.stop();
    }
    if (this.drainPromise) {
      await this.drainPromise;
    }
  }

  private pruneFinished(): void {
    const finished = this.items
      .map((item, idx) => ({ item, idx }))
      .filter(({ item }) => isFinishedStatus(item.status));
    if (finished.length <= this.maxFinishedHistory) {
      return;
    }

    const toDropCount = finished.length - this.maxFinishedHistory;
    finished.sort((a, b) => {
      const aTime = a.item.finishedAt ?? a.item.enqueuedAt;
      const bTime = b.item.finishedAt ?? b.item.enqueuedAt;
      const cmp = aTime.localeCompare(bTime);
      return cmp !== 0 ? cmp : a.idx - b.idx;
    });

    const dropIds = new Set(finished.slice(0, toDropCount).map(({ item }) => item.id));
    this.items = this.items.filter((item) => !dropIds.has(item.id));
  }

  private checkAborted(): void {
    if (this.shutdownRequested) {
      throw new EngineError("cancelled", "shutdown");
    }
    if (this.activeCancelRequested) {
      throw new EngineError("cancelled", "cancelled");
    }
  }

  private async drain(): Promise<void> {
    try {
      while (!this.shuttingDown && this.pauseCount === 0) {
        const next = this.items.find((item) => item.status === "waiting");
        if (!next) {
          break;
        }
        await this.executeItem(next);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.stack || err.message : String(err);
      this.options.logger.error(`Queue drain loop error: ${msg}`);
    } finally {
      this.draining = false;
      this.drainPromise = null;
    }
  }

  private async executeItem(item: QueueItem): Promise<void> {
    this.activeItemId = item.id;
    this.activeCancelRequested = false;
    item.startedAt = new Date().toISOString();
    if (item.request.sourceType === "youtube") {
      item.status = "downloading";
      item.progress = { stage: "downloading", pct: 0 };
    } else {
      item.status = "running";
      item.progress = { stage: "analyzing", pct: 0 };
    }

    const entryId = item.request.regenerateId || crypto.randomUUID();
    const tmpDownloadDir = path.join(this.options.tmpDownloadsRoot, `${item.id}-${Date.now()}`);
    let newMediaDirPath: string | null = null;

    try {
      await this.persist();
      this.emitChange();
      this.checkAborted();

      const settings = await this.options.settingsStore.load();
      this.checkAborted();
      if (!settings.customLevels) {
        throw new EngineError("not_found", "CustomLevels folder is not configured");
      }

      let audioSrc = item.request.sourcePathOrUrl;
      let coverSrc = item.request.coverPath;

      if (item.request.sourceType === "youtube") {
        const dl = await this.options.engine.request<{
          audio_path: string;
          cover_path: string | null;
        }>("download", { url: item.request.sourcePathOrUrl, dest_dir: tmpDownloadDir });
        audioSrc = dl.audio_path;
        coverSrc = item.request.coverPath || dl.cover_path;
        this.checkAborted();

        item.status = "running";
        item.progress = { stage: "analyzing", pct: 0 };
        await this.persist();
        this.emitChange();
        this.checkAborted();
      }

      const cached = await this.options.libraryStore.cacheMedia(entryId, audioSrc, coverSrc);
      if (!item.request.regenerateId) {
        newMediaDirPath = path.dirname(cached.audioCachePath);
      }
      this.checkAborted();

      const cfg = item.request.config;
      const levels =
        cfg.mode === "pack"
          ? cfg.levels ?? DEFAULT_PACK_LEVELS
          : cfg.nps !== null
            ? null
            : cfg.levels;
      const nps = cfg.mode === "pack" ? null : cfg.nps;

      const result = await this.options.engine.request<GenerateOutput>("generate", {
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
        title: item.request.title,
        artist: item.request.artist,
        cover_src: cached.coverCachePath,
        unique_folder: true,
        device: settings.device,
      });

      const entry = buildLibraryEntry(entryId, result, cached, cfg);
      const updatedList = item.request.regenerateId
        ? await this.options.libraryStore.replaceAfterRegenerate(
            item.request.regenerateId,
            entry,
            settings.customLevels,
            this.options.trashItem,
          )
        : await this.options.libraryStore.saveEntry(entry);
      newMediaDirPath = null;

      item.status = "done";
      item.progress = { stage: "writing", pct: 100 };
      item.result = result;
      item.entry = entry;
      item.finishedAt = new Date().toISOString();
      item.cancelling = false;

      try {
        this.options.onLibraryChanged?.(updatedList);
      } catch (cbErr) {
        const msg = cbErr instanceof Error ? cbErr.stack || cbErr.message : String(cbErr);
        this.options.logger.warn(`onLibraryChanged callback failed: ${msg}`);
      }
    } catch (err) {
      if (newMediaDirPath) {
        await fsp.rm(newMediaDirPath, { recursive: true, force: true }).catch(() => {});
      }

      if (this.shutdownRequested) {
        item.status = "failed";
        item.errorCode = "interrupted";
      } else if (
        this.activeCancelRequested ||
        (err instanceof EngineError && err.code === "cancelled")
      ) {
        item.status = "cancelled";
        item.errorCode = null;
      } else {
        const msg = err instanceof Error ? err.stack || err.message : String(err);
        this.options.logger.error(`Queue item ${item.id} failed: ${msg}`);
        item.status = "failed";
        item.errorCode =
          err instanceof EngineError ? err.code : isEnoent(err) ? "not_found" : "internal";
      }
      item.progress = null;
      item.finishedAt = new Date().toISOString();
      item.cancelling = false;
    } finally {
      await fsp.rm(tmpDownloadDir, { recursive: true, force: true }).catch(() => {});
      this.activeItemId = null;
      this.activeCancelRequested = false;
      this.pruneFinished();
      try {
        await this.persist();
      } catch (persistErr) {
        const msg =
          persistErr instanceof Error
            ? persistErr.stack || persistErr.message
            : String(persistErr);
        this.options.logger.error(`Queue persist failed: ${msg}`);
      }
      this.emitChange();
    }
  }
}
