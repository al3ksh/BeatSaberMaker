import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { EngineError, EngineManager } from "./engineManager";
import { LibraryStore } from "./library";
import type { AppLogger } from "./logger";
import { QueueManager } from "./queue";
import { SettingsStore } from "./store";
import type {
  GenerateOutput,
  LibraryEntry,
  MapGenerationConfig,
  QueueEnqueueRequest,
  QueueItem,
  QueueProgressEvent,
} from "./types";

const BASE_CONFIG: MapGenerationConfig = {
  mode: "pack",
  levels: ["Hard", "Expert", "ExpertPlus"],
  nps: null,
  styles: [],
  ornaments: null,
  lights: 2,
  environment: "DefaultEnvironment",
  seed: 42,
  lightsRules: false,
};

function makeLogger(): AppLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    engineStderr: vi.fn(),
    getLogsDir: () => "",
  };
}

function makeGenerateOutput(
  customLevels: string,
  title: string,
  artist = "Artist",
): GenerateOutput {
  return {
    folder: path.join(customLevels, `BSM - ${title}`),
    title,
    artist,
    bpm: 140,
    grid_steady: true,
    n_notes: 300,
    nps: 4.2,
    rank: "Expert",
    levels: [
      {
        rank: "Hard",
        target_nps: 3.0,
        n_notes: 200,
        nps: 3.1,
        n_arcs: 4,
        n_chains: 2,
        n_walls: 8,
        n_bombs: 3,
      },
    ],
    n_walls: 8,
    n_bombs: 3,
    n_lights: 64,
    environment: "DefaultEnvironment",
    warnings: [],
    used_rule_lights: false,
    device_used: "cpu",
  };
}

async function waitFor(predicate: () => boolean, timeoutMs = 4000): Promise<void> {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) {
      throw new Error("Timed out waiting for condition");
    }
    await new Promise((r) => setTimeout(r, 15));
  }
}

async function createTestHarness(opts?: { customLevels?: string }) {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-queue-"));
  const customLevels = opts?.customLevels ?? path.join(tmp, "CustomLevels");
  if (customLevels) {
    await fs.mkdir(customLevels, { recursive: true });
  }
  const mediaDir = path.join(tmp, "media");
  const tmpDownloadsRoot = path.join(tmp, "tmp-downloads");
  const queueFilePath = path.join(tmp, "queue.json");
  const settingsStore = new SettingsStore(path.join(tmp, "settings.json"), path.join(tmp, "export"));
  if (customLevels) {
    await settingsStore.save({ customLevels });
  }
  const libraryStore = new LibraryStore(path.join(tmp, "library.json"), mediaDir);
  const logger = makeLogger();
  const trashed: string[] = [];
  const trashItem = async (folderPath: string) => {
    trashed.push(folderPath);
    await fs.rm(folderPath, { recursive: true, force: true });
  };
  const audioFile = path.join(tmp, "song.wav");
  await fs.writeFile(audioFile, "fake-wav");

  return {
    tmp,
    customLevels,
    mediaDir,
    tmpDownloadsRoot,
    queueFilePath,
    settingsStore,
    libraryStore,
    logger,
    trashed,
    trashItem,
    audioFile,
  };
}

describe("QueueManager", () => {
  it("runs queued items strictly one at a time in FIFO order and lands them in LibraryStore", async () => {
    const h = await createTestHarness();
    let inFlight = 0;
    let maxInFlight = 0;
    const runOrder: string[] = [];
    const libraryUpdates: LibraryEntry[][] = [];

    const engine = {
      request: vi.fn(async (cmd: string, params?: Record<string, unknown>) => {
        if (cmd === "generate") {
          inFlight++;
          maxInFlight = Math.max(maxInFlight, inFlight);
          const title = String(params?.title ?? "");
          runOrder.push(title);
          await new Promise((r) => setTimeout(r, 20));
          inFlight--;
          const out = makeGenerateOutput(h.customLevels, title);
          await fs.mkdir(out.folder, { recursive: true });
          return out;
        }
        throw new Error(`Unexpected cmd: ${cmd}`);
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
      onLibraryChanged: (entries) => libraryUpdates.push(entries),
    });
    await qm.init();

    const makeReq = (title: string): QueueEnqueueRequest => ({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title,
      artist: "Camellia",
      coverPath: null,
      config: { ...BASE_CONFIG, seed: 0 },
    });

    const i1 = await qm.enqueue(makeReq("One"));
    const i2 = await qm.enqueue(makeReq("Two"));
    const i3 = await qm.enqueue(makeReq("Three"));

    expect(i1.request.config.seed).toBeGreaterThan(0);

    await waitFor(() => qm.getItems().every((i) => i.status === "done"));

    expect(maxInFlight).toBe(1);
    expect(runOrder).toEqual(["One", "Two", "Three"]);
    const lib = await h.libraryStore.list();
    expect(lib).toHaveLength(3);
    expect(lib.map((e) => e.title).sort()).toEqual(["One", "Three", "Two"]);
    expect(libraryUpdates).toHaveLength(3);
    expect(qm.getItems().map((i) => [i.id, i.status])).toEqual([
      [i1.id, "done"],
      [i2.id, "done"],
      [i3.id, "done"],
    ]);
  });

  it("tracks YouTube download stage before running generation and cleans up tmp-downloads", async () => {
    const h = await createTestHarness();
    const statuses: string[] = [];
    const progressEvents: QueueProgressEvent[] = [];

    let qm!: QueueManager;
    const engine = {
      request: vi.fn(async (cmd: string, params?: Record<string, unknown>) => {
        if (cmd === "download") {
          const destDir = String(params?.dest_dir);
          await fs.mkdir(destDir, { recursive: true });
          const dlAudio = path.join(destDir, "yt.m4a");
          const dlCover = path.join(destDir, "thumb.jpg");
          await fs.writeFile(dlAudio, "m4a-data");
          await fs.writeFile(dlCover, "jpg-data");
          qm.handleProgress({ stage: "downloading", pct: 55 });
          return { audio_path: dlAudio, cover_path: dlCover };
        }
        if (cmd === "generate") {
          qm.handleProgress({ stage: "notes", pct: 40 });
          const out = makeGenerateOutput(h.customLevels, String(params?.title));
          await fs.mkdir(out.folder, { recursive: true });
          return out;
        }
        throw new Error(`Unexpected cmd: ${cmd}`);
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
      onChange: (items) => {
        if (items[0]) {
          statuses.push(items[0].status);
        }
      },
      onProgress: (ev) => progressEvents.push(ev),
    });
    await qm.init();

    const item = await qm.enqueue({
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtube.com/watch?v=abc",
      title: "GHOST",
      artist: "Camellia",
      coverPath: null,
      thumbnailUrl: "https://i.ytimg.com/vi/abc/hqdefault.jpg",
      config: BASE_CONFIG,
    });

    await waitFor(() => statuses.includes("done"));

    expect(statuses).toContain("waiting");
    expect(statuses).toContain("downloading");
    expect(statuses).toContain("running");
    expect(statuses[statuses.length - 1]).toBe("done");
    expect(progressEvents).toEqual([
      { id: item.id, stage: "downloading", pct: 55 },
      { id: item.id, stage: "notes", pct: 40 },
    ]);
    const leftovers = await fs.readdir(h.tmpDownloadsRoot).catch(() => []);
    expect(leftovers).toEqual([]);
  });

  it("replaces existing library entry on queued regenerate without deleting media on failure", async () => {
    const h = await createTestHarness();
    const oldFolder = path.join(h.customLevels, "BSM - OldMap");
    await fs.mkdir(oldFolder, { recursive: true });
    const cached = await h.libraryStore.cacheMedia("regen-1", h.audioFile, null);
    await h.libraryStore.saveEntry({
      id: "regen-1",
      title: "OldMap",
      artist: "Artist",
      folder: oldFolder,
      createdAt: "2026-11-01T10:00:00Z",
      bpm: 120,
      levels: [],
      nWalls: 0,
      nBombs: 0,
      nLights: 0,
      environment: "DefaultEnvironment",
      audioCachePath: cached.audioCachePath,
      coverCachePath: null,
      config: BASE_CONFIG,
    });

    let shouldFail = false;
    const engine = {
      request: vi.fn(async (cmd: string) => {
        if (cmd === "generate") {
          if (shouldFail) {
            throw new EngineError("internal", "boom");
          }
          const out = makeGenerateOutput(h.customLevels, "NewMap");
          await fs.mkdir(out.folder, { recursive: true });
          return out;
        }
        throw new Error(`Unexpected cmd: ${cmd}`);
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    await qm.enqueue({
      sourceType: "cached",
      sourcePathOrUrl: cached.audioCachePath,
      title: "NewMap",
      artist: "Artist",
      coverPath: null,
      config: BASE_CONFIG,
      regenerateId: "regen-1",
    });

    await waitFor(() => qm.getItems()[0]?.status === "done");
    expect(h.trashed).toEqual([oldFolder]);
    expect((await h.libraryStore.get("regen-1"))?.title).toBe("NewMap");

    shouldFail = true;
    await qm.enqueue({
      sourceType: "cached",
      sourcePathOrUrl: cached.audioCachePath,
      title: "FailRegen",
      artist: "Artist",
      coverPath: null,
      config: BASE_CONFIG,
      regenerateId: "regen-1",
    });

    await waitFor(() => qm.getItems()[1]?.status === "failed");
    expect(await fs.stat(cached.audioCachePath)).toBeTruthy();
  });

  it("cancels a waiting item without disturbing the active item", async () => {
    const h = await createTestHarness();
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((r) => {
      releaseFirst = r;
    });
    let firstStarted!: () => void;
    const firstStartedP = new Promise<void>((r) => {
      firstStarted = r;
    });

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const title = String(params?.title);
        if (title === "First") {
          firstStarted();
          await firstGate;
        }
        const out = makeGenerateOutput(h.customLevels, title);
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const i1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "First",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await firstStartedP;

    const i2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Second",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const i3 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Third",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await qm.cancel(i2.id);
    expect(qm.getItems().find((i) => i.id === i2.id)?.status).toBe("cancelled");
    expect(engine.cancelActive).not.toHaveBeenCalled();

    releaseFirst();
    await waitFor(() => qm.getItems().find((i) => i.id === i3.id)?.status === "done");

    expect(qm.getItems().find((i) => i.id === i1.id)?.status).toBe("done");
    expect(qm.getItems().find((i) => i.id === i2.id)?.status).toBe("cancelled");
    expect(qm.getItems().find((i) => i.id === i3.id)?.status).toBe("done");
  });

  it("cancels the active item (including between download and generate) and advances to the next", async () => {
    const h = await createTestHarness();
    let rejectActiveGen: ((err: unknown) => void) | null = null;
    let genStarted!: () => void;
    const genStartedP = new Promise<void>((r) => {
      genStarted = r;
    });

    const engine = {
      request: vi.fn(async (cmd: string, params?: Record<string, unknown>) => {
        if (cmd === "download") {
          const destDir = String(params?.dest_dir);
          await fs.mkdir(destDir, { recursive: true });
          const dlAudio = path.join(destDir, "yt.m4a");
          await fs.writeFile(dlAudio, "m4a");
          return { audio_path: dlAudio, cover_path: null };
        }
        if (cmd === "generate") {
          const title = String(params?.title);
          if (title === "CancelDuringGen") {
            genStarted();
            return await new Promise<GenerateOutput>((_res, rej) => {
              rejectActiveGen = rej;
            });
          }
          const out = makeGenerateOutput(h.customLevels, title);
          await fs.mkdir(out.folder, { recursive: true });
          return out;
        }
        throw new Error(`Unexpected cmd: ${cmd}`);
      }),
      cancelActive: vi.fn(async () => {
        rejectActiveGen?.(new EngineError("cancelled", "cancelled"));
      }),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    // Case 1: cancel during generate
    const i1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "CancelDuringGen",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const i2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "AfterCancel",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await genStartedP;
    await qm.cancel(i1.id);

    await waitFor(() => qm.getItems().find((i) => i.id === i2.id)?.status === "done");
    expect(qm.getItems().find((i) => i.id === i1.id)?.status).toBe("cancelled");
    expect(await fs.readdir(h.mediaDir)).toHaveLength(1); // only AfterCancel remains

    // Case 2: cancel during download -> cacheMedia gap (engine.activeHeavyJob is null)
    const origCacheMedia = h.libraryStore.cacheMedia.bind(h.libraryStore);
    let cancelTargetId = "";
    const cacheSpy = vi.spyOn(h.libraryStore, "cacheMedia").mockImplementationOnce(async (id, a, c) => {
      const res = await origCacheMedia(id, a, c);
      await qm.cancel(cancelTargetId);
      return res;
    });

    const i3 = await qm.enqueue({
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtube.com/watch?v=gap",
      title: "CancelInGap",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    cancelTargetId = i3.id;
    const i4 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "AfterGapCancel",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await waitFor(() => qm.getItems().find((i) => i.id === i4.id)?.status === "done");
    expect(cacheSpy).toHaveBeenCalledTimes(2);
    expect(engine.request).not.toHaveBeenCalledWith(
      "generate",
      expect.objectContaining({ title: "CancelInGap" }),
    );
    expect(qm.getItems().find((i) => i.id === i3.id)?.status).toBe("cancelled");
    expect(await fs.readdir(h.mediaDir)).toHaveLength(2); // AfterCancel + AfterGapCancel only
  });

  it("marks failed item and continues queue after engine crash mid-item", async () => {
    const h = await createTestHarness();
    const crashPidFile = path.join(h.tmp, "crash-pid.txt");
    const crashScript = path.join(h.tmp, "crash-engine.cjs");
    await fs.writeFile(
      crashScript,
      `
const fs = require("fs");
const path = require("path");
const rl = require("readline").createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.cmd === "generate") {
    if (msg.title === "CrashOne") {
      fs.writeFileSync(${JSON.stringify(crashPidFile)}, String(process.pid), "utf8");
      process.exit(1);
    }
    const folder = path.join(msg.out_dir, "BSM - " + msg.title);
    fs.mkdirSync(folder, { recursive: true });
    console.log(JSON.stringify({
      id: msg.id,
      type: "result",
      data: {
        folder,
        title: msg.title,
        artist: msg.artist,
        bpm: 128,
        grid_steady: true,
        n_notes: 100,
        nps: 3.0,
        rank: "Hard",
        levels: [{ rank: "Hard", target_nps: 3.0, n_notes: 100, nps: 3.0, n_arcs: 0, n_chains: 0 }],
        n_walls: 0,
        n_bombs: 0,
        n_lights: 0,
        environment: "DefaultEnvironment",
        warnings: [String(process.pid)],
        used_rule_lights: false,
        device_used: "cpu"
      }
    }));
  } else if (msg.cmd === "shutdown") {
    process.exit(0);
  }
});
`,
    );

    const engineMgr = new EngineManager({
      resolveLaunch: () => ({ command: process.execPath, args: [crashScript] }),
      cancelTimeoutMs: 500,
      onProgress: (ev) => qm.handleProgress(ev),
      onStderr: () => {},
    });

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine: engineMgr,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const i1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "CrashOne",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const i2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "SurviveTwo",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await waitFor(() => qm.getItems().find((i) => i.id === i2.id)?.status === "done");
    const items = qm.getItems();
    const first = items.find((i) => i.id === i1.id)!;
    const second = items.find((i) => i.id === i2.id)!;
    expect(first.status).toBe("failed");
    expect(first.errorCode).toBe("internal");
    expect(second.status).toBe("done");
    const pid1 = Number(await fs.readFile(crashPidFile, "utf8"));
    const pid2 = Number(second.result?.warnings[0]);
    expect(pid1).toBeGreaterThan(0);
    expect(pid2).toBeGreaterThan(0);
    expect(pid2).not.toBe(pid1);

    await engineMgr.stop();
  });

  it("retries a failed or cancelled item at the end of the waiting queue", async () => {
    const h = await createTestHarness();
    let failFirst = true;
    let releasePause!: () => void;
    let pauseOnTitle = "";
    const runOrder: string[] = [];

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const title = String(params?.title);
        if (title === pauseOnTitle) {
          await new Promise<void>((r) => {
            releasePause = r;
          });
        }
        runOrder.push(title);
        if (title === "Flaky" && failFirst) {
          failFirst = false;
          throw new EngineError("internal", "transient");
        }
        const out = makeGenerateOutput(h.customLevels, title);
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const flaky = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Flaky",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await waitFor(() => qm.getItems()[0]?.status === "failed");

    pauseOnTitle = "Blocker";
    const blocker = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Blocker",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await waitFor(() => qm.getItems().find((i) => i.id === blocker.id)?.status === "running");

    const cancelled = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Cancelled",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await qm.cancel(cancelled.id);
    expect(qm.getItems().find((i) => i.id === cancelled.id)?.status).toBe("cancelled");

    const second = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Second",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await qm.retry(flaky.id);
    await qm.retry(cancelled.id);
    const afterRetry = qm.getItems();
    expect(afterRetry.map((i) => i.id)).toEqual([blocker.id, second.id, flaky.id, cancelled.id]);
    const retriedFailed = afterRetry.find((i) => i.id === flaky.id)!;
    const retriedCancelled = afterRetry.find((i) => i.id === cancelled.id)!;
    expect(retriedFailed.status).toBe("waiting");
    expect(retriedFailed.errorCode).toBeNull();
    expect(retriedFailed.finishedAt).toBeNull();
    expect(retriedCancelled.status).toBe("waiting");
    expect(retriedCancelled.errorCode).toBeNull();
    expect(retriedCancelled.finishedAt).toBeNull();

    releasePause();
    await waitFor(() => qm.getItems().every((i) => i.status === "done"));
    expect(runOrder).toEqual(["Flaky", "Blocker", "Second", "Flaky", "Cancelled"]);
  });

  it("moves waiting items up and down without crossing active or finished items", async () => {
    const h = await createTestHarness();
    let releaseActive!: () => void;
    const activeGate = new Promise<void>((r) => {
      releaseActive = r;
    });
    let activeStarted!: () => void;
    const activeStartedP = new Promise<void>((r) => {
      activeStarted = r;
    });
    const runOrder: string[] = [];

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const title = String(params?.title);
        if (title === "Active") {
          activeStarted();
          await activeGate;
        }
        runOrder.push(title);
        const out = makeGenerateOutput(h.customLevels, title);
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const active = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Active",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await activeStartedP;

    const w1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "W1",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const w2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "W2",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const w3 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "W3",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await qm.move(w3.id, "up");
    expect(qm.getItems().map((i) => i.id)).toEqual([active.id, w1.id, w3.id, w2.id]);

    // Non-noop downward swap
    await qm.move(w1.id, "down");
    expect(qm.getItems().map((i) => i.id)).toEqual([active.id, w3.id, w1.id, w2.id]);

    // No-op boundary checks
    await qm.move(w3.id, "up");
    expect(qm.getItems().map((i) => i.id)).toEqual([active.id, w3.id, w1.id, w2.id]);
    await qm.move(w2.id, "down");
    expect(qm.getItems().map((i) => i.id)).toEqual([active.id, w3.id, w1.id, w2.id]);
    await qm.move(active.id, "down");
    expect(qm.getItems().map((i) => i.id)).toEqual([active.id, w3.id, w1.id, w2.id]);

    releaseActive();
    await waitFor(() => qm.getItems().every((i) => i.status === "done"));
    expect(runOrder).toEqual(["Active", "W3", "W1", "W2"]);
  });

  it("clears only finished items while preserving active and waiting", async () => {
    const h = await createTestHarness();
    let releaseRunning!: () => void;
    const runningGate = new Promise<void>((r) => {
      releaseRunning = r;
    });
    let runningStarted!: () => void;
    const runningStartedP = new Promise<void>((r) => {
      runningStarted = r;
    });

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const title = String(params?.title);
        if (title === "Running") {
          runningStarted();
          await runningGate;
        }
        if (title === "FailMe") {
          throw new EngineError("internal", "err");
        }
        const out = makeGenerateOutput(h.customLevels, title);
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "DoneItem",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "FailMe",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await waitFor(() => qm.getItems().length === 2 && qm.getItems()[1].status === "failed");

    const running = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Running",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await runningStartedP;

    const toCancel = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Cancelled",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const waiting = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Waiting",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await qm.cancel(toCancel.id);

    const remaining = await qm.clearFinished();
    expect(remaining.map((i) => [i.id, i.status])).toEqual([
      [running.id, "running"],
      [waiting.id, "waiting"],
    ]);

    releaseRunning();
    await waitFor(() => qm.getItems().every((i) => i.status === "done"));
  });

  it("persists queue to disk, converts interrupted running/downloading items to failed/interrupted on restart, and auto-resumes waiting", async () => {
    const h = await createTestHarness();
    const staleTmp = path.join(h.tmpDownloadsRoot, "orphan-dir");
    await fs.mkdir(staleTmp, { recursive: true });

    const persisted: QueueItem[] = [
      {
        id: "r1",
        status: "running",
        request: {
          sourceType: "file",
          sourcePathOrUrl: h.audioFile,
          title: "WasRunning",
          artist: "A",
          coverPath: null,
          config: BASE_CONFIG,
        },
        thumbnailUrl: "data:image/png;base64,AAA",
        enqueuedAt: "2026-11-24T10:00:00.000Z",
        startedAt: "2026-11-24T10:01:00.000Z",
        finishedAt: null,
        progress: { stage: "notes", pct: 45 },
        cancelling: true,
        errorCode: null,
        result: null,
        entry: null,
      },
      {
        id: "d1",
        status: "downloading",
        request: {
          sourceType: "youtube",
          sourcePathOrUrl: "https://youtube.com/watch?v=1",
          title: "WasDownloading",
          artist: "A",
          coverPath: null,
          config: BASE_CONFIG,
        },
        thumbnailUrl: "https://i.ytimg.com/vi/1/hqdefault.jpg",
        enqueuedAt: "2026-11-24T10:02:00.000Z",
        startedAt: "2026-11-24T10:02:10.000Z",
        finishedAt: null,
        progress: { stage: "downloading", pct: 20 },
        errorCode: null,
        result: null,
        entry: null,
      },
      {
        id: "w1",
        status: "waiting",
        request: {
          sourceType: "file",
          sourcePathOrUrl: h.audioFile,
          title: "ResumedWaiting",
          artist: "A",
          coverPath: null,
          config: BASE_CONFIG,
        },
        thumbnailUrl: "data:image/png;base64,BBB",
        enqueuedAt: "2026-11-24T10:03:00.000Z",
        startedAt: null,
        finishedAt: null,
        progress: null,
        errorCode: null,
        result: null,
        entry: null,
      },
      {
        id: "done1",
        status: "done",
        request: {
          sourceType: "file",
          sourcePathOrUrl: h.audioFile,
          title: "AlreadyDone",
          artist: "A",
          coverPath: null,
          config: BASE_CONFIG,
        },
        thumbnailUrl: null,
        enqueuedAt: "2026-11-24T09:00:00.000Z",
        startedAt: "2026-11-24T09:00:05.000Z",
        finishedAt: "2026-11-24T09:01:00.000Z",
        progress: { stage: "writing", pct: 100 },
        errorCode: null,
        result: null,
        entry: null,
      },
    ];
    await fs.writeFile(h.queueFilePath, JSON.stringify(persisted), "utf8");

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const out = makeGenerateOutput(h.customLevels, String(params?.title));
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    await waitFor(() => qm.getItems().find((i) => i.id === "w1")?.status === "done");

    // Stale tmp-downloads directory cleaned up on init
    expect(await fs.readdir(h.tmpDownloadsRoot)).toEqual([]);

    const items = qm.getItems();
    const r1 = items.find((i) => i.id === "r1")!;
    const d1 = items.find((i) => i.id === "d1")!;
    const done1 = items.find((i) => i.id === "done1")!;
    expect(r1.status).toBe("failed");
    expect(r1.errorCode).toBe("interrupted");
    expect(r1.progress).toBeNull();
    expect(r1.finishedAt).toBeTruthy();
    expect(d1.status).toBe("failed");
    expect(d1.errorCode).toBe("interrupted");
    expect(done1.status).toBe("done");

    // Verify data: thumbnailUrl and cancelling are stripped on disk
    const onDisk = JSON.parse(await fs.readFile(h.queueFilePath, "utf8")) as QueueItem[];
    expect(onDisk.find((i) => i.id === "r1")?.thumbnailUrl).toBeNull();
    expect(onDisk.find((i) => i.id === "r1")?.cancelling).toBeUndefined();
    expect(onDisk.find((i) => i.id === "d1")?.thumbnailUrl).toBe(
      "https://i.ytimg.com/vi/1/hqdefault.jpg",
    );
    expect(onDisk.find((i) => i.id === "w1")?.thumbnailUrl).toBeNull();

    // Corrupt queue.json is backed up safely
    const corruptDir = path.join(h.tmp, "corrupt-case");
    await fs.mkdir(corruptDir, { recursive: true });
    const corruptPath = path.join(corruptDir, "queue.json");
    await fs.writeFile(corruptPath, "{broken-json", "utf8");
    const qmCorrupt = new QueueManager({
      filePath: corruptPath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qmCorrupt.init();
    expect(qmCorrupt.getItems()).toEqual([]);
    const files = await fs.readdir(corruptDir);
    expect(files.some((f) => f.startsWith("queue.json.corrupt-"))).toBe(true);
  });

  it("shutdown marks active item interrupted and does not start subsequent waiting items", async () => {
    const h = await createTestHarness();
    let rejectGen: ((err: unknown) => void) | null = null;
    let genStarted!: () => void;
    const genStartedP = new Promise<void>((r) => {
      genStarted = r;
    });

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        if (String(params?.title) === "ActiveBeforeQuit") {
          genStarted();
          return await new Promise<GenerateOutput>((_res, rej) => {
            rejectGen = rej;
          });
        }
        throw new Error("Should not start waiting item after shutdown");
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {
        rejectGen?.(new EngineError("cancelled", "engine stopped"));
      }),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const i1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "ActiveBeforeQuit",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    const i2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "StillWaiting",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await genStartedP;
    await qm.shutdown();

    const onDisk = JSON.parse(await fs.readFile(h.queueFilePath, "utf8")) as QueueItem[];
    expect(onDisk.find((i) => i.id === i1.id)?.status).toBe("failed");
    expect(onDisk.find((i) => i.id === i1.id)?.errorCode).toBe("interrupted");
    expect(onDisk.find((i) => i.id === i2.id)?.status).toBe("waiting");
    expect(engine.request).toHaveBeenCalledTimes(1);
  });

  it("maps missing local audio ENOENT to not_found and rejects enqueue without customLevels", async () => {
    const hNoLevels = await createTestHarness({ customLevels: "" });
    const qmNoLevels = new QueueManager({
      filePath: hNoLevels.queueFilePath,
      tmpDownloadsRoot: hNoLevels.tmpDownloadsRoot,
      engine: { request: vi.fn(), cancelActive: vi.fn(), stop: vi.fn() },
      settingsStore: hNoLevels.settingsStore,
      libraryStore: hNoLevels.libraryStore,
      trashItem: hNoLevels.trashItem,
      logger: hNoLevels.logger,
    });
    await qmNoLevels.init();
    await expect(
      qmNoLevels.enqueue({
        sourceType: "file",
        sourcePathOrUrl: hNoLevels.audioFile,
        title: "MissingSettings",
        artist: "A",
        coverPath: null,
        config: BASE_CONFIG,
      }),
    ).rejects.toMatchObject({ code: "not_found" });

    const h = await createTestHarness();
    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine: { request: vi.fn(), cancelActive: vi.fn(), stop: vi.fn() },
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();
    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: path.join(h.tmp, "deleted-audio.wav"),
      title: "Deleted",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await waitFor(() => qm.getItems()[0]?.status === "failed");
    expect(qm.getItems()[0].errorCode).toBe("not_found");
  });

  it("caps finished history to maxFinishedHistory and skips auto-start when canAutoStart is false", async () => {
    const h = await createTestHarness();
    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const out = makeGenerateOutput(h.customLevels, String(params?.title));
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
      maxFinishedHistory: 2,
    });
    await qm.init();

    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Hist1",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Hist2",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "Hist3",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await waitFor(() => qm.getItems().length === 2 && qm.getItems().every((i) => i.status === "done"));
    expect(qm.getItems().map((i) => i.request.title)).toEqual(["Hist2", "Hist3"]);

    // Verify canAutoStart: false holds waiting items until kick()
    const holdPath = path.join(h.tmp, "hold-queue.json");
    await fs.writeFile(
      holdPath,
      JSON.stringify([
        {
          id: "hold-1",
          status: "waiting",
          request: {
            sourceType: "file",
            sourcePathOrUrl: h.audioFile,
            title: "Held",
            artist: "A",
            coverPath: null,
            config: BASE_CONFIG,
          },
          thumbnailUrl: null,
          enqueuedAt: "2026-11-24T10:00:00.000Z",
          startedAt: null,
          finishedAt: null,
          progress: null,
          errorCode: null,
          result: null,
          entry: null,
        },
      ]),
      "utf8",
    );

    const engineHold = {
      request: vi.fn(async () => makeGenerateOutput(h.customLevels, "Held")),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };
    const qmHold = new QueueManager({
      filePath: holdPath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine: engineHold,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qmHold.init({ canAutoStart: async () => false });
    await new Promise((r) => setTimeout(r, 40));
    expect(engineHold.request).not.toHaveBeenCalled();
    expect(qmHold.getItems()[0].status).toBe("waiting");
  });

  it("cancels an item cleanly while settingsStore.load() is pending", async () => {
    const h = await createTestHarness();
    let loadCalls = 0;
    let releaseSettingsLoad!: () => void;
    let settingsLoadStarted!: () => void;
    const settingsLoadStartedP = new Promise<void>((r) => {
      settingsLoadStarted = r;
    });
    const origLoad = h.settingsStore.load.bind(h.settingsStore);
    const settingsStore = {
      load: vi.fn(async () => {
        loadCalls++;
        // Call 1 is enqueue(); Call 2 is inside executeItem() for the first item.
        if (loadCalls === 2) {
          settingsLoadStarted();
          await new Promise<void>((r) => {
            releaseSettingsLoad = r;
          });
        }
        return origLoad();
      }),
    };

    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const out = makeGenerateOutput(h.customLevels, String(params?.title));
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
    });
    await qm.init();

    const enqueued = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "CancelWhileLoading",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    expect(enqueued.status).toBe("waiting");
    expect(enqueued.startedAt).toBeNull();

    await settingsLoadStartedP;
    expect(qm.getItems()[0].status).toBe("running");

    const second = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "NextAfterCancel",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    // second cannot swap above enqueued because enqueued is already running
    await qm.move(second.id, "up");
    expect(qm.getItems().map((i) => i.id)).toEqual([enqueued.id, second.id]);

    await qm.cancel(enqueued.id);
    expect(engine.cancelActive).toHaveBeenCalledTimes(1);

    releaseSettingsLoad();
    await waitFor(() => qm.getItems().find((i) => i.id === second.id)?.status === "done");

    const items = qm.getItems();
    expect(items.find((i) => i.id === enqueued.id)?.status).toBe("cancelled");
    expect(items.find((i) => i.id === second.id)?.status).toBe("done");
    expect(engine.request).toHaveBeenCalledTimes(1);
    expect(engine.request).toHaveBeenCalledWith(
      "generate",
      expect.objectContaining({ title: "NextAfterCancel" }),
    );
  });

  it("preserves saved entry and cached media if onLibraryChanged throws", async () => {
    const h = await createTestHarness();
    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const out = makeGenerateOutput(h.customLevels, String(params?.title));
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
      onLibraryChanged: () => {
        throw new Error("window destroyed");
      },
    });
    await qm.init();

    const item = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "SavedDespiteCallbackError",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    await waitFor(() => qm.getItems()[0]?.status === "done");
    const doneItem = qm.getItems().find((i) => i.id === item.id)!;
    expect(doneItem.status).toBe("done");
    expect(doneItem.entry).not.toBeNull();
    expect(await fs.stat(doneItem.entry!.audioCachePath)).toBeTruthy();
    expect((await h.libraryStore.list()).map((e) => e.id)).toContain(doneItem.entry!.id);
    expect(h.logger.warn).toHaveBeenCalled();
  });

  it("continues draining queue when persist or onChange throws in finally", async () => {
    const h = await createTestHarness();
    let releaseFirstGen!: () => void;
    let firstGenStarted!: () => void;
    const firstGenStartedP = new Promise<void>((r) => {
      firstGenStarted = r;
    });
    const engine = {
      request: vi.fn(async (_cmd: string, params?: Record<string, unknown>) => {
        const title = String(params?.title);
        if (title === "FirstWithFinallyErrors") {
          firstGenStarted();
          await new Promise<void>((r) => {
            releaseFirstGen = r;
          });
        }
        const out = makeGenerateOutput(h.customLevels, title);
        await fs.mkdir(out.folder, { recursive: true });
        return out;
      }),
      cancelActive: vi.fn(async () => {}),
      stop: vi.fn(async () => {}),
    };

    let throwOnNextDone = true;
    const origRename = fs.rename.bind(fs);
    let failNextQueueRename = false;
    const renameSpy = vi.spyOn(fs, "rename").mockImplementation(async (oldPath, newPath) => {
      if (failNextQueueRename && String(newPath) === h.queueFilePath) {
        failNextQueueRename = false;
        const err = new Error("EBUSY") as NodeJS.ErrnoException;
        err.code = "EBUSY";
        throw err;
      }
      return origRename(oldPath, newPath);
    });

    const qm = new QueueManager({
      filePath: h.queueFilePath,
      tmpDownloadsRoot: h.tmpDownloadsRoot,
      engine,
      settingsStore: h.settingsStore,
      libraryStore: h.libraryStore,
      trashItem: h.trashItem,
      logger: h.logger,
      onChange: (items) => {
        if (throwOnNextDone && items[0]?.status === "done") {
          throwOnNextDone = false;
          throw new Error("renderer gone");
        }
      },
    });
    await qm.init();

    const i1 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "FirstWithFinallyErrors",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });
    await firstGenStartedP;

    const i2 = await qm.enqueue({
      sourceType: "file",
      sourcePathOrUrl: h.audioFile,
      title: "SecondStillDrains",
      artist: "A",
      coverPath: null,
      config: BASE_CONFIG,
    });

    failNextQueueRename = true;
    releaseFirstGen();

    await waitFor(() => qm.getItems().find((i) => i.id === i2.id)?.status === "done");
    renameSpy.mockRestore();

    expect(qm.getItems().find((i) => i.id === i1.id)?.status).toBe("done");
    expect(qm.getItems().find((i) => i.id === i2.id)?.status).toBe("done");
    expect(h.logger.error).toHaveBeenCalledTimes(2);
  });
});
