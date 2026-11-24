import fs from "node:fs/promises";
import path from "node:path";
import { atomicWriteJson } from "./store";
import type { LibraryEntry } from "./types";

function normalizeForCompare(p: string): string {
  const resolved = path.resolve(p);
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

export function isSafeBsmMapFolder(folder: string, customLevelsDir: string): boolean {
  if (!folder || !customLevelsDir) {
    return false;
  }
  const resolvedFolder = path.resolve(folder);
  const resolvedCustom = path.resolve(customLevelsDir);
  const parent = path.dirname(resolvedFolder);
  if (normalizeForCompare(parent) !== normalizeForCompare(resolvedCustom)) {
    return false;
  }
  const base = path.basename(resolvedFolder);
  return base.startsWith("BSM - ");
}

async function trashIfSafe(
  folder: string,
  customLevelsDir: string,
  trashFn: (p: string) => Promise<void>,
): Promise<void> {
  if (!isSafeBsmMapFolder(folder, customLevelsDir)) {
    return;
  }
  const resolved = path.resolve(folder);
  try {
    const stat = await fs.lstat(resolved);
    if (stat.isSymbolicLink() || !stat.isDirectory()) {
      return;
    }
  } catch {
    return;
  }
  await trashFn(resolved);
}

export class LibraryStore {
  private writeChain: Promise<void> = Promise.resolve();

  constructor(
    private readonly filePath: string,
    private readonly mediaDir: string,
  ) {}

  private enqueueWrite<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.writeChain.then(fn, fn);
    this.writeChain = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async readAll(): Promise<LibraryEntry[]> {
    try {
      const raw = await fs.readFile(this.filePath, "utf8");
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as LibraryEntry[]) : [];
    } catch (err) {
      const code = (err as NodeJS.ErrnoException)?.code;
      if (code !== "ENOENT") {
        // Back up corrupt library.json instead of crashing or losing evidence.
        const backup = `${this.filePath}.corrupt-${Date.now()}`;
        await fs.copyFile(this.filePath, backup).catch(() => {});
      }
      return [];
    }
  }

  async list(): Promise<LibraryEntry[]> {
    const items = await this.readAll();
    return items.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  async get(id: string): Promise<LibraryEntry | null> {
    const items = await this.readAll();
    return items.find((item) => item.id === id) ?? null;
  }

  saveEntry(entry: LibraryEntry): Promise<LibraryEntry[]> {
    return this.enqueueWrite(async () => {
      const items = await this.readAll();
      const filtered = items.filter((item) => item.id !== entry.id);
      const next = [entry, ...filtered];
      await atomicWriteJson(this.filePath, next);
      return next;
    });
  }

  async cacheMedia(
    id: string,
    audioSrc: string,
    coverSrc: string | null,
  ): Promise<{ audioCachePath: string; coverCachePath: string | null }> {
    const entryDir = path.join(this.mediaDir, id);
    await fs.mkdir(entryDir, { recursive: true });

    const audioExt = path.extname(audioSrc) || ".bin";
    const audioDst = path.join(entryDir, `audio${audioExt}`);
    if (normalizeForCompare(audioSrc) !== normalizeForCompare(audioDst)) {
      await fs.copyFile(audioSrc, audioDst);
    }

    let coverDst: string | null = null;
    if (coverSrc) {
      try {
        const stat = await fs.stat(coverSrc);
        if (stat.isFile()) {
          const coverExt = path.extname(coverSrc) || ".jpg";
          coverDst = path.join(entryDir, `cover${coverExt}`);
          if (normalizeForCompare(coverSrc) !== normalizeForCompare(coverDst)) {
            await fs.copyFile(coverSrc, coverDst);
          }
        }
      } catch {
        coverDst = null;
      }
    }

    return { audioCachePath: audioDst, coverCachePath: coverDst };
  }

  deleteEntry(
    id: string,
    customLevelsDir: string,
    trashFn: (p: string) => Promise<void>,
  ): Promise<LibraryEntry[]> {
    return this.enqueueWrite(async () => {
      const items = await this.readAll();
      const target = items.find((item) => item.id === id);
      if (target) {
        await trashIfSafe(target.folder, customLevelsDir, trashFn);
        await fs.rm(path.join(this.mediaDir, id), { recursive: true, force: true }).catch(() => {});
      }
      const next = items.filter((item) => item.id !== id);
      await atomicWriteJson(this.filePath, next);
      return next.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    });
  }

  replaceAfterRegenerate(
    id: string,
    newEntry: LibraryEntry,
    customLevelsDir: string,
    trashFn: (p: string) => Promise<void>,
  ): Promise<LibraryEntry[]> {
    return this.enqueueWrite(async () => {
      const items = await this.readAll();
      const oldEntry = items.find((item) => item.id === id);
      if (!oldEntry) {
        // Target entry was deleted while regeneration was in flight; do not resurrect it.
        await trashIfSafe(newEntry.folder, customLevelsDir, trashFn);
        return items.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
      }
      if (normalizeForCompare(oldEntry.folder) !== normalizeForCompare(newEntry.folder)) {
        await trashIfSafe(oldEntry.folder, customLevelsDir, trashFn);
      }
      const next = [newEntry, ...items.filter((item) => item.id !== id)];
      await atomicWriteJson(this.filePath, next);
      return next.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    });
  }
}
