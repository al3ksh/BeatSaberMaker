import fs from "node:fs/promises";
import path from "node:path";
import type { Settings } from "./types";

export async function atomicWriteJson(filePath: string, data: unknown): Promise<void> {
  const dir = path.dirname(filePath);
  await fs.mkdir(dir, { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(tmpPath, JSON.stringify(data, null, 2), "utf8");
  try {
    await fs.rename(tmpPath, filePath);
  } catch (err) {
    await fs.rm(tmpPath, { force: true }).catch(() => {});
    throw err;
  }
}

export class SettingsStore {
  private readonly defaults: Settings;

  constructor(
    private readonly filePath: string,
    defaultExportDir: string,
  ) {
    this.defaults = {
      customLevels: "",
      device: "auto",
      exportDir: defaultExportDir,
      engineVersionInstalled: null,
    };
  }

  async load(): Promise<Settings> {
    try {
      const raw = await fs.readFile(this.filePath, "utf8");
      const parsed = JSON.parse(raw) as Partial<Settings>;
      return {
        customLevels: typeof parsed.customLevels === "string" ? parsed.customLevels : this.defaults.customLevels,
        device: parsed.device === "cuda" || parsed.device === "cpu" || parsed.device === "auto"
          ? parsed.device
          : this.defaults.device,
        exportDir: typeof parsed.exportDir === "string" && parsed.exportDir
          ? parsed.exportDir
          : this.defaults.exportDir,
        engineVersionInstalled:
          typeof parsed.engineVersionInstalled === "string" || parsed.engineVersionInstalled === null
            ? parsed.engineVersionInstalled
            : this.defaults.engineVersionInstalled,
      };
    } catch {
      return { ...this.defaults };
    }
  }

  async save(partial: Partial<Settings>): Promise<Settings> {
    const current = await this.load();
    const next: Settings = {
      ...current,
      ...partial,
    };
    await atomicWriteJson(this.filePath, next);
    return next;
  }
}
