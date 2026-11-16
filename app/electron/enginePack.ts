import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import extract from "extract-zip";
import type { EnginePackProgress } from "./types";

export interface EnginePackPart {
  name: string;
  size: number;
  sha256: string;
}

export interface EnginePackManifest {
  engineVersion: string;
  archiveSha256: string;
  totalSize: number;
  parts: EnginePackPart[];
}

export interface EnginePackInstallerOptions {
  engineDir: string;
  downloadDir: string;
  expectedVersion: string;
  manifestUrl: string;
}

export function resolveExpectedEngineVersion(appPath: string): string {
  const candidates = [
    process.resourcesPath ? path.join(process.resourcesPath, "engine-version.json") : "",
    path.join(appPath, "engine-version.json"),
    path.resolve(__dirname, "..", "engine-version.json"),
  ].filter(Boolean);
  for (const cand of candidates) {
    try {
      const parsed = JSON.parse(fs.readFileSync(cand, "utf8")) as { engineVersion?: string };
      if (typeof parsed.engineVersion === "string" && parsed.engineVersion) {
        return parsed.engineVersion;
      }
    } catch {
      // Try next candidate.
    }
  }
  return "1";
}

export function defaultManifestUrl(appVersion: string): string {
  if (process.env.BSM_ENGINE_MANIFEST_URL) {
    return process.env.BSM_ENGINE_MANIFEST_URL;
  }
  const tag = appVersion.startsWith("v") ? appVersion : `v${appVersion}`;
  return `https://github.com/al3ksh/BeatSaberMaker/releases/download/${tag}/engine-pack-manifest.json`;
}

async function fileSha256(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
    stream.on("error", reject);
  });
}

async function fileSizeIfExists(filePath: string): Promise<number> {
  try {
    const st = await fsp.stat(filePath);
    return st.isFile() ? st.size : 0;
  } catch {
    return 0;
  }
}

export class EnginePackInstaller {
  constructor(private readonly opts: EnginePackInstallerOptions) {}

  async getInstalledVersion(): Promise<string | null> {
    const versionFile = path.join(this.opts.engineDir, "installed-version.json");
    try {
      const parsed = JSON.parse(await fsp.readFile(versionFile, "utf8")) as {
        engineVersion?: string;
      };
      return typeof parsed.engineVersion === "string" ? parsed.engineVersion : null;
    } catch {
      return null;
    }
  }

  async needsInstall(): Promise<boolean> {
    if (process.env.BSM_SKIP_ENGINE_PACK === "1" || process.env.BSM_ENGINE_CMD) {
      return false;
    }
    const installed = await this.getInstalledVersion();
    if (installed !== this.opts.expectedVersion) {
      return true;
    }
    const exePath = path.join(this.opts.engineDir, "bsmaker-engine", "bsmaker-engine.exe");
    return !fs.existsSync(exePath);
  }

  async install(onProgress: (p: EnginePackProgress) => void): Promise<void> {
    const res = await fetch(this.opts.manifestUrl);
    if (!res.ok) {
      throw new Error(`Failed to fetch engine manifest (HTTP ${res.status})`);
    }
    const manifest = (await res.json()) as EnginePackManifest;
    if (!manifest || !Array.isArray(manifest.parts) || manifest.parts.length === 0) {
      throw new Error("Invalid engine-pack manifest");
    }

    await fsp.mkdir(this.opts.downloadDir, { recursive: true });
    const totalBytes = manifest.totalSize || manifest.parts.reduce((acc, p) => acc + p.size, 0);
    const partCount = manifest.parts.length;
    let completedBytes = 0;

    for (let i = 0; i < manifest.parts.length; i++) {
      const part = manifest.parts[i];
      const finalPath = path.join(this.opts.downloadDir, part.name);
      const tmpPath = `${finalPath}.part`;

      const existingFinalSize = await fileSizeIfExists(finalPath);
      if (existingFinalSize === part.size) {
        const existingHash = await fileSha256(finalPath);
        if (existingHash.toLowerCase() === part.sha256.toLowerCase()) {
          completedBytes += part.size;
          onProgress({
            stage: "downloading",
            pct: Math.min(85, Math.round((completedBytes / Math.max(totalBytes, 1)) * 85)),
            bytesDownloaded: completedBytes,
            totalBytes,
            partIndex: i + 1,
            partCount,
          });
          continue;
        }
        await fsp.rm(finalPath, { force: true });
      }

      let resumeFrom = await fileSizeIfExists(tmpPath);
      if (resumeFrom >= part.size) {
        await fsp.rm(tmpPath, { force: true });
        resumeFrom = 0;
      }

      const headers: Record<string, string> = {};
      if (resumeFrom > 0) {
        headers.Range = `bytes=${resumeFrom}-`;
      }

      const partUrl = new URL(part.name, this.opts.manifestUrl).toString();
      const partRes = await fetch(partUrl, { headers });
      if (!partRes.ok && partRes.status !== 206) {
        throw new Error(`Failed to download ${part.name} (HTTP ${partRes.status})`);
      }

      const isResumed = partRes.status === 206 && resumeFrom > 0;
      let partBytes = isResumed ? resumeFrom : 0;

      const writeStream = fs.createWriteStream(tmpPath, {
        flags: isResumed ? "a" : "w",
      });

      try {
        if (!partRes.body) {
          throw new Error(`Empty response body for ${part.name}`);
        }
        const reader = partRes.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value && value.byteLength > 0) {
            await new Promise<void>((resolve, reject) => {
              const ok = writeStream.write(Buffer.from(value), (err) => {
                if (err) reject(err);
                else if (!ok) {
                  // Wait for drain handled below.
                }
              });
              if (ok) resolve();
              else writeStream.once("drain", resolve);
            });
            partBytes += value.byteLength;
            const totalDownloaded = completedBytes + partBytes;
            onProgress({
              stage: "downloading",
              pct: Math.min(85, Math.round((totalDownloaded / Math.max(totalBytes, 1)) * 85)),
              bytesDownloaded: totalDownloaded,
              totalBytes,
              partIndex: i + 1,
              partCount,
            });
          }
        }
        await new Promise<void>((resolve, reject) => {
          writeStream.end((err?: Error | null) => (err ? reject(err) : resolve()));
        });
      } catch (err) {
        writeStream.destroy();
        throw err;
      }

      const gotPartSha = await fileSha256(tmpPath);
      if (gotPartSha.toLowerCase() !== part.sha256.toLowerCase()) {
        await fsp.rm(tmpPath, { force: true });
        throw new Error(`SHA256 mismatch for part ${part.name}`);
      }

      await fsp.rename(tmpPath, finalPath);
      completedBytes += part.size;
    }

    onProgress({
      stage: "verifying",
      pct: 88,
      bytesDownloaded: totalBytes,
      totalBytes,
      partIndex: partCount,
      partCount,
    });

    const archivePath = path.join(this.opts.downloadDir, "engine-pack.zip");
    const archiveHash = crypto.createHash("sha256");
    const outStream = fs.createWriteStream(archivePath, { flags: "w" });
    try {
      for (const part of manifest.parts) {
        const partPath = path.join(this.opts.downloadDir, part.name);
        await new Promise<void>((resolve, reject) => {
          const rs = fs.createReadStream(partPath);
          rs.on("data", (chunk) => {
            archiveHash.update(chunk);
            if (!outStream.write(chunk)) {
              rs.pause();
              outStream.once("drain", () => rs.resume());
            }
          });
          rs.on("end", resolve);
          rs.on("error", reject);
        });
      }
      await new Promise<void>((resolve, reject) => {
        outStream.end((err?: Error | null) => (err ? reject(err) : resolve()));
      });
    } catch (err) {
      outStream.destroy();
      await fsp.rm(archivePath, { force: true }).catch(() => {});
      throw err;
    }

    const gotArchiveSha = archiveHash.digest("hex");
    if (gotArchiveSha.toLowerCase() !== manifest.archiveSha256.toLowerCase()) {
      await fsp.rm(archivePath, { force: true }).catch(() => {});
      throw new Error("SHA256 mismatch for combined engine-pack.zip");
    }

    onProgress({
      stage: "extracting",
      pct: 92,
      bytesDownloaded: totalBytes,
      totalBytes,
      partIndex: partCount,
      partCount,
    });

    const tmpEngineDir = `${this.opts.engineDir}.tmp-${Date.now()}`;
    await fsp.rm(tmpEngineDir, { recursive: true, force: true });
    await fsp.mkdir(tmpEngineDir, { recursive: true });

    try {
      await extract(archivePath, { dir: path.resolve(tmpEngineDir) });
      await fsp.writeFile(
        path.join(tmpEngineDir, "installed-version.json"),
        JSON.stringify({ engineVersion: this.opts.expectedVersion }, null, 2),
        "utf8",
      );

      const backupDir = `${this.opts.engineDir}.bak-${Date.now()}`;
      const hadExisting = fs.existsSync(this.opts.engineDir);
      if (hadExisting) {
        await fsp.rename(this.opts.engineDir, backupDir);
      }
      try {
        await fsp.rename(tmpEngineDir, this.opts.engineDir);
      } catch (swapErr) {
        if (hadExisting && fs.existsSync(backupDir)) {
          await fsp.rename(backupDir, this.opts.engineDir).catch(() => {});
        }
        throw swapErr;
      }
      if (hadExisting) {
        await fsp.rm(backupDir, { recursive: true, force: true }).catch(() => {});
      }
    } catch (err) {
      await fsp.rm(tmpEngineDir, { recursive: true, force: true }).catch(() => {});
      throw err;
    }

    await fsp.rm(archivePath, { force: true }).catch(() => {});
    for (const part of manifest.parts) {
      await fsp.rm(path.join(this.opts.downloadDir, part.name), { force: true }).catch(() => {});
    }

    onProgress({
      stage: "extracting",
      pct: 100,
      bytesDownloaded: totalBytes,
      totalBytes,
      partIndex: partCount,
      partCount,
    });
  }
}
