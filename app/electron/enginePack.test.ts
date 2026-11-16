import crypto from "node:crypto";
import fs from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EnginePackInstaller } from "./enginePack";

function sha256(buf: Buffer): string {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makeStoreZip(files: Record<string, string>): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const [name, content] of Object.entries(files)) {
    const nb = Buffer.from(name, "utf8");
    const data = Buffer.from(content, "utf8");
    const crc = crc32(data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4);
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(data.length, 18);
    lh.writeUInt32LE(data.length, 22);
    lh.writeUInt16LE(nb.length, 26);
    locals.push(lh, nb, data);
    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0);
    ch.writeUInt16LE(20, 4);
    ch.writeUInt16LE(20, 6);
    ch.writeUInt32LE(crc, 16);
    ch.writeUInt32LE(data.length, 20);
    ch.writeUInt32LE(data.length, 24);
    ch.writeUInt16LE(nb.length, 28);
    ch.writeUInt32LE(offset, 42);
    centrals.push(ch, nb);
    offset += lh.length + nb.length + data.length;
  }
  const cd = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(Object.keys(files).length, 8);
  eocd.writeUInt16LE(Object.keys(files).length, 10);
  eocd.writeUInt32LE(cd.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, eocd]);
}

describe("EnginePackInstaller", () => {
  it("downloads split parts, resumes via Range, verifies SHA256, and falls back on 200", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-pack-"));
    const zip = makeStoreZip({
      "bsmaker-engine/bsmaker-engine.exe": "EXE",
      "models/timing/best.pt": "PT",
    });
    const split = Math.floor(zip.length / 2);
    const parts = [zip.subarray(0, split), zip.subarray(split)];
    const manifest = {
      engineVersion: "1",
      archiveSha256: sha256(zip),
      totalSize: zip.length,
      parts: parts.map((p, i) => ({ name: `engine.zip.00${i + 1}`, size: p.length, sha256: sha256(p) })),
    };

    let mode: "range" | "no_range" | "corrupt" = "range";
    const rangeHeaders: string[] = [];
    const srv = http.createServer((req, res) => {
      if (req.url === "/manifest.json") {
        res.end(JSON.stringify(manifest));
        return;
      }
      const idx = req.url?.endsWith(".001") ? 0 : 1;
      const body = mode === "corrupt" && idx === 0 ? Buffer.alloc(parts[0].length, 0xff) : parts[idx];
      const rng = req.headers.range;
      if (rng) rangeHeaders.push(rng);
      if (mode === "range" && rng?.startsWith("bytes=")) {
        const start = Number(rng.slice(6).replace("-", ""));
        res.writeHead(206, { "Content-Range": `bytes ${start}-${body.length - 1}/${body.length}` });
        res.end(body.subarray(start));
      } else {
        res.writeHead(200, { "Content-Length": body.length });
        res.end(body);
      }
    });
    await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
    const port = (srv.address() as { port: number }).port;

    try {
      const dlDir = path.join(tmp, "dl");
      const engDir = path.join(tmp, "engine");
      await fs.mkdir(dlDir, { recursive: true });
      await fs.writeFile(path.join(dlDir, `${manifest.parts[0].name}.part`), parts[0].subarray(0, 20));

      const installer = new EnginePackInstaller({
        engineDir: engDir,
        downloadDir: dlDir,
        expectedVersion: "1",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      expect(await installer.needsInstall()).toBe(true);
      await installer.install(() => {});
      expect(rangeHeaders).toContain("bytes=20-");
      expect(await installer.needsInstall()).toBe(false);
      expect(await fs.readFile(path.join(engDir, "bsmaker-engine", "bsmaker-engine.exe"), "utf8")).toBe("EXE");

      // Corrupt download must reject and preserve existing engineDir
      mode = "corrupt";
      const installerV2 = new EnginePackInstaller({
        engineDir: engDir,
        downloadDir: path.join(tmp, "dl2"),
        expectedVersion: "2",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      await expect(installerV2.install(() => {})).rejects.toThrow(/SHA256/i);
      expect(await fs.readFile(path.join(engDir, "bsmaker-engine", "bsmaker-engine.exe"), "utf8")).toBe("EXE");

      // Server ignoring Range (200 OK) resets partial file cleanly
      mode = "no_range";
      const dl3 = path.join(tmp, "dl3");
      await fs.mkdir(dl3, { recursive: true });
      await fs.writeFile(path.join(dl3, `${manifest.parts[0].name}.part`), parts[0].subarray(0, 10));
      const installer3 = new EnginePackInstaller({
        engineDir: path.join(tmp, "engine3"),
        downloadDir: dl3,
        expectedVersion: "1",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      await installer3.install(() => {});
      expect(await installer3.needsInstall()).toBe(false);
    } finally {
      srv.close();
    }
  });
});
