import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EngineManager } from "./engineManager";

const FAKE_SCRIPT = `
const rl = require("readline").createInterface({ input: process.stdin });
let timer = null;
let pendingGenId = null;
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.cmd === "info") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { pid: process.pid, pyutf8: process.env.PYTHONUTF8 } }));
  } else if (msg.cmd === "search") {
    if (msg.crash) {
      process.exit(1);
    } else if (!msg.stayPending) {
      console.log(JSON.stringify({ id: msg.id, type: "result", data: { items: [{ id: "yt1", query: msg.query }] } }));
    }
  } else if (msg.cmd === "resolve") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { id: "yt1", url: msg.url } }));
  } else if (msg.cmd === "export_zip") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { out_zip: msg.out_zip } }));
  } else if (msg.cmd === "generate") {
    console.log(JSON.stringify({ id: msg.id, type: "progress", stage: "notes", pct: 40 }));
    if (msg.hangOnCancel) {
      timer = setTimeout(() => {}, 60000);
    } else if (msg.cancelCleanly) {
      pendingGenId = msg.id;
    } else {
      console.log(JSON.stringify({ id: msg.id, type: "result", data: { folder: "BSM - OK" } }));
    }
  } else if (msg.cmd === "cancel") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { cancelled: true } }));
    if (pendingGenId !== null) {
      const id = pendingGenId;
      pendingGenId = null;
      console.log(JSON.stringify({ id, type: "error", code: "cancelled", detail: "cancelled" }));
    }
  } else if (msg.cmd === "shutdown") {
    process.exit(0);
  }
});
`;

describe("EngineManager", () => {
  it("streams progress, returns result, and force-restarts on hung cancel", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-em-"));
    const script = path.join(tmp, "fake.cjs");
    await fs.writeFile(script, FAKE_SCRIPT);

    const events: Array<{ stage: string; pct: number }> = [];
    const mgr = new EngineManager({
      resolveLaunch: () => ({ command: process.execPath, args: [script] }),
      cancelTimeoutMs: 150,
      onProgress: (ev) => events.push({ stage: ev.stage, pct: ev.pct }),
      onStderr: () => {},
    });

    const info1 = await mgr.request<{ pid: number; pyutf8?: string }>("info", {});
    expect(info1.pyutf8).toBe("1");
    const genOk = await mgr.request<{ folder: string }>("generate", {});
    expect(genOk.folder).toBe("BSM - OK");
    expect(events).toEqual([{ stage: "notes", pct: 40 }]);

    const hanging = mgr.request("generate", { hangOnCancel: true });
    await new Promise((r) => setTimeout(r, 40));
    await mgr.cancelActive();
    // Process tree must already have exited before cancelActive() resolves.
    expect(() => process.kill(info1.pid, 0)).toThrow();
    await expect(hanging).rejects.toMatchObject({ code: "cancelled" });

    // Transparently restarted with a fresh process
    const info2 = await mgr.request<{ pid: number }>("info", {});
    expect(info2.pid).not.toBe(info1.pid);
    await mgr.stop();
  });

  it("runs search/resolve/export_zip concurrently while generate is active and cancels only generate", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-em-"));
    const script = path.join(tmp, "fake.cjs");
    await fs.writeFile(script, FAKE_SCRIPT);

    let genStartedResolve!: () => void;
    const genStarted = new Promise<void>((r) => {
      genStartedResolve = r;
    });

    const mgr = new EngineManager({
      resolveLaunch: () => ({ command: process.execPath, args: [script] }),
      cancelTimeoutMs: 500,
      onProgress: () => genStartedResolve(),
      onStderr: () => {},
    });

    const info1 = await mgr.request<{ pid: number }>("info", {});

    let genSettled = false;
    const genPromise = mgr.request("generate", { cancelCleanly: true }).finally(() => {
      genSettled = true;
    });
    await genStarted;

    const searchRes = await mgr.request<{ items: Array<{ id: string; query: string }> }>("search", { query: "camellia" });
    const resolveRes = await mgr.request<{ id: string; url: string }>("resolve", { url: "https://youtu.be/abc" });
    const zipRes = await mgr.request<{ out_zip: string }>("export_zip", { folder: "C:/map", out_zip: "C:/out.zip" });

    expect(searchRes.items).toEqual([{ id: "yt1", query: "camellia" }]);
    expect(resolveRes).toEqual({ id: "yt1", url: "https://youtu.be/abc" });
    expect(zipRes).toEqual({ out_zip: "C:/out.zip" });
    expect(genSettled).toBe(false);

    await expect(mgr.cancelActive().then(() => genPromise)).rejects.toMatchObject({ code: "cancelled" });

    // Clean cancel keeps the same process alive
    const info2 = await mgr.request<{ pid: number }>("info", {});
    expect(info2.pid).toBe(info1.pid);
    await mgr.stop();
  });

  it("rejects all pending heavy and fast jobs when engine crashes", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-em-"));
    const script = path.join(tmp, "fake.cjs");
    await fs.writeFile(script, FAKE_SCRIPT);

    let genStartedResolve!: () => void;
    const genStarted = new Promise<void>((r) => {
      genStartedResolve = r;
    });

    const mgr = new EngineManager({
      resolveLaunch: () => ({ command: process.execPath, args: [script] }),
      cancelTimeoutMs: 500,
      onProgress: () => genStartedResolve(),
      onStderr: () => {},
    });

    const genPromise = mgr.request("generate", { cancelCleanly: true });
    await genStarted;

    const pendingSearch = mgr.request("search", { stayPending: true });
    const crashTrigger = mgr.request("search", { crash: true });

    await expect(genPromise).rejects.toMatchObject({ code: "internal" });
    await expect(pendingSearch).rejects.toMatchObject({ code: "internal" });
    await expect(crashTrigger).rejects.toMatchObject({ code: "internal" });

    // Respawns cleanly for next request
    const info = await mgr.request<{ pid: number }>("info", {});
    expect(info.pid).toBeGreaterThan(0);
    await mgr.stop();
  });
});
