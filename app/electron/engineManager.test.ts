import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EngineManager } from "./engineManager";

const FAKE_SCRIPT = `
const rl = require("readline").createInterface({ input: process.stdin });
let timer = null;
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.cmd === "info") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { pid: process.pid, pyutf8: process.env.PYTHONUTF8 } }));
  } else if (msg.cmd === "generate") {
    console.log(JSON.stringify({ id: msg.id, type: "progress", stage: "notes", pct: 40 }));
    if (msg.hangOnCancel) {
      timer = setTimeout(() => {}, 60000);
    } else {
      console.log(JSON.stringify({ id: msg.id, type: "result", data: { folder: "BSM - OK" } }));
    }
  } else if (msg.cmd === "cancel") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { cancelled: true } }));
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
    await expect(mgr.cancelActive().then(() => hanging)).rejects.toMatchObject({ code: "cancelled" });

    // Transparently restarted with a fresh process
    const info2 = await mgr.request<{ pid: number }>("info", {});
    expect(info2.pid).not.toBe(info1.pid);
    await mgr.stop();
  });
});
