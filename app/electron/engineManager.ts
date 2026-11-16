import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import type { ProgressEvent } from "./types";

export class EngineError extends Error {
  constructor(
    public readonly code: string,
    public readonly detail = "",
  ) {
    super(detail ? `${code}: ${detail}` : code);
    this.name = "EngineError";
  }
}

export interface LaunchSpec {
  command: string;
  args: string[];
  cwd?: string;
}

export interface EngineManagerOptions {
  resolveLaunch: () => LaunchSpec | null;
  cancelTimeoutMs?: number;
  onProgress?: (ev: ProgressEvent) => void;
  onStderr?: (line: string) => void;
}

interface ActiveJob {
  id: string;
  resolve: (value: unknown) => void;
  reject: (err: unknown) => void;
  settled: boolean;
  promise: Promise<unknown>;
}

export function createDefaultResolveLaunch(userDataDir: string, logsDir: string): () => LaunchSpec | null {
  return () => {
    if (process.env.BSM_ENGINE_CMD) {
      let args: string[] = [];
      if (process.env.BSM_ENGINE_ARGS) {
        args = JSON.parse(process.env.BSM_ENGINE_ARGS) as string[];
      }
      return { command: process.env.BSM_ENGINE_CMD, args };
    }
    const engineDir = path.join(userDataDir, "engine");
    const engineExe = path.join(engineDir, "bsmaker-engine", "bsmaker-engine.exe");
    if (!fs.existsSync(engineExe)) {
      return null;
    }
    const args = [
      "--checkpoints-dir",
      path.join(engineDir, "models", "checkpoints"),
      "--playability",
      path.join(engineDir, "models", "playability.toml"),
      "--layer-stats",
      path.join(engineDir, "models", "layer_stats.json"),
      "--ffmpeg",
      path.join(engineDir, "tools", "ffmpeg.exe"),
      "--ytdlp",
      path.join(engineDir, "tools", "yt-dlp.exe"),
      "--log-file",
      path.join(logsDir, "engine.log"),
    ];
    return { command: engineExe, args, cwd: path.dirname(engineExe) };
  };
}

function killProcessTree(proc: ChildProcessWithoutNullStreams): void {
  if (process.platform === "win32" && proc.pid) {
    try {
      spawn("taskkill", ["/PID", String(proc.pid), "/T", "/F"], {
        windowsHide: true,
        stdio: "ignore",
      });
    } catch {
      try {
        proc.kill();
      } catch {
        // Ignore already-exited process.
      }
    }
    return;
  }
  try {
    proc.kill("SIGKILL");
  } catch {
    // Ignore already-exited process.
  }
}

export class EngineManager {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private rl: readline.Interface | null = null;
  private nextId = 1;
  private activeJob: ActiveJob | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private readonly cancelTimeoutMs: number;

  constructor(private readonly options: EngineManagerOptions) {
    this.cancelTimeoutMs = options.cancelTimeoutMs ?? 5000;
  }

  private ensureRunning(): ChildProcessWithoutNullStreams {
    if (this.proc && !this.proc.killed && this.proc.exitCode === null) {
      return this.proc;
    }
    const spec = this.options.resolveLaunch();
    if (!spec) {
      throw new EngineError("no_models", "engine binary not found");
    }
    const proc = spawn(spec.command, spec.args, {
      cwd: spec.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      env: { ...process.env, PYTHONUTF8: "1" },
    });
    this.proc = proc;

    const rl = readline.createInterface({ input: proc.stdout, crlfDelay: Infinity });
    this.rl = rl;
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(trimmed) as Record<string, unknown>;
      } catch {
        this.options.onStderr?.(`Invalid engine JSON: ${trimmed}`);
        return;
      }
      const id = String(msg.id ?? "");
      const type = String(msg.type ?? "");
      if (type === "progress") {
        this.options.onProgress?.({
          stage: msg.stage as ProgressEvent["stage"],
          pct: Number(msg.pct ?? 0),
        });
        return;
      }
      if (this.activeJob && this.activeJob.id === id && !this.activeJob.settled) {
        const job = this.activeJob;
        job.settled = true;
        this.activeJob = null;
        if (type === "result") {
          job.resolve(msg.data);
        } else if (type === "error") {
          job.reject(new EngineError(String(msg.code ?? "internal"), String(msg.detail ?? "")));
        } else {
          job.reject(new EngineError("internal", `unexpected message type: ${type}`));
        }
      }
    });

    const errRl = readline.createInterface({ input: proc.stderr, crlfDelay: Infinity });
    errRl.on("line", (line) => {
      if (line.trim()) {
        this.options.onStderr?.(line);
      }
    });

    proc.on("exit", (code) => {
      errRl.close();
      rl.close();
      if (this.proc === proc) {
        this.proc = null;
        this.rl = null;
      }
      if (this.activeJob && !this.activeJob.settled) {
        const job = this.activeJob;
        job.settled = true;
        this.activeJob = null;
        job.reject(new EngineError("internal", `engine exited unexpectedly (code ${code ?? "null"})`));
      }
    });

    proc.on("error", (err) => {
      this.options.onStderr?.(`Engine process error: ${err.message}`);
      if (this.activeJob && !this.activeJob.settled) {
        const job = this.activeJob;
        job.settled = true;
        this.activeJob = null;
        job.reject(new EngineError("internal", err.message));
      }
    });

    return proc;
  }

  request<T = unknown>(cmd: string, params: Record<string, unknown> = {}): Promise<T> {
    const run = async (): Promise<T> => {
      const proc = this.ensureRunning();
      const id = String(this.nextId++);
      let resolveFn!: (v: unknown) => void;
      let rejectFn!: (e: unknown) => void;
      const promise = new Promise<unknown>((resolve, reject) => {
        resolveFn = resolve;
        rejectFn = reject;
      });
      const job: ActiveJob = {
        id,
        resolve: resolveFn,
        reject: rejectFn,
        settled: false,
        promise,
      };
      this.activeJob = job;
      try {
        proc.stdin.write(`${JSON.stringify({ id, cmd, ...params })}\n`, "utf8");
      } catch (err) {
        if (!job.settled) {
          job.settled = true;
          this.activeJob = null;
          job.reject(new EngineError("internal", (err as Error).message));
        }
      }
      return (await promise) as T;
    };

    const result = this.queueTail.then(run, run);
    this.queueTail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  async cancelActive(): Promise<void> {
    const proc = this.proc;
    const job = this.activeJob;
    if (!proc || !job || job.settled) {
      return;
    }

    const cancelId = String(this.nextId++);
    try {
      proc.stdin.write(`${JSON.stringify({ id: cancelId, cmd: "cancel" })}\n`, "utf8");
    } catch {
      // Ignore broken pipe; timeout/kill handles cleanup.
    }

    let timer: NodeJS.Timeout | null = null;
    const timeoutPromise = new Promise<void>((resolve) => {
      timer = setTimeout(() => {
        if (!job.settled) {
          job.settled = true;
          if (this.activeJob === job) {
            this.activeJob = null;
          }
          if (this.proc === proc) {
            this.proc = null;
            this.rl?.close();
            this.rl = null;
          }
          killProcessTree(proc);
          job.reject(new EngineError("cancelled", "cancelled after timeout"));
        }
        resolve();
      }, this.cancelTimeoutMs);
    });

    await Promise.race([
      job.promise.then(
        () => undefined,
        () => undefined,
      ),
      timeoutPromise,
    ]);
    if (timer) {
      clearTimeout(timer);
    }
  }

  async stop(): Promise<void> {
    const proc = this.proc;
    if (!proc) {
      return;
    }
    this.proc = null;
    this.rl?.close();
    this.rl = null;

    if (this.activeJob && !this.activeJob.settled) {
      const job = this.activeJob;
      job.settled = true;
      this.activeJob = null;
      job.reject(new EngineError("cancelled", "engine stopped"));
    }

    const stopId = String(this.nextId++);
    const exited = new Promise<void>((resolve) => {
      proc.once("exit", () => resolve());
    });
    try {
      proc.stdin.write(`${JSON.stringify({ id: stopId, cmd: "shutdown" })}\n`, "utf8");
      proc.stdin.end();
    } catch {
      // Ignore broken pipe.
    }

    let timer: NodeJS.Timeout | null = null;
    await Promise.race([
      exited,
      new Promise<void>((resolve) => {
        timer = setTimeout(() => {
          killProcessTree(proc);
          resolve();
        }, 2000);
      }),
    ]);
    if (timer) {
      clearTimeout(timer);
    }
  }
}
