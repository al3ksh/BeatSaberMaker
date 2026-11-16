import fs from "node:fs";
import path from "node:path";

const MAX_LOG_BYTES = 5 * 1024 * 1024;

export interface AppLogger {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
  engineStderr(line: string): void;
  getLogsDir(): string;
}

export function createLogger(logsDir: string): AppLogger {
  const logFile = path.join(logsDir, "app.log");
  const backupFile = path.join(logsDir, "app.log.1");
  let dirReady = false;

  function ensureDir(): void {
    if (!dirReady) {
      fs.mkdirSync(logsDir, { recursive: true });
      dirReady = true;
    }
  }

  function rotateIfNeeded(): void {
    try {
      const stat = fs.statSync(logFile);
      if (stat.size >= MAX_LOG_BYTES) {
        if (fs.existsSync(backupFile)) {
          fs.unlinkSync(backupFile);
        }
        fs.renameSync(logFile, backupFile);
      }
    } catch {
      // File may not exist yet.
    }
  }

  function writeLine(level: string, msg: string): void {
    try {
      ensureDir();
      rotateIfNeeded();
      const line = `[${new Date().toISOString()}] [${level}] ${msg}\n`;
      fs.appendFileSync(logFile, line, "utf8");
    } catch {
      // Logging must never crash the main process.
    }
  }

  return {
    info: (msg: string) => writeLine("INFO", msg),
    warn: (msg: string) => writeLine("WARN", msg),
    error: (msg: string) => writeLine("ERROR", msg),
    engineStderr: (line: string) => writeLine("ENGINE", line),
    getLogsDir: () => logsDir,
  };
}
