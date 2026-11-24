import { describe, expect, it } from "vitest";
import { EN, ENVIRONMENT_LABELS_EN, ERRORS_EN, formatErrorEn } from "./en";

const EXPECTED_CODES = [
  "ffmpeg_missing",
  "ytdlp_missing",
  "download_failed",
  "no_models",
  "cancelled",
  "audio_too_short",
  "audio_silent",
  "invalid_audio",
  "file_exists",
  "not_found",
  "invalid_args",
  "busy",
  "interrupted",
  "internal",
];

describe("English i18n dictionary", () => {
  it("maps every engine error code to non-empty English text without traceback jargon", () => {
    for (const code of EXPECTED_CODES) {
      const msg = ERRORS_EN[code];
      expect(msg, code).toBeTruthy();
      expect(msg).not.toMatch(/traceback|exception|stack/i);
    }
    expect(formatErrorEn("unknown_future_code")).toBe(ERRORS_EN.internal);
  });

  it("includes labels for all supported environments, 2-step wizard, and queue UI", () => {
    expect(Object.keys(ENVIRONMENT_LABELS_EN)).toHaveLength(10);
    expect(EN.wizard.steps.song).toBe("Song");
    expect(EN.wizard.steps.map).toBe("Map");
    expect(EN.nav.queue).toBe("Queue");
    expect(EN.wizard.map.addToQueue).toMatch(/add to queue/i);
    expect(EN.wizard.map.queueRegenerate).toMatch(/queue regeneration/i);
    expect(EN.queue.sections.nowRunning).toBeTruthy();
    expect(EN.queue.sections.upNext).toBeTruthy();
    expect(EN.queue.sections.finished).toBeTruthy();
    expect(EN.queue.status.waiting).toBeTruthy();
    expect(EN.queue.status.downloading).toBeTruthy();
    expect(EN.queue.status.running).toBeTruthy();
    expect(EN.queue.status.done).toBeTruthy();
    expect(EN.queue.status.failed).toBeTruthy();
    expect(EN.queue.status.cancelled).toBeTruthy();
    expect(EN.queue.emptyTitle).toBeTruthy();
    expect(EN.queue.clearFinished).toBeTruthy();
    expect(EN.queue.retry).toBeTruthy();
    expect(EN.queue.openQueue).toBeTruthy();
    expect(EN.settings.cpuNotice).toMatch(/no nvidia gpu|slower/i);
    expect(EN.wizard.done.warnings.missingLayerModels).toMatch(/layer models/i);
    expect(EN.library.confirmDelete).toMatch(/confirm/i);
  });
});
