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

  it("includes labels for all supported environments and wizard steps", () => {
    expect(Object.keys(ENVIRONMENT_LABELS_EN)).toHaveLength(10);
    expect(EN.wizard.steps.song).toBe("Song");
    expect(EN.wizard.steps.map).toBe("Map");
    expect(EN.wizard.steps.done).toBe("Done");
    expect(EN.settings.cpuNotice).toMatch(/no nvidia gpu|slower/i);
    expect(EN.wizard.done.successTitle).toMatch(/map is in the game/i);
    expect(EN.wizard.done.warnings.missingLayerModels).toMatch(/layer models/i);
    expect(EN.library.confirmDelete).toMatch(/confirm/i);
  });
});
