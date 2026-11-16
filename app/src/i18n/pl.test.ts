import { describe, expect, it } from "vitest";
import { ERRORS_PL, ENVIRONMENT_LABELS_PL, formatErrorPl, PL } from "./pl";

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

describe("Polish i18n dictionary", () => {
  it("maps every engine error code to non-empty Polish text without traceback jargon", () => {
    for (const code of EXPECTED_CODES) {
      const msg = ERRORS_PL[code];
      expect(msg, code).toBeTruthy();
      expect(msg).not.toMatch(/traceback|exception|stack/i);
    }
    expect(formatErrorPl("unknown_future_code")).toBe(ERRORS_PL.internal);
  });

  it("includes labels for all supported environments and wizard steps", () => {
    expect(Object.keys(ENVIRONMENT_LABELS_PL)).toHaveLength(10);
    expect(PL.wizard.steps.song).toBe("Utwór");
    expect(PL.wizard.steps.map).toBe("Mapa");
    expect(PL.wizard.steps.done).toBe("Gotowe");
    expect(PL.settings.cpuNotice).toMatch(/brak karty nvidia|wolniej/i);
    expect(PL.wizard.done.successTitle).toMatch(/mapa jest w grze/i);
    expect(PL.wizard.done.warnings.missingLayerModels).toMatch(/brak modeli warstw/i);
    expect(PL.library.confirmDelete).toMatch(/potwierdź/i);
  });
});
