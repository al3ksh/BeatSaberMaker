import { DEFAULT_HITSOUND_VOLUME } from "./hitsound";

export type ViewerQualityMode = "high" | "fast";

const KEY_HITSOUND_ENABLED = "bsm.viewer.hitsoundEnabled";
const KEY_HITSOUND_VOLUME = "bsm.viewer.hitsoundVolume";
const KEY_QUALITY_MODE = "bsm.viewer.qualityMode";

export interface ViewerPrefs {
  hitsoundEnabled: boolean;
  hitsoundVolume: number;
  qualityMode: ViewerQualityMode;
}

export function loadViewerPrefs(): ViewerPrefs {
  let hitsoundEnabled = true;
  let hitsoundVolume = DEFAULT_HITSOUND_VOLUME;
  let qualityMode: ViewerQualityMode = "high";

  try {
    const rawEnabled = window.localStorage.getItem(KEY_HITSOUND_ENABLED);
    if (rawEnabled === "false") hitsoundEnabled = false;
    else if (rawEnabled === "true") hitsoundEnabled = true;

    const rawVol = window.localStorage.getItem(KEY_HITSOUND_VOLUME);
    if (rawVol !== null) {
      const parsed = Number(rawVol);
      if (Number.isFinite(parsed)) {
        hitsoundVolume = Math.max(0, Math.min(1, parsed));
      }
    }

    const rawQuality = window.localStorage.getItem(KEY_QUALITY_MODE);
    if (rawQuality === "fast" || rawQuality === "high") {
      qualityMode = rawQuality;
    }
  } catch {
    // Ignore storage unavailability
  }

  return { hitsoundEnabled, hitsoundVolume, qualityMode };
}

export function saveHitsoundEnabled(enabled: boolean): void {
  try {
    window.localStorage.setItem(KEY_HITSOUND_ENABLED, enabled ? "true" : "false");
  } catch {
    // Ignore
  }
}

export function saveHitsoundVolume(volume: number): void {
  try {
    window.localStorage.setItem(KEY_HITSOUND_VOLUME, String(Math.max(0, Math.min(1, volume))));
  } catch {
    // Ignore
  }
}

export function saveQualityMode(mode: ViewerQualityMode): void {
  try {
    window.localStorage.setItem(KEY_QUALITY_MODE, mode);
  } catch {
    // Ignore
  }
}
