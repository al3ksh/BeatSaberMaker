import { DEFAULT_VOLUME_PERCENT, clampVolumePercent } from "./audioVolume";

export type ViewerQualityMode = "high" | "fast";

// Bumped v2 keys intentionally drop pre-scale (0..1) stored preferences.
const KEY_MUSIC_VOLUME_PERCENT = "bsm.viewer.v2.musicVolumePercent";
const KEY_HITSOUND_ENABLED = "bsm.viewer.v2.hitsoundEnabled";
const KEY_HITSOUND_VOLUME_PERCENT = "bsm.viewer.v2.hitsoundVolumePercent";
const KEY_QUALITY_MODE = "bsm.viewer.v2.qualityMode";

export interface ViewerPrefs {
  musicVolumePercent: number;
  hitsoundEnabled: boolean;
  hitsoundVolumePercent: number;
  qualityMode: ViewerQualityMode;
}

export function loadViewerPrefs(): ViewerPrefs {
  let musicVolumePercent = DEFAULT_VOLUME_PERCENT;
  let hitsoundEnabled = true;
  let hitsoundVolumePercent = DEFAULT_VOLUME_PERCENT;
  let qualityMode: ViewerQualityMode = "high";

  try {
    const rawMusic = window.localStorage.getItem(KEY_MUSIC_VOLUME_PERCENT);
    if (rawMusic !== null) {
      const parsed = Number(rawMusic);
      if (Number.isFinite(parsed)) {
        musicVolumePercent = clampVolumePercent(parsed);
      }
    }

    const rawEnabled = window.localStorage.getItem(KEY_HITSOUND_ENABLED);
    if (rawEnabled === "false") hitsoundEnabled = false;
    else if (rawEnabled === "true") hitsoundEnabled = true;

    const rawVol = window.localStorage.getItem(KEY_HITSOUND_VOLUME_PERCENT);
    if (rawVol !== null) {
      const parsed = Number(rawVol);
      if (Number.isFinite(parsed)) {
        hitsoundVolumePercent = clampVolumePercent(parsed);
      }
    }

    const rawQuality = window.localStorage.getItem(KEY_QUALITY_MODE);
    if (rawQuality === "fast" || rawQuality === "high") {
      qualityMode = rawQuality;
    }
  } catch {
    // Ignore storage unavailability
  }

  return { musicVolumePercent, hitsoundEnabled, hitsoundVolumePercent, qualityMode };
}

export function saveMusicVolumePercent(percent: number): void {
  try {
    window.localStorage.setItem(KEY_MUSIC_VOLUME_PERCENT, String(clampVolumePercent(percent)));
  } catch {
    // Ignore
  }
}

export function saveHitsoundEnabled(enabled: boolean): void {
  try {
    window.localStorage.setItem(KEY_HITSOUND_ENABLED, enabled ? "true" : "false");
  } catch {
    // Ignore
  }
}

export function saveHitsoundVolumePercent(percent: number): void {
  try {
    window.localStorage.setItem(KEY_HITSOUND_VOLUME_PERCENT, String(clampVolumePercent(percent)));
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
