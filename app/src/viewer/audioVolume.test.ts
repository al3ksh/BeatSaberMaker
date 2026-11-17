import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_VOLUME_PERCENT,
  HITSOUND_MAX_GAIN,
  MUSIC_MAX_GAIN,
  sliderToHitsoundGain,
  sliderToMusicGain,
} from "./audioVolume";
import {
  loadViewerPrefs,
  saveHitsoundVolumePercent,
  saveMusicVolumePercent,
} from "./viewerPrefs";

describe("perceptual slider-to-gain mapping & prefs migration", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("maps 0..100 slider perceptually via quadratic curve with comfortable max gains", () => {
    expect(DEFAULT_VOLUME_PERCENT).toBe(40);
    expect(MUSIC_MAX_GAIN).toBeCloseTo(0.6, 6);
    expect(HITSOUND_MAX_GAIN).toBeCloseTo(MUSIC_MAX_GAIN * 0.5, 6);

    // 0% -> silence
    expect(sliderToMusicGain(0)).toBe(0);
    expect(sliderToHitsoundGain(0)).toBe(0);

    // 40% default -> (0.4)^2 * max
    expect(sliderToMusicGain(40)).toBeCloseTo(0.16 * 0.6, 6);
    expect(sliderToHitsoundGain(40)).toBeCloseTo(0.16 * 0.3, 6);

    // 50% -> quarter of max gain (perceptual quadratic curve)
    expect(sliderToMusicGain(50)).toBeCloseTo(0.25 * 0.6, 6);
    expect(sliderToHitsoundGain(50)).toBeCloseTo(0.25 * 0.3, 6);

    // 100% -> max gain (never > 1.0)
    expect(sliderToMusicGain(100)).toBeCloseTo(0.6, 6);
    expect(sliderToHitsoundGain(100)).toBeCloseTo(0.3, 6);

    // Clamps out-of-range and non-finite inputs safely
    expect(sliderToMusicGain(-25)).toBe(0);
    expect(sliderToMusicGain(250)).toBeCloseTo(0.6, 6);
    expect(sliderToMusicGain(Number.NaN)).toBeCloseTo(0.16 * 0.6, 6);
    expect(sliderToHitsoundGain(250)).toBeCloseTo(0.3, 6);
  });

  it("defaults to 40%/40%, drops legacy pre-v2 localStorage values, and preserves v2 saves", () => {
    // Legacy v1 keys from before the scale change must be dropped
    window.localStorage.setItem("bsm.viewer.hitsoundVolume", "0.95");
    window.localStorage.setItem("bsm.viewer.hitsoundEnabled", "false");

    const fresh = loadViewerPrefs();
    expect(fresh.musicVolumePercent).toBe(40);
    expect(fresh.hitsoundVolumePercent).toBe(40);
    expect(fresh.hitsoundEnabled).toBe(true);

    saveMusicVolumePercent(35);
    saveHitsoundVolumePercent(62);
    const saved = loadViewerPrefs();
    expect(saved.musicVolumePercent).toBe(35);
    expect(saved.hitsoundVolumePercent).toBe(62);
  });
});
