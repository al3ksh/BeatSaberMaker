export const DEFAULT_VOLUME_PERCENT = 40;
export const MUSIC_MAX_GAIN = 0.6;
export const HITSOUND_MAX_GAIN = MUSIC_MAX_GAIN * 0.5;

export function clampVolumePercent(
  percent: number,
  fallback = DEFAULT_VOLUME_PERCENT,
): number {
  const safe = Number.isFinite(percent) ? percent : fallback;
  return Math.round(Math.max(0, Math.min(100, safe)));
}

/**
 * Perceptual quadratic volume mapping: gain = (slider / 100)^2 * maxGain.
 * Keeps 100% loud yet comfortable and gives fine control at low volumes.
 */
export function sliderToGain(percent: number, maxGain: number): number {
  const safePercent = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : DEFAULT_VOLUME_PERCENT;
  const norm = safePercent / 100;
  return Math.max(0, Math.min(1, norm * norm * maxGain));
}

export function sliderToMusicGain(percent: number): number {
  return sliderToGain(percent, MUSIC_MAX_GAIN);
}

export function sliderToHitsoundGain(percent: number): number {
  return sliderToGain(percent, HITSOUND_MAX_GAIN);
}
