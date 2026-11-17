import type { AudioContextLike, BufferSourceLike, GainNodeLike } from "./audioPlayer";
import { DEFAULT_VOLUME_PERCENT, sliderToHitsoundGain } from "./audioVolume";
import type { ViewerNote } from "./types";

export const HITSOUND_DEDUP_EPS_S = 0.01;
export const HITSOUND_LOOKAHEAD_S = 0.15;
export const HITSOUND_PEAK_AMPLITUDE = 0.3;
export const DEFAULT_HITSOUND_VOLUME = sliderToHitsoundGain(DEFAULT_VOLUME_PERCENT);

export interface HitsoundClockSnapshot {
  isPlaying: boolean;
  songTimeS: number;
  playbackRate: number;
  anchorCtxTime: number;
  anchorSongTime: number;
}

/**
 * Extract sorted unique note hit timestamps, collapsing simultaneous chords / doubles
 * within `dedupThresholdSec` so notes at the same beat trigger one crisp hitsound.
 */
export function extractHitsoundTimes(
  notes: readonly Pick<ViewerNote, "timeS">[],
  dedupThresholdSec = HITSOUND_DEDUP_EPS_S,
): number[] {
  if (notes.length === 0) return [];
  const sorted = notes.map((n) => n.timeS).filter((t) => Number.isFinite(t)).sort((a, b) => a - b);
  const out: number[] = [];
  let last = -Infinity;
  for (let i = 0; i < sorted.length; i += 1) {
    const t = sorted[i];
    if (t - last > dedupThresholdSec) {
      out.push(t);
      last = t;
    }
  }
  return out;
}

function firstIndexAtOrAfter(sorted: readonly number[], target: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Synthesize a short (~55 ms), crisp percussive Beat Saber-style tick/slice AudioBuffer
 * purely via WebAudio math (no external audio files).
 */
export function synthesizeHitsoundBuffer(ctx: AudioContextLike): AudioBuffer | null {
  if (!ctx.createBuffer) return null;
  const sampleRate = ctx.sampleRate && ctx.sampleRate > 0 ? ctx.sampleRate : 44100;
  const durationSec = 0.055;
  const length = Math.max(1, Math.floor(sampleRate * durationSec));
  const buffer = ctx.createBuffer(1, length, sampleRate);
  const data = buffer.getChannelData(0);

  // Deterministic LCG PRNG for crisp airy high-frequency slice transient
  let rng = 0x13579bdf;
  let prevNoise = 0;
  let phase = 0;
  let peak = 0;

  for (let i = 0; i < length; i += 1) {
    const t = i / sampleRate;
    const attack = Math.min(1, t / 0.0012);

    // Rapid downward pitch sweep (1750 Hz -> 260 Hz) for tactile woodblock/saber tick body
    const freq = 260 + 1490 * Math.exp(-t * 145);
    phase += (2 * Math.PI * freq) / sampleRate;
    const tickBody = Math.sin(phase) * Math.exp(-t * 95);

    // High-passed deterministic noise burst for subtle crisp blade slice
    rng = (Math.imul(1664525, rng) + 1013904223) >>> 0;
    const white = (rng / 0xffffffff) * 2 - 1;
    const hpNoise = white - prevNoise * 0.72;
    prevNoise = white;
    const sliceAir = hpNoise * Math.exp(-t * 115) * 0.42;

    // Gentle envelope tail to zero at buffer end
    const tailFade = i > length - 48 ? (length - i) / 48 : 1;
    const sample = (tickBody * 0.78 + sliceAir) * attack * tailFade;
    data[i] = sample;
    const abs = Math.abs(sample);
    if (abs > peak) peak = abs;
  }

  if (peak > 0) {
    const scale = HITSOUND_PEAK_AMPLITUDE / peak;
    for (let i = 0; i < length; i += 1) {
      data[i] *= scale;
    }
  }

  return buffer;
}

/**
 * Lookahead WebAudio hitsound scheduler pinned to the exact AudioContext clock
 * used by music playback so hitsounds remain sample-accurate across seeks and speed changes.
 */
export class HitsoundScheduler {
  private ctx: AudioContextLike | null = null;
  private gainNode: GainNodeLike | null = null;
  private buffer: AudioBuffer | null = null;
  private noteTimes: readonly number[] = [];
  private nextIndex = 0;
  private enabled = true;
  private volume = DEFAULT_HITSOUND_VOLUME;
  private activeSources = new Set<BufferSourceLike>();

  attachContext(ctx: AudioContextLike): void {
    if (this.ctx === ctx) return;
    this.cancelScheduled();
    this.ctx = ctx;
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(this.enabled ? this.volume : 0, ctx.currentTime);
    this.gainNode.connect(ctx.destination);
    this.buffer = synthesizeHitsoundBuffer(ctx);
  }

  detachContext(): void {
    this.cancelScheduled();
    if (this.gainNode) {
      try {
        this.gainNode.disconnect();
      } catch {
        // Ignore disconnect errors
      }
      this.gainNode = null;
    }
    this.ctx = null;
    this.buffer = null;
  }

  setNoteTimes(times: readonly number[]): void {
    const sorted = [...times].filter((t) => Number.isFinite(t)).sort((a, b) => a - b);
    const deduped: number[] = [];
    let last = -Infinity;
    for (let i = 0; i < sorted.length; i += 1) {
      if (sorted[i] - last > HITSOUND_DEDUP_EPS_S) {
        deduped.push(sorted[i]);
        last = sorted[i];
      }
    }
    this.noteTimes = deduped;
    this.nextIndex = 0;
    this.cancelScheduled();
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.cancelScheduled();
    }
    if (this.ctx && this.gainNode) {
      this.gainNode.gain.setValueAtTime(enabled ? this.volume : 0, this.ctx.currentTime);
    }
  }

  setVolume(vol: number): void {
    this.volume = Math.max(0, Math.min(1, Number.isFinite(vol) ? vol : DEFAULT_HITSOUND_VOLUME));
    if (this.ctx && this.gainNode) {
      this.gainNode.gain.setValueAtTime(this.enabled ? this.volume : 0, this.ctx.currentTime);
    }
  }

  cancelScheduled(): void {
    for (const src of this.activeSources) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        // Ignore already-stopped nodes
      }
      try {
        src.disconnect();
      } catch {
        // Ignore disconnect errors
      }
    }
    this.activeSources.clear();
  }

  /** Cancel queued sounds and reposition lookup cursor after play, seek, or speed change. */
  resetAndSync(songTimeS: number): void {
    this.cancelScheduled();
    this.nextIndex = firstIndexAtOrAfter(this.noteTimes, Math.max(0, songTimeS) - 0.002);
  }

  scheduleTick(clock: HitsoundClockSnapshot): void {
    if (!clock.isPlaying || !this.enabled || !this.ctx || !this.gainNode || !this.buffer) {
      return;
    }
    if (this.noteTimes.length === 0) return;

    const rate = Math.max(0.1, clock.playbackRate);
    const horizonSongTime = clock.songTimeS + HITSOUND_LOOKAHEAD_S * rate;

    while (this.nextIndex < this.noteTimes.length) {
      const hitTime = this.noteTimes[this.nextIndex];
      if (hitTime > horizonSongTime) break;

      if (hitTime >= clock.songTimeS - 0.01) {
        const ctxWhen = Math.max(
          this.ctx.currentTime,
          clock.anchorCtxTime + (hitTime - clock.anchorSongTime) / rate,
        );
        const src = this.ctx.createBufferSource();
        src.buffer = this.buffer;
        src.connect(this.gainNode);
        this.activeSources.add(src);
        src.onended = () => {
          this.activeSources.delete(src);
          try {
            src.disconnect();
          } catch {
            // Ignore
          }
        };
        src.start(ctxWhen);
      }

      this.nextIndex += 1;
    }
  }
}
