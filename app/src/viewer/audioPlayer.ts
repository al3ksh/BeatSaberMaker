import { HitsoundScheduler } from "./hitsound";

export type PlaybackRate = 0.5 | 0.75 | 1 | 1.25 | 1.5;

export const PLAYBACK_RATES: readonly PlaybackRate[] = [0.5, 0.75, 1, 1.25, 1.5];

export interface GainNodeLike {
  gain: {
    value: number;
    setValueAtTime: (value: number, startTime: number) => void;
  };
  connect: (dest: unknown) => void;
  disconnect: () => void;
}

export interface BufferSourceLike {
  buffer: AudioBuffer | null;
  playbackRate: {
    value: number;
    setValueAtTime: (value: number, startTime: number) => void;
  };
  onended: (() => void) | null;
  connect: (dest: unknown) => void;
  disconnect: () => void;
  start: (when?: number, offset?: number) => void;
  stop: () => void;
}

export interface AudioContextLike {
  readonly currentTime: number;
  readonly state: string;
  readonly sampleRate?: number;
  readonly destination: unknown;
  resume: () => Promise<void>;
  close: () => Promise<void>;
  decodeAudioData: (audioData: ArrayBuffer) => Promise<AudioBuffer>;
  createGain: () => GainNodeLike;
  createBuffer?: (numberOfChannels: number, length: number, sampleRate: number) => AudioBuffer;
  createBufferSource: () => BufferSourceLike;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function defaultAudioContextFactory(): AudioContextLike {
  const Ctor =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  return new Ctor() as unknown as AudioContextLike;
}

/**
 * WebAudio player synchronized strictly to `AudioContext.currentTime`,
 * sharing the exact clock with the lookahead hitsound scheduler.
 */
export class ViewerAudioPlayer {
  private ctxFactory: () => AudioContextLike;
  private ctx: AudioContextLike | null = null;
  private gainNode: GainNodeLike | null = null;
  private buffer: AudioBuffer | null = null;
  private source: BufferSourceLike | null = null;
  private hitsounds = new HitsoundScheduler();

  private _duration = 0;
  private _isPlaying = false;
  private _playbackRate: PlaybackRate = 1;
  private _volume = 0.8;
  private anchorCtxTime = 0;
  private anchorSongTime = 0;
  private onEndedCb: (() => void) | null = null;

  constructor(ctxFactory: () => AudioContextLike = defaultAudioContextFactory) {
    this.ctxFactory = ctxFactory;
  }

  get duration(): number {
    return this._duration;
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  get playbackRate(): PlaybackRate {
    return this._playbackRate;
  }

  get volume(): number {
    return this._volume;
  }

  onEnded(cb: (() => void) | null): void {
    this.onEndedCb = cb;
  }

  setNoteHitTimes(times: readonly number[]): void {
    this.hitsounds.setNoteTimes(times);
    if (this._isPlaying) {
      this.hitsounds.resetAndSync(this.getCurrentTime());
      this.tickHitsounds();
    }
  }

  setHitsoundEnabled(enabled: boolean): void {
    this.hitsounds.setEnabled(enabled);
    if (enabled && this._isPlaying) {
      this.hitsounds.resetAndSync(this.getCurrentTime());
      this.tickHitsounds();
    }
  }

  setHitsoundVolume(vol: number): void {
    this.hitsounds.setVolume(vol);
  }

  tickHitsounds(): void {
    if (!this._isPlaying) return;
    const songTimeS = this.getCurrentTime();
    if (!this._isPlaying) return;
    this.hitsounds.scheduleTick({
      isPlaying: this._isPlaying,
      songTimeS,
      playbackRate: this._playbackRate,
      anchorCtxTime: this.anchorCtxTime,
      anchorSongTime: this.anchorSongTime,
    });
  }

  private ensureContext(): AudioContextLike | null {
    if (this.ctx) return this.ctx;
    try {
      const ctx = this.ctxFactory();
      this.ctx = ctx;
      this.gainNode = ctx.createGain();
      this.gainNode.gain.setValueAtTime(this._volume, ctx.currentTime);
      this.gainNode.connect(ctx.destination);
      this.hitsounds.attachContext(ctx);
      return ctx;
    } catch {
      return null;
    }
  }

  private clockNow(): number {
    return this.ctx ? this.ctx.currentTime : performance.now() / 1000;
  }

  private stopSource(): void {
    this.hitsounds.cancelScheduled();
    if (this.source) {
      this.source.onended = null;
      try {
        this.source.stop();
      } catch {
        // Ignore if already stopped.
      }
      try {
        this.source.disconnect();
      } catch {
        // Ignore disconnect failures.
      }
      this.source = null;
    }
  }

  async load(audioBytes: Uint8Array | null, fallbackDurationSec: number): Promise<boolean> {
    this.stopSource();
    this._isPlaying = false;
    this.anchorSongTime = 0;
    this.buffer = null;
    this._duration = Math.max(1, fallbackDurationSec);

    if (!audioBytes || audioBytes.byteLength === 0) {
      return false;
    }

    try {
      const ctx = this.ensureContext();
      if (!ctx) return false;
      const copy = audioBytes.slice().buffer;
      this.buffer = await ctx.decodeAudioData(copy);
      if (this.buffer && Number.isFinite(this.buffer.duration) && this.buffer.duration > 0) {
        this._duration = this.buffer.duration;
      }
      return true;
    } catch {
      this.buffer = null;
      return false;
    }
  }

  getCurrentTime(): number {
    if (!this._isPlaying) {
      return this.anchorSongTime;
    }
    const elapsed = (this.clockNow() - this.anchorCtxTime) * this._playbackRate;
    const raw = this.anchorSongTime + elapsed;
    if (this._duration > 0 && raw >= this._duration) {
      this.stopSource();
      this._isPlaying = false;
      this.anchorSongTime = this._duration;
      return this._duration;
    }
    return clamp(raw, 0, this._duration);
  }

  async play(fromTime?: number): Promise<void> {
    const ctx = this.ensureContext();
    if (ctx && ctx.state === "suspended") {
      await ctx.resume();
    }
    let startAt = clamp(fromTime ?? this.getCurrentTime(), 0, this._duration);
    if (startAt >= this._duration - 0.01) {
      startAt = 0;
    }
    this.stopSource();

    const now = this.clockNow();
    this.anchorCtxTime = now;
    this.anchorSongTime = startAt;
    this._isPlaying = true;

    if (ctx && this.gainNode && this.buffer) {
      const src = ctx.createBufferSource();
      src.buffer = this.buffer;
      src.playbackRate.value = this._playbackRate;
      src.connect(this.gainNode);
      src.onended = () => {
        if (this.source === src && this._isPlaying) {
          this.hitsounds.cancelScheduled();
          this._isPlaying = false;
          this.anchorSongTime = this._duration;
          this.source = null;
          this.onEndedCb?.();
        }
      };
      src.start(0, startAt);
      this.source = src;
    }

    this.hitsounds.resetAndSync(startAt);
    this.tickHitsounds();
  }

  pause(): void {
    if (!this._isPlaying) return;
    const current = this.getCurrentTime();
    this.stopSource();
    this.anchorSongTime = current;
    this._isPlaying = false;
  }

  async seek(targetTimeSec: number): Promise<void> {
    const clamped = clamp(targetTimeSec, 0, this._duration);
    this.anchorSongTime = clamped;
    this.anchorCtxTime = this.clockNow();
    if (this._isPlaying) {
      await this.play(clamped);
    } else {
      this.hitsounds.resetAndSync(clamped);
    }
  }

  setPlaybackRate(rate: PlaybackRate): void {
    const current = this.getCurrentTime();
    this._playbackRate = rate;
    if (this._isPlaying) {
      const now = this.clockNow();
      this.anchorSongTime = current;
      this.anchorCtxTime = now;
      if (this.source) {
        this.source.playbackRate.setValueAtTime(rate, now);
      }
      this.hitsounds.resetAndSync(current);
      this.tickHitsounds();
    }
  }

  setVolume(vol: number): void {
    this._volume = clamp(vol, 0, 1);
    if (this.ctx && this.gainNode) {
      this.gainNode.gain.setValueAtTime(this._volume, this.ctx.currentTime);
    }
  }

  async dispose(): Promise<void> {
    this.stopSource();
    this.hitsounds.detachContext();
    this._isPlaying = false;
    if (this.gainNode) {
      try {
        this.gainNode.disconnect();
      } catch {
        // Ignore
      }
      this.gainNode = null;
    }
    if (this.ctx) {
      try {
        await this.ctx.close();
      } catch {
        // Ignore
      }
      this.ctx = null;
    }
    this.buffer = null;
  }
}
