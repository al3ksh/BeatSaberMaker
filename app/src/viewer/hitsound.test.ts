import { describe, expect, it, vi } from "vitest";
import type { AudioContextLike } from "./audioPlayer";
import { HitsoundScheduler, extractHitsoundTimes } from "./hitsound";

function createHitsoundMockAudio() {
  let ctxTime = 20;
  const started: Array<{ when: number; stopped: boolean }> = [];
  const gainNode = {
    gain: { value: 0.65, setValueAtTime: vi.fn() },
    connect: vi.fn(),
    disconnect: vi.fn(),
  };

  const ctx = {
    get currentTime() {
      return ctxTime;
    },
    state: "running",
    sampleRate: 44100,
    destination: {},
    resume: vi.fn(async () => {}),
    close: vi.fn(async () => {}),
    decodeAudioData: vi.fn(async () => ({ duration: 60 }) as AudioBuffer),
    createGain: () => gainNode,
    createBuffer: (_ch: number, length: number, sampleRate: number) => {
      const data = new Float32Array(length);
      return {
        length,
        sampleRate,
        duration: length / sampleRate,
        numberOfChannels: 1,
        getChannelData: () => data,
      } as unknown as AudioBuffer;
    },
    createBufferSource: () => {
      const rec = { when: -1, stopped: false };
      return {
        buffer: null,
        playbackRate: { value: 1, setValueAtTime: vi.fn() },
        onended: null as (() => void) | null,
        connect: vi.fn(),
        disconnect: vi.fn(),
        start: vi.fn((when = 0) => {
          rec.when = when;
          started.push(rec);
          return undefined;
        }),
        stop: vi.fn(() => {
          rec.stopped = true;
        }),
      };
    },
  } as unknown as AudioContextLike;

  return {
    ctx,
    gainNode,
    started,
    setCtxTime: (t: number) => {
      ctxTime = t;
    },
  };
}

describe("hitsound extraction & scheduling", () => {
  it("deduplicates simultaneous / double notes within 10ms", () => {
    const times = extractHitsoundTimes([
      { timeS: 1.0 },
      { timeS: 1.0 }, // exact double
      { timeS: 1.005 }, // micro-flam within 10ms
      { timeS: 1.25 },
      { timeS: 0.5 }, // unsorted earlier note
    ]);
    expect(times).toEqual([0.5, 1.0, 1.25]);
  });

  it("schedules lookahead hitsounds synced to AudioContext clock and reschedules cleanly after seek/rate change", () => {
    const mock = createHitsoundMockAudio();
    const scheduler = new HitsoundScheduler();
    scheduler.attachContext(mock.ctx);
    scheduler.setEnabled(true);
    scheduler.setVolume(0.75);
    scheduler.setNoteTimes([1.0, 1.0, 1.08, 1.4, 2.5]);

    // Playing at 1.0x from songTime=0.95 at ctxTime=20
    scheduler.resetAndSync(0.95);
    scheduler.scheduleTick({
      isPlaying: true,
      songTimeS: 0.95,
      playbackRate: 1,
      anchorCtxTime: 20,
      anchorSongTime: 0.95,
    });

    // Lookahead is 0.15s -> 0.95..1.10 covers 1.0 (once!) and 1.08, NOT 1.4 or 2.5
    expect(mock.started).toHaveLength(2);
    expect(mock.started[0].when).toBeCloseTo(20.05, 5); // 20 + (1.0 - 0.95)/1
    expect(mock.started[1].when).toBeCloseTo(20.13, 5); // 20 + (1.08 - 0.95)/1

    // Calling tick again immediately does not duplicate already-scheduled notes
    scheduler.scheduleTick({
      isPlaying: true,
      songTimeS: 0.96,
      playbackRate: 1,
      anchorCtxTime: 20,
      anchorSongTime: 0.95,
    });
    expect(mock.started).toHaveLength(2);

    // Seek to 2.45 at 1.5x speed -> cancels previous unplayed hitsounds and schedules 2.5 at scaled time
    mock.setCtxTime(20.02);
    scheduler.resetAndSync(2.45);
    expect(mock.started[0].stopped).toBe(true);
    expect(mock.started[1].stopped).toBe(true);

    scheduler.scheduleTick({
      isPlaying: true,
      songTimeS: 2.45,
      playbackRate: 1.5,
      anchorCtxTime: 20.02,
      anchorSongTime: 2.45,
    });
    expect(mock.started).toHaveLength(3);
    // 20.02 + (2.5 - 2.45) / 1.5 = 20.02 + 0.033333
    expect(mock.started[2].when).toBeCloseTo(20.02 + 0.05 / 1.5, 5);

    // Disabling immediately cancels queued hitsound and suppresses future ticks
    scheduler.setEnabled(false);
    expect(mock.started[2].stopped).toBe(true);
  });
});
