import { describe, expect, it, vi } from "vitest";
import { ViewerAudioPlayer, type AudioContextLike } from "./audioPlayer";
import { DEFAULT_VOLUME_PERCENT, sliderToMusicGain } from "./audioVolume";

function createFakeAudioContext() {
  let currentTime = 10;
  let lastSource: {
    startArgs: number[];
    stopped: boolean;
    playbackRate: { value: number; setValueAtTime: ReturnType<typeof vi.fn> };
    onended: (() => void) | null;
  } | null = null;

  const gainNode = {
    gain: { value: 1, setValueAtTime: vi.fn() },
    connect: vi.fn(),
    disconnect: vi.fn(),
  };

  const ctx: AudioContextLike = {
    get currentTime() {
      return currentTime;
    },
    state: "running",
    destination: {},
    resume: vi.fn(async () => {}),
    close: vi.fn(async () => {}),
    decodeAudioData: vi.fn(async () => ({ duration: 120 }) as AudioBuffer),
    createGain: () => gainNode,
    createBufferSource: () => {
      const src = {
        buffer: null as AudioBuffer | null,
        playbackRate: {
          value: 1,
          setValueAtTime: vi.fn((val: number) => {
            src.playbackRate.value = val;
          }),
        },
        startArgs: [] as number[],
        stopped: false,
        onended: null as (() => void) | null,
        connect: vi.fn(),
        disconnect: vi.fn(),
        start: vi.fn((when = 0, offset = 0) => {
          src.startArgs = [when, offset];
        }),
        stop: vi.fn(() => {
          src.stopped = true;
        }),
      };
      lastSource = src;
      return src;
    },
  };

  return {
    ctx,
    gainNode,
    advanceTime: (dt: number) => {
      currentTime += dt;
    },
    getLastSource: () => lastSource,
  };
}

describe("ViewerAudioPlayer", () => {
  it("tracks AudioContext.currentTime accurately across play, pause, seek, and rate changes", async () => {
    const fake = createFakeAudioContext();
    const player = new ViewerAudioPlayer(() => fake.ctx);

    await player.load(new Uint8Array([1, 2, 3]), 120);
    expect(player.duration).toBe(120);
    expect(player.getCurrentTime()).toBe(0);

    // Play from 0 at 1.0x
    await player.play();
    expect(fake.getLastSource()?.startArgs).toEqual([0, 0]);
    fake.advanceTime(5);
    expect(player.getCurrentTime()).toBeCloseTo(5, 6);

    // Change speed to 1.5x at t=5s, then advance 4s -> 5 + 4*1.5 = 11s
    player.setPlaybackRate(1.5);
    expect(player.getCurrentTime()).toBeCloseTo(5, 6);
    fake.advanceTime(4);
    expect(player.getCurrentTime()).toBeCloseTo(11, 6);

    // Seek to 60s while playing
    await player.seek(60);
    expect(fake.getLastSource()?.startArgs).toEqual([0, 60]);
    fake.advanceTime(2);
    expect(player.getCurrentTime()).toBeCloseTo(63, 6); // 60 + 2*1.5

    // Pause freezes time
    player.pause();
    fake.advanceTime(10);
    expect(player.getCurrentTime()).toBeCloseTo(63, 6);
    expect(player.isPlaying).toBe(false);

    // Volume clamping
    player.setVolume(0.4);
    expect(fake.gainNode.gain.setValueAtTime).toHaveBeenCalledWith(0.4, expect.any(Number));

    await player.dispose();
  });

  it("stops cleanly at duration end", async () => {
    const fake = createFakeAudioContext();
    const player = new ViewerAudioPlayer(() => fake.ctx);
    await player.load(new Uint8Array([1, 2, 3]), 120);

    await player.seek(118);
    await player.play();
    fake.advanceTime(10);
    expect(player.getCurrentTime()).toBe(120);
    expect(player.isPlaying).toBe(false);

    await player.dispose();
  });

  it("defaults music volume to 40% perceptual gain and clamps gain strictly to [0, 1]", async () => {
    const expectedDefaultGain = sliderToMusicGain(DEFAULT_VOLUME_PERCENT);
    const fake = createFakeAudioContext();
    const player = new ViewerAudioPlayer(() => fake.ctx);
    expect(player.volume).toBeCloseTo(expectedDefaultGain, 6);

    await player.load(new Uint8Array([1, 2, 3]), 120);
    expect(fake.gainNode.gain.setValueAtTime).toHaveBeenCalledWith(
      expectedDefaultGain,
      expect.any(Number),
    );

    player.setVolume(1.8);
    expect(player.volume).toBe(1);
    expect(fake.gainNode.gain.setValueAtTime).toHaveBeenLastCalledWith(1, expect.any(Number));

    await player.dispose();
  });
});
