import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EN, ERRORS_EN } from "../../i18n/en";
import type { LibraryEntry, QueueItem } from "../../types";
import { Wizard } from "./Wizard";

const originalBsm = (window as unknown as { bsm?: unknown }).bsm;

afterEach(() => {
  vi.useRealTimers();
  if (originalBsm === undefined) {
    delete (window as unknown as { bsm?: unknown }).bsm;
  } else {
    (window as unknown as { bsm?: unknown }).bsm = originalBsm;
  }
});

function makeQueuedItem(title: string): QueueItem {
  return {
    id: "q-1",
    status: "waiting",
    request: {
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtu.be/1",
      title,
      artist: "Camellia",
      coverPath: null,
      config: {
        mode: "pack",
        levels: ["Hard", "Expert", "ExpertPlus"],
        nps: null,
        styles: [],
        ornaments: null,
        lights: 2,
        environment: "DefaultEnvironment",
        seed: 123,
        lightsRules: false,
      },
    },
    thumbnailUrl: null,
    enqueuedAt: "2026-11-24T12:00:00.000Z",
    startedAt: null,
    finishedAt: null,
    progress: null,
    errorCode: null,
    result: null,
    entry: null,
  };
}

function makeLibraryEntry(): LibraryEntry {
  return {
    id: "lib-1",
    title: "Existing Map",
    artist: "Camellia",
    folder: "C:\\CL\\BSM - Existing Map",
    createdAt: "2026-11-20T12:00:00.000Z",
    bpm: 180,
    levels: [{ rank: "Expert", nps: 5.5, nNotes: 600, nArcs: 12, nChains: 4 }],
    nWalls: 10,
    nBombs: 2,
    nLights: 90,
    environment: "BigMirrorEnvironment",
    audioCachePath: "C:\\media\\lib-1\\audio.ogg",
    coverCachePath: "C:\\media\\lib-1\\cover.png",
    config: {
      mode: "custom",
      levels: ["Expert"],
      nps: null,
      styles: ["tech"],
      ornaments: 2,
      lights: 2,
      environment: "BigMirrorEnvironment",
      seed: 98765,
      lightsRules: false,
    },
  };
}

describe("Wizard", () => {
  it("enqueues from Step 2, resets wizard to Step 1 for the next song, and clears regenerate target", async () => {
    const enqueueGeneration = vi.fn(async () => makeQueuedItem("GHOST"));
    const searchYouTube = vi.fn(async () => [
      {
        id: "1",
        url: "https://youtu.be/1",
        title: "Camellia - GHOST",
        artist: "Camellia",
        track_title: "GHOST",
        channel: "Camellia",
        duration_s: 240,
        thumbnail_url: "https://i.ytimg.com/vi/1/hqdefault.jpg",
      },
    ]);
    (window as unknown as { bsm: unknown }).bsm = {
      searchYouTube,
      resolveYouTube: vi.fn(),
      pickAudioFile: vi.fn(),
      pickCoverImage: vi.fn(),
      enqueueGeneration,
    };

    const onClearRegenerate = vi.fn();
    const onGoToQueue = vi.fn();

    render(
      <Wizard
        regenerateTarget={null}
        onClearRegenerate={onClearRegenerate}
        onGoToQueue={onGoToQueue}
      />,
    );

    expect(screen.queryByText("Done")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText(/search|link/i), {
      target: { value: "ghost" },
    });
    fireEvent.submit(screen.getByRole("search"));
    fireEvent.click(await screen.findByText("Camellia - GHOST"));

    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    const addBtn = await screen.findByRole("button", { name: /add to queue/i });
    fireEvent.click(addBtn);

    await waitFor(() => expect(enqueueGeneration).toHaveBeenCalledTimes(1));
    const payload = (enqueueGeneration.mock.calls as unknown[][])[0][0] as {
      sourceType: string;
      sourcePathOrUrl: string;
      title: string;
      artist: string;
      thumbnailUrl: string | null;
      config: { mode: string; seed: number };
    };
    expect(payload.sourceType).toBe("youtube");
    expect(payload.sourcePathOrUrl).toBe("https://youtu.be/1");
    expect(payload.title).toBe("GHOST");
    expect(payload.artist).toBe("Camellia");
    expect(payload.thumbnailUrl).toBe("https://i.ytimg.com/vi/1/hqdefault.jpg");
    expect(payload.config.mode).toBe("pack");
    expect(payload.config.seed).toBeGreaterThan(0);
    expect(onClearRegenerate).toHaveBeenCalled();

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /next/i })).toBeDisabled(),
    );
    expect(screen.getByText(EN.queue.addedNotice("GHOST"))).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: EN.queue.viewQueue }));
    expect(onGoToQueue).toHaveBeenCalledTimes(1);
  });

  it("stays on Step 2 and shows localized error when enqueueGeneration fails", async () => {
    const enqueueGeneration = vi.fn(async () => {
      throw new Error("BSM_CODE:not_found");
    });
    (window as unknown as { bsm: unknown }).bsm = {
      searchYouTube: vi.fn(),
      resolveYouTube: vi.fn(),
      pickAudioFile: vi.fn(async () => "C:\\Music\\track.mp3"),
      pickCoverImage: vi.fn(),
      enqueueGeneration,
    };

    render(
      <Wizard
        regenerateTarget={null}
        onClearRegenerate={vi.fn()}
        onGoToQueue={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /from file/i }));
    fireEvent.click(screen.getByRole("button", { name: /choose audio file/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /next/i })).not.toBeDisabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /next/i }));

    fireEvent.click(await screen.findByRole("button", { name: /add to queue/i }));

    expect(await screen.findByText(ERRORS_EN.not_found)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /add to queue/i }),
    ).toBeInTheDocument();
  });

  it("opens Step 2 for regenerate target and enqueues with regenerateId", async () => {
    const entry = makeLibraryEntry();
    const enqueueGeneration = vi.fn(async () => makeQueuedItem(entry.title));
    (window as unknown as { bsm: unknown }).bsm = {
      searchYouTube: vi.fn(),
      resolveYouTube: vi.fn(),
      pickAudioFile: vi.fn(),
      pickCoverImage: vi.fn(),
      enqueueGeneration,
    };
    const onClearRegenerate = vi.fn();

    render(
      <Wizard
        regenerateTarget={entry}
        onClearRegenerate={onClearRegenerate}
        onGoToQueue={vi.fn()}
      />,
    );

    expect(await screen.findByDisplayValue("98765")).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: /queue regeneration/i }),
    );

    await waitFor(() => expect(enqueueGeneration).toHaveBeenCalledTimes(1));
    expect(enqueueGeneration).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceType: "cached",
        sourcePathOrUrl: entry.audioCachePath,
        title: entry.title,
        artist: entry.artist,
        coverPath: entry.coverCachePath,
        regenerateId: entry.id,
      }),
    );
    expect(onClearRegenerate).toHaveBeenCalled();
  });
});
