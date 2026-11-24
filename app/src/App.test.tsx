import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EN } from "./i18n/en";
import type { LibraryEntry, QueueItem, QueueProgressEvent } from "./types";
import { App } from "./App";

const originalBsm = (window as unknown as { bsm?: unknown }).bsm;

afterEach(() => {
  if (originalBsm === undefined) {
    delete (window as unknown as { bsm?: unknown }).bsm;
  } else {
    (window as unknown as { bsm?: unknown }).bsm = originalBsm;
  }
});

function makeEntry(id: string, title: string): LibraryEntry {
  return {
    id,
    title,
    artist: "Camellia",
    folder: `C:\\CL\\BSM - ${title}`,
    createdAt: "2026-11-24T12:00:00.000Z",
    bpm: 180,
    levels: [{ rank: "Expert", nps: 5.5, nNotes: 500, nArcs: 5, nChains: 2 }],
    nWalls: 10,
    nBombs: 2,
    nLights: 60,
    environment: "DefaultEnvironment",
    audioCachePath: `C:\\media\\${id}\\audio.ogg`,
    coverCachePath: `C:\\media\\${id}\\cover.png`,
    config: {
      mode: "pack",
      levels: ["Hard", "Expert", "ExpertPlus"],
      nps: null,
      styles: [],
      ornaments: null,
      lights: 2,
      environment: "DefaultEnvironment",
      seed: 42,
      lightsRules: false,
    },
  };
}

function makeQueueItem(overrides: Partial<QueueItem> & { id: string }): QueueItem {
  return {
    status: "waiting",
    request: {
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtu.be/1",
      title: `Track ${overrides.id}`,
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
        seed: 42,
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
    ...overrides,
  };
}

describe("App queue integration", () => {
  it("shows Queue tab badge, bottom QueueStatusBar, live library updates, and Shift+Click library regenerate", async () => {
    const e1 = makeEntry("e1", "GHOST");
    const e2 = makeEntry("e2", "Spin Eternally");

    let queueChangedCb: ((items: QueueItem[]) => void) | null = null;
    let queueProgressCb: ((ev: QueueProgressEvent) => void) | null = null;
    let libraryChangedCb: ((entries: LibraryEntry[]) => void) | null = null;

    const enqueueGeneration = vi.fn(async () => makeQueueItem({ id: "q-new" }));
    const cancelQueueItem = vi.fn(async () => []);
    const readCoverDataUrl = vi.fn(async (p: string) => `data:image/png;base64,${p}`);

    const initialQueue: QueueItem[] = [
      makeQueueItem({
        id: "q1",
        status: "running",
        request: {
          sourceType: "youtube",
          sourcePathOrUrl: "https://youtu.be/ghost",
          title: "Running Track",
          artist: "Camellia",
          coverPath: null,
          config: e1.config,
        },
        progress: { stage: "notes", pct: 40 },
      }),
      makeQueueItem({ id: "q2", status: "waiting" }),
      makeQueueItem({ id: "q3", status: "failed", errorCode: "internal" }),
    ];

    (window as unknown as { bsm: unknown }).bsm = {
      getStartupState: vi.fn(async () => ({
        needsEnginePack: false,
        engineVersion: "4",
        appVersion: "0.1.4",
        settings: {
          customLevels: "C:\\CL",
          device: "auto",
          exportDir: "C:\\Exp",
          engineVersionInstalled: "4",
        },
        info: {
          version: "4",
          cuda_available: true,
          gpu_name: "RTX",
          active_device: "cuda",
          detected_custom_levels: "C:\\CL",
          has_layer_models: true,
          missing_layer_models: [],
          environments: ["DefaultEnvironment"],
          styles: ["tech"],
        },
      })),
      listLibrary: vi.fn(async () => [e1]),
      readCoverDataUrl,
      getQueue: vi.fn(async () => initialQueue),
      enqueueGeneration,
      cancelQueueItem,
      retryQueueItem: vi.fn(async () => initialQueue),
      moveQueueItem: vi.fn(async () => initialQueue),
      clearFinishedQueue: vi.fn(async () => initialQueue),
      onQueueChanged: vi.fn((cb: (items: QueueItem[]) => void) => {
        queueChangedCb = cb;
        return () => {
          queueChangedCb = null;
        };
      }),
      onQueueProgress: vi.fn((cb: (ev: QueueProgressEvent) => void) => {
        queueProgressCb = cb;
        return () => {
          queueProgressCb = null;
        };
      }),
      onLibraryChanged: vi.fn((cb: (entries: LibraryEntry[]) => void) => {
        libraryChangedCb = cb;
        return () => {
          libraryChangedCb = null;
        };
      }),
      onUpdateStatus: vi.fn(() => () => {}),
    };

    render(<App />);

    expect(await screen.findByTestId("queue-status-bar")).toBeInTheDocument();
    expect(screen.getByTestId("queue-tab-count")).toHaveTextContent("2");
    expect(screen.getByTestId("queue-tab-failed-dot")).toBeInTheDocument();
    expect(screen.getByText("Running Track")).toBeInTheDocument();

    act(() => {
      queueProgressCb?.({ id: "q1", stage: "layers", pct: 75 });
    });
    expect(screen.getByText(/75%/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /open queue/i }));
    expect(screen.queryByTestId("queue-status-bar")).not.toBeInTheDocument();
    expect(screen.getByTestId("queue-section-now")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: new RegExp(EN.nav.library, "i") }));
    expect(await screen.findByText("GHOST")).toBeInTheDocument();
    expect(screen.getByTestId("queue-status-bar")).toBeInTheDocument();

    act(() => {
      libraryChangedCb?.([e1, e2]);
    });
    expect(await screen.findByText("Spin Eternally")).toBeInTheDocument();

    const regenButtons = screen.getAllByRole("button", { name: /regenerate/i });
    fireEvent.click(regenButtons[0], { shiftKey: true });

    await waitFor(() => expect(enqueueGeneration).toHaveBeenCalledTimes(1));
    expect(enqueueGeneration).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceType: "cached",
        sourcePathOrUrl: e1.audioCachePath,
        title: e1.title,
        regenerateId: e1.id,
      }),
    );
    expect(screen.getByText("GHOST")).toBeInTheDocument();
    expect(screen.getByText(EN.queue.addedNotice("GHOST"))).toBeInTheDocument();

    act(() => {
      queueChangedCb?.([]);
    });
    expect(screen.queryByTestId("queue-status-bar")).not.toBeInTheDocument();
  });
});
