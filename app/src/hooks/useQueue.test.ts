import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MapGenerationConfig, QueueItem, QueueProgressEvent } from "../types";
import { useQueue } from "./useQueue";

const BASE_CONFIG: MapGenerationConfig = {
  mode: "pack",
  levels: ["Hard", "Expert", "ExpertPlus"],
  nps: null,
  styles: [],
  ornaments: null,
  lights: 2,
  environment: "DefaultEnvironment",
  seed: 123,
  lightsRules: false,
};

function makeItem(overrides: Partial<QueueItem> & { id: string }): QueueItem {
  return {
    status: "waiting",
    request: {
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtube.com/watch?v=abc",
      title: `Track ${overrides.id}`,
      artist: "Camellia",
      coverPath: null,
      config: BASE_CONFIG,
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

describe("useQueue", () => {
  it("hydrates from getQueue, subscribes to queueChanged/queueProgress, and exposes counts and actions", async () => {
    let changedCb: ((items: QueueItem[]) => void) | undefined;
    let progressCb: ((ev: QueueProgressEvent) => void) | undefined;
    const unsubChanged = vi.fn();
    const unsubProgress = vi.fn();

    const initial = [
      makeItem({ id: "q1", status: "running", progress: { stage: "analyzing", pct: 10 } }),
      makeItem({ id: "q2", status: "waiting" }),
      makeItem({ id: "q3", status: "failed", errorCode: "internal" }),
    ];

    const getQueue = vi.fn().mockResolvedValue(initial);
    const enqueueGeneration = vi.fn().mockResolvedValue(makeItem({ id: "q4" }));
    const cancelQueueItem = vi.fn().mockResolvedValue(initial);
    const retryQueueItem = vi.fn().mockResolvedValue(initial);
    const moveQueueItem = vi.fn().mockResolvedValue(initial);
    const clearFinishedQueue = vi.fn().mockResolvedValue(initial.slice(0, 2));

    (window as unknown as { bsm: unknown }).bsm = {
      getQueue,
      enqueueGeneration,
      cancelQueueItem,
      retryQueueItem,
      moveQueueItem,
      clearFinishedQueue,
      onQueueChanged: (cb: (items: QueueItem[]) => void) => {
        changedCb = cb;
        return unsubChanged;
      },
      onQueueProgress: (cb: (ev: QueueProgressEvent) => void) => {
        progressCb = cb;
        return unsubProgress;
      },
    };

    const { result, unmount } = renderHook(() => useQueue());

    await waitFor(() => {
      expect(result.current.items).toHaveLength(3);
    });
    expect(result.current.activeItem?.id).toBe("q1");
    expect(result.current.waitingCount).toBe(1);
    expect(result.current.failedCount).toBe(1);

    act(() => {
      progressCb?.({ id: "q1", stage: "notes", pct: 64 });
    });
    expect(result.current.activeItem?.progress).toEqual({ stage: "notes", pct: 64 });

    act(() => {
      changedCb?.([
        makeItem({ id: "q1", status: "done", progress: { stage: "writing", pct: 100 } }),
        makeItem({ id: "q2", status: "downloading", progress: { stage: "downloading", pct: 25 } }),
      ]);
    });
    expect(result.current.activeItem?.id).toBe("q2");
    expect(result.current.waitingCount).toBe(0);
    expect(result.current.failedCount).toBe(0);

    await act(async () => {
      await result.current.actions.cancel("q2");
      await result.current.actions.retry("q3");
      await result.current.actions.move("q2", "up");
      await result.current.actions.clearFinished();
    });
    expect(cancelQueueItem).toHaveBeenCalledWith("q2");
    expect(retryQueueItem).toHaveBeenCalledWith("q3");
    expect(moveQueueItem).toHaveBeenCalledWith("q2", "up");
    expect(clearFinishedQueue).toHaveBeenCalled();

    unmount();
    expect(unsubChanged).toHaveBeenCalled();
    expect(unsubProgress).toHaveBeenCalled();
  });
});
