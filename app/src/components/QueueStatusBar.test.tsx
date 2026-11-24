import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MapGenerationConfig, QueueItem } from "../types";
import { QueueStatusBar } from "./QueueStatusBar";

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
      sourcePathOrUrl: "https://youtube.com/watch?v=1",
      title: `Track ${overrides.id}`,
      artist: "Artist",
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

describe("QueueStatusBar", () => {
  it("shows active progress and waiting counts when queue is non-empty, hides when empty, and opens queue tab on click", () => {
    const onOpenQueue = vi.fn();
    const onCancelActive = vi.fn();

    const { container, rerender } = render(
      <QueueStatusBar
        items={[]}
        onOpenQueue={onOpenQueue}
        onCancelActive={onCancelActive}
      />,
    );
    expect(container).toBeEmptyDOMElement();

    const items: QueueItem[] = [
      makeItem({
        id: "a1",
        status: "running",
        request: {
          sourceType: "youtube",
          sourcePathOrUrl: "https://youtube.com/watch?v=ghost",
          title: "GHOST",
          artist: "Camellia",
          coverPath: null,
          config: BASE_CONFIG,
        },
        progress: { stage: "notes", pct: 54 },
      }),
      makeItem({ id: "w1", status: "waiting" }),
      makeItem({ id: "w2", status: "waiting" }),
      makeItem({ id: "d1", status: "done" }),
      makeItem({ id: "f1", status: "failed", errorCode: "internal" }),
    ];

    rerender(
      <QueueStatusBar
        items={items}
        onOpenQueue={onOpenQueue}
        onCancelActive={onCancelActive}
      />,
    );

    expect(screen.getByText("GHOST")).toBeInTheDocument();
    expect(screen.getByText(/54%/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "54");
    expect(screen.getByText(/\+2 waiting/i)).toBeInTheDocument();
    expect(screen.getByText(/1 done/i)).toBeInTheDocument();
    expect(screen.getByText(/1 failed/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onCancelActive).toHaveBeenCalledWith("a1");

    fireEvent.click(screen.getByRole("button", { name: /open queue/i }));
    expect(onOpenQueue).toHaveBeenCalled();
  });

  it("renders summary counts without cancel button when no active item is running", () => {
    const onOpenQueue = vi.fn();
    render(
      <QueueStatusBar
        items={[makeItem({ id: "d1", status: "done" })]}
        onOpenQueue={onOpenQueue}
        onCancelActive={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
    expect(screen.getByText(/1 done/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open queue/i }));
    expect(onOpenQueue).toHaveBeenCalled();
  });
});
