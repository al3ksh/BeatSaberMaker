import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EN, ERRORS_EN } from "../i18n/en";
import type { GenerateOutput, LibraryEntry, MapGenerationConfig, QueueItem } from "../types";
import { QueuePanel } from "./QueuePanel";

const originalBsm = (window as unknown as { bsm?: unknown }).bsm;

afterEach(() => {
  if (originalBsm === undefined) {
    delete (window as unknown as { bsm?: unknown }).bsm;
  } else {
    (window as unknown as { bsm?: unknown }).bsm = originalBsm;
  }
});

const PACK_CONFIG: MapGenerationConfig = {
  mode: "pack",
  levels: ["Hard", "Expert", "ExpertPlus"],
  nps: null,
  styles: [],
  ornaments: null,
  lights: 2,
  environment: "DefaultEnvironment",
  seed: 42,
  lightsRules: false,
};

const CUSTOM_CONFIG: MapGenerationConfig = {
  mode: "custom",
  levels: ["Expert"],
  nps: null,
  styles: ["tech"],
  ornaments: null,
  lights: 2,
  environment: "BigMirrorEnvironment",
  seed: 99,
  lightsRules: false,
};

function makeItem(overrides: Partial<QueueItem> & { id: string }): QueueItem {
  return {
    status: "waiting",
    request: {
      sourceType: "youtube",
      sourcePathOrUrl: "https://youtube.com/watch?v=1",
      title: `Song ${overrides.id}`,
      artist: "Artist",
      coverPath: null,
      config: PACK_CONFIG,
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

function makeResult(overrides?: Partial<GenerateOutput>): GenerateOutput {
  return {
    folder: "C:\\CL\\BSM - Song",
    title: "Song",
    artist: "Artist",
    bpm: 120,
    grid_steady: true,
    n_notes: 300,
    nps: 4.0,
    rank: "Hard",
    levels: [
      {
        rank: "Hard",
        target_nps: 4.0,
        n_notes: 300,
        nps: 4.0,
        n_arcs: 0,
        n_chains: 0,
      },
    ],
    n_walls: 0,
    n_bombs: 0,
    n_lights: 40,
    environment: "DefaultEnvironment",
    warnings: [],
    used_rule_lights: false,
    device_used: "cuda",
    ...overrides,
  };
}

describe("QueuePanel", () => {
  it("renders empty state and triggers onGoToCreate", () => {
    const onGoToCreate = vi.fn();
    render(
      <QueuePanel
        items={[]}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onMove={vi.fn()}
        onClearFinished={vi.fn()}
        onGoToCreate={onGoToCreate}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /new map/i }));
    expect(onGoToCreate).toHaveBeenCalled();
  });

  it("renders active progress (including downloading stage for YouTube), waiting items with move/cancel, and finished items", () => {
    const items: QueueItem[] = [
      makeItem({
        id: "active-yt",
        status: "downloading",
        request: {
          sourceType: "youtube",
          sourcePathOrUrl: "https://youtube.com/watch?v=ghost",
          title: "GHOST",
          artist: "Camellia",
          coverPath: null,
          config: PACK_CONFIG,
        },
        startedAt: "2026-11-24T12:01:00.000Z",
        progress: { stage: "downloading", pct: 42 },
      }),
      makeItem({
        id: "wait-1",
        status: "waiting",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\spin.mp3",
          title: "Spin Eternally",
          artist: "Camellia",
          coverPath: null,
          config: PACK_CONFIG,
        },
      }),
      makeItem({
        id: "wait-2",
        status: "waiting",
        request: {
          sourceType: "cached",
          sourcePathOrUrl: "C:\\cache\\reality.wav",
          title: "Reality Check",
          artist: "DM DOKURO",
          coverPath: null,
          config: CUSTOM_CONFIG,
          regenerateId: "lib-1",
        },
      }),
      makeItem({
        id: "done-old",
        status: "done",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\old.mp3",
          title: "Older Finished",
          artist: "Artist A",
          coverPath: null,
          config: PACK_CONFIG,
        },
        finishedAt: "2026-11-24T12:00:10.000Z",
        result: makeResult({ title: "Older Finished", artist: "Artist A" }),
      }),
      makeItem({
        id: "done-new",
        status: "done",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\new.mp3",
          title: "Newer Finished",
          artist: "Artist B",
          coverPath: null,
          config: PACK_CONFIG,
        },
        finishedAt: "2026-11-24T12:05:00.000Z",
        result: makeResult({ title: "Newer Finished", artist: "Artist B" }),
      }),
    ];

    render(
      <QueuePanel
        items={items}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onMove={vi.fn()}
        onClearFinished={vi.fn()}
      />,
    );

    const nowSection = screen.getByTestId("queue-section-now");
    const nextSection = screen.getByTestId("queue-section-next");
    const finishedSection = screen.getByTestId("queue-section-finished");

    // Verify section ordering in DOM
    expect(
      nowSection.compareDocumentPosition(nextSection) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      nextSection.compareDocumentPosition(finishedSection) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    // Active YouTube item shows download stage, stage label, percentage, and progress bar
    expect(within(nowSection).getByText("GHOST")).toBeInTheDocument();
    expect(within(nowSection).getByText(EN.queue.stages.downloading)).toBeInTheDocument();
    expect(within(nowSection).getByText(/42%/)).toBeInTheDocument();
    expect(within(nowSection).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(within(nowSection).getByTestId("stage-step-downloading")).toBeInTheDocument();

    // Waiting items show index, source badge, and bounded move buttons
    expect(within(nextSection).getByText("#1")).toBeInTheDocument();
    expect(within(nextSection).getByText("#2")).toBeInTheDocument();
    expect(within(nextSection).getByText("Spin Eternally")).toBeInTheDocument();
    expect(within(nextSection).getByText("Reality Check")).toBeInTheDocument();
    expect(within(nextSection).getByText("File")).toBeInTheDocument();
    expect(within(nextSection).getByText("Regenerate")).toBeInTheDocument();

    const upButtons = within(nextSection).getAllByRole("button", { name: /move up/i });
    const downButtons = within(nextSection).getAllByRole("button", { name: /move down/i });
    expect(upButtons[0]).toBeDisabled();
    expect(upButtons[1]).not.toBeDisabled();
    expect(downButtons[0]).not.toBeDisabled();
    expect(downButtons[1]).toBeDisabled();

    // Finished section orders newest first
    const finishedTitles = within(finishedSection)
      .getAllByTestId("queue-finished-title")
      .map((el) => el.textContent);
    expect(finishedTitles).toEqual(["Newer Finished", "Older Finished"]);
  });

  it("hides download stage step for non-YouTube active items and disables cancel when cancelling", () => {
    const items: QueueItem[] = [
      makeItem({
        id: "active-file",
        status: "running",
        cancelling: true,
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\track.mp3",
          title: "Local Track",
          artist: "Artist",
          coverPath: null,
          config: PACK_CONFIG,
        },
        progress: { stage: "notes", pct: 55 },
      }),
    ];

    render(
      <QueuePanel
        items={items}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onMove={vi.fn()}
        onClearFinished={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("stage-step-downloading")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancelling/i })).toBeDisabled();
  });

  it("invokes cancel, retry, move up/down, and clear finished IPC actions", () => {
    const onCancel = vi.fn();
    const onRetry = vi.fn();
    const onMove = vi.fn();
    const onClearFinished = vi.fn();

    const items: QueueItem[] = [
      makeItem({
        id: "active-1",
        status: "running",
        progress: { stage: "notes", pct: 30 },
      }),
      makeItem({ id: "wait-1", status: "waiting" }),
      makeItem({ id: "wait-2", status: "waiting" }),
      makeItem({
        id: "fail-1",
        status: "failed",
        errorCode: "download_failed",
        finishedAt: "2026-11-24T12:02:00.000Z",
      }),
      makeItem({
        id: "cancel-1",
        status: "cancelled",
        finishedAt: "2026-11-24T12:01:00.000Z",
      }),
    ];

    render(
      <QueuePanel
        items={items}
        onCancel={onCancel}
        onRetry={onRetry}
        onMove={onMove}
        onClearFinished={onClearFinished}
      />,
    );

    const nowSection = screen.getByTestId("queue-section-now");
    fireEvent.click(within(nowSection).getByRole("button", { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalledWith("active-1");

    const nextSection = screen.getByTestId("queue-section-next");
    const nextCancelButtons = within(nextSection).getAllByRole("button", { name: /cancel/i });
    fireEvent.click(nextCancelButtons[0]);
    expect(onCancel).toHaveBeenCalledWith("wait-1");

    const upButtons = within(nextSection).getAllByRole("button", { name: /move up/i });
    const downButtons = within(nextSection).getAllByRole("button", { name: /move down/i });
    fireEvent.click(upButtons[1]);
    expect(onMove).toHaveBeenCalledWith("wait-2", "up");
    fireEvent.click(downButtons[0]);
    expect(onMove).toHaveBeenCalledWith("wait-1", "down");

    const finishedSection = screen.getByTestId("queue-section-finished");
    const retryButtons = within(finishedSection).getAllByRole("button", { name: /retry/i });
    fireEvent.click(retryButtons[0]);
    expect(onRetry).toHaveBeenCalledWith("fail-1");
    fireEvent.click(retryButtons[1]);
    expect(onRetry).toHaveBeenCalledWith("cancel-1");

    fireEvent.click(within(finishedSection).getByRole("button", { name: /clear finished/i }));
    expect(onClearFinished).toHaveBeenCalled();
  });

  it("expands done item details with per-level walls/bombs table, warnings, preview, and masks internal errors with Open Logs", async () => {
    const openPath = vi.fn();
    const exportMapZip = vi.fn().mockResolvedValue("C:\\CL\\Song.zip");
    const openLogs = vi.fn();
    const onPreview = vi.fn();
    (window as unknown as { bsm: unknown }).bsm = {
      openPath,
      exportMapZip,
      openLogs,
    };

    const level = (rank: string, walls: number, bombs: number) => ({
      rank,
      target_nps: 4.0,
      n_notes: 300,
      nps: 4.0,
      n_arcs: 0,
      n_chains: 0,
      n_walls: walls,
      n_bombs: bombs,
    });

    const items: QueueItem[] = [
      makeItem({
        id: "done-multi",
        status: "done",
        finishedAt: "2026-11-24T12:03:00.000Z",
        result: makeResult({
          folder: "C:\\CL\\BSM - Song",
          rank: "ExpertPlus",
          levels: [
            level("Hard", 58, 52),
            level("Expert", 67, 17),
            level("ExpertPlus", 24, 14),
          ],
          n_walls: 24,
          n_bombs: 14,
          warnings: ["missing ornaments checkpoint"],
          used_rule_lights: true,
          grid_steady: false,
        }),
      }),
      makeItem({
        id: "fail-internal",
        status: "failed",
        errorCode: "internal",
        finishedAt: "2026-11-24T12:02:00.000Z",
      }),
      makeItem({
        id: "fail-interrupted",
        status: "failed",
        errorCode: "interrupted",
        finishedAt: "2026-11-24T12:01:00.000Z",
      }),
    ];

    render(
      <QueuePanel
        items={items}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onMove={vi.fn()}
        onClearFinished={vi.fn()}
        onPreview={onPreview}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /preview/i }));
    expect(onPreview).toHaveBeenCalledWith("C:\\CL\\BSM - Song");

    fireEvent.click(screen.getByRole("button", { name: /open folder/i }));
    expect(openPath).toHaveBeenCalledWith("C:\\CL\\BSM - Song");

    fireEvent.click(screen.getByRole("button", { name: /export zip/i }));
    expect(exportMapZip).toHaveBeenCalledWith("C:\\CL\\BSM - Song", "Song");
    await waitFor(() => {
      expect(screen.getByText(/zip archive saved/i)).toBeInTheDocument();
    });

    // Expand details
    fireEvent.click(screen.getByRole("button", { name: /details/i }));

    const rows = screen.getByTestId("done-levels-table").querySelectorAll("tbody tr");
    expect(rows).toHaveLength(3);
    expect(rows[0].textContent).toContain("58");
    expect(rows[0].textContent).toContain("52");
    expect(rows[1].textContent).toContain("67");
    expect(rows[1].textContent).toContain("17");
    expect(screen.getAllByText("Walls")).toHaveLength(1);

    expect(screen.getByText(/layer models/i)).toBeInTheDocument();
    expect(screen.getByText(/rule-based lights were used/i)).toBeInTheDocument();
    expect(screen.getByText(/unsteady song tempo/i)).toBeInTheDocument();

    // Masked internal error + Open Logs, plus interrupted message
    expect(screen.getByText(ERRORS_EN.internal)).toBeInTheDocument();
    expect(screen.getByText(ERRORS_EN.interrupted)).toBeInTheDocument();
    const openLogsButtons = screen.getAllByRole("button", { name: /open logs/i });
    expect(openLogsButtons).toHaveLength(1);
    fireEvent.click(openLogsButtons[0]);
    expect(openLogs).toHaveBeenCalled();
  });

  it("does not re-call readCoverDataUrl across progress updates when it returns null, and loads coverCachePath once done", async () => {
    const readCoverDataUrl = vi.fn(async (filePath: string) => {
      if (filePath.includes("media")) {
        return "data:image/png;base64,cached-cover";
      }
      return null;
    });
    (window as unknown as { bsm: unknown }).bsm = { readCoverDataUrl };

    const baseItem = makeItem({
      id: "cover-1",
      status: "running",
      request: {
        sourceType: "file",
        sourcePathOrUrl: "C:\\audio\\song.mp3",
        title: "Cover Song",
        artist: "Artist",
        coverPath: "C:\\outside\\cover.jpg",
        config: PACK_CONFIG,
      },
      progress: { stage: "notes", pct: 10 },
    });

    const props = {
      onCancel: vi.fn(),
      onRetry: vi.fn(),
      onMove: vi.fn(),
      onClearFinished: vi.fn(),
    };

    const { rerender } = render(<QueuePanel items={[baseItem]} {...props} />);

    await waitFor(() => {
      expect(readCoverDataUrl).toHaveBeenCalledTimes(1);
    });
    expect(readCoverDataUrl).toHaveBeenCalledWith("C:\\outside\\cover.jpg");

    rerender(
      <QueuePanel
        items={[{ ...baseItem, progress: { stage: "notes", pct: 25 } }]}
        {...props}
      />,
    );
    rerender(
      <QueuePanel
        items={[{ ...baseItem, progress: { stage: "layers", pct: 65 } }]}
        {...props}
      />,
    );

    // Flush microtasks to verify no extra IPC calls were triggered on progress ticks.
    await Promise.resolve();
    expect(readCoverDataUrl).toHaveBeenCalledTimes(1);

    const entry = {
      id: "entry-1",
      title: "Cover Song",
      artist: "Artist",
      folder: "C:\\CL\\Cover Song",
      createdAt: "2026-11-24T12:05:00.000Z",
      bpm: 120,
      levels: [],
      nWalls: 0,
      nBombs: 0,
      nLights: 20,
      environment: "DefaultEnvironment",
      audioCachePath: "C:\\userData\\media\\entry-1\\audio.mp3",
      coverCachePath: "C:\\userData\\media\\entry-1\\cover.jpg",
      config: PACK_CONFIG,
    } as LibraryEntry;

    rerender(
      <QueuePanel
        items={[
          {
            ...baseItem,
            status: "done",
            finishedAt: "2026-11-24T12:05:00.000Z",
            progress: { stage: "writing", pct: 100 },
            result: makeResult({ title: "Cover Song", artist: "Artist" }),
            entry,
          },
        ]}
        {...props}
      />,
    );

    await waitFor(() => {
      expect(readCoverDataUrl).toHaveBeenCalledTimes(2);
    });
    expect(readCoverDataUrl).toHaveBeenLastCalledWith(
      "C:\\userData\\media\\entry-1\\cover.jpg",
    );
    await waitFor(() => {
      expect(screen.getByRole("img", { name: "Cover Song" })).toHaveAttribute(
        "src",
        "data:image/png;base64,cached-cover",
      );
    });
  });

  it("renders default pack difficulties, prioritizes custom NPS over levels, and falls back empty finished artist/title", () => {
    const items: QueueItem[] = [
      makeItem({
        id: "pack-null-levels",
        status: "waiting",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\a.mp3",
          title: "Pack Default Levels",
          artist: "Artist A",
          coverPath: null,
          config: { ...PACK_CONFIG, mode: "pack", levels: null },
        },
      }),
      makeItem({
        id: "custom-nps-and-levels",
        status: "waiting",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\b.mp3",
          title: "Custom NPS Priority",
          artist: "Artist B",
          coverPath: null,
          config: { ...CUSTOM_CONFIG, mode: "custom", levels: ["Expert"], nps: 4.5 },
        },
      }),
      makeItem({
        id: "done-empty-meta",
        status: "done",
        request: {
          sourceType: "file",
          sourcePathOrUrl: "C:\\audio\\c.mp3",
          title: "Request Title",
          artist: "",
          coverPath: null,
          config: PACK_CONFIG,
        },
        finishedAt: "2026-11-24T12:01:00.000Z",
        result: makeResult({ title: "", artist: "" }),
      }),
    ];

    render(
      <QueuePanel
        items={items}
        onCancel={vi.fn()}
        onRetry={vi.fn()}
        onMove={vi.fn()}
        onClearFinished={vi.fn()}
      />,
    );

    const nextSection = screen.getByTestId("queue-section-next");
    expect(within(nextSection).getByText("Hard")).toBeInTheDocument();
    expect(within(nextSection).getAllByText("Expert")).toHaveLength(1);
    expect(within(nextSection).getByText("Expert+")).toBeInTheDocument();
    expect(within(nextSection).getByText("4.5 NPS")).toBeInTheDocument();

    const finishedSection = screen.getByTestId("queue-section-finished");
    expect(within(finishedSection).getByText("Request Title")).toBeInTheDocument();
    expect(within(finishedSection).getByText(EN.common.unknownArtist)).toBeInTheDocument();
  });
});
