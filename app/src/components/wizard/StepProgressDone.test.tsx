import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ERRORS_EN } from "../../i18n/en";
import { StepProgressDone } from "./StepProgressDone";

describe("StepProgressDone", () => {
  it("renders success with info warnings and masks internal errors without tracebacks", () => {
    const cancelActive = vi.fn();
    const openPath = vi.fn();
    const exportMapZip = vi.fn();
    const openLogs = vi.fn();
    (window as unknown as { bsm: unknown }).bsm = {
      cancelActive,
      openPath,
      exportMapZip,
      openLogs,
    };

    const { rerender } = render(
      <StepProgressDone
        hasDownloadStage
        status="running"
        progress={{ stage: "notes", pct: 45 }}
        result={null}
        errorCode={null}
        onReset={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(cancelActive).toHaveBeenCalled();

    rerender(
      <StepProgressDone
        hasDownloadStage={false}
        status="done"
        progress={{ stage: "writing", pct: 100 }}
        result={{
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
          warnings: ["missing ornaments checkpoint"],
          used_rule_lights: true,
          device_used: "cpu",
        }}
        errorCode={null}
        onReset={() => {}}
        onPreview={openPath}
      />,
    );
    expect(screen.getByText(/map is in the game/i)).toBeInTheDocument();
    expect(screen.getByText(/layer models/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /preview/i }));
    expect(openPath).toHaveBeenCalledWith("C:\\CL\\BSM - Song");
    fireEvent.click(screen.getByRole("button", { name: /open folder/i }));
    expect(openPath).toHaveBeenCalledWith("C:\\CL\\BSM - Song");

    rerender(
      <StepProgressDone
        hasDownloadStage={false}
        status="error"
        progress={null}
        result={null}
        errorCode="internal"
        onReset={() => {}}
      />,
    );
    expect(screen.getByText(ERRORS_EN.internal)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open logs/i }));
    expect(openLogs).toHaveBeenCalled();
  });

  it("shows walls and bombs per difficulty when the engine reports them", () => {
    (window as unknown as { bsm: unknown }).bsm = {};
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
    render(
      <StepProgressDone
        hasDownloadStage={false}
        status="done"
        progress={{ stage: "writing", pct: 100 }}
        result={{
          folder: "C:\\CL\\BSM - Song",
          title: "Song",
          artist: "Artist",
          bpm: 120,
          grid_steady: true,
          n_notes: 300,
          nps: 4.0,
          rank: "ExpertPlus",
          levels: [level("Hard", 58, 52), level("Expert", 67, 17), level("ExpertPlus", 24, 14)],
          n_walls: 24,
          n_bombs: 14,
          n_lights: 40,
          environment: "DefaultEnvironment",
          warnings: [],
          used_rule_lights: false,
          device_used: "cuda",
        }}
        errorCode={null}
        onReset={() => {}}
      />,
    );
    const rows = screen.getByTestId("done-levels-table").querySelectorAll("tbody tr");
    expect(rows).toHaveLength(3);
    expect(rows[0].textContent).toContain("58");
    expect(rows[0].textContent).toContain("52");
    expect(rows[1].textContent).toContain("67");
    expect(rows[1].textContent).toContain("17");
    // The single densest-level walls tile is gone in per-level mode
    expect(screen.getAllByText("Walls")).toHaveLength(1);
  });
});
