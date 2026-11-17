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
});
