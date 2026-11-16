import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LibraryView } from "./LibraryView";

describe("LibraryView", () => {
  it("lists cards and confirms before deleting", async () => {
    const entry = {
      id: "e1",
      title: "GHOST",
      artist: "Camellia",
      folder: "C:\\CL\\BSM - GHOST",
      createdAt: "2026-11-16T12:00:00Z",
      bpm: 180,
      levels: [{ rank: "Hard", nps: 4.0, nNotes: 400, nArcs: 10, nChains: 4 }],
      nWalls: 12,
      nBombs: 3,
      nLights: 80,
      environment: "DefaultEnvironment",
      audioCachePath: "C:\\media\\audio.ogg",
      coverCachePath: null,
      config: {
        mode: "pack" as const,
        levels: ["Hard", "Expert", "ExpertPlus"],
        nps: null,
        styles: [],
        ornaments: null,
        lights: 2,
        environment: "DefaultEnvironment",
        seed: 777,
        lightsRules: false,
      },
    };
    const deleteLibraryEntry = vi.fn(async () => []);
    const exportMapZip = vi.fn(async () => "C:\\Exp\\GHOST.zip");
    (window as unknown as { bsm: unknown }).bsm = {
      listLibrary: vi.fn(async () => [entry]),
      deleteLibraryEntry,
      exportMapZip,
      readCoverDataUrl: vi.fn(async () => null),
    };
    const onRegenerate = vi.fn();
    render(<LibraryView onRegenerate={onRegenerate} />);
    expect(await screen.findByText("GHOST")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /wygeneruj ponownie/i }));
    expect(onRegenerate).toHaveBeenCalledWith(entry);

    fireEvent.click(screen.getByRole("button", { name: /eksportuj zip/i }));
    await waitFor(() => expect(exportMapZip).toHaveBeenCalledWith(entry.folder, entry.title));

    fireEvent.click(screen.getByRole("button", { name: /usuń/i }));
    expect(deleteLibraryEntry).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /potwierdź/i }));
    await waitFor(() => expect(deleteLibraryEntry).toHaveBeenCalledWith("e1"));
  });
});
