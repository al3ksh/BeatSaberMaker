import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { LibraryEntry } from "../types";
import { LibraryView } from "./LibraryView";

function makeEntry(id: string, title: string, artist = "Camellia"): LibraryEntry {
  return {
    id,
    title,
    artist,
    folder: `C:\\CL\\BSM - ${title}`,
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
      mode: "pack",
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
}

describe("LibraryView", () => {
  it("lists cards and confirms before deleting", async () => {
    const entry = makeEntry("e1", "GHOST");
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

    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));
    expect(onRegenerate).toHaveBeenCalledWith(entry);

    fireEvent.click(screen.getByRole("button", { name: /export zip/i }));
    await waitFor(() => expect(exportMapZip).toHaveBeenCalledWith(entry.folder, entry.title));

    fireEvent.click(screen.getByRole("button", { name: /delete/i }));
    expect(deleteLibraryEntry).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
    await waitFor(() => expect(deleteLibraryEntry).toHaveBeenCalledWith("e1"));
  });

  it("filters by search query and paginates large libraries", async () => {
    const entries = Array.from({ length: 8 }, (_, idx) =>
      makeEntry(`e${idx + 1}`, `Track ${idx + 1}`, idx === 7 ? "SpecialArtist" : "Artist"),
    );
    (window as unknown as { bsm: unknown }).bsm = {
      listLibrary: vi.fn(async () => entries),
      deleteLibraryEntry: vi.fn(async () => entries),
      exportMapZip: vi.fn(),
      readCoverDataUrl: vi.fn(async () => null),
    };
    render(<LibraryView onRegenerate={vi.fn()} />);

    expect(await screen.findByText("Track 1")).toBeInTheDocument();
    expect(screen.queryByText("Track 7")).not.toBeInTheDocument();
    expect(screen.getByText(/Page 1 of 2/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByText("Track 7")).toBeInTheDocument();
    expect(screen.queryByText("Track 1")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText(/filter by title/i), {
      target: { value: "SpecialArtist" },
    });
    expect(screen.getByText("Track 8")).toBeInTheDocument();
    expect(screen.queryByText("Track 7")).not.toBeInTheDocument();
  });
});
