import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StepSong } from "./StepSong";

describe("StepSong", () => {
  it("searches YouTube, populates editable artist/title on pick, and enables Dalej", async () => {
    const searchYouTube = vi.fn(async () => [
      {
        id: "1",
        url: "https://youtu.be/1",
        title: "Camellia - GHOST",
        artist: "Camellia",
        track_title: "GHOST",
        channel: "Camellia",
        duration_s: 240,
        thumbnail_url: null,
      },
    ]);
    (window as unknown as { bsm: unknown }).bsm = {
      searchYouTube,
      resolveYouTube: vi.fn(),
      pickAudioFile: vi.fn(),
    };
    const onChange = vi.fn();
    const { rerender } = render(
      <StepSong
        song={{
          sourceType: "youtube",
          sourcePathOrUrl: "",
          title: "",
          artist: "",
          coverPath: null,
          thumbnailUrl: null,
        }}
        onChange={onChange}
        onNext={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /dalej/i })).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(/szukaj|link/i), { target: { value: "ghost" } });
    fireEvent.submit(screen.getByRole("search"));
    await waitFor(() => expect(searchYouTube).toHaveBeenCalledWith("ghost"));
    fireEvent.click(await screen.findByText("Camellia - GHOST"));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceType: "youtube",
        sourcePathOrUrl: "https://youtu.be/1",
        title: "GHOST",
        artist: "Camellia",
      }),
    );

    rerender(
      <StepSong
        song={{
          sourceType: "youtube",
          sourcePathOrUrl: "https://youtu.be/1",
          title: "GHOST",
          artist: "Camellia",
          coverPath: null,
          thumbnailUrl: null,
        }}
        onChange={onChange}
        onNext={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /dalej/i })).not.toBeDisabled();
  });
});
