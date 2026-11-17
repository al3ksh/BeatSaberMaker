import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SettingsButton } from "./SettingsButton";

describe("SettingsButton", () => {
  it.each(["idle", "not-available", "checking", "error"] as const)("no dot for %s", (s) => {
    render(<SettingsButton updateState={s} onClick={() => {}} />);
    expect(screen.queryByTestId("settings-update-dot")).toBeNull();
  });
  it("blue pulsing dot while downloading", () => {
    render(<SettingsButton updateState="downloading" onClick={() => {}} />);
    const dot = screen.getByTestId("settings-update-dot");
    expect(dot.className).toContain("bg-blue-500");
    expect(dot.className).toContain("animate-pulse");
    expect(screen.getByRole("button", { name: /update downloading/i })).toBeInTheDocument();
  });
  it("green dot when downloaded", () => {
    render(<SettingsButton updateState="downloaded" onClick={() => {}} />);
    const dot = screen.getByTestId("settings-update-dot");
    expect(dot.className).toContain("bg-emerald-500");
    expect(dot.className).not.toContain("animate-pulse");
    expect(screen.getByRole("button", { name: /ready to install/i })).toBeInTheDocument();
  });
});
