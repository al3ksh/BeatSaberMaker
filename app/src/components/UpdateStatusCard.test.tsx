import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UpdateStatusCard } from "./UpdateStatusCard";
import type { UpdateStatus } from "../types";

const checkForUpdates = vi.fn(async () => {});
const installUpdate = vi.fn(async () => {});

function setup(status: UpdateStatus) {
  render(<UpdateStatusCard status={status} appVersion="0.1.0" />);
  return screen.getByTestId("update-card");
}

describe("UpdateStatusCard", () => {
  beforeEach(() => {
    checkForUpdates.mockClear();
    installUpdate.mockClear();
    (window as unknown as { bsm: unknown }).bsm = { checkForUpdates, installUpdate };
  });

  it("idle", () => {
    const card = setup({ state: "idle" });
    expect(card).toHaveAttribute("data-state", "idle");
    expect(screen.getByText("Updates")).toBeInTheDocument();
    expect(screen.getByText("Current version v0.1.0")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check for updates" }));
    expect(checkForUpdates).toHaveBeenCalled();
  });
  it("checking", () => {
    const card = setup({ state: "checking" });
    expect(card).toHaveAttribute("data-state", "checking");
    expect(screen.getByText("Checking for updates…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check for updates" })).toBeDisabled();
  });
  it("not-available", () => {
    const card = setup({ state: "not-available", version: "0.1.0" });
    expect(card).toHaveAttribute("data-state", "not-available");
    expect(screen.getByText("You're up to date")).toBeInTheDocument();
    expect(screen.getByText("v0.1.0 is the latest version")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    expect(checkForUpdates).toHaveBeenCalled();
  });
  it("available without version or pct", () => {
    const card = setup({ state: "available" });
    expect(card).toHaveAttribute("data-state", "available");
    expect(screen.getByText("New version found")).toBeInTheDocument();
    expect(screen.getByText("Downloading…")).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeNull();
  });
  it("downloading shows progress", () => {
    const card = setup({ state: "downloading", version: "0.1.1", pct: 42 });
    expect(card).toHaveAttribute("data-state", "downloading");
    expect(screen.getByText("Version 0.1.1 found")).toBeInTheDocument();
    expect(screen.getByText("Downloading… 42%")).toBeInTheDocument();
    expect(screen.getByTestId("update-progress")).toHaveStyle({ width: "42%" });
    expect(screen.queryByRole("button")).toBeNull();
  });
  it("downloaded", () => {
    const card = setup({ state: "downloaded", version: "0.1.1" });
    expect(card).toHaveAttribute("data-state", "downloaded");
    expect(screen.getByText("Version 0.1.1 is ready")).toBeInTheDocument();
    expect(screen.getByText("Restart the app to install it.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Install & restart" }));
    expect(installUpdate).toHaveBeenCalled();
  });
  it("error", () => {
    const card = setup({ state: "error" });
    expect(card).toHaveAttribute("data-state", "error");
    expect(screen.getByText("Couldn't check for updates")).toBeInTheDocument();
    expect(screen.getByText("Check your connection or open the logs.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(checkForUpdates).toHaveBeenCalled();
  });
});
