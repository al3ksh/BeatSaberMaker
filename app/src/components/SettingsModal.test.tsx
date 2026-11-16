import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SettingsModal } from "./SettingsModal";

describe("SettingsModal", () => {
  it("shows CPU note when CUDA unavailable and wires save, updates, and logs", async () => {
    const saveSettings = vi.fn(async (p: { device?: "auto" | "cuda" | "cpu" }) => ({
      customLevels: "C:\\CL",
      device: p.device ?? "auto",
      exportDir: "C:\\Exp",
      engineVersionInstalled: "1",
    }));
    const openLogs = vi.fn(async () => {});
    const checkForUpdates = vi.fn(async () => {});
    (window as unknown as { bsm: unknown }).bsm = {
      saveSettings,
      openLogs,
      checkForUpdates,
      pickFolder: vi.fn(async () => null),
      validateCustomLevels: vi.fn(),
      getInfo: vi.fn(),
      installUpdate: vi.fn(),
    };
    render(
      <SettingsModal
        open
        settings={{
          customLevels: "C:\\CL",
          device: "auto",
          exportDir: "C:\\Exp",
          engineVersionInstalled: "1",
        }}
        info={{
          version: "0.1.0",
          cuda_available: false,
          gpu_name: null,
          active_device: "cpu",
          detected_custom_levels: null,
          has_layer_models: true,
          missing_layer_models: [],
          environments: [],
          styles: [],
        }}
        appVersion="0.1.0"
        updateStatus={{ state: "idle" }}
        onClose={() => {}}
        onSettingsSaved={() => {}}
      />,
    );
    expect(screen.getByText(/brak karty nvidia|wolniej/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /otwórz logi/i }));
    expect(openLogs).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /sprawdź aktualizacje/i }));
    expect(checkForUpdates).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /cpu/i }));
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith({ device: "cpu" }));
  });
});
