import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FirstRun } from "./FirstRun";

describe("FirstRun", () => {
  it("downloads engine pack when needed and blocks CustomLevels step until valid folder selected", async () => {
    let packProgressCb: ((ev: unknown) => void) | null = null;
    const installEnginePack = vi.fn(async () => {
      packProgressCb?.({
        stage: "downloading",
        pct: 50,
        bytesDownloaded: 50,
        totalBytes: 100,
        partIndex: 1,
        partCount: 2,
      });
    });
    const pickFolder = vi.fn(async () => "C:\\Games\\Beat Saber");
    const validateCustomLevels = vi.fn(async () => ({
      valid: true,
      resolvedPath: "C:\\Games\\Beat Saber\\Beat Saber_Data\\CustomLevels",
    }));
    const saveSettings = vi.fn(async (p: { customLevels: string }) => ({
      customLevels: p.customLevels,
      device: "auto" as const,
      exportDir: "D:\\Dl",
      engineVersionInstalled: "1",
    }));
    const getInfo = vi.fn(async () => ({
      version: "0.1.0",
      cuda_available: true,
      gpu_name: "RTX 4060 Ti",
      active_device: "cuda",
      detected_custom_levels: null,
      has_layer_models: false,
      missing_layer_models: [],
      environments: [],
      styles: [],
    }));
    (window as unknown as { bsm: unknown }).bsm = {
      installEnginePack,
      pickFolder,
      validateCustomLevels,
      saveSettings,
      getInfo,
      onEnginePackProgress: (cb: (ev: unknown) => void) => {
        packProgressCb = cb;
        return () => {};
      },
    };
    const onComplete = vi.fn();
    render(
      <FirstRun
        startup={{
          needsEnginePack: true,
          engineVersion: "1",
          appVersion: "0.1.0",
          settings: {
            customLevels: "",
            device: "auto",
            exportDir: "D:\\Dl",
            engineVersionInstalled: null,
          },
          info: null,
        }}
        onComplete={onComplete}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /download engine/i }));
    await waitFor(() => expect(installEnginePack).toHaveBeenCalled());
    await waitFor(() => expect(getInfo).toHaveBeenCalled());

    const cont = screen.getByRole("button", { name: /next/i });
    expect(cont).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /choose folder/i }));
    await waitFor(() => expect(validateCustomLevels).toHaveBeenCalledWith("C:\\Games\\Beat Saber"));
    await waitFor(() => expect(cont).not.toBeDisabled());
    fireEvent.click(cont);
    await waitFor(() =>
      expect(saveSettings).toHaveBeenCalledWith({
        customLevels: "C:\\Games\\Beat Saber\\Beat Saber_Data\\CustomLevels",
      }),
    );
    expect(onComplete).toHaveBeenCalled();
  });
});
