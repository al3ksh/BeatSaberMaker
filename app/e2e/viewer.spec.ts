import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const APP_DIR = path.resolve(__dirname, "..");
const FAKE_ENGINE = path.resolve(__dirname, "fake-engine.mjs");
const SHOTS_DIR = path.resolve(__dirname, "viewer-shots");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const electronExe = require("electron") as string;

const CUSTOM_LEVELS =
  "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Beat Saber\\Beat Saber_Data\\CustomLevels";
const MAP_SHUT_UP = path.join(CUSTOM_LEVELS, "BSM - Shut Up and Dance [nps 3, unknown]");
const MAP_GHOST = path.join(
  CUSTOM_LEVELS,
  "BSM - Camellia - GHOST (from Cyphisonia E.P.) [nps 6, unknown]",
);

async function launchViewerApp(customLevelsDir: string) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-viewer-"));
  const userData = path.join(tmp, "userData");
  fs.mkdirSync(userData, { recursive: true });

  const electronApp = await electron.launch({
    executablePath: electronExe,
    args: [APP_DIR],
    env: {
      ...process.env,
      BSM_USER_DATA_DIR: userData,
      BSM_SKIP_ENGINE_PACK: "1",
      BSM_DISABLE_UPDATER: "1",
      BSM_MUTE_AUDIO: "1",
      BSM_ENGINE_CMD: process.execPath,
      BSM_ENGINE_ARGS: JSON.stringify([FAKE_ENGINE]),
      BSM_FAKE_DETECTED_CL: customLevelsDir,
    },
  });
  const page = await electronApp.firstWindow();
  await page.setViewportSize({ width: 1280, height: 800 });
  return { tmp, electronApp, page };
}

test.describe("3D Map Viewer E2E", () => {
  test("opens synthetic map folder, tests YouTube seek overlay, hitsound/quality toggles, cameras, and playback", async () => {
    const tmpMap = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-viewer-synth-"));
    const cl = path.join(tmpMap, "CustomLevels");
    const mapDir = path.join(tmpMap, "SampleMap");
    fs.mkdirSync(cl, { recursive: true });
    fs.mkdirSync(mapDir, { recursive: true });

    fs.writeFileSync(
      path.join(mapDir, "Info.dat"),
      JSON.stringify({
        _version: "2.1.0",
        _songName: "Viewer Synth Test",
        _songAuthorName: "Tester",
        _beatsPerMinute: 120,
        _songTimeOffset: 0,
        _songFilename: "song.egg",
        _environmentName: "DefaultEnvironment",
        _difficultyBeatmapSets: [
          {
            _beatmapCharacteristicName: "Standard",
            _difficultyBeatmaps: [
              {
                _difficulty: "Hard",
                _difficultyRank: 5,
                _beatmapFilename: "HardStandard.dat",
                _noteJumpMovementSpeed: 14,
                _noteJumpStartBeatOffset: 0,
              },
              {
                _difficulty: "Expert",
                _difficultyRank: 7,
                _beatmapFilename: "ExpertStandard.dat",
                _noteJumpMovementSpeed: 16,
                _noteJumpStartBeatOffset: 0,
              },
            ],
          },
        ],
      }),
    );

    const diffPayload = (extraNotes: number) =>
      JSON.stringify({
        version: "3.3.0",
        colorNotes: [
          { b: 10, x: 1, y: 0, c: 0, d: 1, a: 0 },
          { b: 10.5, x: 2, y: 1, c: 1, d: 0, a: 0 },
          { b: 11, x: 0, y: 2, c: 0, d: 8, a: 45 },
          ...Array.from({ length: extraNotes }, (_, idx) => ({
            b: 12 + idx * 0.5,
            x: idx % 4,
            y: idx % 3,
            c: idx % 2,
            d: idx % 9,
            a: 0,
          })),
        ],
        bombNotes: [{ b: 11.5, x: 3, y: 0 }],
        obstacles: [{ b: 10.2, x: 0, y: 2, d: 2, w: 2, h: 3 }],
        sliders: [{ b: 10, c: 0, x: 1, y: 0, d: 1, mu: 1, tb: 12, tx: 2, ty: 1, tc: 0, tmu: 1, m: 0 }],
        burstSliders: [
          { b: 13, x: 2, y: 0, c: 1, d: 3, tb: 13.25, tx: 3, ty: 0, sc: 5, s: 1 },
        ],
        basicBeatmapEvents: [
          { b: 8, et: 0, i: 1, f: 1 },
          { b: 10, et: 1, i: 6, f: 1 },
          { b: 10, et: 2, i: 1, f: 0.9 },
          { b: 10, et: 3, i: 5, f: 0.9 },
          { b: 10, et: 4, i: 10, f: 1 },
        ],
      });

    fs.writeFileSync(path.join(mapDir, "HardStandard.dat"), diffPayload(0));
    fs.writeFileSync(path.join(mapDir, "ExpertStandard.dat"), diffPayload(5));

    const { tmp, electronApp, page } = await launchViewerApp(cl);
    try {
      await page.getByRole("navigation").getByRole("button", { name: "My Maps" }).click();
      await electronApp.evaluate(async ({ dialog }, picked) => {
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [picked] });
      }, mapDir);

      await page.getByRole("button", { name: /open map folder/i }).click();
      await expect(page.getByTestId("map-viewer")).toBeVisible();
      await expect(page.getByText("Viewer Synth Test")).toBeVisible();
      await expect(page.getByText("Notes: 8")).toBeVisible();

      // Switch difficulty to Hard (3 notes)
      await page.getByRole("button", { name: "Hard", exact: true }).click();
      await expect(page.getByText("Notes: 3")).toBeVisible();

      // Seek to 5.0 s (Beat 10.0 at 120 BPM)
      const seek = page.getByTestId("viewer-seek");
      await seek.fill("5");
      await seek.dispatchEvent("change");
      await expect(page.getByTestId("viewer-beat")).toContainText("Beat 10.0");

      // Hover tooltip on seek bar
      const box = await seek.boundingBox();
      if (box) {
        await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
        await expect(page.getByTestId("viewer-seek-tooltip")).toBeVisible();
      }

      // YouTube-style ±5s arrow seek + accumulation
      await page.keyboard.press("ArrowRight");
      await expect(page.getByTestId("viewer-seek-overlay")).toContainText("5 seconds »");
      // Accumulation needs the second press inside the 0.6 s window. On a slow CI runner two
      // separate key presses can land further apart, so let the overlay expire and then fire
      // both presses in one tick.
      await expect(page.getByTestId("viewer-seek-overlay")).toBeHidden();
      await page.evaluate(() => {
        for (let i = 0; i < 2; i++) {
          window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowRight", key: "ArrowRight" }));
        }
      });
      await expect(page.getByTestId("viewer-seek-overlay")).toContainText("10 seconds »");
      await page.keyboard.press("ArrowLeft");
      await expect(page.getByTestId("viewer-seek-overlay")).toContainText("« 5 seconds");

      // Hitsound toggle, volume sliders percentage readout (default 40%/40%), & Quality toggle
      await expect(page.getByTestId("viewer-volume-percent")).toHaveText("40%");
      await expect(page.getByTestId("viewer-hitsound-percent")).toHaveText("40%");
      const musicSlider = page.getByTestId("viewer-volume-slider");
      await musicSlider.fill("35");
      await musicSlider.dispatchEvent("input");
      await expect(page.getByTestId("viewer-volume-percent")).toHaveText("35%");
      const hitsoundSlider = page.getByTestId("viewer-hitsound-volume");
      await hitsoundSlider.fill("65");
      await hitsoundSlider.dispatchEvent("input");
      await expect(page.getByTestId("viewer-hitsound-percent")).toHaveText("65%");

      const hitsoundToggle = page.getByTestId("viewer-hitsound-toggle");
      await expect(hitsoundToggle).toHaveAttribute("aria-pressed", "true");
      await hitsoundToggle.click();
      await expect(hitsoundToggle).toHaveAttribute("aria-pressed", "false");
      await expect(page.getByTestId("viewer-hitsound-percent")).toHaveText("0%");
      await hitsoundToggle.click();
      await expect(page.getByTestId("viewer-hitsound-percent")).toHaveText("65%");

      const qualityToggle = page.getByTestId("viewer-quality-toggle");
      await expect(qualityToggle).toContainText("High");
      await qualityToggle.click();
      await expect(qualityToggle).toContainText("Fast");
      await qualityToggle.click();

      // Camera presets
      await page.getByRole("button", { name: "Side" }).click();
      await page.getByRole("button", { name: "Top" }).click();
      await page.getByRole("button", { name: "Player" }).click();

      // Speed & play/pause toggle with Spacebar + centered flash
      await page.getByRole("button", { name: "1.5x" }).click();
      await page.keyboard.press("Space");
      await expect(page.getByTestId("viewer-play-flash")).toBeVisible();
      await page.waitForTimeout(350);
      await page.keyboard.press("Space");

      const timeText = await page.getByTestId("viewer-time").innerText();
      expect(timeText).not.toMatch(/^0:05\.0/);

      // Back button closes viewer
      await page.getByRole("button", { name: "Back", exact: true }).click();
      await expect(page.getByTestId("map-viewer")).toBeHidden();
    } finally {
      await electronApp.close();
      fs.rmSync(tmp, { recursive: true, force: true });
      fs.rmSync(tmpMap, { recursive: true, force: true });
    }
  });

  test("renders real CustomLevels maps (Shut Up and Dance + GHOST), captures screenshots, and checks FPS", async () => {
    const hasMaps = fs.existsSync(MAP_SHUT_UP) && fs.existsSync(MAP_GHOST);
    test.skip(!hasMaps, "Real CustomLevels sample maps are not present on this machine");

    fs.mkdirSync(SHOTS_DIR, { recursive: true });
    const tmpCl = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-viewer-cl-"));
    const { tmp, electronApp, page } = await launchViewerApp(tmpCl);

    try {
      await page.getByRole("navigation").getByRole("button", { name: "My Maps" }).click();

      // 1. Shut Up and Dance (Hard, BPM ~128, NJS 14)
      await electronApp.evaluate(async ({ dialog }, picked) => {
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [picked] });
      }, MAP_SHUT_UP);
      await page.getByRole("button", { name: /open map folder/i }).click();
      await expect(page.getByTestId("map-viewer")).toBeVisible();
      await expect(page.getByText("Shut Up and Dance")).toBeVisible();
      await expect(page.getByText(/Notes:\s*623/)).toBeVisible();

      for (const sec of [1.2, 10.6, 26, 30, 60, 90]) {
        const seek = page.getByTestId("viewer-seek");
        await seek.fill(String(sec));
        await seek.dispatchEvent("change");
        await page.waitForTimeout(400);
        const tag =
          sec === 1.2
            ? "1.2s-walls-bombs"
            : sec === 10.6
              ? "10.6s-arc"
              : `${sec}s-player`;
        await page.screenshot({ path: path.join(SHOTS_DIR, `shutup-${tag}.png`) });
      }
      await page.getByRole("button", { name: "Side" }).click();
      await page.waitForTimeout(350);
      await page.screenshot({ path: path.join(SHOTS_DIR, "shutup-90s-side.png") });
      await page.getByRole("button", { name: "Top" }).click();
      await page.waitForTimeout(350);
      await page.screenshot({ path: path.join(SHOTS_DIR, "shutup-90s-top.png") });

      // 2. Camellia - GHOST (ExpertPlus, 1831 notes)
      await electronApp.evaluate(async ({ dialog }, picked) => {
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [picked] });
      }, MAP_GHOST);
      await page.getByTestId("map-viewer").getByRole("button", { name: /open map folder/i }).click();
      await expect(page.getByText(/GHOST/)).toBeVisible();
      await expect(page.getByText(/Notes:\s*1831/)).toBeVisible();

      await page.getByRole("button", { name: "Player" }).click();
      for (const sec of [12.8, 30, 60, 90]) {
        const seek = page.getByTestId("viewer-seek");
        await seek.fill(String(sec));
        await seek.dispatchEvent("change");
        await page.waitForTimeout(400);
        const tag = sec === 12.8 ? "12.8s-arcs-walls" : `${sec}s-player`;
        await page.screenshot({ path: path.join(SHOTS_DIR, `ghost-${tag}.png`) });
      }

      // Play GHOST around 88s (dense section) and sample FPS
      const seek = page.getByTestId("viewer-seek");
      await seek.fill("88");
      await seek.dispatchEvent("change");
      await page.keyboard.press("Space");
      await page.waitForTimeout(1600);
      const ghostFps = await page.evaluate(
        () => (window as unknown as { __bsmViewerFps?: number }).__bsmViewerFps ?? 0,
      );
      await page.screenshot({ path: path.join(SHOTS_DIR, "ghost-playing-cut.png") });
      await page.keyboard.press("Space");
      console.log(`[viewer-e2e] GHOST Expert+ High quality FPS: ${ghostFps}`);
      expect(ghostFps).toBeGreaterThan(0);
    } finally {
      await electronApp.close();
      fs.rmSync(tmp, { recursive: true, force: true });
      fs.rmSync(tmpCl, { recursive: true, force: true });
    }
  });
});
