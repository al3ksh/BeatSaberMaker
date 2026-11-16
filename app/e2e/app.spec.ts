import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const APP_DIR = path.resolve(__dirname, "..");
const FAKE_ENGINE = path.resolve(__dirname, "fake-engine.mjs");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const electronExe = require("electron") as string;

async function launchApp(extraEnv: Record<string, string>) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-"));
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
      BSM_ENGINE_CMD: process.execPath,
      BSM_ENGINE_ARGS: JSON.stringify([FAKE_ENGINE]),
      ...extraEnv,
    },
  });
  const page = await electronApp.firstWindow();
  return { tmp, userData, electronApp, page };
}

test.describe("BeatSaberMaker Electron E2E", () => {
  test("1. first-run blocks until valid CustomLevels picked, and auto-detect skips gate", async () => {
    const tmp1 = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-fr-"));
    const bsRoot = path.join(tmp1, "Beat Saber");
    fs.mkdirSync(path.join(bsRoot, "Beat Saber_Data"), { recursive: true });

    const { electronApp, page } = await launchApp({ BSM_FAKE_DETECTED_CL: "" });
    try {
      await expect(page.getByText("First-Time Setup")).toBeVisible();
      const nextButton = page.getByRole("button", { name: "Next" });
      await expect(nextButton).toBeDisabled();

      await electronApp.evaluate(async ({ dialog }, picked) => {
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [picked] });
      }, bsRoot);

      await page.getByRole("button", { name: "Choose Folder" }).click();
      await expect(nextButton).toBeEnabled();
      await nextButton.click();

      await expect(page.getByPlaceholder(/search youtube/i)).toBeVisible();
    } finally {
      await electronApp.close();
      fs.rmSync(tmp1, { recursive: true, force: true });
    }

    // With detected CustomLevels, FirstRun is skipped automatically
    const tmp2 = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-auto-"));
    const detectedCl = path.join(tmp2, "CustomLevels");
    fs.mkdirSync(detectedCl, { recursive: true });
    const app2 = await launchApp({ BSM_FAKE_DETECTED_CL: detectedCl });
    try {
      await expect(app2.page.getByPlaceholder(/search youtube/i)).toBeVisible();
    } finally {
      await app2.electronApp.close();
      fs.rmSync(tmp2, { recursive: true, force: true });
    }
  });

  test("2. happy path, ZIP export, regenerate collision (2) + trash old, and library delete", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-happy-"));
    const cl = path.join(tmp, "CustomLevels");
    const exportZip = path.join(tmp, "exports", "GHOST.zip");
    fs.mkdirSync(cl, { recursive: true });

    const { electronApp, page } = await launchApp({
      BSM_FAKE_DETECTED_CL: cl,
      BSM_TEST_EXPORT_PATH: exportZip,
    });

    try {
      const searchBox = page.getByPlaceholder(/search youtube/i);
      await searchBox.fill("Ghost");
      await searchBox.press("Enter");
      await page.getByText("Camellia - GHOST").click();
      await page.getByRole("button", { name: "Next" }).click();

      await page.getByRole("button", { name: "Generate Map" }).click();
      await expect(page.getByText("Map is in the game!")).toBeVisible();
      await expect(page.getByText(/layer models/i)).toBeVisible();

      const firstFolders = fs.readdirSync(cl);
      expect(firstFolders).toHaveLength(1);
      const firstFolderPath = path.join(cl, firstFolders[0]);

      // Switch to Library
      await page.getByRole("navigation").getByRole("button", { name: "My Maps" }).click();
      await expect(page.getByText("GHOST")).toBeVisible();

      // Export ZIP
      await page.getByRole("button", { name: "Export ZIP" }).click();
      await expect(page.getByText(/zip archive saved/i)).toBeVisible();
      expect(fs.existsSync(exportZip)).toBe(true);

      // Regenerate -> opens Step 2 with saved non-zero seed
      await page.getByRole("button", { name: "Regenerate" }).click();
      await expect(page.getByText(/moved to the recycle bin/i)).toBeVisible();
      const seedInput = page.locator("#bsm-seed-input");
      const savedSeed = Number(await seedInput.inputValue());
      expect(savedSeed).toBeGreaterThan(0);

      await page.getByRole("button", { name: "Generate Map" }).click();
      await expect(page.getByText("Map is in the game!")).toBeVisible();

      const afterRegenFolders = fs.readdirSync(cl);
      expect(afterRegenFolders).toHaveLength(1);
      expect(afterRegenFolders[0]).toContain("(2)");
      expect(fs.existsSync(firstFolderPath)).toBe(false);

      // Delete from Library
      await page.getByRole("navigation").getByRole("button", { name: "My Maps" }).click();
      await page.getByRole("button", { name: "Delete" }).click();
      await page.getByRole("button", { name: "Confirm" }).click();
      await expect(page.getByText(/haven't generated any maps yet/i)).toBeVisible();
      expect(fs.readdirSync(cl)).toHaveLength(0);
    } finally {
      await electronApp.close();
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  test("3. cancel mid-generation returns to Step 2 without broken entry", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-cancel-"));
    const cl = path.join(tmp, "CustomLevels");
    fs.mkdirSync(cl, { recursive: true });

    const { electronApp, page } = await launchApp({
      BSM_FAKE_DETECTED_CL: cl,
      BSM_FAKE_HANG_CANCEL: "1",
    });

    try {
      const searchBox = page.getByPlaceholder(/search youtube/i);
      await searchBox.fill("Ghost");
      await searchBox.press("Enter");
      await page.getByText("Camellia - GHOST").click();
      await page.getByRole("button", { name: "Next" }).click();

      await page.getByRole("button", { name: "Generate Map" }).click();
      await page.getByRole("button", { name: "Cancel" }).click();

      // Returned cleanly to Step 2
      await expect(page.getByRole("button", { name: "Generate Map" })).toBeVisible();
      await page.getByRole("button", { name: "My Maps" }).click();
      await expect(page.getByText(/haven't generated any maps yet/i)).toBeVisible();
    } finally {
      await electronApp.close();
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  test("4. internal engine error displays friendly message without Traceback", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "bsm-e2e-err-"));
    const cl = path.join(tmp, "CustomLevels");
    fs.mkdirSync(cl, { recursive: true });

    const { electronApp, page } = await launchApp({
      BSM_FAKE_DETECTED_CL: cl,
      BSM_FAKE_FAIL_CODE: "internal",
    });

    try {
      const searchBox = page.getByPlaceholder(/search youtube/i);
      await searchBox.fill("Ghost");
      await searchBox.press("Enter");
      await page.getByText("Camellia - GHOST").click();
      await page.getByRole("button", { name: "Next" }).click();

      await page.getByRole("button", { name: "Generate Map" }).click();
      await expect(page.getByText(/unexpected engine error/i)).toBeVisible();
      const bodyText = await page.locator("body").innerText();
      expect(bodyText).not.toMatch(/traceback|secret internal error/i);
    } finally {
      await electronApp.close();
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});
