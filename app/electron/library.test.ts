import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { isSafeBsmMapFolder, LibraryStore } from "./library";
import { SettingsStore } from "./store";

describe("isSafeBsmMapFolder and LibraryStore", () => {
  it("only allows deleting direct BSM - subfolders inside CustomLevels", () => {
    const cl = path.join("C:", "Games", "Beat Saber_Data", "CustomLevels");
    expect(isSafeBsmMapFolder(path.join(cl, "BSM - Song [Hard]"), cl)).toBe(true);
    expect(isSafeBsmMapFolder(path.join(cl, "Other Map"), cl)).toBe(false);
    expect(isSafeBsmMapFolder(path.join(cl, "sub", "BSM - Nested"), cl)).toBe(false);
    expect(isSafeBsmMapFolder(cl, cl)).toBe(false);
    expect(isSafeBsmMapFolder(path.join("C:", "Windows", "BSM - Fake"), cl)).toBe(false);
  });

  it("trashes only safe existing folders, drops missing folders silently, and swaps on regenerate", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-lib-"));
    const cl = path.join(tmp, "CustomLevels");
    const map1 = path.join(cl, "BSM - First [Hard, tech]");
    const map2 = path.join(cl, "BSM - First [Hard, tech] (2)");
    const outsider = path.join(tmp, "NotInCustomLevels");
    await fs.mkdir(map1, { recursive: true });
    await fs.mkdir(map2, { recursive: true });
    await fs.mkdir(outsider, { recursive: true });

    const trashed: string[] = [];
    const trashFn = async (p: string) => {
      trashed.push(p);
      await fs.rm(p, { recursive: true, force: true });
    };

    const store = new LibraryStore(path.join(tmp, "library.json"), path.join(tmp, "media"));
    const audioSrc = path.join(tmp, "song.wav");
    await fs.writeFile(audioSrc, "wav");
    const cached = await store.cacheMedia("id1", audioSrc, null);

    await store.saveEntry({
      id: "id1",
      title: "First",
      artist: "Art",
      folder: map1,
      createdAt: "2026-11-16T10:00:00Z",
      bpm: 120,
      levels: [{ rank: "Hard", nps: 3.5, nNotes: 200, nArcs: 5, nChains: 2 }],
      nWalls: 10,
      nBombs: 4,
      nLights: 50,
      environment: "DefaultEnvironment",
      audioCachePath: cached.audioCachePath,
      coverCachePath: cached.coverCachePath,
      config: {
        mode: "pack",
        levels: ["Hard", "Expert", "ExpertPlus"],
        nps: null,
        styles: [],
        ornaments: null,
        lights: 2,
        environment: "DefaultEnvironment",
        seed: 42,
        lightsRules: false,
      },
    });

    await store.replaceAfterRegenerate(
      "id1",
      {
        ...(await store.get("id1"))!,
        folder: map2,
      },
      cl,
      trashFn,
    );
    expect(trashed).toEqual([map1]);
    expect((await store.get("id1"))?.folder).toBe(map2);

    // Crafted outside folder must NOT be trashed
    await store.saveEntry({ ...(await store.get("id1"))!, id: "bad", folder: outsider });
    await store.deleteEntry("bad", cl, trashFn);
    expect(trashed).toEqual([map1]);

    // Missing map folder is removed from library without error
    await fs.rm(map2, { recursive: true, force: true });
    await store.deleteEntry("id1", cl, trashFn);
    expect(await store.list()).toEqual([]);

    // If old folder was deleted by user beforehand and regeneration lands at the exact same path,
    // replaceAfterRegenerate must NOT trash the newly generated folder!
    const samePath = path.join(cl, "BSM - SamePath");
    await fs.mkdir(samePath, { recursive: true });
    await store.saveEntry({
      id: "same1",
      title: "Same",
      artist: "Art",
      folder: samePath,
      createdAt: "2026-11-16T10:00:00Z",
      bpm: 120,
      levels: [{ rank: "Hard", nps: 3.5, nNotes: 200, nArcs: 5, nChains: 2 }],
      nWalls: 10,
      nBombs: 4,
      nLights: 50,
      environment: "DefaultEnvironment",
      audioCachePath: cached.audioCachePath,
      coverCachePath: cached.coverCachePath,
      config: {
        mode: "pack",
        levels: ["Hard", "Expert", "ExpertPlus"],
        nps: null,
        styles: [],
        ornaments: null,
        lights: 2,
        environment: "DefaultEnvironment",
        seed: 42,
        lightsRules: false,
      },
    });
    const trashedBeforeSame = trashed.length;
    await store.replaceAfterRegenerate(
      "same1",
      { ...(await store.get("same1"))!, folder: samePath },
      cl,
      trashFn,
    );
    expect(trashed.length).toBe(trashedBeforeSame);
    expect(await fs.stat(samePath)).toBeTruthy();

    // Symlink / junction inside CustomLevels must NEVER be followed/trashed
    const linkTarget = path.join(tmp, "RealTarget");
    const linkFolder = path.join(cl, "BSM - Junction");
    await fs.mkdir(linkTarget, { recursive: true });
    await fs.symlink(linkTarget, linkFolder, process.platform === "win32" ? "junction" : "dir");
    await store.saveEntry({ ...(await store.get("same1"))!, id: "sym1", folder: linkFolder });
    await store.deleteEntry("sym1", cl, trashFn);
    expect(trashed.length).toBe(trashedBeforeSame);
    expect(await fs.stat(linkTarget)).toBeTruthy();

    const sStore = new SettingsStore(path.join(tmp, "settings.json"), path.join(tmp, "Downloads"));
    const loaded = await sStore.load();
    expect(loaded.device).toBe("auto");
    await sStore.save({ customLevels: cl });
    expect((await sStore.load()).customLevels).toBe(cl);
  });
});
