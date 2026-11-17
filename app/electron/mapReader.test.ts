import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readMapFolderForViewer } from "./mapReader";

describe("readMapFolderForViewer", () => {
  const dirs: string[] = [];

  afterEach(async () => {
    for (const d of dirs.splice(0)) {
      await fsp.rm(d, { recursive: true, force: true });
    }
  });

  it("reads Info.dat (with BOM), difficulty JSON, cover data URL, and audio bytes read-only", async () => {
    const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), "bsm-mapreader-"));
    dirs.push(tmp);

    const info = {
      _version: "2.1.0",
      _songName: "Test Song",
      _songAuthorName: "Test Artist",
      _levelAuthorName: "BeatSaberMaker",
      _beatsPerMinute: 128,
      _songTimeOffset: 0.1,
      _songFilename: "song.egg",
      _coverImageFilename: "cover.png",
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
              _noteJumpStartBeatOffset: -0.159,
            },
          ],
        },
      ],
    };

    await fsp.writeFile(path.join(tmp, "Info.dat"), "\uFEFF" + JSON.stringify(info), "utf8");
    await fsp.writeFile(
      path.join(tmp, "HardStandard.dat"),
      JSON.stringify({ version: "3.3.0", colorNotes: [{ b: 2, x: 1, y: 0, c: 0, d: 1, a: 0 }] }),
      "utf8",
    );
    await fsp.writeFile(path.join(tmp, "song.egg"), Buffer.from([1, 2, 3, 4]));
    await fsp.writeFile(path.join(tmp, "cover.png"), Buffer.from([0x89, 0x50, 0x4e, 0x47]));

    const beforeFiles = fs.readdirSync(tmp).sort();
    const pkg = await readMapFolderForViewer(tmp);
    const afterFiles = fs.readdirSync(tmp).sort();

    expect(afterFiles).toEqual(beforeFiles);
    expect(pkg.title).toBe("Test Song");
    expect(pkg.artist).toBe("Test Artist");
    expect(pkg.bpm).toBe(128);
    expect(pkg.songTimeOffset).toBe(0.1);
    expect(pkg.environment).toBe("DefaultEnvironment");
    expect(pkg.coverDataUrl).toMatch(/^data:image\/png;base64,/);
    expect(Array.from(pkg.audioBytes ?? [])).toEqual([1, 2, 3, 4]);
    expect(pkg.difficulties).toHaveLength(1);
    expect(pkg.difficulties[0].rank).toBe("Hard");
    expect(pkg.difficulties[0].njs).toBe(14);
    expect(pkg.difficulties[0].njsOffset).toBe(-0.159);
  });

  it("rejects path traversal filenames in Info.dat", async () => {
    const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), "bsm-mapreader-sec-"));
    dirs.push(tmp);

    const info = {
      _songName: "Evil",
      _beatsPerMinute: 120,
      _songFilename: "../secret.ogg",
      _difficultyBeatmapSets: [
        {
          _beatmapCharacteristicName: "Standard",
          _difficultyBeatmaps: [
            {
              _difficulty: "Expert",
              _difficultyRank: 7,
              _beatmapFilename: "../outside.dat",
              _noteJumpMovementSpeed: 16,
              _noteJumpStartBeatOffset: 0,
            },
          ],
        },
      ],
    };

    await fsp.writeFile(path.join(tmp, "info.dat"), JSON.stringify(info), "utf8");
    const pkg = await readMapFolderForViewer(tmp);

    expect(pkg.audioBytes).toBeNull();
    expect(pkg.difficulties).toHaveLength(0);
  });
});
