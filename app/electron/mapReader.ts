import fsp from "node:fs/promises";
import path from "node:path";
import type { ViewerDifficultyRaw, ViewerMapPackage } from "./types";

function isPlainFilename(name: unknown): name is string {
  return (
    typeof name === "string" &&
    name.length > 0 &&
    name !== "." &&
    name !== ".." &&
    !/[/\\:\0]/.test(name)
  );
}

async function findCaseInsensitiveFile(folder: string, name: unknown): Promise<string | null> {
  if (!isPlainFilename(name)) return null;
  const exact = path.join(folder, name);
  try {
    const st = await fsp.stat(exact);
    if (st.isFile()) return exact;
  } catch {
    // Fall back to case-insensitive scan.
  }
  const lower = name.toLowerCase();
  const entries = await fsp.readdir(folder, { withFileTypes: true });
  for (const ent of entries) {
    if (ent.isFile() && ent.name.toLowerCase() === lower) {
      return path.join(folder, ent.name);
    }
  }
  return null;
}

async function readJsonFile(filePath: string): Promise<Record<string, unknown>> {
  const raw = await fsp.readFile(filePath, "utf8");
  const cleaned = raw.replace(/^\uFEFF/, "");
  const parsed: unknown = JSON.parse(cleaned);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Invalid JSON object in ${path.basename(filePath)}`);
  }
  return parsed as Record<string, unknown>;
}

function mimeForImageExt(ext: string): string | null {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".webp":
      return "image/webp";
    default:
      return null;
  }
}

const RANK_ORDER: Record<string, number> = {
  Easy: 1,
  Normal: 3,
  Hard: 5,
  Expert: 7,
  ExpertPlus: 9,
};

/** Read a Beat Saber map folder strictly read-only for the 3D viewer. */
export async function readMapFolderForViewer(folderPath: string): Promise<ViewerMapPackage> {
  const folder = path.resolve(folderPath);
  const stat = await fsp.stat(folder);
  if (!stat.isDirectory()) {
    throw new Error("Map path is not a directory");
  }

  const infoPath = await findCaseInsensitiveFile(folder, "Info.dat");
  if (!infoPath) {
    throw new Error("Info.dat not found in folder");
  }

  const info = await readJsonFile(infoPath);

  let title = "";
  let artist = "";
  let mapper = "";
  let bpm = 0;
  let songTimeOffset = 0;
  let environment = "DefaultEnvironment";
  let audioFilename: unknown = null;
  let coverFilename: unknown = null;

  interface DiffSpec {
    characteristic: string;
    rank: string;
    difficultyRank: number;
    filename: unknown;
    njs: number;
    njsOffset: number;
  }
  const specs: DiffSpec[] = [];

  if ("_difficultyBeatmapSets" in info || "_songName" in info) {
    title = typeof info._songName === "string" ? info._songName : "";
    artist = typeof info._songAuthorName === "string" ? info._songAuthorName : "";
    mapper = typeof info._levelAuthorName === "string" ? info._levelAuthorName : "";
    bpm = Number(info._beatsPerMinute) || 0;
    songTimeOffset = Number(info._songTimeOffset) || 0;
    if (typeof info._environmentName === "string" && info._environmentName) {
      environment = info._environmentName;
    }
    audioFilename = info._songFilename;
    coverFilename = info._coverImageFilename;

    const sets = Array.isArray(info._difficultyBeatmapSets) ? info._difficultyBeatmapSets : [];
    for (const rawSet of sets) {
      if (!rawSet || typeof rawSet !== "object") continue;
      const setObj = rawSet as Record<string, unknown>;
      const characteristic =
        typeof setObj._beatmapCharacteristicName === "string"
          ? setObj._beatmapCharacteristicName
          : "Standard";
      const dbs = Array.isArray(setObj._difficultyBeatmaps) ? setObj._difficultyBeatmaps : [];
      for (const rawDb of dbs) {
        if (!rawDb || typeof rawDb !== "object") continue;
        const db = rawDb as Record<string, unknown>;
        const rank = typeof db._difficulty === "string" ? db._difficulty : "Expert";
        specs.push({
          characteristic,
          rank,
          difficultyRank: Number(db._difficultyRank) || RANK_ORDER[rank] || 5,
          filename: db._beatmapFilename,
          njs: Number(db._noteJumpMovementSpeed) || 0,
          njsOffset: Number(db._noteJumpStartBeatOffset) || 0,
        });
      }
    }
  } else {
    // v4 Info.dat support
    const song = (info.song && typeof info.song === "object"
      ? info.song
      : {}) as Record<string, unknown>;
    const audio = (info.audio && typeof info.audio === "object"
      ? info.audio
      : {}) as Record<string, unknown>;
    title = typeof song.title === "string" ? song.title : "";
    artist = typeof song.author === "string" ? song.author : "";
    bpm = Number(audio.bpm) || 0;
    audioFilename = audio.songFilename;
    coverFilename = info.coverImageFilename;
    const envs = Array.isArray(info.environmentNames) ? info.environmentNames : [];
    if (typeof envs[0] === "string" && envs[0]) {
      environment = envs[0];
    }
    const dbs = Array.isArray(info.difficultyBeatmaps) ? info.difficultyBeatmaps : [];
    for (const rawDb of dbs) {
      if (!rawDb || typeof rawDb !== "object") continue;
      const db = rawDb as Record<string, unknown>;
      const rank = typeof db.difficulty === "string" ? db.difficulty : "Expert";
      specs.push({
        characteristic: typeof db.characteristic === "string" ? db.characteristic : "Standard",
        rank,
        difficultyRank: RANK_ORDER[rank] || 5,
        filename: db.beatmapDataFilename,
        njs: Number(db.noteJumpMovementSpeed) || 0,
        njsOffset: Number(db.noteJumpStartBeatOffset) || 0,
      });
    }
  }

  const difficulties: ViewerDifficultyRaw[] = [];
  for (const spec of specs) {
    const diffPath = await findCaseInsensitiveFile(folder, spec.filename);
    if (!diffPath) continue;
    try {
      const raw = await readJsonFile(diffPath);
      difficulties.push({
        characteristic: spec.characteristic,
        rank: spec.rank,
        difficultyRank: spec.difficultyRank,
        filename: path.basename(diffPath),
        njs: spec.njs,
        njsOffset: spec.njsOffset,
        raw,
      });
    } catch {
      // Skip unreadable difficulty files.
    }
  }

  let audioBytes: Uint8Array | null = null;
  const audioPath =
    (await findCaseInsensitiveFile(folder, audioFilename)) ??
    (await findCaseInsensitiveFile(folder, "song.egg")) ??
    (await findCaseInsensitiveFile(folder, "audio.ogg"));
  if (audioPath) {
    const buf = await fsp.readFile(audioPath);
    audioBytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  let coverDataUrl: string | null = null;
  const coverPath = await findCaseInsensitiveFile(folder, coverFilename);
  if (coverPath) {
    const mime = mimeForImageExt(path.extname(coverPath));
    if (mime) {
      const buf = await fsp.readFile(coverPath);
      coverDataUrl = `data:${mime};base64,${buf.toString("base64")}`;
    }
  }

  return {
    folder,
    title: title || path.basename(folder),
    artist: artist || "Unknown Artist",
    mapper,
    bpm: bpm > 0 ? bpm : 120,
    songTimeOffset,
    environment,
    coverDataUrl,
    audioBytes,
    difficulties,
  };
}
