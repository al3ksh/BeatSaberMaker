import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

let cancelled = false;
let activeTimer = null;

function emit(obj) {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

function uniqueDir(parent, baseName) {
  let candidate = path.join(parent, baseName);
  if (!fs.existsSync(candidate)) return candidate;
  let n = 2;
  while (fs.existsSync(path.join(parent, `${baseName} (${n})`))) {
    n += 1;
  }
  return path.join(parent, `${baseName} (${n})`);
}

rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  const req = JSON.parse(trimmed);
  const { id, cmd } = req;

  if (cmd === "cancel") {
    cancelled = true;
    if (activeTimer) {
      clearTimeout(activeTimer);
      activeTimer = null;
      if (typeof activeTimerCancelCb === "function") {
        activeTimerCancelCb();
      }
    }
    emit({ id, type: "result", data: { cancelled: true } });
    return;
  }

  if (cmd === "shutdown") {
    emit({ id, type: "result", data: { shutdown: true } });
    process.exit(0);
  }

  if (cmd === "info") {
    const detected = process.env.BSM_FAKE_DETECTED_CL || null;
    emit({
      id,
      type: "result",
      data: {
        version: "0.1.0",
        cuda_available: false,
        gpu_name: null,
        active_device: "cpu",
        detected_custom_levels: detected,
        has_layer_models: false,
        missing_layer_models: [
          "ornaments",
          "decor-when",
          "decor-what",
          "lights-when",
          "lights-what",
        ],
        environments: [
          "DefaultEnvironment",
          "BigMirrorEnvironment",
          "NiceEnvironment",
          "TriangleEnvironment",
          "PanicEnvironment",
          "OriginsEnvironment",
          "DragonsEnvironment",
          "TimbalandEnvironment",
          "BTSEnvironment",
          "KaleidoscopeEnvironment",
        ],
        styles: [
          "tech",
          "dance",
          "speed",
          "accuracy",
          "balanced",
          "challenge",
          "fitness",
        ],
      },
    });
    return;
  }

  if (cmd === "search") {
    emit({
      id,
      type: "result",
      data: {
        items: [
          {
            id: "yt1",
            url: "https://youtu.be/yt1",
            title: "Camellia - GHOST",
            artist: "Camellia",
            track_title: "GHOST",
            channel: "Camellia",
            duration_s: 240,
            thumbnail_url: null,
          },
        ],
      },
    });
    return;
  }

  if (cmd === "resolve") {
    emit({
      id,
      type: "result",
      data: {
        id: "yt1",
        url: req.url || "https://youtu.be/yt1",
        title: "Camellia - GHOST",
        artist: "Camellia",
        track_title: "GHOST",
        channel: "Camellia",
        duration_s: 240,
        thumbnail_url: null,
      },
    });
    return;
  }

  if (cmd === "download") {
    const destDir = req.dest_dir;
    fs.mkdirSync(destDir, { recursive: true });
    const audioPath = path.join(destDir, "audio.wav");
    fs.writeFileSync(audioPath, "RIFFfake");
    emit({ id, type: "progress", stage: "downloading", pct: 50 });
    emit({ id, type: "progress", stage: "downloading", pct: 100 });
    emit({
      id,
      type: "result",
      data: {
        audio_path: audioPath,
        cover_path: null,
        title: "GHOST",
        artist: "Camellia",
      },
    });
    return;
  }

  if (cmd === "generate") {
    cancelled = false;
    if (process.env.BSM_FAKE_FAIL_CODE) {
      emit({
        id,
        type: "error",
        code: process.env.BSM_FAKE_FAIL_CODE,
        message_pl: "Błąd",
        detail: "Traceback (most recent call last): secret internal error",
      });
      return;
    }

    if (process.env.BSM_FAKE_HANG_CANCEL === "1") {
      emit({ id, type: "progress", stage: "notes", pct: 35 });
      activeTimerCancelCb = () => {
        activeTimerCancelCb = null;
        emit({
          id,
          type: "error",
          code: "cancelled",
          message_pl: "Anulowano",
          detail: "cancelled",
        });
      };
      activeTimer = setTimeout(() => {
        activeTimer = null;
        activeTimerCancelCb = null;
        if (!cancelled) {
          emit({
            id,
            type: "error",
            code: "cancelled",
            message_pl: "Anulowano",
            detail: "timeout",
          });
        }
      }, 15000);
      return;
    }

    emit({ id, type: "progress", stage: "analyzing", pct: 10 });
    emit({ id, type: "progress", stage: "notes", pct: 50 });
    emit({ id, type: "progress", stage: "layers", pct: 85 });
    emit({ id, type: "progress", stage: "writing", pct: 100 });

    const title = req.title || "untitled";
    const artist = req.artist || "Unknown";
    const levels = Array.isArray(req.levels) && req.levels.length > 0 ? req.levels : ["Hard"];
    const tag = req.nps ? `nps ${req.nps}` : levels.join("+");
    const folderName = `BSM - ${title} [${tag}, unknown]`;
    const folder = req.unique_folder
      ? uniqueDir(req.out_dir, folderName)
      : path.join(req.out_dir, folderName);

    fs.mkdirSync(folder, { recursive: true });
    fs.writeFileSync(path.join(folder, "Info.dat"), JSON.stringify({ _songName: title }));
    fs.writeFileSync(path.join(folder, "song.egg"), "OggS");

    emit({
      id,
      type: "result",
      data: {
        folder,
        title,
        artist,
        bpm: 180,
        grid_steady: true,
        n_notes: 300,
        nps: 4.5,
        rank: levels[levels.length - 1],
        levels: levels.map((rk) => ({
          rank: rk,
          target_nps: 4.5,
          n_notes: 300,
          nps: 4.5,
          n_arcs: 0,
          n_chains: 0,
        })),
        n_walls: 0,
        n_bombs: 0,
        n_lights: 40,
        environment: req.environment || "DefaultEnvironment",
        warnings:
          process.env.BSM_FAKE_LAYER_WARN === "0"
            ? []
            : ["missing ornaments checkpoint: ckpts/layers/ornaments/best.pt"],
        used_rule_lights: true,
        device_used: "cpu",
      },
    });
    return;
  }

  if (cmd === "export_zip") {
    fs.mkdirSync(path.dirname(req.out_zip), { recursive: true });
    fs.writeFileSync(req.out_zip, "PK\x03\x04fakezip");
    emit({ id, type: "result", data: { out_zip: req.out_zip } });
    return;
  }

  emit({ id, type: "error", code: "invalid_args", message_pl: "Unknown cmd", detail: cmd });
});

let activeTimerCancelCb = null;
