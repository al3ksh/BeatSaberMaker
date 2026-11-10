"""Write a generated map as a Beat Saber folder (Info v2.1.0 + difficulty v3.3.0)."""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path

from bsmaker.formats.model import Note

_RANK_NUMBERS = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
_AUTHOR = "BeatSaberMaker"


def _png(size: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size))
            + chunk(b"IEND", b""))


def _encode_audio(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
           "-c:a", "libvorbis", "-q:a", "6", "-f", "ogg", str(dst)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode song.egg") from e
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {res.returncode}): {res.stderr.strip()}")


def write_map(folder: Path, *, title: str, artist: str, bpm: float, rank: str, njs: float,
              njs_offset: float, notes: Sequence[Note], audio_src: Path,
              overwrite: bool = False) -> Path:
    folder = Path(folder)
    if rank not in _RANK_NUMBERS:
        raise ValueError(f"unknown rank {rank!r}")
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"{folder} already exists")
        if not folder.name.startswith("BSM - "):
            raise ValueError(f"refusing to overwrite {folder.name!r}: name must start with 'BSM - '")

    diff_name = f"{rank}Standard.dat"
    info = {
        "_version": "2.1.0",
        "_songName": title,
        "_songSubName": "",
        "_songAuthorName": artist,
        "_levelAuthorName": _AUTHOR,
        "_beatsPerMinute": bpm,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": "song.egg",
        "_coverImageFilename": "cover.png",
        "_environmentName": "DefaultEnvironment",
        "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": rank,
                "_difficultyRank": _RANK_NUMBERS[rank],
                "_beatmapFilename": diff_name,
                "_noteJumpMovementSpeed": njs,
                "_noteJumpStartBeatOffset": njs_offset,
            }],
        }],
    }
    ordered = sorted(notes, key=lambda n: (n.beat, n.color, n.x, n.y))
    diff = {
        "version": "3.3.0",
        "bpmEvents": [], "rotationEvents": [],
        "colorNotes": [{"b": n.beat, "x": n.x, "y": n.y, "c": n.color, "d": n.direction, "a": 0}
                       for n in ordered],
        "bombNotes": [], "obstacles": [], "sliders": [], "burstSliders": [], "waypoints": [],
        "basicBeatmapEvents": [], "colorBoostBeatmapEvents": [],
        "lightColorEventBoxGroups": [], "lightRotationEventBoxGroups": [],
        "lightTranslationEventBoxGroups": [], "vfxEventBoxGroups": [],
        "_fxEventsCollection": {"_il": [], "_fl": []},
        "basicEventTypesWithKeywords": {"d": []},
        "useNormalEventsAsCompatibleEvents": True,
    }

    tmp = folder.with_name(folder.name + ".tmp-writing")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _encode_audio(Path(audio_src), tmp / "song.egg")
        (tmp / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (tmp / diff_name).write_text(json.dumps(diff), encoding="utf-8")
        (tmp / "cover.png").write_bytes(_png(256, (40, 90, 160)))
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return folder
