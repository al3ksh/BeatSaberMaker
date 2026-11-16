"""Write a generated map as a Beat Saber folder (Info v2.1.0 + difficulty v3.3.0)."""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bsmaker.formats.model import Arc, Bomb, Chain, LightEvent, Note, Obstacle

_RANK_NUMBERS = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
_AUTHOR = "BeatSaberMaker"
_FFMPEG_TIMEOUT_S = 300


def _png(size: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size))
            + chunk(b"IEND", b""))


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": _FFMPEG_TIMEOUT_S,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        res = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S} s {what}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg not found on PATH; cannot {what}") from e
    if res.returncode != 0:
        lines = res.stderr.strip().splitlines()
        raise RuntimeError(f"ffmpeg failed (exit {res.returncode}): {lines[-1] if lines else ''}")


def _encode_audio(src: Path, dst: Path, *, ffmpeg_path: Path | None = None) -> None:
    exe = str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg"
    cmd = [exe, "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
           "-c:a", "libvorbis", "-q:a", "6", "-f", "ogg", str(dst)]
    _run_ffmpeg(cmd, "encoding song.egg")


def _convert_cover(src: Path, dst: Path, *, ffmpeg_path: Path | None = None) -> None:
    exe = str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg"
    cmd = [exe, "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
           "-frames:v", "1", "-q:v", "2", str(dst)]
    _run_ffmpeg(cmd, "converting cover.jpg")


@dataclass(frozen=True)
class LevelData:
    rank: str
    njs: float
    njs_offset: float
    notes: Sequence[Note]
    bombs: Sequence[Bomb] = ()
    obstacles: Sequence[Obstacle] = ()
    arcs: Sequence[Arc] = ()
    chains: Sequence[Chain] = ()


def _in_grid(x: int, y: int) -> bool:
    return 0 <= x <= 3 and 0 <= y <= 2


def _time_ok(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def _validate(level: LevelData) -> None:
    r = level.rank
    for i, b in enumerate(level.bombs):
        if not _time_ok(b.time_s):
            raise ValueError(f"{r}: bomb {i} has an invalid time: {b.time_s}")
        if not _in_grid(b.x, b.y):
            raise ValueError(f"{r}: bomb {i} outside the grid: ({b.x}, {b.y})")
    for i, o in enumerate(level.obstacles):
        if not _time_ok(o.time_s) or not math.isfinite(o.duration_s):
            raise ValueError(f"{r}: wall {i} has an invalid time or duration: {o}")
        if not o.duration_s > 0 or o.w <= 0:
            raise ValueError(f"{r}: wall {i} needs a positive duration and width: {o}")
        if not -1 <= o.x <= 4 or not 0 <= o.y <= 2 or not 1 <= o.h <= 5:
            raise ValueError(f"{r}: wall {i} outside the allowed range: {o}")
    for i, a in enumerate(level.arcs):
        if not _time_ok(a.time_s) or not _time_ok(a.tail_time_s):
            raise ValueError(f"{r}: arc {i} has an invalid time: {a}")
        if not _in_grid(a.x, a.y) or not _in_grid(a.tx, a.ty):
            raise ValueError(f"{r}: arc {i} outside the grid: {a}")
        if not 0 <= a.d <= 8 or not 0 <= a.td <= 8:
            raise ValueError(f"{r}: arc {i} has a direction outside 0-8: {a}")
    for i, c in enumerate(level.chains):
        if not _time_ok(c.time_s) or not _time_ok(c.tail_time_s):
            raise ValueError(f"{r}: chain {i} has an invalid time: {c}")
        if not _in_grid(c.x, c.y) or not _in_grid(c.tx, c.ty):
            raise ValueError(f"{r}: chain {i} outside the grid: {c}")
        if not 0 <= c.d <= 8:
            raise ValueError(f"{r}: chain {i} has a direction outside 0-8: {c}")
        if c.segments < 1 or not (math.isfinite(c.squish) and 0 < c.squish <= 1):
            raise ValueError(f"{r}: chain {i} needs segments >= 1 and squish in (0, 1]: {c}")


def _difficulty_json(level: LevelData, bpm: float, lights: Sequence[LightEvent]) -> dict:
    def beat(t: float) -> float:
        return t * bpm / 60.0

    ordered = sorted(level.notes, key=lambda n: (n.beat, n.color, n.x, n.y))
    return {
        "version": "3.3.0",
        "bpmEvents": [], "rotationEvents": [],
        "colorNotes": [{"b": n.beat, "x": n.x, "y": n.y, "c": n.color, "d": n.direction, "a": 0}
                       for n in ordered],
        "bombNotes": [{"b": beat(b.time_s), "x": b.x, "y": b.y}
                      for b in sorted(level.bombs, key=lambda b: (b.time_s, b.x, b.y))],
        "obstacles": [{"b": beat(o.time_s), "x": o.x, "y": o.y, "d": beat(o.duration_s), "w": o.w, "h": o.h}
                      for o in sorted(level.obstacles, key=lambda o: (o.time_s, o.x, o.y))],
        "sliders": [{"b": beat(a.time_s), "c": a.color, "x": a.x, "y": a.y, "d": a.d, "mu": 1,
                     "tb": beat(a.tail_time_s), "tx": a.tx, "ty": a.ty, "tc": a.td, "tmu": 1, "m": 0}
                    for a in sorted(level.arcs, key=lambda a: (a.time_s, a.color, a.x, a.y))],
        "burstSliders": [{"b": beat(c.time_s), "x": c.x, "y": c.y, "c": c.color, "d": c.d,
                          "tb": beat(c.tail_time_s), "tx": c.tx, "ty": c.ty, "sc": c.segments, "s": c.squish}
                         for c in sorted(level.chains, key=lambda c: (c.time_s, c.color, c.x, c.y))],
        "waypoints": [],
        "basicBeatmapEvents": [{"b": beat(e.time_s), "et": e.type, "i": e.value, "f": e.float_value}
                               for e in sorted(lights, key=lambda e: (e.time_s, e.type))],
        "colorBoostBeatmapEvents": [],
        "lightColorEventBoxGroups": [], "lightRotationEventBoxGroups": [],
        "lightTranslationEventBoxGroups": [], "vfxEventBoxGroups": [],
        "_fxEventsCollection": {"_il": [], "_fl": []},
        "basicEventTypesWithKeywords": {"d": []},
        "useNormalEventsAsCompatibleEvents": True,
    }


def write_map_multi(folder: Path, *, title: str, artist: str, bpm: float, levels: Sequence[LevelData],
                    lights: Sequence[LightEvent] = (), environment: str = "DefaultEnvironment",
                    audio_src: Path, overwrite: bool = False,
                    cover_src: Path | None = None,
                    ffmpeg_path: Path | None = None) -> Path:
    """Write one v3 difficulty file per level (ascending rank order in Info).

    Layer objects are placed on the beat grid from their `time_s` and `bpm`; their own
    `beat` fields are ignored.
    """
    folder = Path(folder)
    if not levels:
        raise ValueError("at least one level is required")
    ranks = [lv.rank for lv in levels]
    for r in ranks:
        if r not in _RANK_NUMBERS:
            raise ValueError(f"unknown rank {r!r}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks: {ranks}")
    if not isinstance(environment, str) or not environment:
        raise ValueError("environment must be a non-empty string")
    if cover_src is not None:
        cover_path = Path(cover_src)
        if not cover_path.is_file():
            raise ValueError(f"cover file not found: {cover_src}")
        cover_name = "cover.jpg"
    else:
        cover_path = None
        cover_name = "cover.png"
    for lv in levels:
        _validate(lv)
    for i, e in enumerate(lights):
        if not _time_ok(e.time_s) or not math.isfinite(e.float_value):
            raise ValueError(f"light event {i} has an invalid time or value: {e}")
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"{folder} already exists")
        if not folder.name.startswith("BSM - "):
            raise ValueError(f"refusing to overwrite {folder.name!r}: name must start with 'BSM - '")

    ordered = sorted(levels, key=lambda lv: _RANK_NUMBERS[lv.rank])
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
        "_coverImageFilename": cover_name,
        "_environmentName": environment,
        "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": lv.rank,
                "_difficultyRank": _RANK_NUMBERS[lv.rank],
                "_beatmapFilename": f"{lv.rank}Standard.dat",
                "_noteJumpMovementSpeed": lv.njs,
                "_noteJumpStartBeatOffset": lv.njs_offset,
            } for lv in ordered],
        }],
    }
    diffs = {f"{lv.rank}Standard.dat": _difficulty_json(lv, bpm, lights) for lv in ordered}

    tmp = folder.with_name(folder.name + ".tmp-writing")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _encode_audio(Path(audio_src), tmp / "song.egg", ffmpeg_path=ffmpeg_path)
        (tmp / "Info.dat").write_text(json.dumps(info, indent=2), encoding="utf-8")
        for name, diff in diffs.items():
            (tmp / name).write_text(json.dumps(diff), encoding="utf-8")
        if cover_path is not None:
            if cover_path.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copyfile(cover_path, tmp / "cover.jpg")
            else:
                _convert_cover(cover_path, tmp / "cover.jpg", ffmpeg_path=ffmpeg_path)
        else:
            (tmp / "cover.png").write_bytes(_png(256, (40, 90, 160)))
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return folder


def write_map(folder: Path, *, title: str, artist: str, bpm: float, rank: str, njs: float,
              njs_offset: float, notes: Sequence[Note], audio_src: Path,
              overwrite: bool = False) -> Path:
    return write_map_multi(folder, title=title, artist=artist, bpm=bpm,
                           levels=[LevelData(rank, njs, njs_offset, notes)],
                           audio_src=audio_src, overwrite=overwrite)
