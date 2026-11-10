"""Read Beat Saber map folders (Info + difficulty files, v2/v3/v4 schemas)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bsmaker.formats.model import BeatMap, Difficulty, DifficultyFlags, Note

_RawNote = tuple[float, int, int, int, int, bool]  # beat, x, y, color, direction, has_custom


def map_key(folder_name: str) -> str:
    m = re.match(r"^([0-9a-fA-F]+) ", folder_name)
    if m:
        return m.group(1)
    return "local-" + hashlib.sha1(folder_name.encode("utf-8")).hexdigest()[:10]


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def _find_ci(folder: Path, name: str) -> Path | None:
    exact = folder / name
    if exact.is_file():
        return exact
    low = name.lower()
    for p in folder.iterdir():
        if p.is_file() and p.name.lower() == low:
            return p
    return None


def _detect_schema(data: dict) -> str:
    if "_notes" in data:
        return "v2"
    if "colorNotesData" in data:
        return "v4"
    if "colorNotes" in data:
        return "v3"
    return "v2"  # no notes key at all: treat as an empty v2 map


def _raw_notes(data: dict, schema: str) -> tuple[list[_RawNote], int]:
    out: list[_RawNote] = []
    bombs = 0
    if schema == "v2":
        for n in data.get("_notes") or []:
            t = n.get("_type", 0)
            if t == 3:
                bombs += 1
            elif t in (0, 1):
                out.append((float(n.get("_time", 0)), int(n.get("_lineIndex", 0)),
                            int(n.get("_lineLayer", 0)), t, int(n.get("_cutDirection", 0)), bool(n.get("_customData"))))
    elif schema == "v3":
        for n in data.get("colorNotes") or []:
            out.append((float(n.get("b", 0)), int(n.get("x", 0)), int(n.get("y", 0)),
                        int(n.get("c", 0)), int(n.get("d", 0)), bool(n.get("customData"))))
        bombs = len(data.get("bombNotes") or [])
    else:
        pool = data.get("colorNotesData") or []
        for n in data.get("colorNotes") or []:
            d = pool[n.get("i", 0)]
            out.append((float(n.get("b", 0)), int(d.get("x", 0)), int(d.get("y", 0)),
                        int(d.get("c", 0)), int(d.get("d", 0)),
                        bool(n.get("customData")) or bool(d.get("customData"))))
        bombs = len(data.get("bombNotes") or [])
    return out, bombs


def _counts(data: dict, schema: str) -> tuple[int, int, int]:
    """(obstacles, arcs, chains)."""
    if schema == "v2":
        return len(data.get("_obstacles") or []), 0, 0
    if schema == "v3":
        return (len(data.get("obstacles") or []), len(data.get("sliders") or []),
                len(data.get("burstSliders") or []))
    return (len(data.get("obstacles") or []), len(data.get("arcs") or []),
            len(data.get("chains") or []))


def _bpm_changes(data: dict, schema: str, info_bpm: float, folder: Path) -> bool:
    if schema == "v2":
        cd = data.get("_customData") or {}
        if cd.get("_BPMChanges") or cd.get("_bpmChanges"):
            return True
        if any(e.get("_type") == 100 for e in data.get("_events") or []):
            return True
    elif schema == "v3":
        ev = data.get("bpmEvents") or []
        if len(ev) > 1:
            return True
        if len(ev) == 1 and abs(float(ev[0].get("m", info_bpm)) - info_bpm) > 0.01:
            return True
    for name, key in (("BPMInfo.dat", "_regions"), ("AudioData.dat", "bpmData")):
        p = _find_ci(folder, name)
        if p is not None:
            try:
                if len(_load(p).get(key) or []) > 1:
                    return True
            except (OSError, ValueError, AttributeError):
                pass
    return False


def _read_difficulty(path: Path, folder: Path, characteristic: str, rank: str, njs: float,
                     njs_offset: float, bpm: float, time_offset: float) -> Difficulty:
    data = _load(path)
    if not isinstance(data, dict):
        raise ValueError("difficulty file is not a JSON object")
    schema = _detect_schema(data)
    raw, bombs = _raw_notes(data, schema)
    offgrid = False
    custom = False
    notes: list[Note] = []
    for beat, x, y, c, d, has_custom in raw:
        custom = custom or has_custom
        if not (0 <= x <= 3 and 0 <= y <= 2 and 0 <= d <= 8):
            offgrid = True
            continue
        notes.append(Note(beat * 60.0 / bpm, beat, x, y, c, d))
    notes.sort(key=lambda n: (n.time_s, n.color, n.x, n.y))
    n_obst, n_arcs, n_chains = _counts(data, schema)
    flags = DifficultyFlags(_bpm_changes(data, schema, bpm, folder), offgrid, custom,
                            time_offset != 0)
    return Difficulty(characteristic, rank, schema, njs, njs_offset, tuple(notes), bombs,
                      n_obst, n_arcs, n_chains, flags)


def read_map(folder: Path) -> BeatMap:
    folder = Path(folder)
    problems: list[str] = []
    info_path = _find_ci(folder, "Info.dat")
    if info_path is None:
        raise FileNotFoundError(f"no Info.dat in {folder}")
    info = _load(info_path)

    reqs: list[str] = []
    sugs: list[str] = []
    specs: list[tuple[str, str, str, float, float]] = []  # characteristic, rank, file, njs, offset
    time_offset = 0.0
    if "_difficultyBeatmapSets" in info or "_songName" in info:
        title = info.get("_songName", "")
        artist = info.get("_songAuthorName", "")
        mapper = info.get("_levelAuthorName", "")
        bpm = float(info.get("_beatsPerMinute", 0) or 0)
        audio_name = info.get("_songFilename")
        time_offset = float(info.get("_songTimeOffset", 0) or 0)
        for s in info.get("_difficultyBeatmapSets") or []:
            ch = s.get("_beatmapCharacteristicName", "")
            for d in s.get("_difficultyBeatmaps") or []:
                cd = d.get("_customData") or {}
                for r in cd.get("_requirements") or []:
                    if r not in reqs:
                        reqs.append(r)
                for r in cd.get("_suggestions") or []:
                    if r not in sugs:
                        sugs.append(r)
                specs.append((ch, str(d.get("_difficulty", "")), d.get("_beatmapFilename", ""),
                              float(d.get("_noteJumpMovementSpeed", 0) or 0),
                              float(d.get("_noteJumpStartBeatOffset", 0) or 0)))
    else:
        song = info.get("song") or {}
        audio = info.get("audio") or {}
        title = song.get("title", "")
        artist = song.get("author", "")
        bpm = float(audio.get("bpm", 0) or 0)
        audio_name = audio.get("songFilename")
        mappers: list[str] = []
        for d in info.get("difficultyBeatmaps") or []:
            for m in (d.get("beatmapAuthors") or {}).get("mappers") or []:
                if m not in mappers:
                    mappers.append(m)
            specs.append((d.get("characteristic", ""), str(d.get("difficulty", "")),
                          d.get("beatmapDataFilename", ""),
                          float(d.get("noteJumpMovementSpeed", 0) or 0),
                          float(d.get("noteJumpStartBeatOffset", 0) or 0)))
        mapper = ", ".join(mappers)

    if bpm <= 0:
        problems.append("Info has no usable BPM")

    diffs: list[Difficulty] = []
    for ch, rank, fname, njs, off in specs:
        p = _find_ci(folder, fname) if fname else None
        if p is None:
            problems.append(f"missing difficulty file: {fname!r} ({ch} {rank})")
            continue
        if bpm <= 0:
            problems.append(f"skipped {fname!r}: no BPM")
            continue
        try:
            diffs.append(_read_difficulty(p, folder, ch, rank, njs, off, bpm, time_offset))
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            problems.append(f"cannot read {fname!r}: {type(e).__name__}: {e}")

    audio_path = _find_ci(folder, audio_name) if isinstance(audio_name, str) and audio_name else None
    return BeatMap(folder, map_key(folder.name), title, artist, mapper, bpm, audio_path,
                   tuple(reqs), tuple(sugs), tuple(diffs), tuple(problems))
