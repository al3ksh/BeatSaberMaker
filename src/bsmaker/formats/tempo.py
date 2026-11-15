"""Tempo map: convert beats to seconds through BPM changes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TempoMap:
    info_bpm: float
    points: tuple[tuple[float, float], ...] = ()  # (beat, bpm)

    def __post_init__(self) -> None:
        by_beat: dict[float, float] = {}
        for beat, bpm in self.points:
            by_beat[beat] = bpm  # a later duplicate wins
        object.__setattr__(self, "points", tuple(sorted(by_beat.items())))

    def to_seconds(self, beat: float) -> float:
        seconds = 0.0
        prev_beat = 0.0
        bpm = self.info_bpm
        for p_beat, p_bpm in self.points:
            if beat <= p_beat:
                break
            seconds += (p_beat - prev_beat) * 60.0 / bpm
            prev_beat, bpm = p_beat, p_bpm
        return seconds + (beat - prev_beat) * 60.0 / bpm

    def bpm_range(self) -> tuple[float, float]:
        values = [self.info_bpm] + [bpm for _, bpm in self.points]
        if any(math.isnan(v) for v in values):
            return float("nan"), float("nan")
        return min(values), max(values)


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _tempo_value(value: object) -> float:
    v = _num(value)
    return v if math.isfinite(v) and v > 0 else float("nan")


def _point(beat: object, tempo: object) -> tuple[float, float]:
    """A (beat, bpm) point; a non-finite beat invalidates the point like a bad tempo does."""
    b = _num(beat)
    if not math.isfinite(b):
        return float("nan"), float("nan")
    return b, _tempo_value(tempo)


def _external_signals(folder: Path) -> bool:
    from bsmaker.formats.reader import _find_ci as find_ci, _load as load_json  # avoid import cycle

    for name, key in (("BPMInfo.dat", "_regions"), ("AudioData.dat", "bpmData")):
        p = find_ci(folder, name)
        if p is not None:
            try:
                if len(load_json(p).get(key) or []) > 1:
                    return True
            except (OSError, ValueError, AttributeError):
                pass
    return False


def read_tempo(data: dict, schema: str, info_bpm: float, folder: Path) -> tuple[str, TempoMap]:
    if schema == "v2":
        events = [e for e in data.get("_events") or []
                  if isinstance(e, dict) and e.get("_type") == 100]
        if events:
            pts = tuple(_point(e.get("_time", 0), e.get("_floatValue"))
                        for e in events)
            return "events", TempoMap(info_bpm, pts)
        cd = data.get("_customData") or {}
        changes = cd.get("_BPMChanges") or cd.get("_bpmChanges")
        if changes:
            pts_list: list[tuple[float, float]] = []
            for c in changes:
                if isinstance(c, dict):
                    bpm = c["_BPM"] if "_BPM" in c else c.get("_bpm")
                    pts_list.append(_point(c.get("_time", 0), bpm))
                else:
                    pts_list.append((float("nan"), float("nan")))
            return "editor", TempoMap(info_bpm, tuple(pts_list))
    elif schema == "v3":
        ev = data.get("bpmEvents") or []
        pts = tuple(_point(e.get("b", 0), e.get("m", info_bpm)) if isinstance(e, dict)
                    else (float("nan"), float("nan")) for e in ev)
        changed = len(ev) > 1
        if len(ev) == 1:
            m = pts[0][1]
            changed = math.isnan(m) or abs(m - info_bpm) > 0.01
        if changed:
            return "events", TempoMap(info_bpm, pts)
    if _external_signals(folder):
        return "external", TempoMap(info_bpm)
    return "none", TempoMap(info_bpm)
