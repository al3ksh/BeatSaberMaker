"""Acceptance filters for training difficulties."""
from __future__ import annotations

import math
from collections.abc import Sequence

from bsmaker.formats.model import BeatMap, Difficulty, Note

MIN_NOTES = 80
MIN_DURATION_S = 60.0
MAX_DURATION_S = 600.0
MIN_NPS = 0.5
MAX_NPS = 12.0
MAX_HARD_RATE = 0.02
MIN_BPM = 20.0
MAX_BPM = 1000.0


def _tempo_reason(flags, tempo_verified: bool) -> str | None:
    if flags.tempo_kind == "external":
        return "tempo_external"
    if flags.tempo_kind not in ("editor", "events"):
        return None
    lo, hi = flags.bpm_min, flags.bpm_max
    if (math.isnan(lo) or math.isnan(hi) or lo < MIN_BPM or lo > MAX_BPM
            or hi < MIN_BPM or hi > MAX_BPM):
        return "bpm_trick"
    return None if tempo_verified else "tempo_unverified"


def nps(notes: Sequence[Note]) -> float:
    if len(notes) < 2:
        return 0.0
    times = [n.time_s for n in notes]
    span = max(times) - min(times)
    return len(notes) / span if span > 0 else 0.0


def reject_reasons(
    bm: BeatMap,
    diff: Difficulty,
    *,
    duration_s: float | None,
    hard_rate: float,
    automapper: bool | None,
    tempo_verified: bool = False,
) -> list[str]:
    out: list[str] = []
    if diff.characteristic != "Standard":
        out.append("not_standard")
    if bm.requirements:
        out.append("requirements")
    if diff.flags.has_offgrid_notes or diff.flags.has_custom_note_data:
        out.append("offgrid_or_custom_notes")
    tempo = _tempo_reason(diff.flags, tempo_verified)
    if tempo:
        out.append(tempo)
    if diff.flags.has_time_offset:
        out.append("time_offset")
    if len(diff.notes) < MIN_NOTES:
        out.append("too_few_notes")
    if duration_s is not None and not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        out.append("duration")
    if not (MIN_NPS <= nps(diff.notes) <= MAX_NPS):
        out.append("nps_range")
    if automapper is True:
        out.append("automapper")
    if hard_rate > MAX_HARD_RATE:
        out.append("hard_rate")
    if duration_s is None:
        out.append("no_audio")
    return out
