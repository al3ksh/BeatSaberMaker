"""Token tables for the layers: light events, walls and bomb cell masks."""
from __future__ import annotations

import math

from bsmaker.data.layer_classes import wall_class
from bsmaker.formats.model import Obstacle

LIGHT_CHANNELS = 9
LIGHT_VOCAB = 18
WALL_VOCAB = 24
SUPPORTED_ENVIRONMENTS = (
    "DefaultEnvironment", "BigMirrorEnvironment", "NiceEnvironment", "TriangleEnvironment",
    "PanicEnvironment", "OriginsEnvironment", "DragonsEnvironment", "TimbalandEnvironment",
    "BTSEnvironment", "KaleidoscopeEnvironment")
DURATION_BEATS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
ARC_LENGTH_BEATS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

# channel index -> event type
CHANNEL_TYPES = (0, 1, 2, 3, 4, 8, 9, 12, 13)
_TYPE_TO_CHANNEL = {t: c for c, t in enumerate(CHANNEL_TYPES)}
_SPEED_BASE = 7
_SPIN_TOKEN = 17
# colour A = red (v2 values 5-7), colour B = blue (v2 values 1-3)
_VALUE_TO_TOKEN = {0: 0, 5: 1, 6: 2, 7: 3, 1: 4, 2: 5, 3: 6, 4: 4, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
_TOKEN_TO_VALUE = {0: 0, 1: 5, 2: 6, 3: 7, 4: 1, 5: 2, 6: 3}


def light_event_to_token(type: int, value: int) -> tuple[int, int] | None:
    """(channel, token) for a light event, or None when the event is not representable."""
    ch = _TYPE_TO_CHANNEL.get(type)
    if ch is None:
        return None
    if ch <= 4:
        tok = _VALUE_TO_TOKEN.get(value)
        return None if tok is None else (ch, tok)
    if ch <= 6:
        return ch, _SPIN_TOKEN
    if value < 0:
        return None
    return ch, _SPEED_BASE + min(value, 9)


def token_to_light_event(channel: int, token: int) -> tuple[int, int]:
    """(type, value) for a light token."""
    if not 0 <= channel < LIGHT_CHANNELS:
        raise ValueError(f"invalid light channel: {channel}")
    if channel <= 4:
        if token not in _TOKEN_TO_VALUE:
            raise ValueError(f"invalid light token {token} for channel {channel}")
        return CHANNEL_TYPES[channel], _TOKEN_TO_VALUE[token]
    if channel <= 6:
        if token != _SPIN_TOKEN:
            raise ValueError(f"invalid light token {token} for channel {channel}")
        return CHANNEL_TYPES[channel], 0
    if not _SPEED_BASE <= token < _SPEED_BASE + 10:
        raise ValueError(f"invalid light token {token} for channel {channel}")
    return CHANNEL_TYPES[channel], token - _SPEED_BASE


def duration_bucket(duration_beats: float) -> int:
    """Index into DURATION_BEATS nearest in log scale (non-positive -> 0)."""
    if not duration_beats > 0:
        return 0
    lg = math.log(duration_beats)
    return min(range(len(DURATION_BEATS)), key=lambda i: abs(math.log(DURATION_BEATS[i]) - lg))


def arc_length_bucket(length_beats: float) -> int:
    """1-6: index into ARC_LENGTH_BEATS (1-based), nearest."""
    return 1 + min(range(len(ARC_LENGTH_BEATS)), key=lambda i: abs(ARC_LENGTH_BEATS[i] - length_beats))


def wall_to_token(ob: Obstacle, bpm: float) -> tuple[int, int] | None:
    """(side, token) for a flash or side wall of the supported shapes, else None.

    side 0 = left, 1 = right. Shapes outside the 24-token table (far outside the grid,
    wider than 1, heights other than full/high, centre positions) give None.
    """
    if wall_class(ob) == "centre":
        return None
    w = ob.w if ob.w > 0 else 1
    if w != 1:
        return None
    if ob.x + w - 1 <= 0:
        side, variant = 0, {0: 0, -1: 1}.get(ob.x)
    elif ob.x >= 3:
        side, variant = 1, {3: 0, 4: 1}.get(ob.x)
    else:
        return None
    if variant is None:
        return None
    if (ob.y, ob.h) == (0, 5):
        height = 0
    elif (ob.y, ob.h) == (2, 3):
        height = 1
    else:
        return None
    return side, (variant * 2 + height) * 6 + duration_bucket(ob.duration_beats)


def token_to_wall(side: int, token: int, time_s: float, bpm: float) -> Obstacle:
    if side not in (0, 1) or not 0 <= token < WALL_VOCAB:
        raise ValueError(f"invalid wall side/token: {side}/{token}")
    variant, rest = divmod(token, 12)
    height, bucket = divmod(rest, 6)
    x = (0, -1)[variant] if side == 0 else (3, 4)[variant]
    y, h = ((0, 5), (2, 3))[height]
    beats = DURATION_BEATS[bucket]
    return Obstacle(time_s, time_s * bpm / 60.0, beats * 60.0 / bpm, beats, x, y, 1, h)


def cells_to_mask(cells) -> int:
    mask = 0
    for x, y in cells:
        if not (0 <= x <= 3 and 0 <= y <= 2):
            raise ValueError(f"cell out of the grid: ({x}, {y})")
        mask |= 1 << (y * 4 + x)
    return mask


def mask_to_cells(mask: int) -> list[tuple[int, int]]:
    return [(b % 4, b // 4) for b in range(12) if mask >> b & 1]
