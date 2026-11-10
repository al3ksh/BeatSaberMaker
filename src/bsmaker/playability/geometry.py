"""Swing-angle geometry for note directions."""
from __future__ import annotations

_ANGLES: dict[int, float] = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


def direction_angle(direction: int) -> float | None:
    """Cut angle in degrees (0 = right, counter-clockwise); None for a dot."""
    return _ANGLES.get(direction)


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, 0..180."""
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d
