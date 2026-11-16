"""Wall classes used to pick training walls."""
from __future__ import annotations

from bsmaker.formats.model import Obstacle

MIN_WALL_S = 0.05


def wall_class(ob: Obstacle) -> str:
    """'flash' (degenerate), 'side' (covers neither column 1 nor 2) or 'centre'."""
    if ob.duration_s <= 0 or ob.w <= 0 or ob.duration_s < MIN_WALL_S:
        return "flash"
    if ob.x + ob.w - 1 < 1 or ob.x > 2:
        return "side"
    return "centre"
