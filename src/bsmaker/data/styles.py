"""Map style tags (BeatSaver tags that describe how a map plays)."""
from __future__ import annotations

from collections.abc import Iterable

STYLES = ("tech", "dance", "speed", "accuracy", "balanced", "challenge", "fitness")
UNKNOWN = "unknown"
_TAG_TO_STYLE = {"dance-style": "dance", **{s: s for s in STYLES if s != "dance"}}


def styles_from_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Style names found in `tags` (in STYLES order); other tags are ignored."""
    found = {_TAG_TO_STYLE[t] for t in tags if t in _TAG_TO_STYLE}
    result = tuple(s for s in STYLES if s in found)
    return result or (UNKNOWN,)
