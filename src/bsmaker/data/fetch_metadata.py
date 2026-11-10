"""Fill the bs_* manifest columns from the BeatSaver API."""
from __future__ import annotations

import json
import sqlite3

from bsmaker.data.beatsaver import BATCH_SIZE, BeatSaverClient
from bsmaker.data.manifest import upsert_map

_NOT_FOUND = {"bs_hash": None, "bs_tags": None, "bs_score": None, "bs_upvotes": None,
              "bs_ranked": None, "bs_bl_ranked": None, "bs_curated": None,
              "bs_automapper": None, "bs_fetched": 1}


def fetch_metadata(conn: sqlite3.Connection, client: BeatSaverClient) -> tuple[int, int]:
    """Query BeatSaver for unfetched non-local maps. Returns (updated, not_found)."""
    keys = [r["key"] for r in conn.execute(
        "SELECT key FROM maps WHERE bs_fetched = 0 AND key NOT LIKE 'local-%' ORDER BY key")]
    updated = not_found = 0
    for i in range(0, len(keys), BATCH_SIZE):
        batch = keys[i:i + BATCH_SIZE]
        found = client.get_maps_by_ids(batch)
        for key in batch:
            info = found.get(key.lower())
            if info is None:
                upsert_map(conn, {"key": key, **_NOT_FOUND})
                not_found += 1
                continue
            upsert_map(conn, {
                "key": key, "bs_hash": info.hash, "bs_tags": json.dumps(list(info.tags)),
                "bs_score": info.score, "bs_upvotes": info.upvotes,
                "bs_ranked": int(info.ranked), "bs_bl_ranked": int(info.bl_ranked),
                "bs_curated": int(info.curated), "bs_automapper": int(info.automapper),
                "bs_fetched": 1})
            updated += 1
    return updated, not_found
