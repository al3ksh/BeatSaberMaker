import json
from pathlib import Path

import httpx

from bsmaker.data.beatsaver import BeatSaverClient
from bsmaker.data.fetch_metadata import fetch_metadata
from bsmaker.data.manifest import open_manifest, upsert_map

FIX = Path(__file__).parent.parent / "fixtures" / "beatsaver"


def _setup(tmp_path, keys, fetched=()):
    conn = open_manifest(tmp_path / "m.sqlite")
    for k in keys:
        upsert_map(conn, {"key": k, "source": "local", "bs_fetched": 1 if k in fetched else 0})
    return conn


def _client(requests=None):
    body = (FIX / "maps_by_ids.json").read_text(encoding="utf-8")

    def handler(req):
        if requests is not None:
            requests.append(req.url.path)
        return httpx.Response(200, text=body)

    return BeatSaverClient(transport=httpx.MockTransport(handler))


def test_fetch_metadata_updates_rows(tmp_path):
    conn = _setup(tmp_path, ["2"])
    assert fetch_metadata(conn, _client()) == (1, 0)
    r = conn.execute("SELECT * FROM maps WHERE key='2'").fetchone()
    assert r["bs_fetched"] == 1 and r["bs_hash"] == "1178b3cbb90746a210ebba8b6a1f0ac4117b0654"
    assert json.loads(r["bs_tags"]) == ["anime"]
    assert r["bs_upvotes"] == 287 and abs(r["bs_score"] - 0.8677) < 1e-9
    assert (r["bs_ranked"], r["bs_bl_ranked"], r["bs_curated"], r["bs_automapper"]) == (0, 0, 0, 0)
    assert r["source"] == "local"  # other columns untouched


def test_fetch_metadata_skips_local_keys_and_fetched(tmp_path):
    reqs = []
    conn = _setup(tmp_path, ["local-abc", "2", "3a"], fetched={"3a"})
    assert fetch_metadata(conn, _client(reqs)) == (1, 0)
    assert len(reqs) == 1 and "2" in reqs[0] and "3a" not in reqs[0] and "local" not in reqs[0]
    reqs.clear()
    assert fetch_metadata(conn, _client(reqs)) == (0, 0) and reqs == []


def test_fetch_metadata_marks_not_found(tmp_path):
    conn = _setup(tmp_path, ["2", "dead"])
    assert fetch_metadata(conn, _client()) == (1, 1)
    r = conn.execute("SELECT * FROM maps WHERE key='dead'").fetchone()
    assert r["bs_fetched"] == 1 and r["bs_hash"] is None and r["bs_tags"] is None
