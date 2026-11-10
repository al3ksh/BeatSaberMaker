import json
from pathlib import Path

import httpx
import pytest

from bsmaker.data import beatsaver
from bsmaker.data.beatsaver import BeatSaverClient, BeatSaverError, parse_map_info

FIX = Path(__file__).parent.parent / "fixtures" / "beatsaver"
HOST = "r2cdn.beatsaver.com"


def _client(handler, **kw):
    return BeatSaverClient(transport=httpx.MockTransport(handler), **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(beatsaver.time, "sleep", slept.append)
    return slept


def test_parse_map_info_from_fixture():
    doc = json.loads((FIX / "maps_by_ids.json").read_text(encoding="utf-8"))["2"]
    info = parse_map_info(doc)
    assert info.key == "2"
    assert info.hash == "1178b3cbb90746a210ebba8b6a1f0ac4117b0654"
    assert info.download_url == (
        "https://r2cdn.beatsaver.com/1178b3cbb90746a210ebba8b6a1f0ac4117b0654.zip")
    assert info.title == "Fatima (TV Size) - Kanako Ito (Steins;Gate 0 OP)"
    assert info.mapper == "jugglernaut"
    assert info.tags == ("anime",)
    assert info.score == 0.8677 and info.upvotes == 287
    assert not (info.ranked or info.bl_ranked or info.curated or info.automapper)


def test_parse_search_fixture_flags():
    docs = json.loads((FIX / "search_page.json").read_text(encoding="utf-8"))["docs"]
    infos = [parse_map_info(d) for d in docs]
    assert infos[0].bl_ranked and infos[0].curated and not infos[0].ranked
    assert infos[1].ranked and infos[1].bl_ranked


def test_automapper_or_ai_flag_sets_automapper():
    base = {"id": "a", "versions": [{"hash": "h", "downloadURL": "u", "state": "Published"}]}
    assert parse_map_info({**base, "automapper": True}).automapper
    assert parse_map_info({**base, "declaredAi": "Uploader"}).automapper
    assert not parse_map_info({**base, "declaredAi": "None"}).automapper
    assert not parse_map_info(base).automapper


def test_parse_missing_essentials_raises():
    for doc in ({}, {"id": "a"}, {"id": "a", "versions": []},
                {"id": "a", "versions": [{"hash": "h"}]},
                {"versions": [{"hash": "h", "downloadURL": "u"}]},
                {"id": "a", "versions": "x"}, {"id": "a", "versions": [5]}):
        with pytest.raises(BeatSaverError):
            parse_map_info(doc)


def test_get_maps_by_ids_batches_by_50():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={})

    c = _client(handler)
    assert c.get_maps_by_ids([f"{i:x}" for i in range(120)]) == {}
    assert len(seen) == 3
    assert all(p.startswith("/maps/ids/") for p in seen)
    assert len(seen[0].split("/")[-1].split(",")) == 50


def test_get_maps_by_ids_parses():
    body = (FIX / "maps_by_ids.json").read_text(encoding="utf-8")
    c = _client(lambda r: httpx.Response(200, text=body))
    got = c.get_maps_by_ids(["2", "3a", "zzzz"])
    assert set(got) == {"2", "3a"} and got["3a"].tags == ("rock",)


def test_search_params():
    body = (FIX / "search_page.json").read_text(encoding="utf-8")
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, text=body)

    res = _client(handler).search(2, leaderboard="Ranked")
    assert len(res) == 3
    assert seen[0].url.path == "/search/text/2"
    assert seen[0].url.params["leaderboard"] == "Ranked"


def test_user_agent_header():
    seen = []

    def handler(req):
        seen.append(req.headers["user-agent"])
        return httpx.Response(200, json={})

    _client(handler).get_maps_by_ids(["1"])
    assert seen == ["BeatSaberMaker/0.1"]


def test_retries_on_429_then_succeeds(no_sleep):
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        if n["c"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={})

    _client(handler).get_maps_by_ids(["1"])
    assert n["c"] == 3 and sum(s >= 3 for s in no_sleep) >= 2


def test_gives_up_after_max_retries():
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(503)

    with pytest.raises(BeatSaverError):
        _client(handler, max_retries=3).get_maps_by_ids(["1"])
    assert n["c"] == 4


def test_404_raises_beatsaver_error():
    with pytest.raises(BeatSaverError):
        _client(lambda r: httpx.Response(404, json={})).get_maps_by_ids(["1"])


def test_min_interval_enforced(no_sleep):
    c = _client(lambda r: httpx.Response(200, json={}), min_interval_s=0.25)
    c.get_maps_by_ids(["1"])
    c.get_maps_by_ids(["2"])
    assert no_sleep and 0 < no_sleep[-1] <= 0.25


def test_min_interval_floor():
    with pytest.raises(ValueError):
        BeatSaverClient(min_interval_s=0.1)


@pytest.mark.parametrize("url", [
    "http://r2cdn.beatsaver.com/a.zip", "https://evil.example.com/a.zip",
    "https://r2cdn.beatsaver.com.evil.com/a.zip", "ftp://r2cdn.beatsaver.com/a.zip",
    "file:///etc/passwd", "https://user@evil.com/a.zip"])
def test_download_rejects_bad_url(tmp_path, url):
    called = []
    c = _client(lambda r: called.append(r) or httpx.Response(200, content=b"x"))
    with pytest.raises(BeatSaverError):
        c.download(url, tmp_path / "a.zip")
    assert not called and list(tmp_path.iterdir()) == []


def test_download_success(tmp_path):
    c = _client(lambda r: httpx.Response(200, content=b"hello zip"))
    dest = tmp_path / "a.zip"
    c.download(f"https://{HOST}/a.zip", dest)
    assert dest.read_bytes() == b"hello zip"
    assert [p.name for p in tmp_path.iterdir()] == ["a.zip"]


def test_download_oversize_aborts_and_leaves_no_file(tmp_path):
    c = _client(lambda r: httpx.Response(200, content=b"x" * 1000))
    with pytest.raises(BeatSaverError):
        c.download(f"https://{HOST}/a.zip", tmp_path / "a.zip", max_bytes=100)
    assert list(tmp_path.iterdir()) == []


def test_download_http_error_and_redirect_leave_no_file(tmp_path):
    redirect = httpx.Response(302, headers={"Location": "https://evil.com/x"})
    for resp in (httpx.Response(500), redirect):
        c = _client(lambda r, resp=resp: resp)
        with pytest.raises(BeatSaverError):
            c.download(f"https://{HOST}/a.zip", tmp_path / "a.zip")
        assert list(tmp_path.iterdir()) == []


def _doc(key, versions):
    return {"id": key, "name": key, "versions": versions}


def _ver(h, state, created):
    return {"hash": h, "downloadURL": f"https://{HOST}/{h}.zip", "state": state, "createdAt": created}


def test_parse_picks_newest_published_by_created_at():
    doc = _doc("a", [_ver("new", "Published", "2024-01-01T00:00:00Z"),
                     _ver("old", "Published", "2020-01-01T00:00:00Z"),
                     _ver("test", "Testplay", "2025-01-01T00:00:00Z")])
    assert parse_map_info(doc).hash == "new"


def test_parse_without_published_version_raises():
    with pytest.raises(BeatSaverError):
        parse_map_info(_doc("a", [_ver("t", "Testplay", "2025-01-01T00:00:00Z")]))


def test_by_ids_lowercases_ids_sent():
    seen = []
    c = _client(lambda r: seen.append(r.url.path) or httpx.Response(200, json={}))
    c.get_maps_by_ids(["1A2B", "FF"])
    assert seen == ["/maps/ids/1a2b,ff"]


def test_by_ids_bad_doc_skipped_valid_kept():
    body = {"a": _doc("a", [_ver("h", "Published", "2020-01-01T00:00:00Z")]),
            "b": _doc("b", [_ver("t", "Testplay", "2020-01-01T00:00:00Z")]), "c": None}
    c = _client(lambda r: httpx.Response(200, json=body))
    assert set(c.get_maps_by_ids(["a", "b", "c"])) == {"a"}


def test_by_ids_single_id_shapes():
    d = _doc("a", [_ver("h", "Published", "2020-01-01T00:00:00Z")])
    for body in ({"a": d}, d):
        c = _client(lambda r, body=body: httpx.Response(200, json=body))
        assert set(c.get_maps_by_ids(["a"])) == {"a"}


def test_by_ids_all_unknown_is_empty():
    assert _client(lambda r: httpx.Response(200, json={})).get_maps_by_ids(["x", "y"]) == {}
