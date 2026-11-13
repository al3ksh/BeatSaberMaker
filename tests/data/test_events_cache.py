import json
from types import SimpleNamespace

import numpy as np
import pytest

from bsmaker.data import events_cache as ec
from bsmaker.data.events_cache import EventCache, build_event_cache, training_events
from bsmaker.data.manifest import open_manifest, upsert_difficulty, upsert_map
from bsmaker.formats.model import Note


def _notes(n: int, step: float = 0.5, start: float = 1.0) -> list[Note]:
    return [Note(start + i * step, i, 1, 1, i % 2, 1) for i in range(n)]


def _setup(tmp_path, monkeypatch, maps: dict, with_feat=("a", "b", "c")):
    """maps: key -> {(char, rank): notes}; read_map is faked."""
    conn = open_manifest(tmp_path / "m.sqlite")
    feats = tmp_path / "features"
    feats.mkdir()
    for key, diffs in maps.items():
        upsert_map(conn, {"key": key, "folder": str(tmp_path / key), "bs_tags": '["tech"]'})
        for (ch, rank), _ in diffs.items():
            upsert_difficulty(conn, {"map_key": key, "characteristic": ch, "rank": rank,
                                     "accepted": 1, "split": "val" if key == "b" else "train",
                                     "njs": 18.0, "njs_offset": 0.5})
        if key in with_feat:
            np.save(feats / f"{key}.npy", np.zeros((300, 80), np.float16))

    def fake_read_map(folder):
        diffs = maps[folder.name]
        return SimpleNamespace(difficulties=[
            SimpleNamespace(characteristic=c, rank=r, notes=n) for (c, r), n in diffs.items()])

    monkeypatch.setattr(ec, "read_map", fake_read_map)
    return conn, feats, tmp_path / "events"


def test_cache_roundtrip(tmp_path, monkeypatch):
    na, nb = _notes(30), _notes(25, step=0.4)
    conn, feats, out = _setup(tmp_path, monkeypatch, {
        "a": {("Standard", "Expert"): na}, "b": {("Standard", "Hard"): nb}})
    assert build_event_cache(conn, out, feats) == 0
    cache = EventCache.load(out)
    assert len(cache.entries) == 2
    for entry, notes in zip(cache.entries, (na, nb)):
        exp = training_events(notes)
        t, left, right = cache.events(entry.i)
        assert len(t) == len(exp)
        assert np.allclose(t, [e.time_s for e in exp])
        assert list(left) == [e.left for e in exp]
        assert list(right) == [e.right for e in exp]
        assert entry.n_frames == 300 and entry.styles == ("tech",) and entry.njs == 18.0
    z = np.load(out / "events.npz")
    assert list(z["offsets"]) == [0, 30, 55]
    assert z["time_s"].dtype == np.float32 and z["left"].dtype == np.uint8
    assert len(json.loads((out / "index.json").read_text())) == 2


def test_slider_tails_dropped_in_cache(tmp_path, monkeypatch):
    notes = _notes(30, step=0.5)
    # slider tail: same hand, same direction, 0.1 s after note 0 (color 0)
    notes.append(Note(notes[0].time_s + 0.1, 0.2, 1, 1, 0, 1))
    conn, feats, out = _setup(tmp_path, monkeypatch, {"a": {("Standard", "Expert"): notes}})
    build_event_cache(conn, out, feats)
    cache = EventCache.load(out)
    t, left, right = cache.events(0)
    assert len(t) == 30
    assert np.allclose(t, [1.0 + i * 0.5 for i in range(30)], atol=0.01)  # tail at 1.1 is gone


def test_nps_uses_event_span(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {"a": {("Standard", "Expert"): _notes(21)}})
    build_event_cache(conn, out, feats)
    (e,) = EventCache.load(out).entries
    assert abs(e.nps - 21 / 10.0) < 1e-3


def test_short_or_featureless_difficulty_left_out(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {
        "a": {("Standard", "Expert"): _notes(30), ("Standard", "Easy"): _notes(10)},
        "c": {("Standard", "Hard"): _notes(30)}}, with_feat=("a",))
    assert build_event_cache(conn, out, feats) == 2
    cache = EventCache.load(out)
    assert [(e.map_key, e.rank) for e in cache.entries] == [("a", "Expert")]


def test_select_by_split(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {
        "a": {("Standard", "Expert"): _notes(30)}, "b": {("Standard", "Hard"): _notes(30)}})
    build_event_cache(conn, out, feats)
    cache = EventCache.load(out)
    assert [e.map_key for e in cache.select("val")] == ["b"]
    assert [e.map_key for e in cache.select("train")] == ["a"]


def test_load_rejects_mismatched_index(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {
        "a": {("Standard", "Expert"): _notes(30)}, "b": {("Standard", "Hard"): _notes(25)}})
    build_event_cache(conn, out, feats)
    idx = json.loads((out / "index.json").read_text())
    (out / "index.json").write_text(json.dumps(idx[:1]))
    with pytest.raises(ValueError, match="inconsistent"):
        EventCache.load(out)


def test_quality_fields_and_flags_in_index(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {
        "a": {("Standard", "Expert"): _notes(30)}, "b": {("Standard", "Hard"): _notes(25)}})
    conn.execute("UPDATE maps SET bs_score = 0.69, bs_upvotes = 7 WHERE key = 'a'")
    conn.execute("UPDATE maps SET bs_tags = '[\"tech\", \"ai\"]' WHERE key = 'b'")
    conn.commit()
    build_event_cache(conn, out, feats)
    a, b = EventCache.load(out).entries
    assert (a.score, a.upvotes, a.flags) == (0.69, 7, ("low_score",))
    assert (b.score, b.upvotes, b.flags) == (None, None, ("joke_tag",))


def test_old_index_without_quality_fields_loads(tmp_path, monkeypatch):
    conn, feats, out = _setup(tmp_path, monkeypatch, {"a": {("Standard", "Expert"): _notes(30)}})
    build_event_cache(conn, out, feats)
    raw = json.loads((out / "index.json").read_text())
    for r in raw:
        for k in ("score", "upvotes", "flags"):
            del r[k]
    (out / "index.json").write_text(json.dumps(raw))
    e = EventCache.load(out).entries[0]
    assert (e.score, e.upvotes, e.flags) == (None, None, ())
