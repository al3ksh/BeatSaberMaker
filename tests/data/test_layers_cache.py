import json

import numpy as np
import pytest

from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.layer_stats import compute_layer_stats, level_rate, load_layer_stats, write_layer_stats
from bsmaker.data.layers_cache import LayerCache, build_layer_cache
from bsmaker.data.manifest import open_manifest, upsert_map
from bsmaker.data.stats import NPS_BINS
from bsmaker.formats.events import FRAMES_PER_SECOND

N_FRAMES = int(10 * FRAMES_PER_SECOND)  # about 10 s


def _info(env, fname="E.dat"):
    return {"_version": "2.0.0", "_songName": "S", "_songAuthorName": "A", "_levelAuthorName": "M",
            "_beatsPerMinute": 120, "_songFilename": "song.egg", "_environmentName": env,
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                                        "_difficultyBeatmaps": [{"_difficulty": "Expert",
                                                                 "_beatmapFilename": fname,
                                                                 "_noteJumpMovementSpeed": 16,
                                                                 "_noteJumpStartBeatOffset": 0}]}]}


def _folder(root, name, env, diff):
    f = root / name
    f.mkdir()
    (f / "Info.dat").write_text(json.dumps(_info(env)), encoding="utf-8")
    (f / "E.dat").write_text(json.dumps(diff), encoding="utf-8")
    return f


V2 = {"_version": "2.2.0",
      "_notes": [{"_time": 1, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
                 {"_time": 2, "_lineIndex": 1, "_lineLayer": 1, "_type": 3, "_cutDirection": 0}],
      "_obstacles": [{"_time": 4, "_lineIndex": 3, "_type": 0, "_duration": 1, "_width": 1}],
      "_events": [{"_time": 0, "_type": 1, "_value": 3}]}

LIGHTS = ([{"b": i, "et": 1, "i": 1, "f": 1} for i in range(30)]
          + [{"b": 0, "et": 1, "i": 5, "f": 1},       # duplicate (time, channel): this one wins
             {"b": 2, "et": 7, "i": 1, "f": 1},       # unsupported type: skipped
             {"b": 2, "et": 12, "i": 20, "f": 1}])    # speed clipped to 9

V3 = {"version": "3.2.0",
      "colorNotes": [{"b": 1, "x": 1, "y": 0, "c": 0, "d": 1}, {"b": 1, "x": 2, "y": 0, "c": 1, "d": 1},
                     {"b": 3, "x": 1, "y": 0, "c": 0, "d": 1}, {"b": 4, "x": 3, "y": 2, "c": 1, "d": 8}],
      "bombNotes": [{"b": 2, "x": 0, "y": 0}, {"b": 2, "x": 1, "y": 1}, {"b": 3, "x": 5, "y": 0},
                    {"b": 6, "x": 3, "y": 2}],
      "obstacles": [{"b": 4, "x": 0, "y": 0, "d": 2, "w": 1, "h": 5},     # side left, 2 beats
                    {"b": 2, "x": 4, "y": 2, "d": 0.5, "w": 1, "h": 3},   # side right, outside, high
                    {"b": 5, "x": 1, "y": 0, "d": 2, "w": 2, "h": 5},     # centre
                    {"b": 6, "x": -2, "y": 0, "d": 2, "w": 1, "h": 5},    # too far out
                    {"b": 7, "x": 0, "y": 0, "d": 0, "w": 1, "h": 5}],    # flash
      "sliders": [{"b": 1, "c": 0, "x": 1, "y": 0, "d": 1, "mu": 1, "tb": 3, "tx": 2, "ty": 2, "tc": 4,
                   "tmu": 1, "m": 0},
                  {"b": 9, "c": 1, "x": 0, "y": 0, "d": 1, "tb": 10, "tx": 0, "ty": 0, "tc": 1},
                  {"b": 1, "c": 0, "x": 1, "y": 0, "d": 1, "tb": 5, "tx": 0, "ty": 0, "tc": 1},   # 2nd arc, same note
                  {"b": 3, "c": 0, "x": 1, "y": 0, "d": 1, "tb": 4, "tx": 0, "ty": 0, "tc": 12}],  # bad tail direction
      "burstSliders": [{"b": 4, "c": 1, "x": 3, "y": 2, "d": 8, "tb": 5, "tx": 3, "ty": 0, "sc": 9, "s": 0.5}],
      "basicBeatmapEvents": LIGHTS}

V4 = {"version": "4.0.0", "colorNotesData": [], "colorNotes": []}


@pytest.fixture
def built(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    conn = open_manifest(tmp_path / "m.sqlite")
    folders = {"m2": _folder(maps, "m2", "DefaultEnvironment", V2),
               "m3": _folder(maps, "m3", "NiceEnvironment", V3),
               "m4": _folder(maps, "m4", "NiceEnvironment", V4),
               "gone": maps / "gone"}
    for k, f in folders.items():
        upsert_map(conn, {"key": k, "folder": str(f), "bs_tags": "[]"})
    ev = tmp_path / "events"
    ev.mkdir()
    index = [{"map_key": k, "characteristic": "Standard", "rank": "Expert",
              "split": "train" if k != "gone" else "val", "nps": 5.0, "styles": [], "njs": 16.0,
              "njs_offset": 0.0, "n_frames": N_FRAMES} for k in ("m2", "m3", "m4", "gone")]
    (ev / "index.json").write_text(json.dumps(index))
    np.savez(ev / "events.npz", time_s=np.zeros(0, np.float32), left=np.zeros(0, np.uint8),
             right=np.zeros(0, np.uint8), offsets=np.zeros(5, np.int64))
    out = tmp_path / "layers"
    totals = build_layer_cache(conn, out, ev)
    return LayerCache.load(out), totals, out, ev


def test_records_align_with_event_cache(built):
    cache, totals, _, ev = built
    entries = EventCache.load(ev).entries
    assert [(r["i"], r["map_key"]) for r in cache.records] == [(e.i, e.map_key) for e in entries]
    assert [cache.meta(i)["schema"] for i in range(4)] == ["v2", "v3", "v4", ""]
    assert [cache.meta(i)["has_v3"] for i in range(4)] == [False, True, False, False]
    assert cache.meta(1)["environment"] == "NiceEnvironment" and cache.meta(1)["bpm"] == 120
    assert totals["entries"] == 4 and totals["v4"] == 1 and totals["empty"] == 1
    assert totals["unreadable_maps"] == 1


def test_v3_lights(built):
    cache, *_ = built
    t, ch, tok = cache.lights(1)
    assert len(t) == 31 and t.dtype == np.float32 and ch.dtype == np.uint8
    assert (t[0], ch[0], tok[0]) == (0.0, 1, 1)     # duplicate: last event (value 5 -> token 1) kept
    assert (ch == 7).sum() == 1 and tok[ch == 7][0] == 16
    m = cache.meta(1)
    assert m["lights_skipped"] == 1 and m["lights_duplicate"] == 1 and m["n_lights"] == 31
    assert m["lights_ok"] is True


def test_lights_eligibility(built):
    cache, *_ = built
    assert cache.meta(0)["lights_ok"] is False   # one light event per 10 s
    assert cache.meta(2)["lights_ok"] is False
    assert len(cache.lights(0)[0]) == 1


def test_v3_walls(built):
    cache, *_ = built
    t, side, tok = cache.walls(1)
    assert list(t) == [1.0, 2.0, 3.5]
    assert list(side) == [1, 0, 0]
    assert list(tok) == [(1 * 2 + 1) * 6 + 2, 4, 0]
    m = cache.meta(1)
    assert m["wall_classes"] == {"flash": 1, "side": 3, "centre": 1}
    assert m["walls_unsupported"] == 1 and m["n_walls"] == 3


def test_v3_bombs(built):
    cache, *_ = built
    t, mask = cache.bombs(1)
    assert list(t) == [1.0, 3.0] and mask.dtype == np.uint16
    assert list(mask) == [1 | 1 << 5, 1 << 11]
    assert cache.meta(1)["n_bombs"] == 3 and cache.meta(1)["n_bomb_groups"] == 2


def test_v3_ornaments_matched_to_notes(built):
    cache, *_ = built
    t, hand, token, chain, arc_len, arc_cell, arc_dir = cache.ornaments(1)
    assert list(t) == [0.5, 0.5, 1.5, 2.0]
    assert list(hand) == [0, 1, 0, 1]
    assert token[0] == 1 + 1 * 27 + 0 * 9 + 1 and token[3] == 1 + 3 * 27 + 2 * 9 + 8
    assert list(chain) == [0, 0, 0, 7]            # 9 segments clipped to 7
    assert list(arc_len) == [4, 0, 0, 0]          # 2 beats -> index 4; second arc on the note ignored
    assert list(arc_cell) == [2 * 4 + 2, 0, 0, 0] and list(arc_dir) == [4, 0, 0, 0]
    m = cache.meta(1)
    assert (m["n_arcs"], m["arcs_unmatched"], m["n_chains"], m["chains_unmatched"]) == (1, 3, 1, 0)


def test_v2_and_v4_entries(built):
    cache, *_ = built
    # v2 has one note (the bomb is not a note) and one right-side wall
    assert len(cache.ornaments(0)[0]) == 1
    assert list(cache.walls(0)[1]) == [1]
    assert cache.meta(0)["n_bombs"] == 1 and cache.meta(0)["n_bomb_groups"] == 1
    for i in (2, 3):
        for arr in (*cache.lights(i), *cache.walls(i), *cache.bombs(i), *cache.ornaments(i)):
            assert len(arr) == 0
        m = cache.meta(i)
        assert m["lights_ok"] is False and m["n_walls"] == 0
    assert cache.meta(2)["readable"] is True and cache.meta(3)["readable"] is False


def test_npz_layout(built):
    _, _, out, _ = built
    z = np.load(out / "layers.npz")
    assert list(z["light_offsets"]) == [0, 1, 32, 32, 32]
    assert z["bomb_mask"].dtype == np.uint16 and z["orn_arc_cell"].dtype == np.uint8
    assert len(json.loads((out / "index.json").read_text())) == 4
    assert not list(out.glob("*.tmp"))


def test_inconsistent_index_rejected(built):
    _, _, out, _ = built
    idx = json.loads((out / "index.json").read_text())
    (out / "index.json").write_text(json.dumps(idx[:2]))
    with pytest.raises(ValueError, match="inconsistent"):
        LayerCache.load(out)


def test_unencodable_notes_are_counted():
    # the reader already drops off-grid notes; the counter guards the cache against other callers
    from bsmaker.data.layers_cache import _empty_record, _ornaments
    from bsmaker.formats.model import Difficulty, DifficultyFlags, Layers, Note
    lay = Layers((), (), (), (), (), False, 0)
    d = Difficulty("Standard", "Expert", "v3", 16.0, 0.0, (Note(1.0, 2.0, 1, 0, 0, 9),),
                   0, 0, 0, 0, DifficultyFlags(False, False, False, False), lay)
    rec = _empty_record(0)
    orn = _ornaments(d, rec)
    assert rec["notes_unencodable"] == 1 and orn["orn_token"] == [0]


# ---------------------------------------------------------------- statistics

def _toy(entries):
    """entries: list of dict(split, nps, n_walls, bombs=[masks], lights_ok, ch=[channels])."""
    n = len(entries)
    ec_entries = [Entry(i, f"k{i}", "Standard", "Expert", e["split"], e["nps"], (), 16.0, 0.0,
                        int(round(FRAMES_PER_SECOND * 60))) for i, e in enumerate(entries)]
    ec = EventCache(ec_entries, np.zeros(0, np.float32), np.zeros(0, np.uint8),
                    np.zeros(0, np.uint8), np.zeros(n + 1, np.int64))
    records, data = [], {}
    cols = {"light": ["light_t", "light_ch", "light_tok"], "bomb": ["bomb_t", "bomb_mask"],
            "wall": ["wall_t", "wall_side", "wall_tok"],
            "orn": ["orn_t", "orn_hand", "orn_token", "orn_chain", "orn_arc_len", "orn_arc_cell",
                    "orn_arc_dir"]}
    lengths = {g: [] for g in cols}
    for e in entries:
        records.append({"n_arcs": 0, "n_chains": 0, "n_walls": e.get("n_walls", 0),
                        "n_bomb_groups": len(e.get("bombs", [])), "lights_ok": e.get("lights_ok", False)})
        lengths["light"].append(len(e.get("ch", [])))
        lengths["bomb"].append(len(e.get("bombs", [])))
        lengths["wall"].append(0)
        lengths["orn"].append(0)
    for g, ks in cols.items():
        off = np.zeros(n + 1, np.int64)
        np.cumsum(lengths[g], out=off[1:])
        data[f"{g}_offsets"] = off
        for k in ks:
            data[k] = np.zeros(int(off[-1]), np.float32)
    data["light_ch"] = np.array([c for e in entries for c in e.get("ch", [])], np.uint8)
    data["bomb_mask"] = np.array([m for e in entries for m in e.get("bombs", [])], np.uint16)
    return LayerCache(records, data), ec


def test_layer_stats_percentiles_and_share():
    bombs = [5] * 18 + [7, 9]
    cache, ec = _toy([
        {"split": "train", "nps": 5.0, "n_walls": 1, "bombs": bombs},
        {"split": "train", "nps": 5.0, "n_walls": 2},
        {"split": "train", "nps": 5.0, "n_walls": 3, "lights_ok": True, "ch": [0, 0, 0, 1]},
        {"split": "train", "nps": 5.0, "n_walls": 4},
        {"split": "train", "nps": 5.0},
        {"split": "val", "nps": 5.0, "n_walls": 99}])
    st = compute_layer_stats(cache, ec, {"nps_bins": NPS_BINS})
    assert st["bomb_vocab"] == [5, 7]
    assert st["bomb_coverage"] == pytest.approx(0.95)
    b = 4  # nps 5.0 lies in [4.5, 5.5)
    assert st["rates"]["walls"][b] == pytest.approx([1.75, 2.5, 3.25], rel=1e-3)
    assert st["share"]["walls"] == pytest.approx(4 / 5)
    assert st["share"]["arcs"] == 0.0
    assert st["share"]["light_ch0"] == 1.0 and st["share"]["light_ch2"] == 0.0
    assert st["rates"]["light_ch0"][b][1] == pytest.approx(3.0, rel=1e-3)
    assert st["rates"]["arcs"][b] == [0.0, 0.0, 0.0]
    # empty bins are filled from the nearest bin
    assert st["rates"]["walls"][0] == st["rates"]["walls"][b]


def test_level_rate(tmp_path):
    cache, ec = _toy([{"split": "train", "nps": 5.0, "n_walls": n} for n in (1, 2, 3, 4)])
    st = compute_layer_stats(cache, ec, {"nps_bins": NPS_BINS})
    path = tmp_path / "layer_stats.json"
    write_layer_stats(st, path)
    st = load_layer_stats(path)
    assert level_rate(st, "walls", 5.0, 0) == 0.0
    assert [level_rate(st, "walls", 5.0, k) for k in (1, 2, 3)] == pytest.approx([1.75, 2.5, 3.25], rel=1e-3)
    assert level_rate(st, "walls", 0.8, 2) == pytest.approx(2.5, rel=1e-3)   # empty bin, nearest fill
    assert level_rate(st, "arcs", 5.0, 3) == 0.0
    with pytest.raises(ValueError):
        level_rate(st, "walls", 5.0, 4)
