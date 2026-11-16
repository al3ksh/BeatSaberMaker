import json

import numpy as np
import pytest
from typer.testing import CliRunner

from bsmaker.cli import app
from bsmaker.data.manifest import open_manifest, upsert_map


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "bsmaker.toml").write_text(
        f"custom_levels = '{tmp_path / 'levels'}'\ndata_dir = '{data}'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path, data


def test_build_layers_needs_event_cache(env):
    r = CliRunner().invoke(app, ["data", "build-layers"])
    assert r.exit_code == 1 and "build-events" in r.output


def test_build_layers_needs_dataset_stats(env):
    tmp, data = env
    (data / "events").mkdir()
    (data / "events" / "index.json").write_text("[]")
    (data / "events" / "events.npz").write_bytes(b"")
    r = CliRunner().invoke(app, ["data", "build-layers"])
    assert r.exit_code == 1 and "dataset_stats.json" in r.output


def test_build_layers_end_to_end(env):
    tmp, data = env
    folder = tmp / "m1"
    folder.mkdir()
    info = {"_version": "2.0.0", "_songName": "S", "_beatsPerMinute": 120, "_songFilename": "s.egg",
            "_environmentName": "DefaultEnvironment",
            "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
                {"_difficulty": "Expert", "_beatmapFilename": "E.dat", "_noteJumpMovementSpeed": 16,
                 "_noteJumpStartBeatOffset": 0}]}]}
    diff = {"version": "3.0.0", "colorNotes": [{"b": 1, "x": 1, "y": 0, "c": 0, "d": 1}],
            "bombNotes": [{"b": 2, "x": 0, "y": 0}], "obstacles": [], "sliders": [], "burstSliders": [],
            "basicBeatmapEvents": []}
    (folder / "Info.dat").write_text(json.dumps(info))
    (folder / "E.dat").write_text(json.dumps(diff))
    conn = open_manifest(data / "manifest.sqlite")
    upsert_map(conn, {"key": "m1", "folder": str(folder), "bs_tags": "[]"})
    conn.close()
    ev = data / "events"
    ev.mkdir()
    (ev / "index.json").write_text(json.dumps([{
        "map_key": "m1", "characteristic": "Standard", "rank": "Expert", "split": "train", "nps": 5.0,
        "styles": [], "njs": 16.0, "njs_offset": 0.0, "n_frames": 1000}]))
    np.savez(ev / "events.npz", time_s=np.zeros(0, np.float32), left=np.zeros(0, np.uint8),
             right=np.zeros(0, np.uint8), offsets=np.zeros(2, np.int64))
    (tmp / "dataset_stats.json").write_text(json.dumps({"nps_bins": [0.5, 4.5, 12]}))
    r = CliRunner().invoke(app, ["data", "build-layers"])
    assert r.exit_code == 0, r.output
    assert "bomb vocabulary: 1 masks" in r.output and "lightshow-eligible difficulties: 0" in r.output
    stats = json.loads((tmp / "layer_stats.json").read_text())
    assert stats["bomb_vocab"] == [1] and stats["share"]["bomb_groups"] == 1.0
    assert (data / "layers" / "layers.npz").is_file()
