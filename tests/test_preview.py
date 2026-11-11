import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.cli import app
from bsmaker.formats.model import Note
from bsmaker.formats.writer import write_map
from bsmaker.preview import render_preview

SR = SAMPLE_RATE


def _song(path, secs=5.0, amp=0.01, silent=False):
    rng = np.random.default_rng(0)
    x = np.zeros(int(secs * SR)) if silent else amp * rng.standard_normal(int(secs * SR))
    sf.write(str(path), np.clip(x, -1, 1).astype(np.float32), SR, subtype="PCM_16")
    return path


def _map(tmp_path, notes, name="map", rank="Expert", **song):
    audio = _song(tmp_path / "src.wav", **song)
    return write_map(tmp_path / name, title="T", artist="A", bpm=120.0, rank=rank,
                     njs=16.0, njs_offset=0.0, notes=notes, audio_src=audio)


def _n(t, color):
    return Note(t, t * 2, 1, 1, color, 8)


def _band(x, f, t0, n):
    seg = x[int(t0 * SR):int(t0 * SR) + n]
    w = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * w))
    k = int(round(f * len(seg) / SR))
    return spec[k - 2:k + 3].max()


def test_clicks_at_note_times(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0), _n(3.0, 0)])
    out = render_preview(folder, tmp_path / "o.wav")
    x, sr = sf.read(str(out))
    assert sr == SR
    n = int(0.015 * SR)
    at = [_band(x, 1000, t, n) for t in (1.0, 3.0)]
    between = [_band(x, 1000, t, n) for t in (2.0, 4.0)]
    assert min(at) > 10 * max(between)


def test_left_and_right_pitches_differ(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0), _n(3.0, 1)])
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")))
    n = int(0.015 * SR)
    assert _band(x, 1000, 1.0, n) > 3 * _band(x, 1500, 1.0, n)
    assert _band(x, 1500, 3.0, n) > 3 * _band(x, 1000, 3.0, n)


def test_output_not_clipped(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0), _n(1.005, 1), _n(2.0, 0)], amp=1.0)
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")), dtype="int16")
    assert np.max(np.abs(x.astype(int))) <= int(0.9 * 32767) + 50
    assert np.max(np.abs(x.astype(int))) < 32767


def test_missing_audio_raises(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)])
    for f in folder.iterdir():
        if f.suffix in (".ogg", ".egg", ".wav", ".mp3"):
            f.unlink()
    with pytest.raises(ValueError, match="no audio"):
        render_preview(folder, tmp_path / "o.wav")


def test_default_output_outside_map_folder(tmp_path, monkeypatch):
    folder = _map(tmp_path, [_n(1.0, 0)])
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    before = sorted(p.name for p in folder.iterdir())
    res = CliRunner().invoke(app, ["preview", str(folder)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "work" / "preview - map.wav").is_file()
    assert sorted(p.name for p in folder.iterdir()) == before


def test_bad_folder_exits_1(tmp_path):
    res = CliRunner().invoke(app, ["preview", str(tmp_path / "nope")])
    assert res.exit_code == 1


def test_cwd_inside_map_folder_refused(tmp_path, monkeypatch):
    folder = _map(tmp_path, [_n(1.0, 0)])
    before = sorted(p.name for p in folder.iterdir())
    monkeypatch.chdir(folder)
    res = CliRunner().invoke(app, ["preview", "."])
    assert res.exit_code == 1
    assert sorted(p.name for p in folder.iterdir()) == before


def test_explicit_out_inside_map_folder_refused(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)])
    before = sorted(p.name for p in folder.iterdir())
    res = CliRunner().invoke(app, ["preview", str(folder), "--out", str(folder / "x.wav")])
    assert res.exit_code == 1
    assert sorted(p.name for p in folder.iterdir()) == before
    with pytest.raises(ValueError, match="inside the map folder"):
        render_preview(folder, folder)


def test_bad_out_paths(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)])
    with pytest.raises(ValueError):
        render_preview(folder, tmp_path / "nodir" / "o.wav")
    (tmp_path / "d").mkdir()
    with pytest.raises(ValueError):
        render_preview(folder, tmp_path / "d")
    res = CliRunner().invoke(app, ["preview", str(folder), "--out", str(tmp_path / "nodir" / "o.wav")])
    assert res.exit_code == 1


def test_no_temp_file_left(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)])
    render_preview(folder, tmp_path / "o.wav")
    assert not (tmp_path / "o.wav.tmp").exists()




def test_rank_selection_unit():
    from bsmaker.preview import _rank_key
    assert _rank_key("Expert+") == _rank_key("expertplus") == _rank_key("ExpertPlus")
    assert _rank_key("Easy") < _rank_key("Hard") < _rank_key("Expert") < _rank_key("ExpertPlus")
    assert _rank_key("Weird") < _rank_key("Easy")


def test_rank_explicit_and_unknown(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)], rank="ExpertPlus")
    render_preview(folder, tmp_path / "a.wav", rank="Expert+")
    render_preview(folder, tmp_path / "b.wav", rank="expertplus")
    render_preview(folder, tmp_path / "c.wav")
    with pytest.raises(ValueError):
        render_preview(folder, tmp_path / "d.wav", rank="Easy")
    with pytest.raises(ValueError):
        render_preview(folder, tmp_path / "d.wav", rank="Bogus")


def test_out_of_range_notes_ignored(tmp_path):
    folder = _map(tmp_path, [_n(-1.0, 0), _n(1.0, 0), _n(99.0, 1)])
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")))
    assert len(x) == 5 * SR
    n = int(0.015 * SR)
    assert _band(x, 1000, 1.0, n) > 10 * _band(x, 1000, 2.0, n)


def test_simultaneous_left_right(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0), _n(1.0, 1)])
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")))
    n = int(0.015 * SR)
    for f in (1000, 1500):
        assert _band(x, f, 1.0, n) > 10 * _band(x, f, 3.0, n)


def test_sample_accuracy(tmp_path):
    t = 1.2345
    folder = _map(tmp_path, [_n(t, 0)], silent=True)
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")), dtype="int16")
    first = int(np.flatnonzero(x)[0])
    # first sample of a linear fade-in is 0, so allow a couple of samples
    assert abs(first - round(t * SR)) <= 2


def test_preview_refuses_output_in_custom_levels(tmp_path, monkeypatch):
    folder = _map(tmp_path, [_n(1.0, 0)])
    levels = tmp_path / "levels"
    levels.mkdir()
    cfg = tmp_path / "bsmaker.toml"
    cfg.write_text(f"custom_levels = '{levels}'\ndata_dir = '{tmp_path / 'data'}'\n",
                   encoding="utf-8")
    from bsmaker import cli
    from bsmaker.config import load_config
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: load_config(cfg))
    monkeypatch.chdir(levels)
    res = CliRunner().invoke(app, ["preview", str(folder)])
    assert res.exit_code == 1 and "CustomLevels" in res.output
    assert list(levels.iterdir()) == []
    res = CliRunner().invoke(app, ["preview", str(folder), "--out", str(levels / "x.wav")])
    assert res.exit_code == 1 and list(levels.iterdir()) == []
