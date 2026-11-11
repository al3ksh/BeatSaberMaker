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


def _song(path, secs=5.0):
    rng = np.random.default_rng(0)
    sf.write(str(path), (0.01 * rng.standard_normal(int(secs * SR))).astype(np.float32),
             SR, subtype="PCM_16")
    return path


def _map(tmp_path, notes, name="map"):
    audio = _song(tmp_path / "src.wav")
    return write_map(tmp_path / name, title="T", artist="A", bpm=120.0, rank="Expert",
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
    folder = _map(tmp_path, [_n(1.0, 0), _n(1.005, 1), _n(2.0, 0)])
    x, _ = sf.read(str(render_preview(folder, tmp_path / "o.wav")))
    assert np.max(np.abs(x)) == pytest.approx(0.9, abs=0.01)


def test_missing_audio_raises(tmp_path):
    folder = _map(tmp_path, [_n(1.0, 0)])
    (folder / "song.ogg").unlink(missing_ok=True)
    for f in folder.iterdir():
        if f.suffix in (".ogg", ".egg", ".wav", ".mp3"):
            f.unlink()
    with pytest.raises(ValueError):
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
