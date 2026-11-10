import numpy as np
import pytest
import soundfile as sf

from bsmaker.audio.decode import SAMPLE_RATE, AudioDecodeError, decode_mono, write_compact


def _tone(sr=44100, secs=2.0, channels=2):
    t = np.arange(int(sr * secs)) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return np.stack([x] * channels, axis=1)


def test_decode_resamples_to_mono_22050(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, _tone(), 44100)
    a = decode_mono(p)
    assert a.ndim == 1 and a.dtype == np.float32
    assert abs(len(a) - 2 * SAMPLE_RATE) <= 0.01 * 2 * SAMPLE_RATE


def test_decode_ogg(tmp_path):
    p = tmp_path / "a.ogg"
    sf.write(p, _tone(), 44100, format="OGG")
    q = tmp_path / "a.egg"
    p.rename(q)
    a = decode_mono(q)
    assert a.ndim == 1 and abs(len(a) - 2 * SAMPLE_RATE) <= 0.01 * 2 * SAMPLE_RATE


def test_decode_garbage_raises(tmp_path):
    p = tmp_path / "x.egg"
    p.write_text("not audio at all")
    with pytest.raises(AudioDecodeError, match="x.egg"):
        decode_mono(p)


def test_decode_empty_raises(tmp_path):
    p = tmp_path / "e.egg"
    p.write_bytes(b"")
    with pytest.raises(AudioDecodeError, match="e.egg"):
        decode_mono(p)


def test_write_compact_is_mono_22050(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(src, _tone(), 44100)
    out = tmp_path / "c.ogg"
    write_compact(src, out)
    a = decode_mono(out)
    assert abs(len(a) - 2 * SAMPLE_RATE) <= 0.02 * 2 * SAMPLE_RATE
    info = sf.info(out)
    assert info.channels == 1 and info.samplerate == SAMPLE_RATE
