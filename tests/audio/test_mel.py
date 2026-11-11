import numpy as np
import soundfile as sf

import bsmaker.audio.mel as mel_mod
import bsmaker.formats.events as events
from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import log_mel, time_to_frame, cached_log_mel


def test_log_mel_shape_and_dtype():
    m = log_mel(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    assert m.dtype == np.float16
    assert abs(m.shape[0] - 173) <= 1 and m.shape[1] == 80


def test_click_lands_on_expected_frame():
    a = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    a[SAMPLE_RATE] = 1.0
    m = log_mel(a).astype(np.float32)
    energy = np.exp(m).sum(axis=1)
    assert abs(int(np.argmax(energy)) - time_to_frame(1.0)) <= 1


def test_frame_rate_matches_events():
    assert mel_mod.FRAMES_PER_SECOND == events.FRAMES_PER_SECOND


def test_cache_hit_and_invalidation(tmp_path, monkeypatch):
    src = tmp_path / "a.wav"
    sf.write(src, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    cache = tmp_path / "a.npy"
    first = cached_log_mel(src, cache)
    calls = []
    real = mel_mod.decode_mono
    monkeypatch.setattr(mel_mod, "decode_mono", lambda p: calls.append(p) or real(p))
    second = cached_log_mel(src, cache)
    assert calls == [] and np.array_equal(first, second)
    sf.write(src, np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    third = cached_log_mel(src, cache)
    assert len(calls) == 1 and third.shape[0] > first.shape[0]


def test_missing_sidecar_is_a_miss(tmp_path, monkeypatch):
    src = tmp_path / "a.wav"
    sf.write(src, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    cache = tmp_path / "a.npy"
    cached_log_mel(src, cache)
    (tmp_path / "a.npy.json").unlink()
    calls = []
    real = mel_mod.decode_mono
    monkeypatch.setattr(mel_mod, "decode_mono", lambda p: calls.append(p) or real(p))
    cached_log_mel(src, cache)
    assert len(calls) == 1


def test_corrupt_cached_npy_is_a_miss(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(src, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    cache = tmp_path / "a.npy"
    first = cached_log_mel(src, cache)
    cache.write_bytes(b"truncated garbage")  # sidecar still matches the source
    again = cached_log_mel(src, cache)
    assert np.array_equal(first, again)
    assert np.array_equal(np.load(cache), first)
