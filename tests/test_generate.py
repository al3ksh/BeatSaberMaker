import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from bsmaker import generate as G
from bsmaker.formats.events import to_events
from bsmaker.formats.model import Note
from bsmaker.formats.reader import read_map
from bsmaker.generate import GenerateError, generate_map, pick_njs, pick_rank, safe_title
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import LoadedPattern, SampleReport
from bsmaker.models.timing.decode import LoadedTiming
from bsmaker.models.timing.model import TimingConfig, TimingModel
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

STATS = {
    "mel_mean": 0.0, "mel_std": 1.0, "coverage_median": 0.8, "nps_bins": [0.0, 3.0, 100.0],
    "double_share_by_bin": [0.2, 0.3], "njs_by_bin": [12.0, 16.0], "njs_offset_by_bin": [0.0, 0.5],
    "rank_median_nps": {"Easy": 1.0, "Hard": 3.0, "ExpertPlus": 6.0},
}
MASK = PlayabilityConfig()


def make_models(seed: int = 0) -> tuple[LoadedTiming, LoadedPattern]:
    torch.manual_seed(seed)
    tcfg = TimingConfig(d_model=32, n_layers=2, n_heads=4, window=128, dropout=0.0)
    pcfg = PatternConfig(d_model=32, n_layers=2, n_heads=2, context=16, dropout=0.0)
    return (LoadedTiming(TimingModel(tcfg).eval(), tcfg, STATS),
            LoadedPattern(PatternModel(pcfg).eval(), pcfg, STATS))


def write_audio(path: Path, seconds: float = 40.0, silent: bool = False) -> Path:
    sr = 22050
    y = np.zeros(int(seconds * sr), np.float32)
    if not silent:
        rng = np.random.default_rng(0)
        for t in np.arange(0.5, seconds - 0.1, 0.5):
            i = int(t * sr)
            y[i:i + 1000] = rng.normal(size=1000) * 0.5
    sf.write(path, y, sr)
    return path


@pytest.fixture(scope="module")
def models():
    return make_models()


def _gen(models, audio, out, **kw):
    args = dict(nps=3.0, styles=("tech",), seed=1, out_dir=out, timing=models[0],
                pattern=models[1], mask_cfg=MASK, stats=STATS, device="cpu")
    args.update(kw)
    return generate_map(audio, **args)


def test_generate_end_to_end(models, tmp_path):
    res = _gen(models, write_audio(tmp_path / "My Song.wav"), tmp_path / "out")
    assert res.folder.is_dir() and res.folder.name == "BSM - My Song [nps 3, tech]"
    assert res.rank == "Hard" and res.n_notes > 20
    m = read_map(res.folder)
    assert not m.problems
    assert len(m.difficulties) == 1 and m.difficulties[0].characteristic == "Standard"
    notes = m.difficulties[0].notes
    assert [n.beat for n in notes] == sorted(n.beat for n in notes)
    events, dropped = to_events(list(notes))
    assert dropped == 0 and check(events, MASK).hard == ()


def test_generated_folder_has_only_expected_files(models, tmp_path):
    res = _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out")
    assert {p.name for p in res.folder.iterdir()} == {
        "Info.dat", "HardStandard.dat", "song.egg", "cover.png"}
    assert [p.name for p in (tmp_path / "out").iterdir()] == [res.folder.name]


def test_same_seed_same_map(models, tmp_path):
    audio = write_audio(tmp_path / "a.wav")
    a = _gen(models, audio, tmp_path / "o1", seed=5)
    b = _gen(models, audio, tmp_path / "o2", seed=5)
    assert ((a.folder / "HardStandard.dat").read_bytes()
            == (b.folder / "HardStandard.dat").read_bytes())


def test_short_audio_rejected(models, tmp_path):
    with pytest.raises(GenerateError):
        _gen(models, write_audio(tmp_path / "s.wav", seconds=10), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_silent_audio_rejected(models, tmp_path):
    with pytest.raises(GenerateError):
        _gen(models, write_audio(tmp_path / "s.wav", silent=True), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_safe_title():
    assert safe_title("a / b") == "a b"
    assert safe_title("tab	inside") == "tab inside"
    assert safe_title("AC/DC: Live?") == "AC DC Live"
    assert safe_title("  two   spaces ") == "two spaces"
    assert safe_title("name... ") == "name"
    assert safe_title("x" * 200) == "x" * 80
    assert safe_title("🎵 song") == "🎵 song"
    assert not safe_title(("a" * 79) + ". .").endswith((".", " "))
    assert safe_title("CON") == "CON"
    assert safe_title("???") == "untitled" and safe_title("") == "untitled"


def test_info_keeps_original_title(models, tmp_path):
    audio = write_audio(tmp_path / "a.wav")
    res = _gen(models, audio, tmp_path / "out", title="AC/DC: Live?")
    assert res.folder.name == "BSM - AC DC Live [nps 3, tech]"
    info = json.loads((res.folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_songName"] == "AC/DC: Live?"


def test_to_planned_times_maps_frames_and_rejects_unknown():
    fps = 22050 / 256
    t = 100 / fps  # exactly on a frame
    planned = np.array([t, t + 0.2 / fps, 2.0])  # first two share a frame; first wins
    notes = [Note(t, 0.0, 0, 0, 0, 1), Note(round(2.0 * fps) / fps, 0.0, 0, 0, 0, 1)]
    out = G._to_planned_times(notes, planned)
    assert [n.time_s for n in out] == [t, 2.0]
    with pytest.raises(RuntimeError):
        G._to_planned_times([Note(5.0, 0.0, 0, 0, 0, 1)], planned)


def test_later_round_below_min_events_keeps_best_attempt(models, tmp_path, monkeypatch):
    calls = []

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0):
        calls.append(count_scale)
        n = 240 if len(calls) == 1 else G.MIN_EVENTS - 2  # round 1 overshoots 2x
        return 2.0 + np.arange(n) * (30.0 / n)

    monkeypatch.setattr(G, "predict_events", fake_predict)
    _stub_sampler(monkeypatch)
    res = _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out", nps=4.0)
    assert len(calls) == 2 and res.density_rounds == 2 and res.n_notes == 240
    assert any("stopped early" in w for w in res.warnings)
    assert any("density" in w and "outside" in w for w in res.warnings)


def test_existing_folder_needs_overwrite(models, tmp_path):
    audio = write_audio(tmp_path / "a.wav")
    first = _gen(models, audio, tmp_path / "out")
    with pytest.raises(FileExistsError):
        _gen(models, audio, tmp_path / "out")
    assert _gen(models, audio, tmp_path / "out", overwrite=True).folder == first.folder


def _stub_sampler(monkeypatch):
    def fake_sample(loaded, times, mel, nps, styles, seed, cfg, device, **kw):
        notes = [Note(round(t * 22050 / 256) * 256 / 22050, 0.0, 0, 0, 0, 1) for t in times]
        return notes, SampleReport(len(times), 0, 0, 0.0)

    monkeypatch.setattr(G, "sample_notes", fake_sample)
    # the stub ignores the mask, so its notes are not meant to pass the final playability check
    monkeypatch.setattr(G, "check", lambda events, cfg: SimpleNamespace(hard=()))


def test_density_loop_rescales(models, tmp_path, monkeypatch):
    calls = []

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0):
        calls.append(count_scale)
        n = int(120 * count_scale)  # 30 s span: 4 nps at scale 1
        return 2.0 + np.arange(n) * (30.0 / n)

    monkeypatch.setattr(G, "predict_events", fake_predict)
    _stub_sampler(monkeypatch)
    res = _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out", nps=3.0)
    assert 2 <= len(calls) <= G.MAX_DENSITY_ROUNDS and res.density_rounds == len(calls)
    assert abs(res.nps / 3.0 - 1) <= G.DENSITY_TOLERANCE
    assert not any("density" in w for w in res.warnings)


def test_density_warning_when_not_reached(models, tmp_path, monkeypatch):
    calls = []

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0):
        calls.append(count_scale)
        return 2.0 + np.arange(120) * 0.25  # ignores count_scale: always 4 nps

    monkeypatch.setattr(G, "predict_events", fake_predict)
    _stub_sampler(monkeypatch)
    res = _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out", nps=2.0)
    assert len(calls) == G.MAX_DENSITY_ROUNDS and res.density_rounds == len(calls)
    assert any("density" in w for w in res.warnings)


def test_pick_rank_and_njs_from_stats():
    assert pick_rank(1.2, STATS) == "Easy"
    assert pick_rank(4.0, STATS) == "Hard"
    assert pick_rank(9.0, STATS) == "ExpertPlus"
    assert pick_rank(4.0, {}) == "Expert"
    assert pick_njs(2.0, STATS) == (12.0, 0.0)
    assert pick_njs(5.0, STATS) == (16.0, 0.5)


def test_require_stats_names_missing_keys():
    with pytest.raises(GenerateError, match="coverage_median"):
        G.require_stats({})
    G.require_stats(STATS, ("mel_mean", "mel_std"))
