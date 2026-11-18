import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from bsmaker import generate as G
from bsmaker.formats.events import to_events
from bsmaker.formats.layer_tokens import WALL_VOCAB
from bsmaker.formats.model import Bomb, Layers, Note, Obstacle
from bsmaker.formats.reader import read_map
from bsmaker.generate import (
    GenerateError, auto_ornaments_level, base_half_jump_beats, generate_map, pick_njs,
    pick_rank, reaction_offset, safe_title)
from bsmaker.layers_generate import LayerModels
from bsmaker.models.layers.spec import LIGHTS, channel_mask, decor_spec
from bsmaker.models.layers.what import LoadedWhat, WhatConfig, WhatModel
from bsmaker.models.layers.when import LoadedWhen, WhenConfig, WhenModel
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel
from bsmaker.models.ornaments.sample import LoadedOrnaments
from bsmaker.models.pattern.model import PatternConfig, PatternModel
from bsmaker.models.pattern.sample import LoadedPattern, SampleReport
from bsmaker.models.timing.decode import LoadedTiming
from bsmaker.models.timing.model import TimingConfig, TimingModel
from bsmaker.playability.checker import Violation, check
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.layers import check_layers

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


def test_later_round_below_min_events_keeps_best_attempt(models, tmp_path, monkeypatch):
    calls = []

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0, **kw):
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

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0, **kw):
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

    def fake_predict(loaded, mel, nps, styles, device, count_scale=1.0, **kw):
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


def test_check_exact_groups_identical_times_without_frame_rounding():
    t = 2.0
    ok = [Note(t, 0.0, 0, 0, 0, 1), Note(t, 0.0, 3, 0, 1, 1)]
    assert G.check_exact(ok, MASK).n_notes == 2
    # two same-hand notes 87.9 ms apart: one frame-rounded merge would hide it
    close = [Note(t, 0.0, 0, 0, 0, 1), Note(t + 0.0879, 0.0, 1, 0, 0, 1)]
    assert any(v.rule == "hand_speed" for v in G.check_exact(close, MASK).hard)


@pytest.mark.parametrize("nps", [0.5, 12.0])
def test_density_extremes_real_pipeline(models, tmp_path, nps):
    audio = write_audio(tmp_path / "a.wav", seconds=60.0)
    try:
        res = _gen(models, audio, tmp_path / "out", nps=nps)
    except GenerateError as e:
        assert str(e)
        return
    notes = list(read_map(res.folder).difficulties[0].notes)
    events, dropped = to_events(notes)
    assert dropped == 0 and check(events, MASK).hard == ()
    assert G.check_exact(notes, MASK).hard == ()


def test_events_are_aligned_then_snapped_before_sampling(models, tmp_path, monkeypatch):
    from bsmaker.audio.beats import Grid

    true_off, bpm = 0.2, 120.0  # beat 0.5 s, 1/4 beat 125 ms
    rng = np.random.default_rng(4)
    k = np.unique(rng.integers(20, 300, 200))
    raw = true_off + k * 0.125 + rng.uniform(-0.015, 0.015, len(k))
    # detect_grid is 60 ms off: only aligning to the events can bring the snap back on the grid
    monkeypatch.setattr(G, "detect_grid", lambda audio: Grid(bpm, true_off + 0.06, True))
    monkeypatch.setattr(G, "predict_events", lambda *a, **k: raw)
    seen = []

    def fake_sample(loaded, times, mel, nps, styles, seed, cfg, device, **kw):
        seen.append(np.asarray(times))
        assert not {"double_bias", "dot_brake"} & kw.keys()  # generate_map relies on the sampler defaults
        notes = [Note(round(t * 22050 / 256) * 256 / 22050, 0.0, 0, 0, 0, 1) for t in times]
        return notes, SampleReport(len(times), 0, 0, 0.0)

    monkeypatch.setattr(G, "sample_notes", fake_sample)
    monkeypatch.setattr(G, "check", lambda events, cfg: SimpleNamespace(hard=()))
    _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out", nps=5.0)
    planned = seen[0]
    err = (planned - true_off + 0.0625) % 0.125 - 0.0625
    assert np.max(np.abs(err)) < 0.004


def _reaction_s(bpm: float, njs: float, offset: float) -> float:
    return max(base_half_jump_beats(bpm, njs) + offset, 0.25) * 60.0 / bpm


def test_reaction_offset_gives_the_same_time_at_half_and_full_tempo():
    # nps 6 -> 0.580 s whatever BPM the grid detection reports
    for bpm in (110.0, 220.0, 174.0, 87.0):
        assert _reaction_s(bpm, 18.0, reaction_offset(6.0, 18.0, bpm)) == pytest.approx(0.580, abs=0.002)


def test_reaction_offset_follows_density():
    slow = _reaction_s(128.0, 14.0, reaction_offset(3.0, 14.0, 128.0))
    fast = _reaction_s(128.0, 19.0, reaction_offset(8.0, 19.0, 128.0))
    assert slow == pytest.approx(0.863, abs=0.002) and fast == pytest.approx(0.528, abs=0.002)


def test_reaction_offset_clamps_outside_the_bins_and_at_the_game_minimum():
    assert reaction_offset(0.1, 10.0, 120.0) == reaction_offset(1.0, 10.0, 120.0)
    assert reaction_offset(20.0, 21.0, 120.0) == reaction_offset(10.0, 21.0, 120.0)
    # 30 BPM: 0.483 s is 0.24 beats, below the game's 0.25-beat floor
    off = reaction_offset(10.0, 21.0, 30.0)
    assert base_half_jump_beats(30.0, 21.0) + off == pytest.approx(0.25, abs=1e-3)


GOLDEN_GEN = Path(__file__).parent / "fixtures" / "generate_golden"
BOMB_VOCAB = [1, 2, 1 << 5]
LAYER_STATS = {
    "nps_bins": [0.0, 100.0],
    "bomb_vocab": BOMB_VOCAB,
    "rates": {
        "arcs": [[6.0, 12.0, 18.0]],
        "chains": [[6.0, 12.0, 18.0]],
        "walls": [[6.0, 12.0, 18.0]],
        "bomb_groups": [[4.0, 8.0, 12.0]],
        **{f"light_ch{c}": [[4.0, 8.0, 12.0]] for c in range(9)},
    },
}


def make_layer_models(seed: int = 0) -> LayerModels:
    torch.manual_seed(seed)
    ocfg = OrnamentConfig(d_model=16, n_layers=1, n_heads=2, context=16, dropout=0.0)
    orn = LoadedOrnaments(OrnamentModel(ocfg).eval(), ocfg, STATS)

    dspec = decor_spec(LAYER_STATS)
    dstats = dict(STATS, bomb_masks=BOMB_VOCAB)
    dw_cfg = WhenConfig(n_channels=dspec.n_channels, d_model=16, n_layers=1, n_heads=2, window=128, dropout=0.0)
    dv_cfg = WhatConfig(n_channels=dspec.n_channels, vocab_size=dspec.vocab_size,
                        d_model=16, n_layers=1, n_heads=2, context=16, dropout=0.0)
    lw_cfg = WhenConfig(n_channels=LIGHTS.n_channels, d_model=16, n_layers=1, n_heads=2, window=128, dropout=0.0)
    lv_cfg = WhatConfig(n_channels=LIGHTS.n_channels, vocab_size=LIGHTS.vocab_size,
                        d_model=16, n_layers=1, n_heads=2, context=16, dropout=0.0)
    return LayerModels(
        ornaments=orn,
        decor_when=LoadedWhen(WhenModel(dw_cfg).eval(), dw_cfg, dstats, "decor", BOMB_VOCAB),
        decor_what=LoadedWhat(WhatModel(dv_cfg, channel_mask(dspec)).eval(), dv_cfg, dstats, "decor", BOMB_VOCAB),
        lights_when=LoadedWhen(WhenModel(lw_cfg).eval(), lw_cfg, STATS, "lights", None),
        lights_what=LoadedWhat(WhatModel(lv_cfg, channel_mask(LIGHTS)).eval(), lv_cfg, STATS, "lights", None),
    )


def test_all_layers_off_is_byte_identical_to_pre_change_output(models, tmp_path, monkeypatch):
    # the golden predates F7 phrase repetition (it changes notes, not layers): keep it off here
    monkeypatch.setattr(G, "plan_phrases", lambda planned, probs, repeats, seed, **kw: (planned, None, 0))
    # it also predates F6 (loudness-aware doubles, double geometry prior): keep both off here
    from dataclasses import replace as _replace
    from bsmaker.models.pattern import sample as PS
    real_sample = G.sample_notes

    def sample_pre_f6(*a, **kw):
        kw.setdefault("double_bias", _replace(PS.DEFAULT_DOUBLE_BIAS, loud=0.0))
        kw.setdefault("double_geometry", None)
        return real_sample(*a, **kw)

    monkeypatch.setattr(G, "sample_notes", sample_pre_f6)
    audio = write_audio(tmp_path / "My Song.wav")
    res_default = _gen(models, audio, tmp_path / "out_default")
    for name in ("Info.dat", "HardStandard.dat"):
        assert (res_default.folder / name).read_bytes() == (GOLDEN_GEN / name).read_bytes()

    lm = make_layer_models()
    res_off = _gen(
        models, audio, tmp_path / "out_off",
        ornaments=0, lights=0, environment="DefaultEnvironment",
        layer_models=lm, layer_stats=LAYER_STATS,
    )
    for name in ("Info.dat", "HardStandard.dat"):
        assert (res_off.folder / name).read_bytes() == (GOLDEN_GEN / name).read_bytes()


def test_multi_level_produces_two_files_and_one_info(models, tmp_path):
    audio = write_audio(tmp_path / "My Song.wav")
    res = generate_map(
        audio, levels=("ExpertPlus", "easy"), styles=("tech",), seed=1,
        out_dir=tmp_path / "out", timing=models[0], pattern=models[1],
        mask_cfg=MASK, stats=STATS, device="cpu", environment="NiceEnvironment",
    )
    assert res.folder.name == "BSM - My Song [Easy+ExpertPlus, tech]"
    assert {p.name for p in res.folder.iterdir()} == {
        "Info.dat", "EasyStandard.dat", "ExpertPlusStandard.dat", "song.egg", "cover.png",
    }
    assert [lv.rank for lv in res.levels] == ["Easy", "ExpertPlus"]
    assert [lv.target_nps for lv in res.levels] == [1.0, 6.0]

    info = json.loads((res.folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_environmentName"] == "NiceEnvironment"
    bms = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
    assert [bm["_difficulty"] for bm in bms] == ["Easy", "ExpertPlus"]
    assert bms[0]["_noteJumpMovementSpeed"] == 12.0
    assert bms[1]["_noteJumpMovementSpeed"] == 16.0

    # Same seed on single-level matches the corresponding difficulty's notes
    single_easy = _gen(models, audio, tmp_path / "single", nps=1.0, seed=1)
    multi_map = read_map(res.folder, layers=True)
    single_map = read_map(single_easy.folder, layers=True)
    assert multi_map.difficulties[0].notes == single_map.difficulties[0].notes


def test_all_layers_present_and_rule_clean_with_tiny_models(models, tmp_path):
    audio = write_audio(tmp_path / "My Song.wav", seconds=60.0)
    lm = make_layer_models()
    res = generate_map(
        audio, levels=("Easy", "Hard"), styles=("tech",), seed=2,
        out_dir=tmp_path / "out", timing=models[0], pattern=models[1],
        mask_cfg=MASK, stats=STATS, device="cpu",
        ornaments=None, lights=2, environment="BigMirrorEnvironment",
        layer_models=lm, layer_stats=LAYER_STATS,
    )
    m = read_map(res.folder, layers=True)
    assert m.environment == "BigMirrorEnvironment" and not m.problems
    assert len(m.difficulties) == 2
    for d in m.difficulties:
        assert d.layers is not None
        assert check_layers(d.notes, d.layers) == []
        assert len(d.layers.arcs) > 0 and len(d.layers.chains) > 0
        assert len(d.layers.obstacles) > 0 and len(d.layers.bombs) > 0
        assert len(d.layers.lights) > 0
    assert m.difficulties[0].layers.lights == m.difficulties[1].layers.lights
    assert res.n_walls == len(m.difficulties[-1].layers.obstacles)  # densest level
    assert res.n_bombs == len(m.difficulties[-1].layers.bombs)
    assert [lv.n_walls for lv in res.levels] == [len(d.layers.obstacles) for d in m.difficulties]
    assert res.n_lights == len(m.difficulties[0].layers.lights)
    assert res.lights_report is not None and res.lights_report["used_rules"] is False


def test_auto_ornaments_level_and_per_level_ornaments(models, tmp_path):
    assert auto_ornaments_level(0.5) == 2
    assert auto_ornaments_level(4.49) == 2
    assert auto_ornaments_level(4.5) == 1
    assert auto_ornaments_level(7.0) == 1
    assert auto_ornaments_level(7.01) == 0

    stats_hi = {
        **STATS,
        "rank_median_nps": {"Easy": 2.0, "ExpertPlus": 8.0},
    }
    audio = write_audio(tmp_path / "My Song.wav", seconds=45.0)
    lm = make_layer_models()
    res = generate_map(
        audio, levels=("Easy", "ExpertPlus"), styles=("tech",), seed=1,
        out_dir=tmp_path / "out", timing=models[0], pattern=models[1],
        mask_cfg=MASK, stats=stats_hi, device="cpu",
        ornaments=None, lights=0, layer_models=lm, layer_stats=LAYER_STATS,
    )
    by_rank = {lv.rank: lv for lv in res.levels}
    assert by_rank["Easy"].n_arcs > 0 and by_rank["Easy"].n_chains > 0
    # ExpertPlus is > 7.0 nps -> auto ornaments level 0 (both ornaments and shared decor)
    assert by_rank["ExpertPlus"].n_arcs == 0 and by_rank["ExpertPlus"].n_chains == 0
    assert res.n_walls == 0 and res.n_bombs == 0


def test_generate_map_validation_errors(models, tmp_path):
    audio = write_audio(tmp_path / "a.wav")
    base = dict(styles=("tech",), seed=0, out_dir=tmp_path / "out", timing=models[0],
                pattern=models[1], mask_cfg=MASK, stats=STATS, device="cpu")
    with pytest.raises(ValueError):
        generate_map(audio, **base)
    with pytest.raises(ValueError):
        generate_map(audio, nps=3.0, levels=("Hard",), **base)
    with pytest.raises(ValueError):
        generate_map(audio, levels=(), **base)
    with pytest.raises(ValueError):
        generate_map(audio, levels=("Hard", "hard"), **base)
    with pytest.raises(ValueError):
        generate_map(audio, levels=("Insane",), **base)
    with pytest.raises(GenerateError):
        generate_map(audio, levels=("Normal",), **base)
    with pytest.raises(ValueError):
        generate_map(audio, nps=3.0, environment="SpaceEnvironment", **base)
    with pytest.raises(ValueError):
        generate_map(audio, nps=3.0, ornaments=4, **base)
    with pytest.raises(ValueError):
        generate_map(audio, nps=3.0, lights=4, **base)
    with pytest.raises(ValueError):
        generate_map(audio, nps=3.0, flow=4, **base)


def test_lights_rules_flag_and_check_layers_guard(models, tmp_path, monkeypatch):
    audio = write_audio(tmp_path / "a.wav")
    lm = make_layer_models()
    res = _gen(models, audio, tmp_path / "out_rules",
               lights=2, lights_rules=True, layer_models=lm, layer_stats=LAYER_STATS)
    assert res.lights_report is not None and res.lights_report["used_rules"] is True
    assert res.n_lights > 0

    monkeypatch.setattr(G, "check_layers", lambda notes, layers: [Violation(1.0, -1, "wall_on_note")])
    with pytest.raises(RuntimeError, match="layer rule"):
        _gen(models, audio, tmp_path / "out_bad", ornaments=0, lights=0)


def test_multi_level_decor_is_generated_per_level(models, tmp_path, monkeypatch):
    audio = write_audio(tmp_path / "a.wav", seconds=45.0)
    lm = make_layer_models()
    calls = []

    def fake_decor(models_, mel, notes, nps, bpm, duration_s, level, layer_stats, seed, device, **k):
        calls.append((list(notes), nps, level))
        # a different amount of decor per level, placed on a free spot
        n = len(calls)
        obs = [Obstacle(1.0 + 3.0 * i, 2.0 + 3.0 * i, 0.2, 0.4, -1, 0, 1, 5) for i in range(n)]
        bombs = [Bomb(0.2, 0.4, 3, 2)] * n
        rep = {
            "walls_requested": n, "bomb_groups_requested": n,
            "walls_placed": n, "decor_walls_placed": n, "gameplay_walls_placed": 0,
            "bomb_groups_placed": n, "bombs_placed": n,
            "walls_skipped": 0, "bombs_skipped": 0, "skipped_by_rules": 0,
            "dropped_by_gameplay": 0, "missing_models": False, "missing_stats": False,
        }
        return obs, bombs, rep

    monkeypatch.setattr(G, "generate_decor", fake_decor)
    res = generate_map(
        audio, levels=("Easy", "ExpertPlus"), styles=("tech",), seed=1,
        out_dir=tmp_path / "out", timing=models[0], pattern=models[1],
        mask_cfg=MASK, stats=STATS, device="cpu",
        ornaments=2, lights=0, layer_models=lm, layer_stats=LAYER_STATS,
    )
    assert len(calls) == 2  # once per level
    m = read_map(res.folder, layers=True)
    by_rank = {d.rank: d for d in m.difficulties}
    for i, lv in enumerate(res.levels):
        notes, nps, level = calls[i]
        assert nps == lv.target_nps and level == 2
        assert len(notes) == lv.n_notes
        assert lv.n_walls == i + 1 and lv.n_bombs == i + 1
    assert [len(by_rank[lv.rank].layers.obstacles) for lv in res.levels] == [1, 2]
    for d in m.difficulties:
        assert check_layers(d.notes, d.layers) == []
    assert res.decor_report["walls_placed"] == 3
    assert res.n_walls == res.levels[-1].n_walls  # densest level


def test_generate_map_artist_cover_unique_folder_and_progress(models, tmp_path):
    audio = write_audio(tmp_path / "My Song.wav")
    cover = tmp_path / "art.jpg"
    cover.write_bytes(b"\xff\xd8\xff\xe0thumb")
    events: list[tuple[str, int]] = []

    res1 = _gen(
        models, audio, tmp_path / "out",
        title="My: Song?", artist="Cool/Band", cover_src=cover,
        unique_folder=True, progress=lambda s, p: events.append((s, p)),
    )
    info = json.loads((res1.folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_songAuthorName"] == "Cool/Band"
    assert info["_coverImageFilename"] == "cover.jpg"
    assert (res1.folder / "cover.jpg").read_bytes() == cover.read_bytes()
    stages = [s for s, _ in events]
    assert stages[0] == "analyzing" and "notes" in stages and "layers" in stages and stages[-1] == "writing"
    pcts = [p for _, p in events]
    assert pcts == sorted(pcts) and pcts[0] >= 0 and pcts[-1] == 100

    res2 = _gen(
        models, audio, tmp_path / "out",
        title="My: Song?", artist="Cool/Band", unique_folder=True,
    )
    assert res2.folder != res1.folder
    assert res2.folder.name == f"{res1.folder.name} (2)"
    assert res1.folder.is_dir() and res2.folder.is_dir()


def test_generate_map_cancel_mid_sampling_cleans_up(models, tmp_path):
    from bsmaker.generate import GenerateCancelled

    audio = write_audio(tmp_path / "a.wav")
    out = tmp_path / "out"
    calls = 0

    def cancel_after_notes_start() -> bool:
        nonlocal calls
        calls += 1
        return calls > 4

    with pytest.raises(GenerateCancelled):
        _gen(models, audio, out, is_cancelled=cancel_after_notes_start)
    assert not out.exists() or list(out.iterdir()) == []


def test_generate_map_monotonic_progress_across_density_rounds_and_layer_cancel(models, tmp_path, monkeypatch):
    from bsmaker import generate as G
    from bsmaker.generate import GenerateCancelled
    from bsmaker.models.layers import what as W
    from bsmaker.models.pattern.sample import SampleReport

    audio = write_audio(tmp_path / "rounds.wav")
    orig_sample = G.sample_notes
    calls = 0

    def two_round_sample(pattern, planned, mel, target_nps, styles, seed, mask_cfg, device, *, is_cancelled=None, on_progress=None, phrase=None):
        nonlocal calls
        calls += 1
        if on_progress is not None:
            on_progress(1, 2)
            on_progress(2, 2)
        notes, rep = orig_sample(pattern, planned, mel, target_nps, styles, seed, mask_cfg, device)
        if calls == 1:
            return notes[:2], SampleReport(len(planned), len(notes[:2]), 0, 0, 0)
        return notes, rep

    monkeypatch.setattr(G, "sample_notes", two_round_sample)
    events: list[tuple[str, int]] = []
    _gen(models, audio, tmp_path / "out_rounds", progress=lambda s, p: events.append((s, p)))
    pcts = [p for _, p in events]
    assert calls >= 2
    assert pcts == sorted(pcts)

    seen_cancel_args = []
    monkeypatch.setattr(G, "sample_notes", orig_sample)
    orig_generate_lights = G.generate_lights

    def spy_lights(*a, is_cancelled=None, **kw):
        seen_cancel_args.append(is_cancelled)
        if is_cancelled and is_cancelled():
            raise GenerateCancelled("cancelled")
        return orig_generate_lights(*a, is_cancelled=is_cancelled, **kw)

    monkeypatch.setattr(G, "generate_lights", spy_lights)
    cancel_fn = lambda: True
    # Also verify sample_layer_values raises GenerateCancelled directly when cancelled
    import numpy as np
    from bsmaker.models.layers.what import sample_layer_values
    lm = make_layer_models()
    with pytest.raises(GenerateCancelled):
        sample_layer_values(
            lm.lights_what, np.array([1.0]), np.array([0]), np.zeros((100, 80), dtype=np.float32),
            [], 3.0, 0, "cpu", is_cancelled=cancel_fn,
        )


def test_generate_map_flow_per_level(models, tmp_path):
    audio = write_audio(tmp_path / "a.wav", seconds=45.0)
    res = generate_map(
        audio, levels=("Easy", "Hard"), styles=("dance",), seed=1,
        out_dir=tmp_path / "out", timing=models[0], pattern=models[1],
        mask_cfg=MASK, stats=STATS, device="cpu", flow=1,
    )
    assert len(res.levels) == 2
    for lv in res.levels:
        assert lv.flow_report is not None
        assert lv.flow_report.level == 1


def test_generate_map_flow_defaults_to_zero(models, tmp_path):
    res = _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out")
    assert all(lv.flow_report is not None and lv.flow_report.level == 0 for lv in res.levels)
    with pytest.raises(ValueError):
        _gen(models, write_audio(tmp_path / "b.wav"), tmp_path / "out_none", flow=None)


def test_generate_map_flow_final_check_raises(models, tmp_path, monkeypatch):
    from bsmaker.flow_rules import FlowReport

    def flow_stub(notes, *, bpm, onset_env, onset_hop_s, level, seed, mask_cfg, phrase=None):
        return list(notes), FlowReport(level=level)

    seen = []

    def bad_final(pre, final, cfg):
        seen.append((len(pre), len(final), cfg))
        return False

    monkeypatch.setattr(G, "apply_flow", flow_stub)
    monkeypatch.setattr(G, "final_flow_ok", bad_final)
    with pytest.raises(RuntimeError, match="flow layer"):
        _gen(models, write_audio(tmp_path / "a.wav"), tmp_path / "out", flow=2)
    assert seen and seen[0][2] is MASK


def test_generate_map_flow_zero_same_as_before(models, tmp_path, monkeypatch):
    audio = write_audio(tmp_path / "a.wav", seconds=45.0)
    res0 = _gen(models, audio, tmp_path / "out0", flow=0)

    def no_flow(notes, *, bpm, onset_env, onset_hop_s, level, seed, mask_cfg, phrase=None):
        from bsmaker.flow_rules import FlowReport
        return list(notes), FlowReport(level=0)

    monkeypatch.setattr(G, "apply_flow", no_flow)
    res_patched = _gen(models, audio, tmp_path / "out_patched", flow=2)
    assert (res0.folder / "HardStandard.dat").read_bytes() == (
        res_patched.folder / "HardStandard.dat"
    ).read_bytes()
    notes0 = read_map(res0.folder).difficulties[0].notes
    notes_p = read_map(res_patched.folder).difficulties[0].notes
    assert sum(n.direction == 8 for n in notes0) == sum(n.direction == 8 for n in notes_p)

