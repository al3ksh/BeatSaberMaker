from dataclasses import asdict
import logging

import numpy as np
import pytest
import torch

from bsmaker.audio.decode import SAMPLE_RATE
from bsmaker.audio.mel import HOP
from bsmaker.formats.layer_tokens import CHANNEL_TYPES, WALL_VOCAB
from bsmaker.formats.model import Layers, LightEvent, Note, Obstacle
from bsmaker.layers_generate import LayerModels, generate_decor, generate_lights, load_layer_models
from bsmaker.lights_rules import rule_lights
from bsmaker.obstacles_rules import gameplay_obstacles
from bsmaker.models.common import save_checkpoint
from bsmaker.models.layers.spec import LIGHTS, channel_mask, decor_spec
from bsmaker.models.layers.what import WhatConfig, WhatModel
from bsmaker.models.layers.when import WhenConfig, WhenModel
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel
from bsmaker.playability.layers import check_layers, wall_ok

STATS = {"mel_mean": 0.0, "mel_std": 1.0}
BOMB_VOCAB = [1, 2, 1 << 5, (1 << 0) | (1 << 3)]  # cells (0,0), (1,0), (1,1), (0,0)+(3,0)
LAYER_STATS = {
    "nps_bins": [0.0, 100.0],
    "bomb_vocab": BOMB_VOCAB,
    "rates": {
        "arcs": [[2.0, 6.0, 12.0]],
        "chains": [[3.0, 9.0, 18.0]],
        "walls": [[6.0, 12.0, 24.0]],
        "bomb_groups": [[4.0, 8.0, 16.0]],
        **{f"light_ch{c}": [[4.0, 8.0, 16.0]] for c in range(9)},
    },
}
BPM = 120.0
TINY_WHEN = dict(d_model=16, n_layers=1, n_heads=2, window=128, dropout=0.0)
TINY_WHAT = dict(d_model=16, n_layers=1, n_heads=2, context=16, dropout=0.0)
TINY_ORN = dict(d_model=16, n_layers=1, n_heads=2, context=16, dropout=0.0)


def _mel(seconds: float, seed: int = 0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE / HOP)
    rng = np.random.default_rng(seed)
    mel = rng.normal(size=(n, 80)).astype(np.float32)
    mel[::40] += 3.0
    return mel.astype(np.float16)


def _write_all_checkpoints(root):
    layers_dir = root / "layers"
    torch.manual_seed(0)

    ocfg = OrnamentConfig(**TINY_ORN)
    save_checkpoint(
        layers_dir / "ornaments" / "best.pt",
        {"model": OrnamentModel(ocfg).state_dict(), "model_config": asdict(ocfg), "stats": STATS},
    )

    dspec = decor_spec(LAYER_STATS)
    dstats = dict(STATS, bomb_masks=BOMB_VOCAB)
    dw_cfg = WhenConfig(n_channels=dspec.n_channels, **TINY_WHEN)
    save_checkpoint(
        layers_dir / "decor-when" / "best.pt",
        {
            "model": WhenModel(dw_cfg).state_dict(),
            "model_config": asdict(dw_cfg),
            "stats": dstats,
            "train_options": {"layer": "decor"},
        },
    )
    dv_cfg = WhatConfig(n_channels=dspec.n_channels, vocab_size=dspec.vocab_size, **TINY_WHAT)
    save_checkpoint(
        layers_dir / "decor-what" / "best.pt",
        {
            "model": WhatModel(dv_cfg, channel_mask(dspec)).state_dict(),
            "model_config": asdict(dv_cfg),
            "stats": dstats,
            "train_options": {"layer": "decor"},
        },
    )

    lw_cfg = WhenConfig(n_channels=LIGHTS.n_channels, **TINY_WHEN)
    save_checkpoint(
        layers_dir / "lights-when" / "best.pt",
        {
            "model": WhenModel(lw_cfg).state_dict(),
            "model_config": asdict(lw_cfg),
            "stats": STATS,
            "train_options": {"layer": "lights"},
        },
    )
    lv_cfg = WhatConfig(n_channels=LIGHTS.n_channels, vocab_size=LIGHTS.vocab_size, **TINY_WHAT)
    save_checkpoint(
        layers_dir / "lights-what" / "best.pt",
        {
            "model": WhatModel(lv_cfg, channel_mask(LIGHTS)).state_dict(),
            "model_config": asdict(lv_cfg),
            "stats": STATS,
            "train_options": {"layer": "lights"},
        },
    )


@pytest.fixture(scope="module")
def all_models(tmp_path_factory):
    ckpt_dir = tmp_path_factory.mktemp("ckpts")
    _write_all_checkpoints(ckpt_dir)
    models, warnings = load_layer_models(ckpt_dir, "cpu")
    assert warnings == []
    return models


def _side_notes(duration_s: float, x_choices=(2, 3), step: float = 0.5) -> list[Note]:
    out = []
    t = 2.0
    while t < duration_s - 1.0:
        x = x_choices[len(out) % len(x_choices)]
        out.append(Note(t, t * BPM / 60.0, x, 1, len(out) % 2, 1))
        t += step
    return out


def _random_notes(rng: np.random.Generator, n: int, span: float) -> list[Note]:
    times = np.sort(rng.uniform(0.5, span - 0.5, n))
    return [
        Note(
            float(t),
            float(t) * BPM / 60.0,
            int(rng.integers(0, 4)),
            int(rng.integers(0, 3)),
            int(rng.integers(0, 2)),
            int(rng.integers(0, 9)),
        )
        for t in times
    ]


def test_load_layer_models_all_and_missing(tmp_path):
    empty = LayerModels()
    assert empty == LayerModels(None, None, None, None, None)

    models, warnings = load_layer_models(tmp_path, "cpu")
    assert models == LayerModels()
    assert len(warnings) == 5

    _write_all_checkpoints(tmp_path)
    (tmp_path / "layers" / "lights-what" / "best.pt").unlink()
    partial, p_warnings = load_layer_models(tmp_path, "cpu")
    assert partial.ornaments is not None
    assert partial.decor_when is not None and partial.decor_what is not None
    assert partial.lights_when is not None and partial.lights_what is None
    assert len(p_warnings) == 1 and "lights-what" in p_warnings[0]


def test_load_layer_models_rejects_mismatched_layer_field(tmp_path):
    _write_all_checkpoints(tmp_path)
    layers_dir = tmp_path / "layers"

    # Put a lights checkpoint in decor-what, a decor checkpoint in lights-when, and empty layer in decor-when
    lv_cfg = WhatConfig(n_channels=LIGHTS.n_channels, vocab_size=LIGHTS.vocab_size, **TINY_WHAT)
    save_checkpoint(
        layers_dir / "decor-what" / "best.pt",
        {
            "model": WhatModel(lv_cfg, channel_mask(LIGHTS)).state_dict(),
            "model_config": asdict(lv_cfg),
            "stats": STATS,
            "train_options": {"layer": "lights"},
        },
    )
    dspec = decor_spec(LAYER_STATS)
    dw_cfg = WhenConfig(n_channels=dspec.n_channels, **TINY_WHEN)
    save_checkpoint(
        layers_dir / "lights-when" / "best.pt",
        {
            "model": WhenModel(dw_cfg).state_dict(),
            "model_config": asdict(dw_cfg),
            "stats": dict(STATS, bomb_masks=BOMB_VOCAB),
            "train_options": {"layer": "decor"},
        },
    )
    save_checkpoint(
        layers_dir / "decor-when" / "best.pt",
        {
            "model": WhenModel(dw_cfg).state_dict(),
            "model_config": asdict(dw_cfg),
            "stats": dict(STATS, bomb_masks=BOMB_VOCAB),
            "train_options": {},
        },
    )

    models, warnings = load_layer_models(tmp_path, "cpu")
    assert models.ornaments is not None
    assert models.lights_what is not None
    assert models.decor_what is None
    assert models.lights_when is None
    assert models.decor_when is None
    assert len(warnings) == 3
    assert any("decor-what" in w for w in warnings)
    assert any("lights-when" in w for w in warnings)
    assert any("decor-when" in w for w in warnings)


def test_level_zero_returns_empty(all_models):
    mel = _mel(40.0)
    notes = _side_notes(40.0)
    obs, bombs, drep = generate_decor(all_models, mel, notes, 2.0, BPM, 40.0, 0, LAYER_STATS, 0, "cpu")
    lights, lrep = generate_lights(all_models, mel, notes, 2.0, BPM, 0.0, 40.0, 0, LAYER_STATS, 0, "cpu")
    assert obs == [] and bombs == []
    assert (
        drep["walls_requested"] == 0
        and drep["walls_placed"] == 0
        and drep["missing_models"] is False
        and drep["missing_stats"] is False
    )
    assert lights == []
    assert lrep["requested"] == 0 and lrep["placed"] == 0 and lrep["missing_stats"] is False


def test_missing_models_behaviour(all_models):
    mel = _mel(60.0)
    notes = _side_notes(60.0)

    # Missing decor models: level 1 has no gameplay walls; levels 2-3 still emit rule gameplay obstacles
    only_lights = LayerModels(lights_when=all_models.lights_when, lights_what=all_models.lights_what)
    obs1, bombs1, drep1 = generate_decor(only_lights, mel, notes, 2.0, BPM, 60.0, 1, LAYER_STATS, 1, "cpu")
    assert obs1 == [] and bombs1 == []
    assert drep1["missing_models"] is True and drep1["missing_stats"] is False

    obs3, bombs3, drep3 = generate_decor(only_lights, mel, notes, 2.0, BPM, 60.0, 3, LAYER_STATS, 1, "cpu")
    expected_gp = gameplay_obstacles(notes, BPM, 60.0, 3, 1)
    assert len(expected_gp) > 0
    assert obs3 == expected_gp and bombs3 == []
    assert drep3["gameplay_walls_placed"] == len(expected_gp)
    assert drep3["decor_walls_placed"] == 0
    assert drep3["walls_placed"] == len(expected_gp)
    assert drep3["missing_models"] is True and drep3["missing_stats"] is False
    assert check_layers(notes, Layers(tuple(bombs3), tuple(obs3), (), (), (), False, 0)) == []

    # Missing lights models: falls back to rule_lights
    only_decor = LayerModels(decor_when=all_models.decor_when, decor_what=all_models.decor_what)
    lights, lrep = generate_lights(only_decor, mel, notes, 2.0, BPM, 0.25, 60.0, 2, LAYER_STATS, 1, "cpu")
    expected = rule_lights(mel, notes, BPM, 0.25, 60.0, 2, 1)
    assert lights == expected and len(lights) > 0
    assert lrep["used_rules"] is True and lrep["missing_models"] is True and lrep["missing_stats"] is False


def test_missing_layer_stats_sets_missing_stats_and_runs_rules(all_models):
    mel = _mel(60.0)
    notes = _side_notes(60.0)

    obs, bombs, drep = generate_decor(all_models, mel, notes, 2.0, BPM, 60.0, 2, None, 1, "cpu")
    expected_gp = gameplay_obstacles(notes, BPM, 60.0, 2, 1)
    assert len(expected_gp) > 0
    assert obs == expected_gp and bombs == []
    assert drep["missing_stats"] is True and drep["missing_models"] is False
    assert drep["gameplay_walls_placed"] == len(expected_gp)

    lights, lrep = generate_lights(all_models, mel, notes, 2.0, BPM, 0.25, 60.0, 2, None, 1, "cpu")
    assert lights == rule_lights(mel, notes, BPM, 0.25, 60.0, 2, 1) and len(lights) > 0
    assert lrep["used_rules"] is True
    assert lrep["missing_stats"] is True
    assert lrep["missing_models"] is False


def test_predict_layer_events_receives_expected_per_channel_counts(all_models, monkeypatch):
    import bsmaker.layers_generate as lg

    stats = {
        "nps_bins": [0.0, 4.0, 10.0],
        "bomb_vocab": BOMB_VOCAB,
        "rates": {
            "walls": [[7.0, 11.0, 15.0], [2.0, 4.0, 6.0]],
            "bomb_groups": [[3.0, 5.0, 9.0], [1.0, 2.0, 3.0]],
            **{f"light_ch{c}": [[float(c + 1), float(2 * c + 3), float(3 * c + 5)], [1.0, 2.0, 3.0]]
               for c in range(9)},
        },
    }
    captured: list[tuple[int, ...]] = []
    orig_predict = lg.predict_layer_events

    def spy_predict(loaded, mel, notes, nps, counts, device):
        captured.append(tuple(int(x) for x in counts))
        return orig_predict(loaded, mel, notes, nps, counts, device)

    monkeypatch.setattr(lg, "predict_layer_events", spy_predict)
    mel = _mel(60.0)
    notes = _side_notes(60.0)

    _, _, drep = generate_decor(all_models, mel, notes, 2.0, BPM, 60.0, 2, stats, 0, "cpu")
    _, lrep = generate_lights(all_models, mel, notes, 2.0, BPM, 0.0, 60.0, 2, stats, 0, "cpu")

    # nps=2.0 -> bin 0, level=2, 60 s -> walls=11 -> (5, 6, 5), lights=[2*c+3]
    assert captured[0] == (11 // 2, 11 - 11 // 2, 5)
    assert drep["walls_requested"] == 11 and drep["bomb_groups_requested"] == 5
    expected_lights = tuple(int(round(stats["rates"][el][0][1])) for el in LIGHTS.rate_elements)
    assert captured[1] == expected_lights
    assert lrep["requested"] == sum(expected_lights)


def test_use_rules_forces_rule_lights_with_models_present(all_models):
    mel = _mel(45.0)
    notes = _side_notes(45.0)
    lights, lrep = generate_lights(
        all_models, mel, notes, 2.0, BPM, 0.1, 45.0, 2, LAYER_STATS, 3, "cpu", use_rules=True
    )
    assert lights == rule_lights(mel, notes, BPM, 0.1, 45.0, 2, 3)
    assert lrep["used_rules"] is True and lrep["missing_models"] is False


def test_decor_passes_check_layers_and_respects_rules(all_models):
    mel = _mel(60.0)
    for seed, n_notes in [(0, 20), (1, 200), (2, 480)]:
        notes = _random_notes(np.random.default_rng(seed), n_notes, 60.0)
        for level in (1, 2, 3):
            obs, bombs, rep = generate_decor(
                all_models, mel, notes, n_notes / 60.0, BPM, 60.0, level, LAYER_STATS, seed, "cpu"
            )
            assert rep["missing_models"] is False
            assert check_layers(notes, Layers(tuple(bombs), tuple(obs), (), (), (), False, 0)) == []
            assert rep["walls_placed"] == len(obs)
            assert rep["bombs_placed"] == len(bombs)


def test_decor_rule_mask_forbids_all_bombs_logs_warning_and_never_loops(all_models, caplog):
    mel = _mel(31.0)
    # Place notes densely in every cell of row 0 and row 1 across the full song so all bomb masks
    # in BOMB_VOCAB are forbidden everywhere.
    notes = [
        Note(t, t * BPM / 60.0, x, y, x % 2, 1)
        for t in np.arange(1.0, 31.5, 0.25)
        for x in range(4)
        for y in (0, 1)
    ]
    with caplog.at_level(logging.WARNING):
        obs, bombs, rep = generate_decor(
            all_models, mel, notes, 8.0, BPM, 31.0, 3, LAYER_STATS, 0, "cpu"
        )
    assert bombs == []
    assert rep["bomb_groups_requested"] > 0
    assert rep["bombs_skipped"] > 0
    assert rep["skipped_by_rules"] >= rep["bombs_skipped"]
    assert check_layers(notes, Layers(tuple(bombs), tuple(obs), (), (), (), False, 0)) == []
    assert any("decor" in r.message for r in caplog.records)


def test_decor_rule_mask_skips_blocked_inner_walls(tmp_path, caplog):
    _write_all_checkpoints(tmp_path)
    models, _ = load_layer_models(tmp_path, "cpu")
    # Restrict channel_mask to inner-grid wall tokens (0..11) so notes in cols 0 & 3 block all walls
    cm = models.decor_what.model.channel_mask.clone()
    cm[:2, 12:WALL_VOCAB] = False
    models.decor_what.model.channel_mask.copy_(cm)

    mel = _mel(31.0)
    notes = [
        Note(t, t * BPM / 60.0, x, y, 0 if x == 0 else 1, 1)
        for t in np.arange(1.0, 31.5, 0.25)
        for x in (0, 3)
        for y in (0, 1, 2)
    ]
    with caplog.at_level(logging.WARNING):
        obs, bombs, rep = generate_decor(
            models, mel, notes, 4.0, BPM, 31.0, 2, LAYER_STATS, 0, "cpu"
        )
    assert rep["walls_skipped"] > 0
    assert rep["decor_walls_placed"] == 0
    assert check_layers(notes, Layers(tuple(bombs), tuple(obs), (), (), (), False, 0)) == []
    assert any("decor" in r.message for r in caplog.records)


def test_empty_mel_empty_notes_and_invalid_level(all_models):
    empty_mel = np.zeros((0, 80), dtype=np.float16)
    obs, bombs, drep = generate_decor(all_models, empty_mel, [], 2.0, BPM, 0.0, 2, LAYER_STATS, 0, "cpu")
    lights, lrep = generate_lights(all_models, empty_mel, [], 2.0, BPM, 0.0, 0.0, 2, LAYER_STATS, 0, "cpu")
    assert obs == [] and bombs == [] and drep["walls_placed"] == 0
    assert lights == [] and lrep["placed"] == 0

    with pytest.raises(ValueError):
        generate_decor(all_models, _mel(31.0), [], 2.0, BPM, 31.0, 4, LAYER_STATS, 0, "cpu")
    with pytest.raises(ValueError):
        generate_lights(all_models, _mel(31.0), [], 2.0, BPM, 0.0, 31.0, 4, LAYER_STATS, 0, "cpu")


def test_decor_determinism(all_models):
    mel = _mel(40.0)
    notes = _random_notes(np.random.default_rng(4), 40, 40.0)
    a = generate_decor(all_models, mel, notes, 2.0, BPM, 40.0, 2, LAYER_STATS, 9, "cpu")
    b = generate_decor(all_models, mel, notes, 2.0, BPM, 40.0, 2, LAYER_STATS, 9, "cpu")
    assert a == b


def test_decor_gameplay_obstacles_never_overlap_decor_walls_on_same_side(tmp_path, monkeypatch):
    # Create a tiny decor model pair; monkeypatch predict_layer_events to hit both sides inside and outside
    # the gameplay obstacle window, so we test exact overlap removal by side.
    _write_all_checkpoints(tmp_path)
    models, _ = load_layer_models(tmp_path, "cpu")
    # Also force wall_what to emit inside-column or outside-column walls cleanly (default channel_mask allows 0..23)
    mel = _mel(100.0)
    # All notes on the right side (cols 2-3): gameplay_obstacles at level 2 picks left side-step (x=0, w=2)
    notes = _side_notes(100.0, x_choices=(2, 3), step=0.5)
    from bsmaker.obstacles_rules import gameplay_obstacles

    gp = gameplay_obstacles(notes, BPM, 100.0, 2, 0)
    assert len(gp) >= 1 and gp[0].x == 0 and gp[0].w == 2
    g0, g1 = gp[0].time_s, gp[0].time_s + gp[0].duration_s
    mid = (g0 + g1) / 2.0

    import bsmaker.layers_generate as lg

    # Channel 0 (left): one overlapping mid, one far away at 4.0 s
    # Channel 1 (right): one overlapping mid (must be KEPT because different side!), one at 4.0 s
    monkeypatch.setattr(
        lg,
        "predict_layer_events",
        lambda *a, **k: [
            np.array([4.0, mid], dtype=np.float64),
            np.array([4.0, mid], dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        ],
    )

    # Force deterministic wall tokens via sampler spy (channel 0 -> x=0 token 0; channel 1 -> x=4 token 12)
    def spy_sample(loaded, times, channels, mel, notes, nps, seed, device, allowed=None, **kw):
        toks = np.array([0 if ch == 0 else 12 for ch in channels], dtype=np.int64)
        for i, (t, ch, tok) in enumerate(zip(times, channels, toks)):
            assert allowed(i, float(t), int(ch))[tok]
        return toks, 0

    monkeypatch.setattr(lg, "sample_layer_values", spy_sample)
    obs, bombs, rep = generate_decor(models, mel, notes, 2.0, BPM, 100.0, 2, LAYER_STATS, 0, "cpu")
    assert bombs == []
    assert rep["gameplay_walls_placed"] == len(gp)
    assert rep["dropped_by_gameplay"] == 1
    assert rep["decor_walls_placed"] + rep["walls_skipped"] == 4 - rep["dropped_by_gameplay"]
    assert rep["decor_walls_placed"] == 3

    # Verify the kept right-side wall at `mid` survived, while the left-side wall at `mid` was dropped
    decor_only = [o for o in obs if o not in gp]
    assert len(decor_only) == 3
    for d in decor_only:
        for g in gp:
            same_side = (d.x < 2 and g.x < 2) or (d.x + d.w > 2 and g.x + g.w > 2)
            overlaps_time = d.time_s < g.time_s + g.duration_s and g.time_s < d.time_s + d.duration_s
            assert not (same_side and overlaps_time)
    assert any(o.x >= 3 and abs(o.time_s - mid) < 1e-9 for o in decor_only)


def test_decor_duck_obstacle_drops_overlapping_decor_walls_on_both_sides(all_models, monkeypatch):
    # Notes everywhere except a wide gap 40..45 s -> gameplay_obstacles produces a duck wall (x=0, w=4)
    mel = _mel(70.0)
    notes = [
        n
        for n in _side_notes(70.0, x_choices=(0, 3), step=0.5)
        if not 39.9 < n.time_s < 45.1
    ]
    from bsmaker.obstacles_rules import gameplay_obstacles

    gp = gameplay_obstacles(notes, BPM, 70.0, 2, 0)
    assert len(gp) == 1 and (gp[0].x, gp[0].w, gp[0].y, gp[0].h) == (0, 4, 2, 3)
    mid = gp[0].time_s + gp[0].duration_s / 2.0

    import bsmaker.layers_generate as lg

    monkeypatch.setattr(
        lg,
        "predict_layer_events",
        lambda *a, **k: [
            np.array([4.0, mid], dtype=np.float64),
            np.array([4.0, mid], dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        ],
    )

    def spy_sample(loaded, times, channels, mel, notes, nps, seed, device, allowed=None, **kw):
        toks = np.full(len(channels), 12, dtype=np.int64)  # x=-1 / x=4 outside grid
        for i, (t, ch, tok) in enumerate(zip(times, channels, toks)):
            assert allowed(i, float(t), int(ch))[tok]
        return toks, 0

    monkeypatch.setattr(lg, "sample_layer_values", spy_sample)
    obs, _, rep = generate_decor(all_models, mel, notes, 2.0, BPM, 70.0, 2, LAYER_STATS, 0, "cpu")
    assert rep["dropped_by_gameplay"] == 2
    assert rep["decor_walls_placed"] + rep["walls_skipped"] == 4 - rep["dropped_by_gameplay"]
    assert rep["decor_walls_placed"] == 2
    assert all(abs(o.time_s - 4.0) < 1e-9 for o in obs if o not in gp)


def test_lights_model_path_and_determinism(all_models):
    mel = _mel(45.0)
    notes = _side_notes(45.0)
    a, rep_a = generate_lights(all_models, mel, notes, 2.0, BPM, 0.0, 45.0, 2, LAYER_STATS, 7, "cpu")
    b, _ = generate_lights(all_models, mel, notes, 2.0, BPM, 0.0, 45.0, 2, LAYER_STATS, 7, "cpu")
    assert a == b and len(a) > 0
    assert rep_a["used_rules"] is False and rep_a["missing_models"] is False
    assert rep_a["skipped_by_rules"] == 0 and rep_a["placed"] == len(a)
    valid_types = set(CHANNEL_TYPES)
    for e in a:
        assert isinstance(e, LightEvent)
        assert e.type in valid_types
        assert e.float_value == 1.0
        assert abs(e.beat - e.time_s * BPM / 60.0) < 1e-9
    keys = [(e.time_s, e.type) for e in a]
    assert keys == sorted(keys)


def test_unsteady_grid_and_31s_song_work(all_models):
    mel = _mel(31.0)
    # Unsteady grid: arbitrary detected BPM, offset_s = 0.0, no beat snapping
    bpm = 137.4
    notes = [Note(1.5 + 0.5 * k, (1.5 + 0.5 * k) * bpm / 60.0, k % 4, 1, k % 2, 1) for k in range(55)]
    obs, bombs, drep = generate_decor(all_models, mel, notes, 2.0, bpm, 31.0, 3, LAYER_STATS, 0, "cpu")
    lights, lrep = generate_lights(all_models, mel, notes, 2.0, bpm, 0.0, 31.0, 2, LAYER_STATS, 0, "cpu")
    assert drep["missing_models"] is False
    assert check_layers(notes, Layers(tuple(bombs), tuple(obs), (), (), (), False, 0)) == []
    for o in obs:
        assert abs(o.duration_s - o.duration_beats * 60.0 / bpm) < 1e-9
    assert lrep["placed"] == len(lights) and len(lights) > 0


def test_runs_under_inference_mode(all_models, monkeypatch):
    import bsmaker.layers_generate as lg

    seen = []
    orig_predict = lg.predict_layer_events

    def spy_predict(*a, **k):
        seen.append(torch.is_inference_mode_enabled())
        return orig_predict(*a, **k)

    monkeypatch.setattr(lg, "predict_layer_events", spy_predict)
    mel = _mel(31.0)
    notes = _side_notes(31.0)
    generate_decor(all_models, mel, notes, 2.0, BPM, 31.0, 2, LAYER_STATS, 0, "cpu")
    generate_lights(all_models, mel, notes, 2.0, BPM, 0.0, 31.0, 2, LAYER_STATS, 0, "cpu")
    assert seen == [True, True]
