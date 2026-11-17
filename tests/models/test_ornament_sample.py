import numpy as np
import pytest
import torch

from bsmaker.formats.model import Arc, Layers, Note
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel
from bsmaker.models.ornaments.sample import (
    LoadedOrnaments, load_ornaments, sample_ornaments, snap_arc_tail)
from bsmaker.playability.layers import NoteIndex, arc_ok, chain_ok, check_layers

STATS = {"mel_mean": 0.0, "mel_std": 1.0}
BPM = 120.0
# arcs and chains per minute for levels 1-3, one nps bin
LAYER_STATS = {"nps_bins": [0.0, 100.0],
               "rates": {"arcs": [[2.0, 6.0, 12.0]], "chains": [[3.0, 9.0, 18.0]]}}


@pytest.fixture(scope="module")
def loaded():
    torch.manual_seed(0)
    cfg = OrnamentConfig(d_model=32, n_layers=1, n_heads=2, context=16, dropout=0.0)
    model = OrnamentModel(cfg).eval()
    return LoadedOrnaments(model, cfg, STATS)


def _mel(seconds):
    return np.zeros((int(seconds * 22050 / 256) + 1, 80), np.float16)


def _random_notes(rng, n, span):
    times = np.sort(rng.uniform(0.0, span, n))
    return [Note(float(t), float(t) * BPM / 60, int(rng.integers(0, 4)), int(rng.integers(0, 3)),
                 int(rng.integers(0, 2)), int(rng.integers(0, 9))) for t in times]


def _run(loaded, notes, span, nps=3.0, level=3, seed=0):
    return sample_ornaments(loaded, notes, _mel(span), nps, level, LAYER_STATS, BPM, seed, "cpu")


def _check_rules(notes, arcs, chains):
    idx = NoteIndex(notes)
    assert check_layers(notes, Layers((), (), tuple(arcs), tuple(chains), (), False, 0)) == []
    used = set()
    for c in chains:
        head = next(n for n in idx.notes if (n.time_s, n.color, n.x, n.y) == (c.time_s, c.color, c.x, c.y))
        assert head.direction != 8
        assert chain_ok(c, idx)
        assert (c.tx, c.ty) != (c.x, c.y) and 0 <= c.tx <= 3 and 0 <= c.ty <= 2
        assert 3 <= c.segments <= 7 and c.squish == 1.0
        used.add((c.time_s, c.color))
    for a in arcs:
        assert arc_ok(a, idx)
        assert a.tail_time_s > a.time_s and 0 <= a.tx <= 3 and 0 <= a.ty <= 2
        assert 0 <= a.td <= 8
        assert (a.time_s, a.color) not in used
        used.add((a.time_s, a.color))


def test_level_zero_is_empty(loaded):
    notes = _random_notes(np.random.default_rng(0), 40, 60.0)
    arcs, chains, rep = _run(loaded, notes, 60.0, level=0)
    assert arcs == [] and chains == []
    assert (rep.arcs_placed, rep.chains_placed) == (0, 0)


def test_counts_equal_target_when_enough_candidates(loaded):
    notes = [Note(i * 1.0, i * 2.0, i % 4, 1, i % 2, 1) for i in range(120)]
    arcs, chains, rep = _run(loaded, notes, 120.0, level=2)
    assert (rep.arcs_requested, rep.chains_requested) == (12, 18)  # 2 min x 6 and 9 per min
    assert len(arcs) == 12 and len(chains) == 18
    assert (rep.arcs_placed, rep.chains_placed) == (12, 18)
    _check_rules(notes, arcs, chains)


@pytest.mark.parametrize("n,span", [(3, 31.0), (3, 2.0), (16, 31.0), (40, 80.0), (400, 33.0),
                                    (2, 5.0), (1, 5.0)])
def test_rules_hold_on_random_notes(loaded, n, span):
    for seed in range(8):
        notes = _random_notes(np.random.default_rng(seed), n, span)
        for level in (1, 3):
            arcs, chains, rep = _run(loaded, notes, span, nps=n / span, level=level, seed=seed)
            _check_rules(notes, arcs, chains)
            assert len(arcs) <= rep.arcs_requested and len(chains) <= rep.chains_requested


def test_counts_never_exceed_target_on_dense_song(loaded):
    notes = _random_notes(np.random.default_rng(1), 360, 30.0)  # 12 nps
    arcs, chains, rep = _run(loaded, notes, 30.0, nps=12.0, level=3)
    assert len(arcs) <= rep.arcs_requested and len(chains) <= rep.chains_requested
    _check_rules(notes, arcs, chains)


def test_empty_notes(loaded):
    arcs, chains, rep = _run(loaded, [], 31.0)
    assert arcs == [] and chains == []


def test_deterministic_per_seed(loaded):
    notes = _random_notes(np.random.default_rng(3), 80, 60.0)
    a = _run(loaded, notes, 60.0, seed=5)
    b = _run(loaded, notes, 60.0, seed=5)
    assert a[:2] == b[:2]
    c = _run(loaded, notes, 60.0, seed=6)
    assert [(x.tx, x.ty, x.td) for x in a[0]] != [(x.tx, x.ty, x.td) for x in c[0]] or not a[0]


def test_dot_notes_never_chains(loaded):
    notes = [Note(i * 1.0, i * 2.0, i % 4, 1, i % 2, 8) for i in range(60)]
    arcs, chains, rep = _run(loaded, notes, 60.0)
    assert chains == [] and rep.rejected_chain == 60
    assert arcs  # dots can still start arcs


def test_chain_blocked_by_note_in_tail_cell(loaded):
    # the only same-hand pair is 1 s apart; a note of the other hand sits in the tail cell between
    notes = [Note(0.0, 0.0, 1, 1, 0, 3), Note(0.5, 1.0, 2, 1, 1, 8), Note(1.0, 2.0, 0, 0, 0, 8)]
    _, chains, _ = _run(loaded, notes, 31.0, level=3)
    assert all(c.time_s != 0.0 for c in chains)


def test_arc_length_capped_by_next_note(loaded):
    notes = [Note(0.0, 0.0, 0, 0, 0, 1), Note(0.3, 0.6, 1, 0, 0, 1), Note(5.0, 10.0, 2, 0, 0, 1)]
    arcs, _, _ = _run(loaded, notes, 31.0, level=3)
    for a in arcs:
        if a.time_s == 0.0:
            assert a.tail_time_s <= 0.3  # 0.5 beat at 120 bpm = 0.25 s


def test_load_ornaments_roundtrip(tmp_path, loaded):
    from dataclasses import asdict

    from bsmaker.models.common import save_checkpoint
    path = tmp_path / "best.pt"
    save_checkpoint(path, {"model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
                           "stats": STATS})
    got = load_ornaments(path, "cpu")
    assert got.config == loaded.config and got.stats == STATS
    for a, b in zip(got.model.parameters(), loaded.model.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("n", [9, 12])
def test_softmax_sharp_float32_logits_sum_to_one(n):
    from bsmaker.models.ornaments.sample import _softmax
    rng = np.random.default_rng(0)
    x = rng.choice([-80.0, 80.0], size=(50, n)).astype(np.float32)
    p = _softmax(x, 0.8)
    assert p.dtype == np.float64
    for row in p:
        rng.choice(n, p=row)  # must not raise "probabilities do not sum to 1"


def test_last_note_and_exclusive_next_note_follow_playability_layers(loaded):
    chains_only = {"nps_bins": [0.0, 100.0],
                   "rates": {"arcs": [[0.0, 0.0, 0.0]], "chains": [[3.0, 9.0, 18.0]]}}
    # Single non-dot note (no subsequent same-hand note): allowed by playability.layers.chain_ok
    single = [Note(0.0, 0.0, 1, 1, 0, 3)]
    _, chains, rep = sample_ornaments(loaded, single, _mel(31.0), 3.0, 3, chains_only, BPM, 0, "cpu")
    assert rep.rejected_chain == 0 and len(chains) == 1 and chains[0].time_s == 0.0
    # Next same-hand note sits in the tail cell (2, 1) >= 0.25 s later: exclusive bound allows chain
    pair = [Note(0.0, 0.0, 1, 1, 0, 3), Note(0.5, 1.0, 2, 1, 0, 8)]
    _, chains, rep = sample_ornaments(loaded, pair, _mel(31.0), 3.0, 3, chains_only, BPM, 0, "cpu")
    assert rep.rejected_chain == 1  # only the dot note is rejected
    assert len(chains) == 1 and chains[0].time_s == 0.0


def _arc(tail_beat, tx, ty, td=8):
    return Arc(0.0, 0.0, 0, 0, 0, 8, tail_beat * 0.5, tail_beat, tx, ty, td)


def test_arc_tail_just_short_of_next_note_snaps_to_it():
    notes = [Note(0.0, 0.0, 0, 0, 0, 1), Note(2.0, 4.0, 2, 1, 0, 5), Note(1.0, 2.0, 3, 2, 1, 0)]
    snapped = snap_arc_tail(_arc(3.5, 1, 1), notes)  # 0.5 beat short, one cell away
    assert (snapped.tail_beat, snapped.tail_time_s, snapped.tx, snapped.ty, snapped.td) == (4.0, 2.0, 2, 1, 5)
    assert arc_ok(snapped, NoteIndex(notes))


def test_arc_tail_far_from_next_note_is_unchanged():
    notes = [Note(0.0, 0.0, 0, 0, 0, 1), Note(2.0, 4.0, 2, 1, 0, 5)]
    arc = _arc(2.0, 1, 1)  # 2 beats short
    assert snap_arc_tail(arc, notes) == arc
    arc = _arc(3.5, 0, 2)  # 0.5 beat short but 2 cells away
    assert snap_arc_tail(arc, notes) == arc
    arc = _arc(3.5, 0, 0)  # distance 2 in x
    assert snap_arc_tail(arc, notes) == arc
    assert snap_arc_tail(_arc(3.5, 1, 1), [Note(0.0, 0.0, 0, 0, 0, 1)]) == _arc(3.5, 1, 1)  # no next note
