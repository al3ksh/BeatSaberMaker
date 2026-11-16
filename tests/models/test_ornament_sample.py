import numpy as np
import pytest
import torch

from bsmaker.formats.model import Note
from bsmaker.models.ornaments.model import OrnamentConfig, OrnamentModel
from bsmaker.models.ornaments.sample import (
    LoadedOrnaments, load_ornaments, sample_ornaments)

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
    notes = sorted(notes, key=lambda n: n.time_s)

    def next_same(color, t):
        later = [n.time_s for n in notes if n.color == color and n.time_s > t]
        return min(later) if later else None

    used = set()
    for c in chains:
        head = next(n for n in notes if (n.time_s, n.color, n.x, n.y) == (c.time_s, c.color, c.x, c.y))
        assert head.direction != 8
        nt = next_same(c.color, c.time_s)
        assert nt is not None and nt - c.time_s >= 0.25
        assert (c.tx, c.ty) != (c.x, c.y) and 0 <= c.tx <= 3 and 0 <= c.ty <= 2
        assert 3 <= c.segments <= 7 and c.squish == 1.0
        for o in notes:
            if o is not head and c.time_s <= o.time_s <= nt:
                assert (o.x, o.y) not in {(c.x, c.y), (c.tx, c.ty)}
        used.add((c.time_s, c.color))
    for a in arcs:
        nt = next_same(a.color, a.time_s)
        assert nt is None or a.tail_time_s <= nt
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
