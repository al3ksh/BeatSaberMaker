import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from bsmaker.data.events_cache import Entry
from bsmaker.models.sampling import entry_flags, entry_weights
from bsmaker.models.train_loop import TrainSettings, run_training

BINS = {"nps_bins": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 12]}


def _e(i, score=0.9, nps=3.0, flags=()):
    return Entry(i, f"k{i}", "Standard", "Expert", "train", nps, (), 18.0, 0.0, 100,
                 score=score, flags=flags)


@pytest.mark.parametrize("score,want", [(None, 0.6), (0.70, 0.25), (0.80, 0.25),
                                        (0.875, 0.625), (0.95, 1.0), (0.99, 1.0)])
def test_quality_weight(score, want):
    # all in one bin: the density factor is 1
    w = entry_weights([_e(0, score), _e(1, score)], BINS)
    assert w[0] == pytest.approx(want)


def test_flags():
    assert entry_flags(0.69, []) == ("low_score",)
    assert entry_flags(0.70, []) == ()
    assert entry_flags(None, []) == ()
    for t in ("comedy-meme", "challenge", "ai"):
        assert entry_flags(0.9, ["tech", t]) == ("joke_tag",)
    assert entry_flags(0.5, ["ai"]) == ("low_score", "joke_tag")


def test_flagged_weighs_zero():
    w = entry_weights([_e(0, flags=("low_score",)), _e(1, flags=("joke_tag",)), _e(2)], BINS)
    assert w[0] == 0.0 and w[1] == 0.0 and w[2] > 0


def test_density_factor_and_clips():
    # bin 1: 4 entries, bin 3: 1 entry, bin 5: 1 entry -> median count 1 (nonempty bins only)
    es = [_e(i, 0.95, 1.0) for i in range(4)] + [_e(4, 0.95, 3.0), _e(5, 0.95, 5.0)]
    w = entry_weights(es, BINS)
    assert w[:4] == pytest.approx([0.5] * 4)  # (1/4)**0.5 = 0.5
    assert w[4:] == pytest.approx([1.0, 1.0])
    # clip at 3.0: 1 entry vs median of 100-entry bins
    es = [_e(i, 0.95, 1.0) for i in range(100)] + [_e(100 + i, 0.95, 3.0) for i in range(100)]
    es.append(_e(999, 0.95, 5.0))
    w = entry_weights(es, BINS)
    assert w[-1] == pytest.approx(3.0) and w[0] == pytest.approx(1.0)
    # clip at 0.5 from below
    es = [_e(i, 0.95, 1.0) for i in range(100)] + [_e(200, 0.95, 3.0), _e(201, 0.95, 5.0)]
    assert entry_weights(es, BINS)[0] == pytest.approx(0.5)


def test_out_of_range_nps_goes_to_nearest_bin():
    es = [_e(0, 0.95, 0.1), _e(1, 0.95, 0.6), _e(2, 0.95, 50.0), _e(3, 0.95, 11.0)]
    w = entry_weights(es, BINS)
    # bins: {0: 2 entries, 7: 2 entries}, median 2 -> density 1 for all
    assert w == pytest.approx([1.0] * 4)


def test_empty():
    assert entry_weights([], BINS).shape == (0,)


class Idx(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i)), "y": torch.zeros(1)}


def _train(out, weights, max_steps=5, options=None, log=None, ds=Idx):
    seen = []

    def step(model, batch):
        seen.extend(int(v) for v in batch["x"][:, 0])
        return ((model(batch["x"]) - batch["y"]) ** 2).mean()

    torch.manual_seed(0)
    s = TrainSettings(max_steps=max_steps, batch_size=4, warmup=1, eval_every=1000,
                      save_every=1000, num_workers=0, amp=False, device="cpu")
    run_training(nn.Linear(4, 1), ds(), ds(), step,
                 lambda m, l: {"loss": 1.0}, out, s, {"d": 4},
                 lambda n, b: b is None, (log if log is not None else []).append,
                 sample_weights=weights, train_options=options)
    return seen


def test_weights_select_only_positive_entries(tmp_path):
    w = np.zeros(16)
    w[0] = 1.0
    assert set(_train(tmp_path, w)) == {0}


def test_no_weights_keeps_previous_shuffle(tmp_path):
    # expected: DataLoader(shuffle=True, generator=gen) with the same seeding as before
    from torch.utils.data import DataLoader
    gen = torch.Generator().manual_seed(
        int(np.random.SeedSequence([0, 0]).generate_state(1, dtype=np.uint64)[0] >> 1))
    want = []
    for b in DataLoader(Idx(), batch_size=4, shuffle=True, drop_last=True, generator=gen):
        want.extend(int(v) for v in b["x"][:, 0])
    assert _train(tmp_path, None, max_steps=4) == want


def test_train_options_resume(tmp_path):
    _train(tmp_path, None, 3, {"weighting": "quality"})
    from bsmaker.models.common import load_checkpoint
    assert load_checkpoint(tmp_path / "last.pt")["train_options"] == {"weighting": "quality"}
    _train(tmp_path, None, 6, {"weighting": "quality"})  # equal: fine
    with pytest.raises(ValueError) as ei:
        _train(tmp_path, None, 9, {"weighting": "none"})
    msg = str(ei.value)
    assert "weighting" in msg and "quality" in msg and "none" in msg and "--out" in msg


def test_checkpoint_without_train_options_resumes(tmp_path):
    from bsmaker.models.common import load_checkpoint
    _train(tmp_path, None, 3)
    ck = load_checkpoint(tmp_path / "last.pt")
    assert ck["train_options"] == {}
    del ck["train_options"]
    torch.save(ck, tmp_path / "last.pt")
    logs = []
    _train(tmp_path, None, 6, {"weighting": "quality"}, log=logs)
    assert any("no train_options" in m for m in logs)
