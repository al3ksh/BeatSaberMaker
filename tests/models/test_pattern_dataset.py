import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from bsmaker.data.events_cache import Entry
from bsmaker.models.pattern.dataset import PatternSequences
from bsmaker.models.train_loop import TrainSettings, run_training

STATS = {"mel_mean": 0.0, "mel_std": 1.0}


class FakeCache:
    def __init__(self, ev):
        self.ev = ev

    def events(self, i):
        return self.ev


def _ds(tmp_path, n_events, context, fixed=False, corrupt_p=None, seed=0):
    n_frames = 400
    np.save(tmp_path / "m.npy", np.zeros((n_frames, 80), np.float16))
    times = np.arange(1, n_events + 1, dtype=np.float32)
    left = np.arange(1, n_events + 1, dtype=np.uint8)
    right = np.arange(255, 255 - n_events, -1, dtype=np.uint8)  # left + right == 256
    entry = Entry(0, "m", "Standard", "Expert", "train", 3.0, ("tech",), 18.0, 0.0, n_frames)
    kw = {} if corrupt_p is None else {"corrupt_p": corrupt_p}
    return PatternSequences(FakeCache((times, left, right)), [entry], tmp_path, STATS, context,
                            seed=seed, fixed=fixed, **kw)


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_zero_corruption_reproduces_todays_batches(tmp_path):
    a, b = _ds(tmp_path, 60, 8), _ds(tmp_path, 60, 8, corrupt_p=0.0)
    for _ in range(5):
        assert _same(a[0], b[0])


def test_half_corruption_statistics(tmp_path):
    ds = _ds(tmp_path, 200, 100, corrupt_p=0.5)
    changed = total = 0
    for _ in range(20):
        it = ds[0]
        left = it["left"].numpy()
        pl, pr = it["prev_left"].numpy(), it["prev_right"].numpy()
        assert (pl[1:] + pr[1:] == 256).all()  # replaced pairs stay intact
        assert set(pl[1:].tolist()) <= set(left.tolist())  # and exist elsewhere in the run
        changed += int((pl[1:] != left[:-1]).sum())
        total += 99
    assert 0.45 < changed / total < 0.55


def test_targets_and_position_zero_untouched(tmp_path):
    for seed in range(10):  # fresh pair per seed: corruption advances the generator
        ia = _ds(tmp_path, 200, 50, corrupt_p=0.0, seed=seed)[0]
        ib = _ds(tmp_path, 200, 50, corrupt_p=0.5, seed=seed)[0]
        for k in ("left", "right", "valid", "gaps", "mel_slices"):
            assert torch.equal(ia[k], ib[k])
        assert ia["prev_left"][0] == ib["prev_left"][0] and ia["prev_right"][0] == ib["prev_right"][0]
        assert not torch.equal(ia["prev_left"], ib["prev_left"])


def test_replacement_is_never_the_true_predecessor_position(tmp_path):
    ds = _ds(tmp_path, 200, 50, corrupt_p=0.5)
    for _ in range(20):
        it = ds[0]
        left, pl = it["left"].tolist(), it["prev_left"].tolist()
        for j in range(1, 50):  # tokens are unique per position, so equal token = same position
            if pl[j] != left[j - 1]:
                assert pl[j] in left


@pytest.mark.parametrize("n", [1, 2, 3])
def test_short_runs_do_not_crash(tmp_path, n):
    ds = _ds(tmp_path, n, 8, corrupt_p=0.5)
    for _ in range(10):
        it = ds[0]
        if n < 3:  # nothing to corrupt: identical to the true predecessors
            assert it["prev_left"][1:n].tolist() == it["left"][:n - 1].tolist()
        else:  # 3 events: replacement for j=1 can only be position 1 or 2
            assert int(it["valid"].sum()) == 3


def test_fixed_ignores_corrupt_p(tmp_path):
    a = _ds(tmp_path, 60, 8, fixed=True, corrupt_p=0.5)
    b = _ds(tmp_path, 60, 8, fixed=True, corrupt_p=0.0)
    assert _same(a[0], b[0])


class Toy(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i)), "y": torch.zeros(1)}


def _train(out, steps, options):
    s = TrainSettings(max_steps=steps, batch_size=4, warmup=1, eval_every=1000, save_every=1000,
                      num_workers=0, amp=False, device="cpu")
    run_training(nn.Linear(4, 1), Toy(), Toy(), lambda m, b: ((m(b["x"]) - b["y"]) ** 2).mean(),
                 lambda m, l: {"loss": 1.0}, out, s, {"d": 4}, lambda n, b: b is None,
                 lambda m: None, train_options=options)


def test_resume_with_other_corrupt_p_fails(tmp_path):
    _train(tmp_path, 3, {"weighting": "none", "corrupt_p": 0.1})
    _train(tmp_path, 6, {"weighting": "none", "corrupt_p": 0.1})
    with pytest.raises(ValueError) as ei:
        _train(tmp_path, 9, {"weighting": "none", "corrupt_p": 0.2})
    msg = str(ei.value)
    assert "corrupt_p" in msg and "0.1" in msg and "0.2" in msg
