import json

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from bsmaker.models.common import load_checkpoint
from bsmaker.models.train_loop import TrainSettings, run_training


class Toy(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(4, generator=g)
        return {"x": x, "y": x.sum()[None]}


def _step(model, batch):
    return ((model(batch["x"]) - batch["y"]) ** 2).mean()


def _eval(model, loader):
    dev = next(model.parameters()).device
    with torch.no_grad():
        losses = [float(_step(model, {k: v.to(dev) for k, v in b.items()})) for b in loader]
    return {"loss": sum(losses) / len(losses)}


def _better(new, best):
    return best is None or new["loss"] < best["loss"]


def _run(out, max_steps, model=None):
    torch.manual_seed(0)
    model = model or nn.Linear(4, 1)
    logs = []
    s = TrainSettings(max_steps=max_steps, batch_size=4, warmup=2, eval_every=10, save_every=10,
                      num_workers=0, amp=False, device="cpu")
    r = run_training(model, Toy(), Toy(), _step, _eval, out, s, {"d": 4}, _better, logs.append,
                     stats={"mel_mean": 0.5})
    return model, r, logs


def test_checkpoint_atomic_and_resume(tmp_path):
    model, r, logs = _run(tmp_path, 20)
    assert r["step"] == 20
    assert not list(tmp_path.glob("*.tmp"))
    ck = load_checkpoint(tmp_path / "last.pt")
    assert ck["step"] == 20 and ck["stats"] == {"mel_mean": 0.5} and ck["model_config"] == {"d": 4}
    assert load_checkpoint(tmp_path / "prev.pt")["step"] == 10
    best = load_checkpoint(tmp_path / "best.pt")
    assert best["best"] is not None and "loss" in best["best"]
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert [json.loads(x)["step"] for x in lines] == [10, 20]
    assert any(x.startswith("step 1 ") for x in logs) and any(x.startswith("eval") for x in logs)

    resumed = nn.Linear(4, 1)
    _, r2, logs2 = _run(tmp_path, 20, model=resumed)
    assert r2["step"] == 20 and logs2[0] == "resumed from step 20"
    for a, b in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
    _, r3, _ = _run(tmp_path, 30, model=nn.Linear(4, 1))
    assert r3["step"] == 30 and load_checkpoint(tmp_path / "last.pt")["step"] == 30


def test_resume_falls_back_when_last_is_truncated(tmp_path):
    _run(tmp_path, 20)
    last = tmp_path / "last.pt"
    last.write_bytes(last.read_bytes()[:100])
    _, r, logs = _run(tmp_path, 30, model=nn.Linear(4, 1))
    assert logs[0] == "resumed from step 10"
    assert r["step"] == 30


def test_resume_fails_when_nothing_loads(tmp_path):
    _run(tmp_path, 20)
    for n in ("last.pt", "prev.pt"):
        (tmp_path / n).write_bytes(b"junk")
    with pytest.raises(RuntimeError):
        _run(tmp_path, 30)


def test_rotation_after_fallback_keeps_good_prev(tmp_path):
    _run(tmp_path, 20)
    last = tmp_path / "last.pt"
    last.write_bytes(last.read_bytes()[:100])
    _run(tmp_path, 30, model=nn.Linear(4, 1))  # resumes from prev (step 10), saves 20 and 30
    assert load_checkpoint(tmp_path / "last.pt")["step"] == 30
    assert load_checkpoint(tmp_path / "prev.pt")["step"] == 20


def test_best_pt_kept_when_better_than_resumed_record(tmp_path):
    _run(tmp_path, 20)
    ck = load_checkpoint(tmp_path / "best.pt")
    ck["best"] = {"loss": -1.0}  # unbeatable
    torch.save(ck, tmp_path / "best.pt")
    last = tmp_path / "last.pt"
    last.write_bytes(last.read_bytes()[:100])
    _, r, _ = _run(tmp_path, 30, model=nn.Linear(4, 1))
    assert r["best"] == {"loss": -1.0}
    assert load_checkpoint(tmp_path / "best.pt")["best"] == {"loss": -1.0}
