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
    with torch.no_grad():
        losses = [float(_step(model, b)) for b in loader]
    return {"loss": sum(losses) / len(losses)}


def _better(new, best):
    return best is None or new["loss"] < best["loss"]


def _run(out, max_steps, model=None, cfg=None, options=None, init_ckpt=None, lr=3e-4, logs=None):
    torch.manual_seed(0)
    model = model or nn.Linear(4, 1)
    s = TrainSettings(max_steps=max_steps, batch_size=4, warmup=2, eval_every=10, save_every=10,
                      num_workers=0, amp=False, device="cpu", lr=lr)
    r = run_training(model, Toy(), Toy(), _step, _eval, out, s, cfg or {"d": 4}, _better,
                     (logs if logs is not None else []).append, train_options=options,
                     init_ckpt=init_ckpt)
    return model, r


def test_init_ckpt_loads_weights_only_and_starts_at_step_zero(tmp_path):
    base = tmp_path / "base"
    trained, _ = _run(base, 40, lr=0.05, options={"lr": 0.05})
    init = base / "best.pt"
    src = load_checkpoint(init)
    assert src["step"] > 0 and src["best"] is not None

    ft = tmp_path / "ft"
    model = nn.Linear(4, 1)
    logs = []
    # lr 0: the weights stay those of the init checkpoint
    _, r = _run(ft, 3, model=model, options={"lr": 0.0}, init_ckpt=init, lr=0.0, logs=logs)
    assert r["step"] == 3
    for k, v in src["model"].items():
        assert torch.equal(model.state_dict()[k], v)
    last = load_checkpoint(ft / "last.pt")
    assert last["step"] == 3 and last["train_options"] == {"lr": 0.0, "init_ckpt": str(init)}
    # fresh optimizer: its state only has the 3 fine-tune steps
    assert all(st["step"] == 3 for st in last["optimizer"]["state"].values())
    assert any("init" in m and str(init) in m for m in logs)
    # the base folder is untouched
    assert load_checkpoint(base / "best.pt")["step"] == src["step"]


def test_init_ckpt_best_starts_fresh(tmp_path):
    _run(tmp_path / "base", 20, lr=0.05)
    _, r = _run(tmp_path / "ft", 10, init_ckpt=tmp_path / "base" / "best.pt", lr=0.0)
    # best is the first fine-tune eval, not the base record
    assert r["best"] is not None
    assert load_checkpoint(tmp_path / "ft" / "best.pt")["step"] == 10


def test_init_ckpt_refused_when_out_has_checkpoints(tmp_path):
    _run(tmp_path / "base", 10)
    _run(tmp_path / "ft", 10)
    with pytest.raises(ValueError, match="last.pt"):
        _run(tmp_path / "ft", 20, init_ckpt=tmp_path / "base" / "best.pt")


def test_init_ckpt_restarts_after_a_cut_off_run_with_only_best(tmp_path):
    _run(tmp_path / "base", 10)
    init = tmp_path / "base" / "best.pt"
    _run(tmp_path / "ft", 10, init_ckpt=init)
    (tmp_path / "ft" / "last.pt").unlink()
    (tmp_path / "ft" / "prev.pt").unlink(missing_ok=True)
    _, r = _run(tmp_path / "ft", 10, init_ckpt=init)
    assert r["step"] == 10 and r["best"] is not None


def test_init_ckpt_with_other_model_config_is_refused(tmp_path):
    _run(tmp_path / "base", 10)
    with pytest.raises(ValueError, match="model_config"):
        _run(tmp_path / "ft", 10, cfg={"d": 5}, init_ckpt=tmp_path / "base" / "best.pt")
    assert not (tmp_path / "ft" / "last.pt").exists()


def test_fine_tune_resumes_without_init_ckpt(tmp_path):
    _run(tmp_path / "base", 10)
    init = tmp_path / "base" / "best.pt"
    _run(tmp_path / "ft", 10, options={"lr": 1.0}, init_ckpt=init)
    logs = []
    _, r = _run(tmp_path / "ft", 20, options={"lr": 1.0}, logs=logs)
    assert r["step"] == 20 and "resumed from step 10" in logs
    assert load_checkpoint(tmp_path / "ft" / "last.pt")["train_options"]["init_ckpt"] == str(init)
