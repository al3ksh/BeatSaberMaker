from dataclasses import asdict
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
from test_generate import LAYER_STATS, STATS, make_layer_models, make_models  # noqa: E402
from bsmaker.models.common import load_checkpoint, save_checkpoint
from bsmaker.models.layers.what import load_what
from bsmaker.models.pattern.sample import load_pattern
from bsmaker.models.timing.decode import load_timing
from slim_checkpoints import slim_checkpoints


def test_slim_checkpoints_strips_optimizer_and_keeps_train_options(tmp_path: Path) -> None:
    src = tmp_path / "src_ckpt"
    dst = tmp_path / "dst_ckpt"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        fake_opt = {"state": {0: {"exp_avg": loaded.model.state_dict()}}}
        save_checkpoint(src / name / "best.pt", {
            "model": loaded.model.state_dict(), "optimizer": fake_opt,
            "scaler": {"scale": 65536.0}, "step": 1000,
            "model_config": asdict(loaded.config), "stats": STATS, "best": {"f1": 0.8},
        })
    lm = make_layer_models()
    save_checkpoint(src / "layers" / "lights-what" / "best.pt", {
        "model": lm.lights_what.model.state_dict(),
        "optimizer": {"state": {0: {"exp_avg": lm.lights_what.model.state_dict()}}},
        "model_config": asdict(lm.lights_what.config),
        "stats": STATS,
        "train_options": {"layer": "lights"},
    })

    report = slim_checkpoints(src, dst, allow_missing_layers=True)
    assert set(report["slimmed"]) == {"timing", "pattern", "layers/lights-what"}
    assert len(report["missing_layers"]) == 4
    for name in ("timing", "pattern", "layers/lights-what"):
        raw = load_checkpoint(dst / name / "best.pt")
        assert "optimizer" not in raw and "scaler" not in raw
        assert (dst / name / "best.pt").stat().st_size < (src / name / "best.pt").stat().st_size
    load_timing(dst / "timing" / "best.pt", "cpu")
    load_pattern(dst / "pattern" / "best.pt", "cpu")
    loaded_lw = load_what(dst / "layers" / "lights-what" / "best.pt", "cpu")
    assert loaded_lw.layer == "lights"
