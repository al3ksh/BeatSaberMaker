from dataclasses import asdict
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_generate import LAYER_STATS, STATS, make_layer_models, make_models, write_audio  # noqa: E402
from bsmaker.config import Config
from bsmaker.data.layer_stats import write_layer_stats
from bsmaker.models.common import save_checkpoint
from bsmaker.pipeline import (
    MissingModelsError, load_pipeline, pipeline_to_cpu, pipeline_to_device, run_generate,
)


def _setup_pack(tmp_path: Path, *, with_layers: bool = False) -> Config:
    ckpts = tmp_path / "checkpoints"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        save_checkpoint(ckpts / name / "best.pt", {
            "model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
            "stats": STATS, "step": 1, "best": {},
        })
    play = tmp_path / "playability.toml"
    play.write_text("[mask]\n[data]\n", encoding="utf-8")
    lstats = tmp_path / "layer_stats.json"
    write_layer_stats(LAYER_STATS, lstats)
    if with_layers:
        lm = make_layer_models()
        save_checkpoint(ckpts / "layers" / "ornaments" / "best.pt", {
            "model": lm.ornaments.model.state_dict(),
            "model_config": asdict(lm.ornaments.config), "stats": STATS,
        })
    return Config.from_paths(
        tmp_path / "CustomLevels", checkpoints_dir=ckpts,
        playability_path=play, layer_stats_path=lstats,
    )


def test_load_pipeline_missing_core_checkpoint_raises(tmp_path: Path) -> None:
    cfg = Config.from_paths(tmp_path / "CustomLevels", checkpoints_dir=tmp_path / "empty")
    with pytest.raises(MissingModelsError, match="checkpoint not found"):
        load_pipeline(cfg, device="cpu")


def test_load_pipeline_records_custom_missing_layer_stats_path(tmp_path: Path) -> None:
    cfg = _setup_pack(tmp_path, with_layers=False)
    missing_stats = tmp_path / "custom" / "my_layer_stats.json"
    cfg_custom = Config.from_paths(
        cfg.custom_levels,
        checkpoints_dir=cfg.checkpoints_dir,
        playability_path=cfg.playability_path,
        layer_stats_path=missing_stats,
    )
    pipe = load_pipeline(cfg_custom, device="cpu", load_layers=True)
    assert pipe.layer_stats is None
    assert pipe.layer_stats_missing_path == missing_stats


def test_run_generate_filters_layer_warnings_by_options(tmp_path: Path) -> None:
    cfg = _setup_pack(tmp_path, with_layers=False)
    pipe = load_pipeline(cfg, device="cpu", load_layers=True)
    audio = write_audio(tmp_path / "song.wav")

    res_off = run_generate(
        pipe, audio, out_dir=tmp_path / "out1", nps=3.0, ornaments=0, lights=0,
    )
    assert not any("checkpoint" in w for w in res_off.warnings)

    res_on = run_generate(
        pipe, audio, out_dir=tmp_path / "out2", nps=3.0, ornaments=2, lights=2,
    )
    assert any("ornaments" in w for w in res_on.warnings)
    assert any("lights-when" in w for w in res_on.warnings)
    assert res_on.lights_report is not None and res_on.lights_report["used_rules"] is True

    pipeline_to_cpu(pipe)
    pipeline_to_device(pipe, "cpu")
