import pickle

import numpy as np
import pytest
import torch

from bsmaker.models.common import (
    STYLE_DIM, Conditioning, load_checkpoint, save_checkpoint, style_vector)


def test_style_vector():
    v = style_vector(["tech", "speed"])
    assert v.dtype == np.float32 and v.shape == (STYLE_DIM,)
    assert v.tolist() == [1, 0, 1, 0, 0, 0, 0, 0]
    for names in ([], ["unknown"], ["nonsense"]):
        assert style_vector(names).tolist() == [0] * 7 + [1]


def test_conditioning_shape():
    out = Conditioning(16)(torch.zeros(3), torch.zeros(3, STYLE_DIM))
    assert out.shape == (3, 16)


def test_load_checkpoint_weights_only(tmp_path):
    payload = {"model": {"w": torch.ones(2)}, "step": 3, "model_config": {"d": 8, "x": [1, 2]},
               "stats": {"mel_mean": -1.0, "r": {"a": 1.0}}, "best": None, "scaler": {}}
    p = tmp_path / "c.pt"
    save_checkpoint(p, payload)
    assert not (tmp_path / "c.pt.tmp").exists()
    got = load_checkpoint(p)
    assert got["step"] == 3 and got["model_config"] == payload["model_config"]
    assert torch.equal(got["model"]["w"], torch.ones(2))


def test_load_checkpoint_rejects_pickled_objects(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"x": pickle.Pickler}, p)
    with pytest.raises(Exception):
        load_checkpoint(p)
