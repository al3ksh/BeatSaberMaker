import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bsmaker.data.align import (
    alignment_score,
    calibrate_threshold,
    onset_envelope,
    render_report,
    verify_tempo,
)
from bsmaker.data.features import build_features
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import get_meta, open_manifest, set_meta
from bsmaker.formats.events import FRAMES_PER_SECOND
from bsmaker.formats.reader import read_map
from bsmaker.playability.config import PlayabilityConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "maps"
FRAMES = 6000


def _mel(times, seed=0):
    rng = np.random.default_rng(seed)
    mel = rng.normal(0.0, 0.3, size=(FRAMES, 80)).astype(np.float32)
    for t in times:
        f = round(t * FRAMES_PER_SECOND)
        if 0 <= f < FRAMES:
            mel[f] += 5.0
    return mel


def _times(n=40, step=0.5, start=1.0):
    return np.array([start + i * step for i in range(n)])


def test_onset_envelope_is_nonnegative_with_zero_minimum():
    env = onset_envelope(_mel(_times()))
    assert env.shape == (FRAMES,) and env.min() == 0.0 and (env >= 0).all()


def test_alignment_score_aligned_shifted_stretched():
    t = _times()
    env = onset_envelope(_mel(t))
    aligned = alignment_score(t, env)
    shifted = alignment_score(t + 0.15, env)
    stretched = alignment_score(t[0] + (t - t[0]) * 1.03, env)
    assert aligned > 5
    assert shifted < 2
    assert stretched < aligned - 2


def test_alignment_score_nan_cases():
    t = _times()
    env = onset_envelope(_mel(t))
    assert math.isnan(alignment_score(t[:19], env))
    assert math.isnan(alignment_score(t, np.zeros(FRAMES, dtype=np.float32)))
    assert math.isnan(alignment_score(t + 1000.0, env))  # beyond the envelope: nothing usable
    # notes beyond the end are ignored, the rest still count
    assert alignment_score(np.concatenate([t, t + 1000.0]), env) > 5


def _write_map(folder: Path, bpm_events=None, n=40) -> None:
    shutil.copytree(FIX / "v3_basic", folder)
    p = folder / "ExpertStandard.dat"
    d = json.loads(p.read_text("utf-8"))
    d["colorNotes"] = [{"b": 2 + i, "x": i % 4, "y": i % 3, "c": i % 2, "d": 8, "a": 0}
                       for i in range(n)]
    d["bombNotes"] = []
    d["obstacles"] = []
    if bpm_events is not None:
        d["bpmEvents"] = bpm_events
    p.write_text(json.dumps(d), "utf-8")


def _setup(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    _write_map(root / "1f00 (Aligned)", [{"b": 0, "m": 120}, {"b": 20, "m": 90}])
    _write_map(root / "2f00 (Misaligned)", [{"b": 0, "m": 120}, {"b": 20, "m": 90}])
    _write_map(root / "3f00 (NoFeatures)", [{"b": 0, "m": 120}, {"b": 20, "m": 90}])
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, PlayabilityConfig())
    # the fixture songs are too short and require mods; force the state under test
    conn.execute("UPDATE difficulties SET accepted=0, reject_reasons='[\"tempo_unverified\"]'")
    conn.commit()
    assert conn.execute("SELECT DISTINCT tempo_kind FROM difficulties").fetchall()[0][0] == "events"
    feats = tmp_path / "features"
    feats.mkdir()
    times = [n.time_s for n in read_map(root / "1f00 (Aligned)").difficulties[0].notes]
    np.save(feats / "1f00.npy", _mel(times).astype(np.float16))
    np.save(feats / "2f00.npy", _mel(np.array(times) + 0.15, seed=1).astype(np.float16))
    return root, conn, SimpleNamespace(features_dir=feats)


def _reasons(conn, key):
    return conn.execute("SELECT reject_reasons FROM difficulties WHERE map_key=?",
                        (key,)).fetchone()[0]


def test_verify_tempo_accepts_aligned_keeps_misaligned_and_reports_missing(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    assert _reasons(conn, "1f00") == '["tempo_unverified"]'
    set_meta(conn, "align_threshold", "3.0")
    res = verify_tempo(conn, cfg)
    ev = res["kinds"]["events"]
    assert ev == {"accepted": 1, "failed": 1, "missing": 1, "nan": 0}
    assert res["threshold"] == 3.0
    row = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchone()
    assert row["accepted"] == 1 and row["reject_reasons"] == "[]"
    assert row["align_score"] > 3.0
    assert row["align_score_const"] < row["align_score"]  # tempo map beats constant BPM
    bad = conn.execute("SELECT * FROM difficulties WHERE map_key='2f00'").fetchone()
    assert bad["accepted"] == 0 and bad["reject_reasons"] == '["tempo_unverified"]'
    assert bad["align_score"] is not None and bad["align_score"] < 3.0
    miss = conn.execute("SELECT * FROM difficulties WHERE map_key='3f00'").fetchone()
    assert miss["align_score"] is None and miss["accepted"] == 0
    assert res["events_scored"] == 2 and res["const_higher"] >= 0
    # safe to repeat: accepted stays accepted, failed is re-scored
    res2 = verify_tempo(conn, cfg)
    assert res2["kinds"]["events"] == {"accepted": 0, "failed": 1, "missing": 1, "nan": 0}
    assert conn.execute("SELECT accepted FROM difficulties WHERE map_key='1f00'"
                        ).fetchone()[0] == 1
    text = render_report(res, calibrated=False)
    assert "read from the manifest" in text and "events: accepted 1" in text


def test_verify_tempo_nan_is_a_fail_stored_as_null(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    np.save(cfg.features_dir / "1f00.npy", np.zeros((FRAMES, 80), dtype=np.float16))
    set_meta(conn, "align_threshold", "0.0")
    res = verify_tempo(conn, cfg)
    assert res["kinds"]["events"]["nan"] >= 1
    row = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchone()
    assert row["accepted"] == 0 and row["align_score"] is None


def test_verify_tempo_without_threshold_raises(tmp_path):
    _, conn, cfg = _setup(tmp_path)
    with pytest.raises(ValueError, match="align_threshold"):
        verify_tempo(conn, cfg)


def _calib_setup(tmp_path, n_maps):
    root = tmp_path / "levels"
    root.mkdir()
    for i in range(n_maps):
        _write_map(root / f"{i:x}a0 (M{i})")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, PlayabilityConfig())
    conn.execute("UPDATE difficulties SET split='train', accepted=1, reject_reasons='[]'")
    conn.commit()
    feats = tmp_path / "features"
    feats.mkdir()
    for r in conn.execute("SELECT map_key FROM difficulties").fetchall():
        t = [n.time_s for n in read_map(Path(conn.execute(
            "SELECT folder FROM maps WHERE key=?", (r[0],)).fetchone()[0])).difficulties[0].notes]
        np.save(feats / f"{r[0]}.npy", _mel(t, seed=hash(r[0]) % 1000).astype(np.float16))
    return conn, feats


def test_calibrate_threshold_stores_percentile(tmp_path):
    conn, feats = _calib_setup(tmp_path, 6)
    thr = calibrate_threshold(conn, feats, n=6, min_scores=5)
    assert thr > 3.0
    assert float(get_meta(conn, "align_threshold")) == thr


def test_calibrate_threshold_too_few_scores(tmp_path):
    conn, feats = _calib_setup(tmp_path, 3)
    with pytest.raises(ValueError, match="usable"):
        calibrate_threshold(conn, feats, n=10)
    assert get_meta(conn, "align_threshold") is None


def test_build_features_tempo_candidates_selection(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    conn.execute("UPDATE difficulties SET accepted=1, reject_reasons='[]' WHERE map_key='1f00'")
    conn.commit()
    seen = []

    import bsmaker.data.features as f

    orig = f.cached_log_mel
    f.cached_log_mel = lambda audio, out: seen.append(Path(out).stem) or Path(out).write_bytes(b"")
    try:
        build_features(conn, tmp_path / "f1", free_bytes=lambda p: 10**12)
        assert seen == ["1f00"]
        seen.clear()
        build_features(conn, tmp_path / "f2", free_bytes=lambda p: 10**12, tempo_candidates=True)
        assert sorted(seen) == ["1f00", "2f00", "3f00"]
    finally:
        f.cached_log_mel = orig
