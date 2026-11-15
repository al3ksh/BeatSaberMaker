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
    calibration_summary,
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
    assert 0.7 < shifted < 1.3
    assert stretched < aligned / 2


def test_alignment_score_nan_cases():
    t = _times()
    env = onset_envelope(_mel(t))
    assert math.isnan(alignment_score(t[:19], env))
    assert math.isnan(alignment_score(t, np.zeros(FRAMES, dtype=np.float32)))
    assert math.isnan(alignment_score(t + 1000.0, env))  # beyond the envelope: nothing usable
    # more than 2% of the events beyond the envelope: not covered
    assert math.isnan(alignment_score(np.concatenate([t, t + 1000.0]), env))


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
    assert ev == {"accepted": 1, "failed": 1, "missing": 1, "unreadable": 0, "nan": 0}
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
    assert res["events_scored"] == 2
    expected = conn.execute("SELECT COUNT(*) FROM difficulties WHERE "
                            "align_score_const > align_score").fetchone()[0]
    assert res["const_higher"] == expected and expected <= 1  # 1f00 is never const-higher
    # safe to repeat: accepted stays accepted, failed is re-scored
    res2 = verify_tempo(conn, cfg)
    assert res2["kinds"]["events"] == {"accepted": 0, "failed": 1, "missing": 1, "unreadable": 0, "nan": 0}
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


def test_dense_case_scores_above_one_and_offset_events_below():
    # dense events, 6-11 frames apart; the envelope is bright exactly on them. A +-1 frame
    # window cannot separate a strictly periodic envelope from its shifts, so gaps vary.
    rng = np.random.default_rng(3)
    frames = np.cumsum(rng.integers(6, 12, size=550)) + 40
    env = np.zeros(FRAMES, dtype=np.float32)
    env[frames] = 1.0
    t = frames / FRAMES_PER_SECOND
    assert alignment_score(t, env) > 1.5
    assert alignment_score(t + 3 / FRAMES_PER_SECOND, env) < 1.0


def test_short_feature_file_is_nan_and_stays_unverified(tmp_path):
    t = _times()
    env = onset_envelope(_mel(t))
    assert math.isnan(alignment_score(t, env[: 20 * 86]))  # cut at ~30 s
    root, conn, cfg = _setup(tmp_path)
    times = [n.time_s for n in read_map(root / "1f00 (Aligned)").difficulties[0].notes]
    cut = int(max(times) * FRAMES_PER_SECOND * 0.6)
    np.save(cfg.features_dir / "1f00.npy", _mel(times)[:cut].astype(np.float16))
    set_meta(conn, "align_threshold", "0.0")
    res = verify_tempo(conn, cfg)
    assert res["kinds"]["events"]["nan"] == 1
    row = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchone()
    assert row["accepted"] == 0 and row["reject_reasons"] == '["tempo_unverified"]'
    assert row["align_score"] is None


def test_verify_tempo_ignores_difficulty_with_other_reasons(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    conn.execute("UPDATE difficulties SET reject_reasons='[\"tempo_unverified\",\"duration\"]' "
                 "WHERE map_key='1f00'")
    conn.commit()
    set_meta(conn, "align_threshold", "0.0")
    verify_tempo(conn, cfg)
    row = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchone()
    assert row["accepted"] == 0 and row["align_score"] is None


def test_const_score_stored_for_editor_kind_and_report_lists_all_kinds(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    conn.execute("UPDATE difficulties SET tempo_kind='editor' WHERE map_key='1f00'")
    conn.commit()
    set_meta(conn, "align_threshold", "0.0")
    res = verify_tempo(conn, cfg)
    row = conn.execute("SELECT * FROM difficulties WHERE map_key='1f00'").fetchone()
    assert row["align_score"] is not None and row["align_score_const"] is not None
    assert res["editor_scored"] == 1
    assert res["kinds"]["editor"]["accepted"] == 1
    res["kinds"]["external"] = {"accepted": 0, "failed": 2, "missing": 0, "unreadable": 0,
                                "nan": 0}
    assert "external: accepted 0, failed 2" in render_report(res, calibrated=True)


def test_unreadable_map_is_counted_apart_from_missing_features(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    shutil.rmtree(root / "1f00 (Aligned)")
    set_meta(conn, "align_threshold", "0.0")
    res = verify_tempo(conn, cfg)
    assert res["kinds"]["events"]["unreadable"] == 1
    assert res["kinds"]["events"]["missing"] == 1


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
        np.save(feats / f"{r[0]}.npy", _mel(t, seed=int(r[0][:2], 16)).astype(np.float16))
    return conn, feats


def test_calibrate_threshold_stores_percentile(tmp_path):
    conn, feats = _calib_setup(tmp_path, 6)
    thr = calibrate_threshold(conn, feats, n=6, min_scores=5)
    assert thr > 3.0
    assert float(get_meta(conn, "align_threshold")) == thr


def test_calibrate_threshold_skips_events_and_editor_rows(tmp_path):
    conn, feats = _calib_setup(tmp_path, 6)
    keys = [r[0] for r in conn.execute("SELECT map_key FROM difficulties ORDER BY map_key")]
    conn.execute("UPDATE difficulties SET tempo_kind='events' WHERE map_key=?", (keys[0],))
    conn.execute("UPDATE difficulties SET tempo_kind='editor' WHERE map_key=?", (keys[1],))
    conn.commit()
    calibrate_threshold(conn, feats, n=6, min_scores=4)
    with pytest.raises(ValueError, match="only 4 usable"):
        calibrate_threshold(conn, feats, n=6, min_scores=5)


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


def _dump(conn):
    return ([tuple(r) for r in conn.execute("SELECT * FROM difficulties ORDER BY rowid")],
            [tuple(r) for r in conn.execute("SELECT * FROM maps ORDER BY rowid")])


def test_dry_run_leaves_every_row_identical_and_returns_same_counts(tmp_path):
    root, conn, cfg = _setup(tmp_path)
    set_meta(conn, "align_threshold", "3.0")
    before = _dump(conn)
    dry = verify_tempo(conn, cfg, write=False)
    assert _dump(conn) == before
    assert dry["dry_run"] is True
    text = render_report(dry, calibrated=False)
    assert text.splitlines()[0].startswith("DRY RUN")
    wet = verify_tempo(conn, cfg)
    assert _dump(conn) != before
    assert wet["kinds"] == dry["kinds"] and wet["events_scored"] == dry["events_scored"]
    assert not render_report(wet, calibrated=False).startswith("DRY RUN")


def test_events_ties_are_excluded_from_const_higher(tmp_path):
    root = tmp_path / "levels"
    root.mkdir()
    # a "tempo map" that never changes the tempo: times equal constant BPM -> tie
    _write_map(root / "1f00 (Tie)", [{"b": 0, "m": 120}, {"b": 30, "m": 120}])
    _write_map(root / "2f00 (Real)", [{"b": 0, "m": 120}, {"b": 20, "m": 90}])
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, PlayabilityConfig())
    conn.execute("UPDATE difficulties SET accepted=0, reject_reasons='[\"tempo_unverified\"]'")
    conn.commit()
    feats = tmp_path / "features"
    feats.mkdir()
    for key, name in (("1f00", "1f00 (Tie)"), ("2f00", "2f00 (Real)")):
        times = [n.time_s for n in read_map(root / name).difficulties[0].notes]
        np.save(feats / f"{key}.npy", _mel(times).astype(np.float16))
    set_meta(conn, "align_threshold", "3.0")
    res = verify_tempo(conn, SimpleNamespace(features_dir=feats))
    assert res["events_scored"] == 2 and res["ties"] == 1 and res["const_higher"] == 0
    text = render_report(res, calibrated=False)
    assert "events: 2 scored, 1 ties" in text and "0 of 1 non-tie pairs" in text


def test_editor_variant_with_tempo_map_is_scored(tmp_path):
    from bsmaker.formats.tempo import TempoMap
    root = tmp_path / "levels"
    root.mkdir()
    folder = root / "1f00 (Editor)"
    shutil.copytree(FIX / "v2_basic", folder)
    p = folder / "ExpertStandard.dat"
    d = json.loads(p.read_text("utf-8-sig"))
    d["_notes"] = [{"_time": 2 + i, "_lineIndex": i % 4, "_lineLayer": i % 3, "_type": i % 2,
                    "_cutDirection": 8} for i in range(40)]
    d["_events"] = []
    d["_obstacles"] = []
    d["_customData"] = {"_BPMChanges": [{"_time": 0, "_BPM": 120}, {"_time": 20, "_BPM": 90}]}
    p.write_text(json.dumps(d), "utf-8")
    conn = open_manifest(tmp_path / "m.sqlite")
    index_folder(conn, root, PlayabilityConfig())
    conn.execute("UPDATE difficulties SET accepted=0, reject_reasons='[\"tempo_unverified\"]'")
    conn.commit()
    assert conn.execute("SELECT DISTINCT tempo_kind FROM difficulties").fetchone()[0] == "editor"
    bm = read_map(folder)
    tm = TempoMap(bm.bpm, ((0.0, 120.0), (20.0, 90.0)))
    feats = tmp_path / "features"
    feats.mkdir()
    np.save(feats / "1f00.npy", _mel([tm.to_seconds(n.beat) for n in bm.difficulties[0].notes]
                                     ).astype(np.float16))
    set_meta(conn, "align_threshold", "1.0")
    res = verify_tempo(conn, SimpleNamespace(features_dir=feats))
    assert res["editor_scored"] == 1 and res["editor_ties"] == 0
    assert res["editor_map_higher"] == 1
    row = conn.execute("SELECT * FROM difficulties").fetchone()
    assert row["align_score_const"] > row["align_score"]
    assert "the tempo map scores higher than constant BPM in 1 of 1 non-tie pairs" in         render_report(res, calibrated=False)
    assert "other variant" in render_report(res, calibrated=False)


def test_calibration_stores_negative_control_and_summary(tmp_path):
    conn, feats = _calib_setup(tmp_path, 6)
    thr = calibrate_threshold(conn, feats, n=6, min_scores=5)
    s = calibration_summary(conn)
    assert s["threshold"] == thr and s["n"] == 6
    assert sorted(s["percentiles"]) == [5, 25, 50, 75, 95]
    assert s["percentiles"][5] == thr
    assert 0.0 <= s["negative_pass_share"] <= 0.05  # stretched copies do not pass
    assert 0.0 <= float(get_meta(conn, "align_negative_pass_share")) <= 1.0


def test_report_warns_on_low_threshold_or_leaky_negative_control():
    base = {"threshold": 3.0, "kinds": {}, "events_scored": 0, "const_higher": 0,
            "calibration": {"threshold": 3.0, "n": 10, "negative_pass_share": 0.0,
                            "percentiles": {5: 3.0, 25: 4.0, 50: 5.0, 75: 6.0, 95: 7.0}}}
    ok = render_report(base, calibrated=True)
    assert "WARNING:" not in ok and "p95 7.000" in ok and "negative control" in ok
    low = render_report({**base, "threshold": 1.1}, calibrated=True)
    assert any(l.startswith("WARNING:") for l in low.splitlines())
    leaky = render_report({**base, "calibration": {**base["calibration"],
                                                   "negative_pass_share": 0.2}}, calibrated=True)
    assert any(l.startswith("WARNING:") for l in leaky.splitlines())
