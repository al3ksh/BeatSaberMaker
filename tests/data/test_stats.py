import numpy as np
import pytest

from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.stats import NPS_BINS, compute_stats, load_stats, lookup, write_stats


def _cache(specs):
    """specs: (rank, nps, njs, n_events, n_double, split)."""
    entries, ts, ls, rs, offs = [], [], [], [], [0]
    for i, (rank, nps, njs, n, d, split) in enumerate(specs):
        entries.append(Entry(i, f"m{i}", "Standard", rank, split, nps, ("tech",), njs, 0.1, 1000))
        ts.append(np.arange(n, dtype=np.float32))
        ls.append(np.full(n, 5, np.uint8))
        rs.append(np.array([5] * d + [0] * (n - d), np.uint8))
        offs.append(offs[-1] + n)
    return EventCache(entries, np.concatenate(ts), np.concatenate(ls), np.concatenate(rs),
                      np.array(offs, np.int64))


@pytest.fixture
def feats(tmp_path):
    for i in range(4):
        np.save(tmp_path / f"m{i}.npy", np.full((10, 80), 2.0, np.float16))
    return tmp_path


def test_stats_double_share(feats):
    cache = _cache([("Expert", 2.0, 18, 10, 5, "train"), ("Hard", 2.0, 20, 10, 1, "train"),
                    ("Expert", 2.0, 99, 10, 10, "val")])
    s = compute_stats(cache, feats)
    assert s["double_share_by_bin"][1] == pytest.approx(6 / 20)
    assert s["njs_by_bin"][1] == 19.0  # val map ignored
    assert s["mel_mean"] == pytest.approx(2.0) and s["mel_std"] == pytest.approx(0.0)


def test_stats_bins_cover_range(feats):
    cache = _cache([("Expert", 2.0, 18, 10, 5, "train"), ("Expert", 10.0, 22, 10, 0, "train")])
    s = compute_stats(cache, feats)
    n = len(NPS_BINS) - 1
    for table in ("njs_by_bin", "njs_offset_by_bin", "double_share_by_bin"):
        assert len(s[table]) == n and all(v is not None for v in s[table])
    assert s["njs_by_bin"][0] == 18 and s["njs_by_bin"][-1] == 22 and s["njs_by_bin"][4] == 18


def test_rank_names(feats):
    cache = _cache([("Expert+", 5.0, 18, 10, 0, "train"), ("normal", 2.0, 18, 10, 0, "train"),
                    ("Lawless", 9.0, 18, 10, 0, "train")])
    assert compute_stats(cache, feats)["rank_median_nps"] == {"Normal": 2.0, "ExpertPlus": 5.0}


def test_stats_json_roundtrip(feats, tmp_path):
    s = compute_stats(_cache([("Expert", 2.0, 18, 10, 5, "train")]), feats)
    write_stats(s, tmp_path / "s.json")
    assert load_stats(tmp_path / "s.json") == s


def test_lookup_clamps_outside_bins():
    s = {"nps_bins": NPS_BINS, "njs_by_bin": list(range(8))}
    assert lookup(s, "njs_by_bin", 0.0) == 0
    assert lookup(s, "njs_by_bin", 2.0) == 1
    assert lookup(s, "njs_by_bin", 50.0) == 7
