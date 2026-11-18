import numpy as np
import pytest

from bsmaker.data.events_cache import Entry
from bsmaker.data.pattern_metrics import PatternMetrics
from bsmaker.formats.tokens import encode_token
from bsmaker.models.sampling import entry_weights, pattern_factor, pattern_factors

BINS = {"nps_bins": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 12]}
REF = PatternMetrics(diag_share=0.3, bottom_up_share=0.1, top_row_share=0.2, flat_flip_share=0.2)
FLAT = PatternMetrics(diag_share=0.1, bottom_up_share=0.5, top_row_share=0.05, flat_flip_share=0.6)
MID = PatternMetrics(diag_share=0.1, bottom_up_share=0.3, top_row_share=0.05, flat_flip_share=0.6)


def _pe(i=0, rank="Expert", styles=("unknown",), flags=(), score=0.95):
    return Entry(i, f"k{i}", "Standard", rank, "train", 3.0, styles, 18.0, 0.0, 100,
                 score=score, flags=flags)


@pytest.mark.parametrize("rank", ["Hard", "Expert", "ExpertPlus"])
def test_pattern_factor_ranks_it_applies_to(rank):
    assert pattern_factor(_pe(rank=rank), REF) == 1.5
    assert pattern_factor(_pe(rank=rank), FLAT) == 0.5
    assert pattern_factor(_pe(rank=rank), MID) == 1.0


@pytest.mark.parametrize("rank", ["Easy", "Normal"])
def test_pattern_factor_is_one_below_hard(rank):
    assert pattern_factor(_pe(rank=rank), REF) == 1.0
    assert pattern_factor(_pe(rank=rank), FLAT) == 1.0


def test_flat_penalty_only_for_unknown_style_above_035():
    assert pattern_factor(_pe(styles=("balanced",)), FLAT) == 1.0
    assert pattern_factor(_pe(styles=("tech",)), REF) == 1.5
    assert pattern_factor(_pe(), PatternMetrics(0.1, 0.35, 0.05, 0.6)) == 1.0
    assert pattern_factor(_pe(), PatternMetrics(0.1, 0.3501, 0.05, 0.6)) == 0.5
    assert pattern_factor(_pe(), None) == 1.0


def test_factors_multiply_weights_and_flagged_stay_zero():
    es = [_pe(0), _pe(1), _pe(2, flags=("low_score",)), _pe(3, rank="Easy")]
    f = np.array([pattern_factor(e, m) for e, m in zip(es, [REF, FLAT, REF, FLAT])])
    assert f.tolist() == [1.5, 0.5, 1.5, 1.0]
    base = entry_weights(es, BINS)
    w = entry_weights(es, BINS, factors=f)
    assert w[2] == 0.0 and base[2] == 0.0
    assert w[[0, 1, 3]] == pytest.approx(base[[0, 1, 3]] * f[[0, 1, 3]])


def test_quality_only_weighting_unchanged_without_factors():
    es = [_pe(0), _pe(1, score=0.875), _pe(2, score=None)]
    assert entry_weights(es, BINS) == pytest.approx([1.0, 0.625, 0.6])
    assert np.array_equal(entry_weights(es, BINS), entry_weights(es, BINS, factors=None))
    with pytest.raises(ValueError):
        entry_weights(es, BINS, factors=np.ones(2))


class _Cache:
    def __init__(self, streams):
        self.streams = streams

    def events(self, i):
        left, right = self.streams[i]
        return np.zeros(len(left), dtype=np.float32), left, right


def test_pattern_factors_reads_token_streams_from_the_cache():
    ref_l = np.array([encode_token(1, 2, 6), encode_token(1, 0, 7)] * 10, dtype=np.uint8)
    ref_r = np.array([encode_token(2, 2, 7), encode_token(2, 0, 6)] * 10, dtype=np.uint8)
    flat_l = np.array([encode_token(1, 0, 1), encode_token(1, 0, 0)] * 10, dtype=np.uint8)
    flat_r = np.array([encode_token(2, 0, 1), encode_token(2, 0, 0)] * 10, dtype=np.uint8)
    cache = _Cache({0: (ref_l, ref_r), 1: (flat_l, flat_r), 2: (flat_l, flat_r)})
    es = [_pe(0), _pe(1), _pe(2, rank="Normal")]
    assert pattern_factors(cache, es).tolist() == [1.5, 0.5, 1.0]
