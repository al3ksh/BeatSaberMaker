import numpy as np
import pytest

from bsmaker.data.pattern_metrics import PatternMetrics, compute_pattern_metrics, is_ref_like
from bsmaker.formats.tokens import NONE_TOKEN, encode_token

U, D, L, R, UL, UR, DL, DR, DOT = range(9)


def _arr(*toks):
    return np.array(toks, dtype=np.uint8)


def _hand(notes):
    """(x, y, d) notes -> tokens, None -> NONE_TOKEN."""
    return _arr(*[NONE_TOKEN if n is None else encode_token(*n) for n in notes])


def test_empty_and_single_note_do_not_divide_by_zero():
    assert compute_pattern_metrics(_arr(), _arr()) == PatternMetrics(0.0, 0.0, 0.0, 0.0)
    assert compute_pattern_metrics(_arr(0, 0), _arr(0, 0)) == PatternMetrics(0.0, 0.0, 0.0, 0.0)
    m = compute_pattern_metrics(_hand([(1, 0, U)]), _hand([None]))
    assert m == PatternMetrics(0.0, 1.0, 0.0, 0.0)


def test_diag_and_top_row_share_over_all_notes():
    left = _hand([(0, 2, UL), (1, 0, D), None, (1, 1, DOT)])
    right = _hand([(2, 2, DR), None, (3, 0, DL), (2, 1, U)])
    m = compute_pattern_metrics(left, right)
    assert m.diag_share == pytest.approx(3 / 6)   # dots are not diagonals
    assert m.top_row_share == pytest.approx(2 / 6)


def test_bottom_up_share_is_over_up_cuts_only():
    # 4 straight up cuts, 1 on the bottom row; diagonal ups and other notes do not count
    left = _hand([(1, 0, U), (1, 1, U), (1, 0, UL), (1, 0, D)])
    right = _hand([(2, 2, U), (2, 1, U), (2, 0, UR), (2, 0, DOT)])
    assert compute_pattern_metrics(left, right).bottom_up_share == pytest.approx(1 / 4)
    no_ups = compute_pattern_metrics(_hand([(1, 0, D), (1, 0, UL)]), _hand([None, None]))
    assert no_ups.bottom_up_share == 0.0


def test_flat_flip_share_is_over_up_down_flips_per_hand():
    # left: D(y0) U(y0) D(y1) U(y1) -> 3 U/D flips, 2 stay on their row (U0->D1 does not)
    # right: U(y0) D(y1) R U(y2) U(y2) -> one flip (U->D, rows differ), R breaks, U->U no flip
    left = _hand([(1, 0, D), (1, 0, U), (1, 1, D), (1, 1, U), None])
    right = _hand([(2, 0, U), (2, 1, D), (2, 1, R), (2, 2, U), (2, 2, U)])
    m = compute_pattern_metrics(left, right)
    assert m.flat_flip_share == pytest.approx(2 / 4)


def test_flips_skip_empty_slots_but_not_dots():
    # empty slots are skipped (same hand, consecutive notes); a dot sits between U and D
    left = _hand([(1, 0, D), None, None, (1, 0, U), (1, 0, DOT), (1, 0, D)])
    m = compute_pattern_metrics(left, _hand([None] * 6))
    assert m.flat_flip_share == pytest.approx(1.0)  # only D->U counts as a pair, flat
    # hands are not mixed: L D then R U at the same row is no flip
    m2 = compute_pattern_metrics(_hand([(1, 0, D), None]), _hand([None, (2, 0, U)]))
    assert m2.flat_flip_share == 0.0


@pytest.mark.parametrize("field,value,want", [
    (None, None, True),
    ("diag_share", 0.25, True), ("diag_share", 0.2499, False),
    ("bottom_up_share", 0.25, True), ("bottom_up_share", 0.2501, False),
    ("top_row_share", 0.15, True), ("top_row_share", 0.1499, False),
    ("flat_flip_share", 0.35, True), ("flat_flip_share", 0.3501, False),
])
def test_ref_like_boundaries(field, value, want):
    base = dict(diag_share=0.3, bottom_up_share=0.1, top_row_share=0.2, flat_flip_share=0.2)
    if field:
        base[field] = value
    assert is_ref_like(PatternMetrics(**base)) is want
