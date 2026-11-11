from dataclasses import replace
from pathlib import Path

from bsmaker.formats.events import Event
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
import pytest

from bsmaker.playability.calibrate import (
    LADDER,
    calibrate,
    measure,
    rare_tokens_from,
    render_report,
    write_playability_toml,
)
from bsmaker.playability.config import (
    DATA_DEFAULTS,
    PROFILES,
    PlayabilityConfig,
    load_playability_config,
)

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def _right(i: int, d: int) -> Event:
    return Event(i * 0.25, NONE_TOKEN, encode_token(2, 1, d))


def clean_list(n: int = 20) -> list[Event]:
    return [_right(i, DOWN if i % 2 == 0 else UP) for i in range(n)]


def double_down_list(n: int = 20) -> list[Event]:
    return [_right(i, DOWN) for i in range(n)]


def turn90_list(n: int = 20) -> list[Event]:
    """Every transition is a 90 degree turn: fails strict 45, passes strict 90."""
    return [_right(i, DOWN if i % 2 == 0 else LEFT) for i in range(n)]


def same_cell_list(n: int = 20) -> list[Event]:
    return [Event(i * 0.25, encode_token(2, 1, d), encode_token(2, 1, d))
            for i, d in ((i, DOWN if i % 2 == 0 else UP) for i in range(n))]


def test_measure_counts_pass_share() -> None:
    lists = [clean_list(), clean_list(), clean_list(), double_down_list()]
    r = measure(lists, PlayabilityConfig())
    assert r.n_files == 4
    assert r.pass_share == 0.75
    assert r.rule_note_share["parity"] > 0


def test_rare_tokens_from() -> None:
    rare, common, frequent = encode_token(0, 0, UP), encode_token(1, 1, UP), encode_token(2, 1, DOWN)
    events = [Event(i * 0.25, NONE_TOKEN, rare if i == 0 else common if i < 201 else frequent)
              for i in range(20_000)]
    # rare: 1/20000 = 5e-5 < 1e-4; common: 200/20000 = 1%.
    left, right = rare_tokens_from([events])
    assert rare in right and common not in right and frequent not in right
    assert encode_token(3, 2, RIGHT) in right  # never seen
    assert NONE_TOKEN not in left and NONE_TOKEN not in right
    assert len(left) == 108  # left hand has no notes: every token is blocked


def test_ladder_stops_at_first_passing_step() -> None:
    lists = [[Event(i * 0.4, NONE_TOKEN, encode_token(2, 1, DOWN)) for i in range(20)]] * 3
    cal = calibrate(lists)
    assert cal.data.pass_share == 1.0
    assert [label for label, _ in cal.steps] == [LADDER[0][0]]
    assert cal.data.config.reset_gap_s == 0.25
    assert cal.data.config.forbid_crossover is True


def test_ladder_exhausted_reports_below_target() -> None:
    lists = [same_cell_list()] * 3
    cal = calibrate(lists)
    assert len(cal.steps) == 5 == len(LADDER)
    assert cal.data.pass_share < 0.95
    assert cal.data.config.slider_gap_s == 0.12 and cal.data.config.forbid_crossover is False
    assert "FAILED" in render_report(cal)


def test_mask_profile_keeps_default_thresholds_and_rare_tokens() -> None:
    cal = calibrate([clean_list()] * 2)
    assert replace(cal.mask.config, rare_tokens=PlayabilityConfig().rare_tokens) == PlayabilityConfig()
    assert cal.mask.config.rare_tokens == rare_tokens_from([clean_list()] * 2, 1e-4)
    assert cal.mask.rare_tokens[1]  # unseen tokens are blocked


def test_data_profile_has_no_rare_tokens() -> None:
    cal = calibrate([turn90_list()] * 2)
    assert cal.data.config.rare_tokens == (frozenset(), frozenset())


def test_pass_share_02() -> None:
    base = [_right(i, DOWN if i % 2 == 0 else UP) for i in range(101)]
    flipped = list(base)
    flipped[30] = _right(30, UP)  # two parity violations in 101 notes: rate 0.0198
    r = measure([base, flipped], PlayabilityConfig())
    assert r.pass_share == 0.5
    assert r.pass_share_02 == 1.0


def test_toml_roundtrip_two_profiles(tmp_path: Path) -> None:
    mask = replace(PlayabilityConfig(), rare_tokens=(frozenset({3, 9}), frozenset({1, 108})))
    data = replace(DATA_DEFAULTS, min_hand_gap_s=0.0733)
    path = tmp_path / "playability.toml"
    write_playability_toml(mask, data, path)
    assert load_playability_config(path, "mask") == mask
    assert load_playability_config(path, "data") == data


def test_load_missing_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "old.toml"
    path.write_text("strict_angle_deg = 90.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="old.toml.*mask"):
        load_playability_config(path)
    with pytest.raises(ValueError, match="data"):
        load_playability_config(path, "data")
    with pytest.raises(ValueError):
        load_playability_config(path, "nope")


def test_load_none_gives_profile_defaults() -> None:
    assert PROFILES == ("mask", "data")
    assert load_playability_config(None) == PlayabilityConfig()
    assert load_playability_config(None, "data") == DATA_DEFAULTS
    assert DATA_DEFAULTS.slider_gap_s == 0.12


def test_render_report_lists_steps() -> None:
    cal = calibrate([turn90_list()])
    text = render_report(cal)
    assert "pass share" in text and LADDER[0][0] in text and "mask" in text
