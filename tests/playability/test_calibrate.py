from dataclasses import replace
from pathlib import Path

from bsmaker.formats.events import Event
from bsmaker.formats.tokens import NONE_TOKEN, encode_token
from bsmaker.playability.calibrate import (
    calibrate,
    measure,
    rare_tokens_from,
    render_report,
    write_playability_toml,
)
from bsmaker.playability.config import PlayabilityConfig, load_playability_config

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


def test_relaxation_order_parity() -> None:
    lists = [turn90_list(), turn90_list()]
    start = PlayabilityConfig()
    first, steps = calibrate(lists, start, max_rounds=1)
    assert first.config.strict_angle_deg == 67.5 and len(steps) == 1
    final, steps = calibrate(lists, start, max_rounds=6)
    assert final.config.strict_angle_deg == 90.0
    assert final.pass_share == 1.0 and len(steps) == 2


def test_never_relaxes_same_cell() -> None:
    lists = [same_cell_list()] * 3
    start = PlayabilityConfig()
    result, steps = calibrate(lists, start, max_rounds=6)
    assert result.rule_note_share["same_cell"] > 0
    assert steps == []
    assert replace(result.config, rare_tokens=start.rare_tokens) == start
    assert result.pass_share == 0.0


def test_unrelaxable_top_rule_falls_through_to_next() -> None:
    lists = [same_cell_list(), turn90_list(10)]
    result, steps = calibrate(lists, PlayabilityConfig(), max_rounds=1)
    share = measure(lists, PlayabilityConfig()).rule_note_share
    assert share["same_cell"] > share["parity"] > 0
    assert len(steps) == 1 and result.config.strict_angle_deg == 67.5


def test_toml_roundtrip(tmp_path: Path) -> None:
    cfg = replace(PlayabilityConfig(), strict_angle_deg=67.5, slow_gap_s=0.3, forbid_crossover=False,
                  min_hand_gap_s=0.0733, rare_tokens=(frozenset({3, 9}), frozenset({1, 108})))
    path = tmp_path / "playability.toml"
    write_playability_toml(cfg, path)
    assert load_playability_config(path) == cfg
    write_playability_toml(PlayabilityConfig(), path)
    assert load_playability_config(path) == PlayabilityConfig()


def test_render_report_lists_steps() -> None:
    lists = [turn90_list()]
    before = measure(lists, PlayabilityConfig())
    after, steps = calibrate(lists, PlayabilityConfig())
    text = render_report(before, after, steps)
    assert "pass_share" in text and steps[0] in text
