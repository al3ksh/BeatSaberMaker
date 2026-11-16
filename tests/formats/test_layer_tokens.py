import pytest

from bsmaker.data.layer_classes import wall_class
from bsmaker.formats.layer_tokens import (
    CHANNEL_TYPES, DURATION_BEATS, LIGHT_CHANNELS, LIGHT_VOCAB, WALL_VOCAB, cells_to_mask,
    duration_bucket, light_event_to_token, mask_to_cells, token_to_light_event, token_to_wall,
    wall_to_token)
from bsmaker.formats.model import Obstacle


def test_constants():
    assert (LIGHT_CHANNELS, LIGHT_VOCAB, WALL_VOCAB) == (9, 18, 24)
    assert DURATION_BEATS == (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


@pytest.mark.parametrize("value,token", [(0, 0), (5, 1), (6, 2), (7, 3), (1, 4), (2, 5), (3, 6),
                                         (4, 4), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1)])
def test_colour_channel_values(value, token):
    for ch in range(5):
        assert light_event_to_token(CHANNEL_TYPES[ch], value) == (ch, token)


def test_light_token_roundtrip_every_token():
    for ch in range(5):
        for tok in range(7):
            t, v = token_to_light_event(ch, tok)
            assert light_event_to_token(t, v) == (ch, tok)
    for ch in (7, 8):
        for tok in range(7, 17):
            t, v = token_to_light_event(ch, tok)
            assert v == tok - 7 and light_event_to_token(t, v) == (ch, tok)
    for ch in (5, 6):
        t, v = token_to_light_event(ch, 17)
        assert light_event_to_token(t, v) == (ch, 17)


def test_speed_clipped_and_unknown_dropped():
    assert light_event_to_token(12, 25) == (7, 16)
    assert light_event_to_token(13, -1) is None
    assert light_event_to_token(8, 123) == (5, 17)
    assert light_event_to_token(5, 1) is None     # unsupported type
    assert light_event_to_token(0, 13) is None    # unknown value
    with pytest.raises(ValueError):
        token_to_light_event(0, 17)


def test_duration_bucketing_log_scale():
    assert [duration_bucket(d) for d in DURATION_BEATS] == [0, 1, 2, 3, 4, 5]
    assert duration_bucket(0) == 0 and duration_bucket(-3) == 0 and duration_bucket(100) == 5
    assert duration_bucket(1.4) == 3      # sqrt(2)=1.414 is the log midpoint of 1 and 2
    assert duration_bucket(1.5) == 4


def _wall(x, y=0, h=5, w=1, beats=2.0, bpm=120.0):
    return Obstacle(1.0, 2.0, beats * 60 / bpm, beats, x, y, w, h)


def test_every_wall_variant_roundtrips():
    seen = set()
    for side in (0, 1):
        for token in range(WALL_VOCAB):
            ob = token_to_wall(side, token, 3.0, 120.0)
            assert ob.w == 1 and ob.time_s == 3.0 and ob.beat == 6.0
            assert wall_class(ob) == "side"
            assert wall_to_token(ob, 120.0) == (side, token)
            seen.add((side, token))
    assert len(seen) == 48


def test_wall_shapes():
    assert wall_to_token(_wall(0), 120) == (0, 4)
    assert wall_to_token(_wall(-1, y=2, h=3, beats=0.5), 120) == (0, (1 * 2 + 1) * 6 + 2)
    assert wall_to_token(_wall(3), 120) == (1, 4)
    assert wall_to_token(_wall(4, y=2, h=3, beats=4), 120) == (1, (1 * 2 + 1) * 6 + 5)
    assert wall_to_token(_wall(-2), 120) is None        # too far out
    assert wall_to_token(_wall(5), 120) is None
    assert wall_to_token(_wall(-1, w=2), 120) is None   # wider than 1
    assert wall_to_token(_wall(1), 120) is None         # centre
    assert wall_to_token(_wall(0, y=1, h=3), 120) is None  # unsupported height


def test_wall_flash_gets_smallest_duration_token():
    ob = Obstacle(1.0, 2.0, 0.0, 0.0, 0, 0, 1, 5)
    assert wall_class(ob) == "flash" and wall_to_token(ob, 120) == (0, 0)


def test_wall_classes():
    assert wall_class(Obstacle(0, 0, 0.0, 0.0, 1, 0, 1, 5)) == "flash"
    assert wall_class(Obstacle(0, 0, 1.0, 2.0, 1, 0, 0, 5)) == "flash"
    assert wall_class(Obstacle(0, 0, 0.049, 0.1, 1, 0, 1, 5)) == "flash"
    assert wall_class(Obstacle(0, 0, 0.05, 0.1, 0, 0, 1, 5)) == "side"
    assert wall_class(Obstacle(0, 0, 1.0, 2.0, 3, 0, 1, 5)) == "side"
    assert wall_class(Obstacle(0, 0, 1.0, 2.0, -1, 0, 2, 5)) == "side"
    assert wall_class(Obstacle(0, 0, 1.0, 2.0, 0, 0, 2, 5)) == "centre"
    assert wall_class(Obstacle(0, 0, 1.0, 2.0, 2, 0, 2, 5)) == "centre"


def test_masks():
    cells = [(0, 0), (1, 1), (3, 2)]
    mask = cells_to_mask(cells)
    assert mask == 1 | 1 << 5 | 1 << 11
    assert mask_to_cells(mask) == [(0, 0), (1, 1), (3, 2)]
    for m in range(1 << 12):
        assert cells_to_mask(mask_to_cells(m)) == m
    with pytest.raises(ValueError):
        cells_to_mask([(4, 0)])
