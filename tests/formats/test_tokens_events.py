import pytest

from bsmaker.formats.events import FRAMES_PER_SECOND, Event, to_events
from bsmaker.formats.model import Note
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE, decode_token, encode_token


def note(t, color=0, x=1, y=0, d=1):
    return Note(t, t * 2, x, y, color, d)


def test_token_roundtrip():
    seen = set()
    for x in range(4):
        for y in range(3):
            for d in range(9):
                tok = encode_token(x, y, d)
                assert 1 <= tok < VOCAB_SIZE
                assert decode_token(tok) == (x, y, d)
                seen.add(tok)
    assert len(seen) == 108 and VOCAB_SIZE == 109


@pytest.mark.parametrize("bad", [0, 109, -1])
def test_decode_none_raises(bad):
    with pytest.raises(ValueError):
        decode_token(bad)


def test_frame_rate_constant():
    assert FRAMES_PER_SECOND == 22050 / 256


def test_to_events_groups_doubles():
    events, dropped = to_events([note(1.0, 0, 1, 0, 1), note(1.0, 1, 2, 0, 1), note(2.0, 1, 3, 1, 3)])
    assert dropped == 0 and len(events) == 2
    assert events[0] == Event(round(1.0 * FRAMES_PER_SECOND) / FRAMES_PER_SECOND,
                              encode_token(1, 0, 1), encode_token(2, 0, 1))
    assert events[1].left == NONE_TOKEN and events[1].right == encode_token(3, 1, 3)


def test_to_events_merges_stack():
    events, dropped = to_events([note(1.0, 0, 1, 0, 1), note(1.0, 0, 2, 1, 1)])
    assert len(events) == 1 and dropped == 1
    assert events[0].left == encode_token(1, 0, 1)


def test_to_events_merges_close_slider():
    events, dropped = to_events([note(1.0, 0), note(1.02, 0, 2, 0, 1)])
    assert len(events) == 1 and dropped == 1


def test_to_events_keeps_distant_same_color():
    events, dropped = to_events([note(1.0, 0), note(1.1, 0, 2, 0, 1)])
    assert len(events) == 2 and dropped == 0
