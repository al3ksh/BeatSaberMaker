import random

import numpy as np

from bsmaker.formats.events import Event
from bsmaker.formats.tokens import NONE_TOKEN, VOCAB_SIZE
from bsmaker.playability.checker import advance, allowed, check
from bsmaker.playability.config import PlayabilityConfig
from bsmaker.playability.state import initial_state

CFG = PlayabilityConfig()


def test_allowed_none_always_true():
    state = initial_state()
    for hand in (0, 1):
        m = allowed(state, 0.0, hand, CFG)
        assert m.shape == (VOCAB_SIZE,) and m.dtype == bool and m[NONE_TOKEN]


def test_mask_agrees_with_check():
    for seed in range(200):
        rng = random.Random(seed)
        state, t, events = initial_state(), 0.0, []
        for _ in range(64):
            t += rng.uniform(0.06, 1.2)
            ml = allowed(state, t, 0, CFG)
            left = rng.choice(np.flatnonzero(ml).tolist())
            mr = allowed(state, t, 1, CFG, other_token=left)
            right = rng.choice(np.flatnonzero(mr).tolist())
            if left == NONE_TOKEN and right == NONE_TOKEN:
                continue
            events.append(Event(t, left, right))
            state = advance(state, t, left, right)
        assert check(events, CFG).hard == (), seed


def test_doubles_symmetric_in_hand_order():
    state = initial_state()
    as_right = {t: allowed(state, 1.0, 1, CFG, other_token=t) for t in range(1, VOCAB_SIZE)}
    as_left = {t: allowed(state, 1.0, 0, CFG, other_token=t) for t in range(1, VOCAB_SIZE)}
    for left in range(1, VOCAB_SIZE):
        for right in range(1, VOCAB_SIZE):
            assert as_right[left][right] == as_left[right][left]
    # both verdicts occur, so the check is not vacuous
    assert any(not v for m in as_right.values() for v in m[1:])
    assert any(v for m in as_right.values() for v in m[1:])


def test_blocked_token_is_reported():
    rng = random.Random(1)
    state, t = initial_state(), 0.0
    for _ in range(20):
        t += 0.2
        left = rng.choice(np.flatnonzero(allowed(state, t, 0, CFG)).tolist())
        state = advance(state, t, left, NONE_TOKEN)
    t += 0.1
    blocked = np.flatnonzero(~allowed(state, t, 0, CFG))
    assert len(blocked) > 0
    rng = random.Random(1)
    events, t2 = [], 0.0
    state = initial_state()
    for _ in range(20):
        t2 += 0.2
        left = rng.choice(np.flatnonzero(allowed(state, t2, 0, CFG)).tolist())
        events.append(Event(t2, left, NONE_TOKEN))
        state = advance(state, t2, left, NONE_TOKEN)
    events.append(Event(t, int(blocked[0]), NONE_TOKEN))
    assert any(v.time_s == t for v in check(events, CFG).hard)
