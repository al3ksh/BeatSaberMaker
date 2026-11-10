"""Note token vocabulary: 0 = no note, 1..108 = (x, y, direction)."""
from __future__ import annotations

NONE_TOKEN = 0
VOCAB_SIZE = 109


def encode_token(x: int, y: int, direction: int) -> int:
    if not (0 <= x <= 3 and 0 <= y <= 2 and 0 <= direction <= 8):
        raise ValueError(f"note out of range: x={x} y={y} direction={direction}")
    return 1 + x * 27 + y * 9 + direction


def decode_token(token: int) -> tuple[int, int, int]:
    if not 1 <= token < VOCAB_SIZE:
        raise ValueError(f"invalid note token: {token}")
    t = token - 1
    return t // 27, (t % 27) // 9, t % 9
