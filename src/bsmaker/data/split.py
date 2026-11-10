"""Deterministic train/val/test split by map key."""
from __future__ import annotations

import hashlib


def assign_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    if h < 95:
        return "val"
    return "test"
