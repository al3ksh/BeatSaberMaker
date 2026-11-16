"""Shared exceptions for BeatSaberMaker."""
from __future__ import annotations


class GenerateCancelled(Exception):
    """Generation or download was cancelled by the caller."""
