"""Frozen protocol for the G3 P4 source-union generalization cycle."""

from __future__ import annotations

from pathlib import Path


CYCLE_NAME = "chemvector-g3-p4-source-union-cycle"
OUTPUT = Path(f"outputs/audits/{CYCLE_NAME}")
SEED = 20260840
VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
PILOT_TRAIN_MOLECULES = 5_000
FULL_TRAIN_MOLECULES = 20_000
CONFIRM_V2_BUCKET_MOLECULES = {
    "1-20": 50,
    "21-40": 50,
    "41-60": 50,
    "61-80": 20,
    "81-100": 5,
}
COUNT_BUCKETS = (
    ("1-20", 1, 20),
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
)


def count_bucket(value: int) -> str:
    for name, lower, upper in COUNT_BUCKETS:
        if lower <= value <= upper:
            return name
    raise ValueError(f"P4 Count is outside 1..100: {value}")


__all__ = [
    "CONFIRM_V2_BUCKET_MOLECULES",
    "COUNT_BUCKETS",
    "CYCLE_NAME",
    "FULL_TRAIN_MOLECULES",
    "OUTPUT",
    "PILOT_TRAIN_MOLECULES",
    "SEED",
    "VARIANTS",
    "count_bucket",
]
