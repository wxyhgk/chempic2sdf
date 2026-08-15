"""Canonical encoding and capacity helpers shared by R0 research tools."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any, Iterable


LIMITS = {
    "max_adjacency_edges": 32_768,
    "max_ambiguous_variables": 20,
    "max_atomic_fragments": 64,
    "max_carriers": 8,
    "max_constraints": 32_768,
    "max_domain_classes": 64,
    "max_intersections": 8,
    "max_output_bytes": 16_777_216,
    "max_owner_options": 16,
    "max_pixel_variables": 4_096,
    "max_raw_assignments": 1_048_576,
    "max_total_rle_runs": 65_536,
}


class CandidateDomainError(ValueError):
    """Closed R0 failure raised before any partial artifact is returned."""


def fail(code: str) -> None:
    raise CandidateDomainError(code)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_runs(pixels: Iterable[tuple[int, int]]) -> list[list[int]]:
    ordered = sorted(set(pixels))
    if not ordered:
        return []
    row, start = ordered[0]
    end = start + 1
    runs: list[list[int]] = []
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
            continue
        runs.append([row, start, end - start])
        row, start, end = next_row, column, column + 1
    runs.append([row, start, end - start])
    return runs


def decode_runs(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        fail("RLE_NOT_ARRAY")
    pixels: set[tuple[int, int]] = set()
    previous: tuple[int, int, int] | None = None
    for run in value:
        if not (
            isinstance(run, list)
            and len(run) == 3
            and all(type(item) is int for item in run)
        ):
            fail("RLE_RUN_INVALID")
        row, start, length = run
        if not (0 <= row < 768 and 0 <= start < 768 and 1 <= length <= 768 - start):
            fail("RLE_RUN_OUT_OF_RANGE")
        if previous is not None:
            old_row, old_start, old_length = previous
            if (row, start) <= (old_row, old_start):
                fail("RLE_ORDER_NONCANONICAL")
            if row == old_row and start <= old_start + old_length:
                fail("RLE_NOT_MAXIMAL")
        for column in range(start, start + length):
            pixel = row, column
            if pixel in pixels:
                fail("RLE_PIXEL_DUPLICATE")
            pixels.add(pixel)
        previous = row, start, length
    return pixels


def framed_items_sha256(items: list[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(len(items).to_bytes(4, "big"))
    for item in items:
        encoded = canonical_bytes(item)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def ratio(numerator: int, denominator: int) -> list[int]:
    if denominator <= 0 or numerator < 0:
        fail("RATIO_INVALID")
    value = Fraction(numerator, denominator)
    return [value.numerator, value.denominator]


def capacity_gate(name: str, value: int) -> None:
    if name not in LIMITS:
        fail("UNKNOWN_CAPACITY")
    if type(value) is not int or value < 0:
        fail("CAPACITY_VALUE_INVALID")
    if value > LIMITS[name]:
        fail(f"CAPACITY_EXCEEDED:{name}")
