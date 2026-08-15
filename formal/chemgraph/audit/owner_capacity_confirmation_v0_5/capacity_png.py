"""Deterministically render the union of parent supports as a grayscale PNG."""

from __future__ import annotations

import binascii
import struct
import zlib
from typing import Any


WIDTH = 768
HEIGHT = 768


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body))


def support_union(parent_supports: list[dict[str, Any]]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for support in parent_supports:
        for row, start, length in support["support_runs"]:
            pixels.update((row, column) for column in range(start, start + length))
    return pixels


def render_support_png(parent_supports: list[dict[str, Any]]) -> bytes:
    foreground = support_union(parent_supports)
    rows = bytearray()
    for row in range(HEIGHT):
        rows.append(0)
        rows.extend(0 if (row, column) in foreground else 255 for column in range(WIDTH))
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join((
        _chunk(b"IHDR", header),
        _chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
        _chunk(b"IEND", b""),
    ))
