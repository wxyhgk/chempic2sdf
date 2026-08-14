#!/usr/bin/env python3
"""Frozen Gray8 PNG decoder and translation-canonical raster preprocessor."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import struct
import zlib

try:
    from ..challenge_v0_4.rasterizer_v0_4 import encode_canonical_png
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from challenge_v0_4.rasterizer_v0_4 import encode_canonical_png  # type: ignore


CANVAS_SIZE = 768
GRAY_BYTES = CANVAS_SIZE * CANVAS_SIZE
BACKGROUND = 255
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CONTRACT_SHA256 = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"


class CanonicalRasterError(ValueError):
    """Raised when a v0.5 in-domain raster violates the frozen contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_contract(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise CanonicalRasterError("contract_sha256_mismatch")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalRasterError(f"contract_duplicate_key:{key}")
            result[key] = value
        return result

    try:
        contract = json.loads(payload, object_pairs_hook=reject_duplicate)
    except json.JSONDecodeError as error:
        raise CanonicalRasterError(f"contract_json_error:{error}") from error
    if contract.get("schema_version") != "chemgraph-observation-closure-p0-v0.5":
        raise CanonicalRasterError("contract_schema_version_mismatch")
    return contract


def decode_canonical_png(payload: bytes) -> bytes:
    if not payload.startswith(PNG_SIGNATURE):
        raise CanonicalRasterError("png_bad_signature")
    chunks: list[tuple[bytes, bytes]] = []
    cursor = len(PNG_SIGNATURE)
    while cursor < len(payload):
        if cursor + 12 > len(payload):
            raise CanonicalRasterError("png_truncated_chunk")
        length = struct.unpack(">I", payload[cursor:cursor + 4])[0]
        kind = payload[cursor + 4:cursor + 8]
        end = cursor + 12 + length
        if end > len(payload):
            raise CanonicalRasterError("png_truncated_payload")
        chunk_payload = payload[cursor + 8:end - 4]
        expected_crc = struct.unpack(">I", payload[end - 4:end])[0]
        actual_crc = binascii.crc32(chunk_payload, binascii.crc32(kind)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise CanonicalRasterError("png_crc_mismatch")
        chunks.append((kind, chunk_payload))
        cursor = end
        if kind == b"IEND":
            break
    if cursor != len(payload):
        raise CanonicalRasterError("png_trailing_bytes")
    if [kind for kind, _ in chunks] != [b"IHDR", b"IDAT", b"IEND"]:
        raise CanonicalRasterError("png_noncanonical_chunks")
    if len(chunks[0][1]) != 13:
        raise CanonicalRasterError("png_bad_ihdr_length")
    if struct.unpack(">IIBBBBB", chunks[0][1]) != (768, 768, 8, 0, 0, 0, 0):
        raise CanonicalRasterError("png_wrong_profile")
    try:
        scanlines = zlib.decompress(chunks[1][1])
    except zlib.error as error:
        raise CanonicalRasterError(f"png_zlib_error:{error}") from error
    if len(scanlines) != CANVAS_SIZE * (CANVAS_SIZE + 1):
        raise CanonicalRasterError("png_wrong_decoded_length")
    gray = bytearray(GRAY_BYTES)
    for row in range(CANVAS_SIZE):
        source = row * (CANVAS_SIZE + 1)
        if scanlines[source] != 0:
            raise CanonicalRasterError("png_nonzero_filter")
        begin = row * CANVAS_SIZE
        gray[begin:begin + CANVAS_SIZE] = scanlines[source + 1:source + 1 + CANVAS_SIZE]
    decoded = bytes(gray)
    if encode_canonical_png(decoded) != payload:
        raise CanonicalRasterError("png_not_byte_canonical")
    return decoded


def foreground_bbox(gray: bytes) -> tuple[int, int, int, int]:
    if len(gray) != GRAY_BYTES:
        raise CanonicalRasterError("gray_wrong_length")
    min_x = CANVAS_SIZE
    min_y = CANVAS_SIZE
    max_x = -1
    max_y = -1
    for y in range(CANVAS_SIZE):
        row = gray[y * CANVAS_SIZE:(y + 1) * CANVAS_SIZE]
        without_left_background = row.lstrip(b"\xff")
        if not without_left_background:
            continue
        left = CANVAS_SIZE - len(without_left_background)
        right = len(row.rstrip(b"\xff")) - 1
        min_x = min(min_x, left)
        max_x = max(max_x, right)
        min_y = min(min_y, y)
        max_y = y
    if max_x < 0:
        raise CanonicalRasterError("empty_foreground")
    return min_x, min_y, max_x, max_y


def require_white_border(gray: bytes, minimum: int) -> tuple[int, int, int, int]:
    if minimum < 0:
        raise CanonicalRasterError("negative_border")
    min_x, min_y, max_x, max_y = foreground_bbox(gray)
    margins = (min_x, min_y, CANVAS_SIZE - 1 - max_x, CANVAS_SIZE - 1 - max_y)
    if min(margins) < minimum:
        raise CanonicalRasterError(f"foreground_border_too_small:{margins}")
    return min_x, min_y, max_x, max_y


def preprocess_gray(gray: bytes) -> bytes:
    min_x, min_y, max_x, max_y = require_white_border(gray, 1)
    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    paste_left = (CANVAS_SIZE - crop_width) // 2
    paste_top = (CANVAS_SIZE - crop_height) // 2
    canonical = bytearray([BACKGROUND]) * GRAY_BYTES
    for source_y in range(min_y, max_y + 1):
        source = source_y * CANVAS_SIZE + min_x
        target_y = paste_top + source_y - min_y
        target = target_y * CANVAS_SIZE + paste_left
        canonical[target:target + crop_width] = gray[source:source + crop_width]
    return bytes(canonical)


def embed_canonical(canonical: bytes) -> bytes:
    if len(canonical) != GRAY_BYTES:
        raise CanonicalRasterError("canonical_wrong_length")
    return bytes(canonical)


def integer_shift(gray: bytes, dx: int, dy: int) -> bytes:
    if len(gray) != GRAY_BYTES:
        raise CanonicalRasterError("gray_wrong_length")
    if not (-32 <= dx <= 32 and -32 <= dy <= 32):
        raise CanonicalRasterError("translation_out_of_domain")
    require_white_border(gray, 33)
    shifted = bytearray([BACKGROUND]) * GRAY_BYTES
    source_x0 = max(0, -dx)
    source_x1 = min(CANVAS_SIZE, CANVAS_SIZE - dx)
    width = source_x1 - source_x0
    for source_y in range(max(0, -dy), min(CANVAS_SIZE, CANVAS_SIZE - dy)):
        target_y = source_y + dy
        source = source_y * CANVAS_SIZE + source_x0
        target = target_y * CANVAS_SIZE + source_x0 + dx
        shifted[target:target + width] = gray[source:source + width]
    result = bytes(shifted)
    require_white_border(result, 1)
    return result


def preprocess_png(payload: bytes) -> bytes:
    return preprocess_gray(decode_canonical_png(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--input-png", type=Path, required=True)
    parser.add_argument("--output-gray", type=Path, required=True)
    args = parser.parse_args()
    verify_contract(args.contract, args.expected_contract_sha256)
    if args.expected_contract_sha256 != CONTRACT_SHA256:
        raise CanonicalRasterError("binary_frozen_contract_mismatch")
    canonical = preprocess_png(args.input_png.read_bytes())
    args.output_gray.write_bytes(canonical)


if __name__ == "__main__":
    main()
