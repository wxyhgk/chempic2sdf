"""Independent confirmation oracle for the frozen representative fixture."""

from __future__ import annotations

import binascii
import hashlib
import itertools
import json
import struct
import zlib
from pathlib import Path
from typing import Any

from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (
    CANVAS_SIZE,
    Q,
    LineStyle,
    Segment,
    rasterize_gray,
    rasterize_support,
)


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "representative_edge_w5_round_box4.png"
WITNESS_PATH = ROOT / "representative_witness.json"
EXPECTED_PNG_SHA256 = "0152efc7791f03515ef554739b545088b36d654aa081da9718d0c31eef3ecb4e"
EXPECTED_WITNESS_SHA256 = "e00421918b62adc1b76c42ca1a9cb1b5c9981d3ee33d7e6f1f96dcebf63c268d"
EXPECTED_RENDERER_SHA256 = "5d673ae1c33648b5498b4ebacfc1b91136fee5abdbae4fa1503a144a59816943"
RENDERER_PATH = Path(__file__).resolve().parents[2] / "challenge_v0_4" / "rasterizer_v0_4.py"


class ConfirmationError(ValueError):
    """Closed independent confirmation failure."""


def _fail(code: str) -> None:
    raise ConfirmationError(code)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode_canonical_gray_png(png: bytes) -> bytes:
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        _fail("PNG_SIGNATURE_INVALID")
    cursor = 8
    ihdr: bytes | None = None
    idat = bytearray()
    seen_iend = False
    chunk_names: list[bytes] = []
    while cursor < len(png):
        if cursor + 12 > len(png):
            _fail("PNG_CHUNK_TRUNCATED")
        size = struct.unpack(">I", png[cursor:cursor + 4])[0]
        kind = png[cursor + 4:cursor + 8]
        payload_end = cursor + 8 + size
        chunk_end = payload_end + 4
        if chunk_end > len(png):
            _fail("PNG_PAYLOAD_TRUNCATED")
        payload = png[cursor + 8:payload_end]
        expected_crc = struct.unpack(">I", png[payload_end:chunk_end])[0]
        actual_crc = binascii.crc32(payload, binascii.crc32(kind)) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            _fail("PNG_CRC_MISMATCH")
        chunk_names.append(kind)
        if kind == b"IHDR":
            if ihdr is not None:
                _fail("PNG_IHDR_DUPLICATE")
            ihdr = payload
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            if payload:
                _fail("PNG_IEND_NONEMPTY")
            seen_iend = True
            cursor = chunk_end
            break
        else:
            _fail("PNG_UNEXPECTED_CHUNK")
        cursor = chunk_end
    if cursor != len(png) or not seen_iend or chunk_names != [b"IHDR", b"IDAT", b"IEND"]:
        _fail("PNG_CHUNK_SEQUENCE_INVALID")
    if ihdr is None or len(ihdr) != 13:
        _fail("PNG_IHDR_INVALID")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if (width, height, depth, color, compression, filtering, interlace) != (
        768, 768, 8, 0, 0, 0, 0
    ):
        _fail("PNG_FORMAT_MISMATCH")
    try:
        scanlines = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise ConfirmationError("PNG_ZLIB_INVALID") from error
    if len(scanlines) != (width + 1) * height:
        _fail("PNG_SCANLINE_LENGTH_MISMATCH")
    gray = bytearray(width * height)
    for row in range(height):
        begin = row * (width + 1)
        if scanlines[begin] != 0:
            _fail("PNG_FILTER_NOT_ZERO")
        gray[row * width:(row + 1) * width] = scanlines[begin + 1:begin + 1 + width]
    return bytes(gray)


def _decode_runs(value: Any) -> set[int]:
    if not isinstance(value, list):
        _fail("RLE_NOT_ARRAY")
    pixels: set[int] = set()
    previous: tuple[int, int, int] | None = None
    for run in value:
        if not (
            isinstance(run, list)
            and len(run) == 3
            and all(type(item) is int for item in run)
        ):
            _fail("RLE_RUN_INVALID")
        row, start, length = run
        if not (0 <= row < 768 and 0 <= start < 768 and 1 <= length <= 768 - start):
            _fail("RLE_RUN_OUT_OF_RANGE")
        if previous is not None:
            old_row, old_start, old_length = previous
            if (row, start) <= (old_row, old_start):
                _fail("RLE_ORDER_NONCANONICAL")
            if row == old_row and start <= old_start + old_length:
                _fail("RLE_NOT_MAXIMAL")
        for column in range(start, start + length):
            pixel = row * 768 + column
            if pixel in pixels:
                _fail("RLE_PIXEL_DUPLICATE")
            pixels.add(pixel)
        previous = row, start, length
    return pixels


def _connected(pixels: set[int]) -> bool:
    if not pixels:
        return False
    unseen = set(pixels)
    queue = [unseen.pop()]
    while queue:
        pixel = queue.pop()
        row, column = divmod(pixel, 768)
        for next_row in range(max(0, row - 1), min(767, row + 1) + 1):
            for next_column in range(max(0, column - 1), min(767, column + 1) + 1):
                neighbor = next_row * 768 + next_column
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return not unseen


def _admissible(
    universe: set[int], left: set[int], right: set[int],
    fibers: dict[int, frozenset[int]],
) -> bool:
    left_seed, right_seed = 20 * 768 + 18, 20 * 768 + 22
    return (
        left_seed in left
        and right_seed in right
        and not left & right
        and left | right == universe
        and _connected(left)
        and _connected(right)
        and all({0, 1} <= fibers[pixel] for pixel in left)
        and all({0, 2} <= fibers[pixel] for pixel in right)
    )


def _fixture() -> tuple[list[Segment], LineStyle]:
    return [
        Segment("H", 10 * Q, 20 * Q, 30 * Q, 20 * Q),
        Segment("V1", 18 * Q, 10 * Q, 18 * Q, 20 * Q),
        Segment("V2", 22 * Q, 20 * Q, 22 * Q, 30 * Q),
    ], LineStyle(width_q16=5 * Q, cap="round", antialias_profile="box_4x4")


def verify_payload(
    png: bytes, witness: Any, *, enforce_frozen_identity: bool,
) -> dict[str, Any]:
    if enforce_frozen_identity:
        if _sha256(png) != EXPECTED_PNG_SHA256:
            _fail("PNG_IDENTITY_MISMATCH")
        if _sha256(canonical_bytes(witness)) != EXPECTED_WITNESS_SHA256:
            _fail("WITNESS_IDENTITY_MISMATCH")
    if not isinstance(witness, dict) or set(witness) != {
        "configuration", "schema_version", "training_authorized"
    }:
        _fail("WITNESS_NOT_CLOSED")
    if witness["schema_version"] != "png-ownership-realization-witness-v0.5-research":
        _fail("WITNESS_SCHEMA_MISMATCH")
    if witness["training_authorized"] is not False:
        _fail("TRAINING_AUTHORIZED")

    segments, style = _fixture()
    supports = [rasterize_support([segment], style) for segment in segments]
    foreground = set().union(*supports)
    fibers = {
        pixel: frozenset(index for index, support in enumerate(supports) if pixel in support)
        for pixel in foreground
    }
    universe = {pixel for pixel, fiber in fibers.items() if len(fiber) >= 2}
    left_eligible = {pixel for pixel in universe if {0, 1} <= fibers[pixel]}
    right_eligible = {pixel for pixel in universe if {0, 2} <= fibers[pixel]}
    left_fixed = left_eligible - right_eligible
    right_fixed = right_eligible - left_eligible
    ambiguous = left_eligible & right_eligible
    unowned = universe - left_eligible - right_eligible

    gray = rasterize_gray(segments, style)
    decoded_gray = _decode_canonical_gray_png(png)
    if decoded_gray != gray:
        _fail("PNG_GRAY_RENDER_MISMATCH")
    nonwhite = {index for index, value in enumerate(decoded_gray) if value != 255}
    if foreground != nonwhite:
        _fail("SUPPORT_UNION_PNG_MISMATCH")

    configuration = witness["configuration"]
    expected_configuration_keys = {
        "ambiguity_within_capacity", "antialias_profile", "cap",
        "carrier_supports", "cross_carrier_overlap_pixel_count",
        "enumeration_status", "foreground_pixel_count",
        "left_fixed_pixel_count", "locally_admissible_partition_count",
        "ownership_sets", "ownership_universe_pixel_count",
        "partition_annotations", "phase", "png_sha256", "raw_gray_sha256",
        "right_fixed_pixel_count", "seeds_available", "support_union_exact",
        "triple_overlap_pixel_count", "unowned_overlap_pixel_count", "width_pixels",
    }
    if not isinstance(configuration, dict) or set(configuration) != expected_configuration_keys:
        _fail("CONFIGURATION_NOT_CLOSED")
    expected_scalars = {
        "ambiguity_within_capacity": True,
        "antialias_profile": "box_4x4",
        "cap": "round",
        "cross_carrier_overlap_pixel_count": len(universe),
        "enumeration_status": "ENUMERATED",
        "foreground_pixel_count": len(foreground),
        "left_fixed_pixel_count": len(left_fixed),
        "locally_admissible_partition_count": 873,
        "ownership_universe_pixel_count": len(universe),
        "phase": "edge",
        "png_sha256": _sha256(png),
        "raw_gray_sha256": _sha256(gray),
        "right_fixed_pixel_count": len(right_fixed),
        "seeds_available": True,
        "support_union_exact": True,
        "triple_overlap_pixel_count": len(ambiguous),
        "unowned_overlap_pixel_count": len(unowned),
        "width_pixels": 5,
    }
    if any(
        type(configuration[name]) is not type(value) or configuration[name] != value
        for name, value in expected_scalars.items()
    ):
        _fail("CONFIGURATION_SCALAR_MISMATCH")
    declared_carriers = configuration.get("carrier_supports")
    if not isinstance(declared_carriers, list) or len(declared_carriers) != 3:
        _fail("CARRIER_SUPPORTS_INVALID")
    for index, (declared, actual) in enumerate(zip(declared_carriers, supports, strict=True)):
        if not isinstance(declared, dict) or set(declared) != {
            "carrier_id", "pixel_count", "support_runs"
        }:
            _fail("CARRIER_SUPPORT_NOT_CLOSED")
        if declared["carrier_id"] != ("H", "V1", "V2")[index]:
            _fail("CARRIER_ID_MISMATCH")
        if type(declared["pixel_count"]) is not int or declared["pixel_count"] != len(actual):
            _fail("CARRIER_COUNT_MISMATCH")
        if _decode_runs(declared["support_runs"]) != actual:
            _fail("CARRIER_SUPPORT_MISMATCH")

    declared_sets = configuration.get("ownership_sets")
    if not isinstance(declared_sets, dict) or set(declared_sets) != {
        "ambiguous_runs", "left_fixed_runs", "ownership_universe_runs",
        "right_fixed_runs", "unowned_overlap_runs"
    }:
        _fail("OWNERSHIP_SETS_INVALID")
    expected_sets = {
        "ambiguous_runs": ambiguous,
        "left_fixed_runs": left_fixed,
        "ownership_universe_runs": universe,
        "right_fixed_runs": right_fixed,
        "unowned_overlap_runs": unowned,
    }
    for name, expected in expected_sets.items():
        if _decode_runs(declared_sets[name]) != expected:
            _fail("OWNERSHIP_SET_MISMATCH")

    count = 0
    ambiguous_order = sorted(ambiguous)
    for assignment in itertools.product((False, True), repeat=len(ambiguous_order)):
        left = left_fixed | {
            pixel for pixel, choose_left in zip(ambiguous_order, assignment, strict=True)
            if choose_left
        }
        right = right_fixed | (ambiguous - left)
        if _admissible(universe, left, right, fibers):
            count += 1

    annotations = configuration.get("partition_annotations")
    if not isinstance(annotations, list) or len(annotations) != 2:
        _fail("ANNOTATIONS_INVALID")
    decoded_annotations: list[tuple[set[int], set[int]]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or set(annotation) != {
            "left_region_runs", "right_region_runs"
        }:
            _fail("ANNOTATION_NOT_CLOSED")
        left = _decode_runs(annotation["left_region_runs"])
        right = _decode_runs(annotation["right_region_runs"])
        if not _admissible(universe, left, right, fibers):
            _fail("ANNOTATION_NOT_ADMISSIBLE")
        decoded_annotations.append((left, right))
    if decoded_annotations[0] == decoded_annotations[1]:
        _fail("ANNOTATIONS_NOT_DISTINCT")

    checks = {
        "ambiguous_pixel_count": len(ambiguous),
        "left_fixed_pixel_count": len(left_fixed),
        "locally_admissible_partition_count": count,
        "ownership_universe_pixel_count": len(universe),
        "right_fixed_pixel_count": len(right_fixed),
        "unowned_overlap_pixel_count": len(unowned),
    }
    expected = {
        "ambiguous_pixel_count": 10,
        "left_fixed_pixel_count": 24,
        "locally_admissible_partition_count": 873,
        "ownership_universe_pixel_count": 58,
        "right_fixed_pixel_count": 24,
        "unowned_overlap_pixel_count": 0,
    }
    if checks != expected:
        _fail("CONFIRMATION_VALUE_MISMATCH")
    return {
        "checks": checks,
        "confirmation_protocol_commit": "61b7779",
        "png_sha256": _sha256(png),
        "renderer_sha256": _sha256(RENDERER_PATH.read_bytes()),
        "schema_version": "png-ownership-independent-confirmation-v0.5",
        "status": "passed",
        "support_union_png_exact": True,
        "training_authorized": False,
        "witness_sha256": _sha256(canonical_bytes(witness)),
    }


def verify_frozen_confirmation() -> dict[str, Any]:
    if _sha256(RENDERER_PATH.read_bytes()) != EXPECTED_RENDERER_SHA256:
        _fail("RENDERER_IDENTITY_MISMATCH")
    png = PNG_PATH.read_bytes()
    raw_witness = WITNESS_PATH.read_bytes()
    try:
        witness = json.loads(raw_witness)
    except json.JSONDecodeError as error:
        raise ConfirmationError("WITNESS_JSON_INVALID") from error
    if raw_witness != canonical_bytes(witness):
        _fail("WITNESS_NONCANONICAL")
    return verify_payload(png, witness, enforce_frozen_identity=True)


def main() -> None:
    print(canonical_bytes(verify_frozen_confirmation()).decode("ascii"), end="")


if __name__ == "__main__":
    main()
