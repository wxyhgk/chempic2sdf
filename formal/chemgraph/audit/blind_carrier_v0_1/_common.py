"""Strict JSON, frozen-schema, and PNG helpers for blind carrier validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import struct
from typing import Any
import zlib

from jsonschema import Draft202012Validator
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_ROOT = ROOT / "Data" / "benchmarks" / "blind_carrier_v0_1" / "schemas"
INPUT_SCHEMA_SHA256 = "2a6e3ac1d3553b86989bb63ac705e1ba41c754accb16ab7d834f26b0d8fbf751"
PREDICTION_SCHEMA_SHA256 = "7bab948af313a43cf38b6c23e863501869798c51f33c40ebfe3dc9606963c5e2"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


class DocumentError(ValueError):
    def __init__(self, issue: ValidationIssue):
        super().__init__(issue.message)
        self.issue = issue


class _DuplicateKey(ValueError):
    pass


def _canonical_png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def canonical_grayscale_png(decoded: bytes, width: int = 768, height: int = 768) -> bytes:
    if len(decoded) != width * height:
        raise ValueError("decoded grayscale byte length does not match dimensions")
    scanlines = b"".join(
        b"\x00" + decoded[row * width:(row + 1) * width]
        for row in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (PNG_SIGNATURE + _canonical_png_chunk(b"IHDR", ihdr)
            + _canonical_png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + _canonical_png_chunk(b"IEND", b""))


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _check_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DocumentError(ValidationIssue("nonfinite_number", path, "non-finite number"))
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, f"{path}/{index}")


def parse_canonical_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DocumentError(ValidationIssue("invalid_utf8", "$", str(error))) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except _DuplicateKey as error:
        raise DocumentError(
            ValidationIssue("duplicate_key", "$", f"duplicate object key: {error}"),
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise DocumentError(ValidationIssue("invalid_json", "$", str(error))) from error
    _check_finite(value)
    if raw != canonical_json_bytes(value):
        raise DocumentError(
            ValidationIssue("noncanonical_json", "$", "JSON bytes are not canonical"),
        )
    return value


def load_frozen_schema(filename: str, expected_sha256: str) -> dict[str, Any]:
    import hashlib

    path = SCHEMA_ROOT / filename
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise DocumentError(ValidationIssue(
            "schema_hash_mismatch", str(path),
            f"expected {expected_sha256}, got {actual}",
        ))
    try:
        return json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise DocumentError(ValidationIssue("invalid_schema", str(path), str(error))) from error


def schema_issues(value: Any, schema: dict[str, Any]) -> list[ValidationIssue]:
    validator = Draft202012Validator(schema)
    issues = []
    for error in sorted(
        validator.iter_errors(value),
        key=lambda item: tuple(str(token) for token in item.absolute_path),
    ):
        pointer = "$" + "".join(f"/{token}" for token in error.absolute_path)
        issues.append(ValidationIssue("jsonschema", pointer, error.message))
    return issues


def _png_chunks(raw: bytes) -> tuple[int, int, int]:
    if not raw.startswith(PNG_SIGNATURE):
        raise DocumentError(ValidationIssue("invalid_png_signature", "$/image", "invalid PNG signature"))
    offset = len(PNG_SIGNATURE)
    index = 0
    width = height = -1
    color_type = -1
    saw_iend = False
    saw_idat = False
    idat_closed = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise DocumentError(ValidationIssue("truncated_png_chunk", "$/image", "truncated PNG chunk"))
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunk_type = raw[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            raise DocumentError(ValidationIssue("truncated_png_chunk", "$/image", "truncated PNG payload"))
        payload = raw[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", raw[offset + 8 + length:end])[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise DocumentError(ValidationIssue("png_crc_mismatch", "$/image", "PNG chunk CRC mismatch"))
        if index == 0:
            if chunk_type != b"IHDR" or length != 13:
                raise DocumentError(ValidationIssue("invalid_png_ihdr", "$/image", "IHDR must be first and 13 bytes"))
            width, height = struct.unpack(">II", payload[:8])
            bit_depth, color_type, compression, filter_method, interlace = payload[8:13]
            if (bit_depth, color_type, compression, filter_method, interlace) != (8, 0, 0, 0, 0):
                raise DocumentError(ValidationIssue(
                    "unsupported_png_profile", "$/image",
                    "only noninterlaced 8-bit grayscale PNG is allowed",
                ))
        elif chunk_type == b"IHDR":
            raise DocumentError(ValidationIssue("duplicate_png_ihdr", "$/image", "duplicate IHDR"))
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise DocumentError(ValidationIssue("noncontiguous_png_idat", "$/image", "IDAT chunks must be contiguous"))
            saw_idat = True
        elif chunk_type != b"IEND":
            raise DocumentError(ValidationIssue(
                "forbidden_png_chunk", "$/image",
                f"only IHDR, IDAT, and IEND are allowed; got {chunk_type.decode('latin-1')}",
            ))
        if chunk_type == b"IEND":
            if length != 0:
                raise DocumentError(ValidationIssue("invalid_png_iend", "$/image", "IEND must be empty"))
            saw_iend = True
            offset = end
            break
        if saw_idat and chunk_type != b"IDAT":
            idat_closed = True
        offset = end
        index += 1
    if not saw_iend:
        raise DocumentError(ValidationIssue("missing_png_iend", "$/image", "missing IEND"))
    if not saw_idat:
        raise DocumentError(ValidationIssue("missing_png_idat", "$/image", "missing IDAT"))
    if offset != len(raw):
        raise DocumentError(ValidationIssue("png_trailing_bytes", "$/image", "bytes follow IEND"))
    return width, height, color_type


def decode_png_foreground(raw: bytes) -> set[tuple[int, int]]:
    width, height, color_type = _png_chunks(raw)
    if (width, height) != (768, 768):
        raise DocumentError(ValidationIssue(
            "png_dimensions", "$/image", f"expected 768x768, got {width}x{height}",
        ))
    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.size != (768, 768):
                raise ValueError(f"decoded size mismatch: {image.size}")
            if image.mode != "L" or color_type != 0:
                raise ValueError(f"expected decoded grayscale mode L, got {image.mode}")
            decoded = image.tobytes()
            if raw != canonical_grayscale_png(decoded):
                raise DocumentError(ValidationIssue(
                    "noncanonical_png_bytes", "$/image",
                    "PNG bytes differ from the frozen grayscale re-encoding",
                ))
            foreground = {
                (index % 768, index // 768)
                for index, intensity in enumerate(decoded)
                if intensity != 255
            }
            return foreground
    except DocumentError:
        raise
    except (OSError, ValueError) as error:
        raise DocumentError(ValidationIssue("png_decode_failed", "$/image", str(error))) from error


def success_report(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        **fields,
        "document_kind": kind,
        "status": "STRUCTURALLY_VALID_NOT_SCORED",
        "training_authorized": False,
    }


def error_report(issues: list[ValidationIssue]) -> dict[str, Any]:
    return {
        "errors": [issue.record() for issue in issues],
        "status": "INVALID",
        "training_authorized": False,
    }


def emit_report(report: dict[str, Any], *, error: bool) -> int:
    import sys

    stream = sys.stderr if error else sys.stdout
    stream.buffer.write(canonical_json_bytes(report))
    return 2 if error else 0


class StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        import sys

        report = error_report([
            ValidationIssue("invalid_arguments", "$", message),
        ])
        sys.stderr.buffer.write(canonical_json_bytes(report))
        raise SystemExit(2)
