"""Frozen-schema bindings plus the unchanged v0.1 JSON and PNG primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.blind_carrier_v0_1._common import (
    DocumentError,
    StructuredArgumentParser,
    ValidationIssue,
    canonical_grayscale_png,
    canonical_json_bytes,
    decode_png_foreground,
    emit_report,
    error_report,
    parse_canonical_json,
    schema_issues,
    success_report,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_ROOT = ROOT / "Data" / "benchmarks" / "blind_carrier_v0_2" / "schemas"
INPUT_SCHEMA_SHA256 = "0e1478985f29c1c78f0d88c1c4149972a8a01c60986569dfdd27f286951a6a96"
PREDICTION_SCHEMA_SHA256 = "8970b3317b0b6c0c3d9479bc260ee36858e22b446a990e00d3e057d1faaf7d40"


def load_frozen_schema(filename: str, expected_sha256: str) -> dict[str, Any]:
    path = SCHEMA_ROOT / filename
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise DocumentError(ValidationIssue(
            "schema_hash_mismatch", str(path),
            f"expected {expected_sha256}, got {actual}",
        ))
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise DocumentError(ValidationIssue(
            "invalid_schema", str(path), str(error),
        )) from error
    if not isinstance(value, dict):
        raise DocumentError(ValidationIssue(
            "invalid_schema", str(path), "schema root must be an object",
        ))
    return value


__all__ = [
    "DocumentError", "INPUT_SCHEMA_SHA256", "PREDICTION_SCHEMA_SHA256",
    "StructuredArgumentParser", "ValidationIssue", "canonical_grayscale_png",
    "canonical_json_bytes", "decode_png_foreground", "emit_report", "error_report",
    "load_frozen_schema", "parse_canonical_json", "schema_issues", "success_report",
]
