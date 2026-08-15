"""Frozen schema bindings and canonical framing for candidate-set v0.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from formal.chemgraph.audit.blind_carrier_v0_2._common import (
    DocumentError,
    StructuredArgumentParser,
    ValidationIssue,
    canonical_json_bytes,
    decode_png_foreground,
    emit_report,
    error_report,
    parse_canonical_json,
    schema_issues,
    success_report,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_ROOT = ROOT / "Data" / "benchmarks" / "carrier_candidate_set_v0_3" / "schemas"
INPUT_SCHEMA_SHA256 = "d2552f04fe91438cb0d2ff6d88d88ed9fa7f59d5752d2766e9f901f1d0277bf7"
CANDIDATE_SCHEMA_SHA256 = "bb69ee2b054ecc93e1c563dfb6638a533f4559f787af96645d494d615da7752b"
PROTOCOL_SHA256 = "f4aeb651e531087eea18bc3b96eebd38bdc2956f5dd635bd0baa6b677824e37c"


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


def framed_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = canonical_json_bytes(value)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "CANDIDATE_SCHEMA_SHA256",
    "DocumentError",
    "INPUT_SCHEMA_SHA256",
    "PROTOCOL_SHA256",
    "StructuredArgumentParser",
    "ValidationIssue",
    "canonical_json_bytes",
    "decode_png_foreground",
    "emit_report",
    "error_report",
    "framed_sha256",
    "load_frozen_schema",
    "parse_canonical_json",
    "schema_issues",
    "success_report",
]
