#!/usr/bin/env python3
"""Validate one anonymous Carrier Candidate Set v0.3 input."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

from ._common import (
    DocumentError,
    INPUT_SCHEMA_SHA256,
    StructuredArgumentParser,
    ValidationIssue,
    decode_png_foreground,
    emit_report,
    error_report,
    load_frozen_schema,
    parse_canonical_json,
    schema_issues,
    success_report,
)


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    value: dict[str, Any]
    png_bytes: bytes
    foreground: set[tuple[int, int]]


def validated_input(raw: bytes) -> ValidatedInput:
    value = parse_canonical_json(raw)
    issues = schema_issues(
        value, load_frozen_schema("input.schema.json", INPUT_SCHEMA_SHA256),
    )
    if issues:
        raise DocumentError(issues[0])
    try:
        png_bytes = base64.b64decode(value["image"]["bytes_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise DocumentError(ValidationIssue(
            "invalid_base64", "$/image/bytes_base64", str(error),
        )) from error
    foreground = decode_png_foreground(png_bytes)
    if not foreground:
        raise DocumentError(ValidationIssue(
            "empty_foreground", "$/image/bytes_base64",
            "decoded PNG must contain at least one non-white pixel",
        ))
    return ValidatedInput(value, png_bytes, foreground)


def validate_input_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = validated_input(raw)
    except DocumentError as error:
        return error_report([error.issue])
    return success_report(
        "carrier_candidate_input_v0.3",
        foreground_pixel_count=len(value.foreground),
        png_sha256=hashlib.sha256(value.png_bytes).hexdigest(),
        schema_sha256=INPUT_SCHEMA_SHA256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("input", help="canonical input JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
    except OSError as error:
        return emit_report(error_report([
            ValidationIssue("input_read_failed", args.input, str(error)),
        ]), error=True)
    report = validate_input_bytes(raw)
    return emit_report(report, error=report["status"] == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
