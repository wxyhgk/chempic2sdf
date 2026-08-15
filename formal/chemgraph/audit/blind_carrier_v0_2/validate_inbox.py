#!/usr/bin/env python3
"""Validate one anonymous Blind Carrier v0.2 inbox without reading GT."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from formal.chemgraph.audit.blind_carrier_v0_2._common import (  # type: ignore
        DocumentError, INPUT_SCHEMA_SHA256, StructuredArgumentParser,
        ValidationIssue, decode_png_foreground, emit_report, error_report,
        load_frozen_schema, parse_canonical_json, schema_issues, success_report,
    )
else:
    from ._common import (
        DocumentError, INPUT_SCHEMA_SHA256, StructuredArgumentParser,
        ValidationIssue, decode_png_foreground, emit_report, error_report,
        load_frozen_schema, parse_canonical_json, schema_issues, success_report,
    )


@dataclass(frozen=True, slots=True)
class ValidatedInbox:
    value: dict[str, Any]
    png_bytes: bytes
    foreground: set[tuple[int, int]]


def _validated_inbox(raw: bytes) -> ValidatedInbox:
    value = parse_canonical_json(raw)
    issues = schema_issues(
        value, load_frozen_schema("input.schema.json", INPUT_SCHEMA_SHA256),
    )
    if issues:
        raise DocumentError(issues[0])
    try:
        png_bytes = base64.b64decode(value["image"]["bytes_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise DocumentError(ValidationIssue(
            "invalid_base64", "$/image/bytes_base64", str(error),
        )) from error
    foreground = decode_png_foreground(png_bytes)
    if not foreground:
        raise DocumentError(ValidationIssue(
            "empty_foreground", "$/image/bytes_base64",
            "decoded PNG must contain at least one non-white pixel",
        ))
    return ValidatedInbox(value, png_bytes, foreground)


def validate_inbox_bytes(raw: bytes) -> dict[str, Any]:
    try:
        validated = _validated_inbox(raw)
    except DocumentError as error:
        return error_report([error.issue])
    return success_report(
        "blind_carrier_inbox_v0.2",
        foreground_pixel_count=len(validated.foreground),
        png_sha256=hashlib.sha256(validated.png_bytes).hexdigest(),
        schema_sha256=INPUT_SCHEMA_SHA256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("input", help="canonical inbox JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
    except OSError as error:
        return emit_report(error_report([
            ValidationIssue("input_read_failed", args.input, str(error)),
        ]), error=True)
    report = validate_inbox_bytes(raw)
    return emit_report(report, error=report["status"] == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
