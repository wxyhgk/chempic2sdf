#!/usr/bin/env python3
"""Adversarial stability and bounded-output checks for the public P3 extractor."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAX_BYTES = 65_536


class StabilityAuditError(ValueError):
    """The extractor violates a stability or schema invariant."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def run(extractor: Path, payload: bytes) -> tuple[bytes, dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(extractor)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise StabilityAuditError(result.stderr.decode("utf-8", errors="replace"))
    value = json.loads(result.stdout)
    if result.stdout != canonical_bytes(value) or len(result.stdout) > MAX_BYTES:
        raise StabilityAuditError("extractor output is non-canonical or over capacity")
    return result.stdout, value


def validate_reject(value: dict[str, Any]) -> None:
    if set(value) != {"reject_reason", "schema_version", "status"}:
        raise StabilityAuditError("reject record is not closed")
    if value["schema_version"] != "chemgraph-evidence-core-v0.4" or value["status"] != "reject":
        raise StabilityAuditError("reject record has the wrong discriminator")
    if not isinstance(value["reject_reason"], str) or not value["reject_reason"]:
        raise StabilityAuditError("reject_reason must be a non-empty string")


def validate_reject_against_schema(schema: dict[str, Any], value: dict[str, Any]) -> None:
    matches = 0
    for branch in schema.get("oneOf", []):
        if branch.get("type") != "object" or branch.get("additionalProperties") is not False:
            continue
        properties = branch.get("properties", {})
        required = set(branch.get("required", []))
        if set(value) != required or set(value) - set(properties):
            continue
        valid = True
        for key, item in value.items():
            rule = properties[key]
            if "const" in rule and item != rule["const"]:
                valid = False
            if rule.get("type") == "string" and (
                not isinstance(item, str)
                or len(item) < rule.get("minLength", 0)
                or len(item) > rule.get("maxLength", 2**31)
            ):
                valid = False
        matches += int(valid)
    if matches != 1:
        raise StabilityAuditError("reject record does not match exactly one closed schema branch")


def angle_probe(extractor: Path) -> dict[str, Any]:
    sys.path.insert(0, str(extractor.parent))
    from extract_evidence_core_v0_4 import RawLine, cluster_hough_lines  # type: ignore

    cases = {
        "negative_diagonal": [
            RawLine(-49.95, 50.0, 49.95, -50.0),
            RawLine(-50.05, 50.0, 50.05, -50.0),
        ],
        "vertical": [
            RawLine(0.0, -50.0, -0.001, 50.0),
            RawLine(0.0, -50.0, 0.0, 50.0),
            RawLine(0.0, -50.0, 0.001, 50.0),
        ],
    }
    result = {}
    for name, lines in cases.items():
        clusters = cluster_hough_lines(lines)
        if len(clusters) != 1:
            raise StabilityAuditError(f"{name} crossed an unoriented angle seam")
        result[name] = {
            "cluster_count": 1,
            "cluster_angle_degrees": math.degrees(clusters[0].angle),
        }
    return result


def audit(pack: Path, extractor: Path) -> dict[str, Any]:
    rows = []
    for image_path in sorted((pack / "sealed_image_pack" / "images").glob("*.png")):
        raw = image_path.read_bytes()
        first_bytes, first = run(extractor, raw)
        replay_bytes, _replay = run(extractor, raw)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image[1, 1] < 250:
            raise StabilityAuditError(f"{image_path.name}: fixed perturbation pixel is not blank")
        image[1, 1] = 0
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise StabilityAuditError("OpenCV failed to encode the perturbation")
        _mutated_bytes, mutated = run(extractor, encoded.tobytes())
        geometry_stable = (
            first["stroke_primitives"] == mutated["stroke_primitives"]
            and first["event_regions"] == mutated["event_regions"]
        )
        original_residual = first["residual"]
        mutated_residual = mutated["residual"]
        residual_exact = (
            mutated_residual["foreground_pixel_count"]
            == original_residual["foreground_pixel_count"] + 1
            and mutated_residual["component_count"] == original_residual["component_count"] + 1
            and mutated_residual["bbox_q4"][:2] == [16, 16]
            and mutated_residual["bbox_q4"][2:] == original_residual["bbox_q4"][2:]
        )
        rows.append(
            {
                "delivery_id": image_path.stem,
                "event_count": len(first["event_regions"]),
                "geometry_stable_under_corner_pixel": geometry_stable,
                "primitive_count": len(first["stroke_primitives"]),
                "replay_byte_exact": first_bytes == replay_bytes,
                "residual_delta_exact": residual_exact,
                "stdout_bytes": len(first_bytes),
            }
        )

    _reject_bytes, reject = run(extractor, b"not-a-png")
    validate_reject(reject)
    schema = json.loads((extractor.parent / "evidence_core.schema.json").read_text(encoding="utf-8"))
    validate_reject_against_schema(schema, reject)
    passed = all(
        row["geometry_stable_under_corner_pixel"]
        and row["replay_byte_exact"]
        and row["residual_delta_exact"]
        for row in rows
    )
    return {
        "angle_probe": angle_probe(extractor),
        "capacity": {
            "max_event_count": max(row["event_count"] for row in rows),
            "max_primitive_count": max(row["primitive_count"] for row in rows),
            "max_stdout_bytes": max(row["stdout_bytes"] for row in rows),
        },
        "corner_pixel_geometry_stable_count": sum(
            bool(row["geometry_stable_under_corner_pixel"]) for row in rows
        ),
        "observation_count": len(rows),
        "reject_record_closed": True,
        "replay_byte_exact_count": sum(bool(row["replay_byte_exact"]) for row in rows),
        "residual_delta_exact_count": sum(bool(row["residual_delta_exact"]) for row in rows),
        "rows": rows,
        "schema_version": "evidence-core-public-stability-audit-v0.4",
        "status": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.pack.resolve(), args.extractor.resolve())
    encoded = canonical_bytes(result)
    if args.output:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
