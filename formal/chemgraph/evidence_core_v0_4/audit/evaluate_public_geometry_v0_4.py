#!/usr/bin/env python3
"""Independently score bounded EvidenceCore geometry on a revealed public pack."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


MAX_OUTPUT_BYTES = 65_536
MAX_PRIMITIVES = 64
MAX_EVENTS = 64
ENDPOINT_TOLERANCE_Q4 = 12 * 16
SCHEMA_VERSION = "chemgraph-evidence-core-v0.4"


class AuditError(ValueError):
    """The public pack or extractor output violates the audit contract."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def point_distance(left: tuple[int, int], right: tuple[int, int]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def unordered_endpoint_cost(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[float, float]:
    direct = (point_distance(left[0], right[0]), point_distance(left[1], right[1]))
    reverse = (point_distance(left[0], right[1]), point_distance(left[1], right[0]))
    return min((max(direct), sum(direct)), (max(reverse), sum(reverse)))


def maximum_matching(
    truth: list[tuple[tuple[int, int], tuple[int, int]]],
    predicted: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[list[tuple[int, int]], float]:
    """Maximize match count, then minimize endpoint error; truth has at most four lanes."""
    compatible = [
        [
            (predicted_id, unordered_endpoint_cost(gt, pred)[1])
            for predicted_id, pred in enumerate(predicted)
            if unordered_endpoint_cost(gt, pred)[0] <= ENDPOINT_TOLERANCE_Q4
        ]
        for gt in truth
    ]
    best_pairs: list[tuple[int, int]] = []
    best_cost = math.inf

    def visit(gt_id: int, used: set[int], pairs: list[tuple[int, int]], cost: float) -> None:
        nonlocal best_pairs, best_cost
        if gt_id == len(truth):
            if len(pairs) > len(best_pairs) or (len(pairs) == len(best_pairs) and cost < best_cost):
                best_pairs = list(pairs)
                best_cost = cost
            return
        visit(gt_id + 1, used, pairs, cost)
        for predicted_id, pair_cost in compatible[gt_id]:
            if predicted_id not in used:
                used.add(predicted_id)
                pairs.append((gt_id, predicted_id))
                visit(gt_id + 1, used, pairs, cost + pair_cost)
                pairs.pop()
                used.remove(predicted_id)

    visit(0, set(), [], 0.0)
    return best_pairs, 0.0 if not best_pairs else best_cost


def gt_segments(ink_program: dict[str, Any]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    result = []
    for primitive in ink_program["primitives"]:
        points = primitive["points_q16"]
        if len(points) != 2:
            raise AuditError("public geometry audit only accepts two-point primitives")
        # Q16.16 pixels -> Q4 pixels, rounded to the nearest 1/16 pixel.
        converted = [
            (int(round(point[0] / 4096)), int(round(point[1] / 4096))) for point in points
        ]
        result.append((converted[0], converted[1]))
    return result


def validate_evidence(value: dict[str, Any], encoded: bytes) -> None:
    expected_root = {
        "event_regions", "image", "residual", "schema_version", "status", "stroke_primitives"
    }
    if set(value) != expected_root or value["schema_version"] != SCHEMA_VERSION or value["status"] != "ok":
        raise AuditError("extractor did not emit the frozen successful EvidenceCore shape")
    if encoded != canonical_bytes(value):
        raise AuditError("EvidenceCore stdout is not canonical JSON")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise AuditError("EvidenceCore stdout exceeds the byte bound")
    primitives = value["stroke_primitives"]
    events = value["event_regions"]
    if not isinstance(primitives, list) or len(primitives) > MAX_PRIMITIVES:
        raise AuditError("primitive capacity exceeded")
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise AuditError("event capacity exceeded")
    forbidden = {"rle", "bitmap", "png", "sha256", "pixels", "skeleton", "base64", "blob"}
    for key in value:
        if any(token in key.lower() for token in forbidden):
            raise AuditError(f"forbidden root field: {key}")


def run_extractor(extractor: Path, image: Path) -> bytes:
    completed = subprocess.run(
        [sys.executable, str(extractor)],
        input=image.read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AuditError(f"extractor failed for {image.name}: {completed.stderr.decode('utf-8')}")
    return completed.stdout


def evaluate(pack: Path, extractor: Path) -> dict[str, Any]:
    observations = read_jsonl(pack / "sealed_gt" / "observations.jsonl")
    inks = {
        item["ink_program_id"]: item
        for item in read_jsonl(pack / "sealed_gt" / "ink_programs.jsonl")
    }
    rows = []
    replay_exact = True
    for observation in observations:
        image = pack / "sealed_image_pack" / "images" / f'{observation["delivery_id"]}.png'
        first = run_extractor(extractor, image)
        second = run_extractor(extractor, image)
        replay_exact = replay_exact and first == second
        evidence = json.loads(first)
        validate_evidence(evidence, first)
        truth = gt_segments(inks[observation["ink_program_id"]])
        predicted = [
            (tuple(item["begin_q4"]), tuple(item["end_q4"]))
            for item in evidence["stroke_primitives"]
        ]
        pairs, endpoint_cost = maximum_matching(truth, predicted)
        matched_truth = {item[0] for item in pairs}
        matched_predicted = {item[1] for item in pairs}
        rows.append(
            {
                "delivery_id": observation["delivery_id"],
                "family_id": observation["family_id"],
                "false_negative_count": len(truth) - len(matched_truth),
                "false_positive_count": len(predicted) - len(matched_predicted),
                "geometry_exact": len(matched_truth) == len(truth) == len(predicted),
                "gt_primitive_count": len(truth),
                "matched_primitive_count": len(pairs),
                "mean_matched_endpoint_error_q4": (
                    0.0 if not pairs else endpoint_cost / (2 * len(pairs))
                ),
                "observation_id": observation["observation_id"],
                "predicted_primitive_count": len(predicted),
                "stdout_bytes": len(first),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        gt = sum(item["gt_primitive_count"] for item in items)
        predicted = sum(item["predicted_primitive_count"] for item in items)
        matched = sum(item["matched_primitive_count"] for item in items)
        return {
            "false_negative_count": gt - matched,
            "false_positive_count": predicted - matched,
            "geometry_exact_count": sum(bool(item["geometry_exact"]) for item in items),
            "geometry_exact_rate": sum(bool(item["geometry_exact"]) for item in items) / len(items),
            "gt_primitive_count": gt,
            "observation_count": len(items),
            "precision": 1.0 if predicted == 0 and gt == 0 else matched / max(1, predicted),
            "predicted_primitive_count": predicted,
            "recall": 1.0 if gt == 0 else matched / gt,
        }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["family_id"]].append(row)
    overall = summarize(rows)
    return {
        "bucket_metrics": {key: summarize(value) for key, value in sorted(buckets.items())},
        "criteria": {
            "endpoint_tolerance_q4": ENDPOINT_TOLERANCE_Q4,
            "matching": "maximum cardinality, then minimum unordered endpoint L2 error",
        },
        "overall": overall,
        "replay_byte_exact": replay_exact,
        "rows": rows,
        "schema_version": "evidence-core-public-geometry-audit-v0.4",
        "status": "passed" if replay_exact and overall["geometry_exact_rate"] == 1.0 else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.pack.resolve(), args.extractor.resolve())
    encoded = canonical_bytes(result)
    if args.output:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
