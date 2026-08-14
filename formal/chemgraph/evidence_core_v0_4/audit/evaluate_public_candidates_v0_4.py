#!/usr/bin/env python3
"""Verify extractor-to-candidate graph-set equality on a revealed public pack."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


MAX_BYTES = 65_536


class CandidateAuditError(ValueError):
    """A process or graph-set output violates the public audit contract."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_process(command: list[str], payload: bytes, label: str) -> bytes:
    result = subprocess.run(
        command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        raise CandidateAuditError(
            f"{label} exited {result.returncode}: {result.stderr.decode('utf-8', errors='replace')}"
        )
    if len(result.stdout) > MAX_BYTES:
        raise CandidateAuditError(f"{label} output exceeds {MAX_BYTES} bytes")
    return result.stdout


def graph_signature(graph: dict[str, Any], gt: bool = False) -> str:
    nodes = graph["nodes"]
    if gt:
        source_ids = [item["node_id"] for item in nodes]
        roles = {
            item["node_id"]: "implicit" if item["chemical_label"] == "C" else "text_stroke"
            for item in nodes
        }
        edges = [
            (item["begin_node_id"], item["end_node_id"], item["order"])
            for item in graph["edges"]
        ]
    else:
        source_ids = [item["id"] for item in nodes]
        roles = {item["id"]: item["role"] for item in nodes}
        edges = [(item["begin"], item["end"], item["order"]) for item in graph["edges"]]
    if len(source_ids) > 10:
        raise CandidateAuditError("public exact verifier refuses graphs above 10 nodes")
    incident: dict[Any, list[int]] = {node_id: [] for node_id in source_ids}
    for begin, end, order in edges:
        if begin not in incident or end not in incident or begin == end:
            raise CandidateAuditError("graph has an invalid edge endpoint")
        incident[begin].append(order)
        incident[end].append(order)
    groups: dict[tuple[str, tuple[int, ...]], list[Any]] = defaultdict(list)
    for node_id in source_ids:
        groups[(roles[node_id], tuple(sorted(incident[node_id])))].append(node_id)
    ordered_groups = [groups[key] for key in sorted(groups)]
    best: str | None = None
    for permutations in itertools.product(*(itertools.permutations(group) for group in ordered_groups)):
        order = [node_id for group in permutations for node_id in group]
        mapping = {node_id: index for index, node_id in enumerate(order)}
        value = {
            "edges": sorted(
                (min(mapping[begin], mapping[end]), max(mapping[begin], mapping[end]), edge_order)
                for begin, end, edge_order in edges
            ),
            "roles": [roles[node_id] for node_id in order],
        }
        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        if best is None or encoded < best:
            best = encoded
    if best is None:
        best = '{"edges":[],"roles":[]}'
    return best


def validate_candidate_set(value: dict[str, Any], encoded: bytes) -> None:
    if encoded != canonical_bytes(value):
        raise CandidateAuditError("CandidateSet is not canonical JSON")
    if set(value) != {"candidates", "decode_kind", "schema_version"}:
        raise CandidateAuditError("CandidateSet root is not closed")
    if value["schema_version"] != "anonymous-candidate-set-v0.4":
        raise CandidateAuditError("unsupported CandidateSet schema")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 64:
        raise CandidateAuditError("candidate capacity exceeded")
    expected_kind = "Reject" if not candidates else "Graph" if len(candidates) == 1 else "Ambiguous"
    if value["decode_kind"] != expected_kind:
        raise CandidateAuditError("decode_kind disagrees with candidate cardinality")


def evaluate(pack: Path, extractor: Path, candidate: Path) -> dict[str, Any]:
    observations = read_jsonl(pack / "sealed_gt" / "observations.jsonl")
    graph_sets = {
        item["graph_set_id"]: item
        for item in read_jsonl(pack / "sealed_gt" / "graph_sets.jsonl")
    }
    rows = []
    extractor_replay = True
    candidate_replay = True
    for observation in observations:
        png = (pack / "sealed_image_pack" / "images" / f'{observation["delivery_id"]}.png').read_bytes()
        extractor_command = [sys.executable, str(extractor)]
        candidate_command = [sys.executable, str(candidate)]
        evidence = run_process(extractor_command, png, "extractor")
        extractor_replay = extractor_replay and evidence == run_process(
            extractor_command, png, "extractor replay"
        )
        candidate_bytes = run_process(candidate_command, evidence, "candidate")
        candidate_replay = candidate_replay and candidate_bytes == run_process(
            candidate_command, evidence, "candidate replay"
        )
        candidate_set = json.loads(candidate_bytes)
        validate_candidate_set(candidate_set, candidate_bytes)
        expected_graph_set = graph_sets[observation["graph_set_id"]]
        expected = sorted({graph_signature(graph, gt=True) for graph in expected_graph_set["graphs"]})
        actual = sorted({graph_signature(graph) for graph in candidate_set["candidates"]})
        expected_kind = "Graph" if len(expected) == 1 else "Ambiguous"
        rows.append(
            {
                "actual_candidate_count": len(actual),
                "actual_decode_kind": candidate_set["decode_kind"],
                "candidate_set_exact": actual == expected,
                "decision_exact": candidate_set["decode_kind"] == expected_kind,
                "delivery_id": observation["delivery_id"],
                "expected_candidate_count": len(expected),
                "expected_decode_kind": expected_kind,
                "family_id": observation["family_id"],
                "observation_id": observation["observation_id"],
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidate_set_exact_count": sum(bool(item["candidate_set_exact"]) for item in items),
            "candidate_set_exact_rate": sum(bool(item["candidate_set_exact"]) for item in items) / len(items),
            "decision_exact_count": sum(bool(item["decision_exact"]) for item in items),
            "decision_exact_rate": sum(bool(item["decision_exact"]) for item in items) / len(items),
            "observation_count": len(items),
        }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["family_id"]].append(row)
    overall = summarize(rows)
    passed = (
        extractor_replay
        and candidate_replay
        and overall["candidate_set_exact_rate"] == 1.0
        and overall["decision_exact_rate"] == 1.0
    )
    return {
        "bucket_metrics": {key: summarize(value) for key, value in sorted(buckets.items())},
        "candidate_replay_byte_exact": candidate_replay,
        "extractor_replay_byte_exact": extractor_replay,
        "overall": overall,
        "rows": rows,
        "schema_version": "evidence-core-public-candidate-audit-v0.4",
        "status": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.pack.resolve(), args.extractor.resolve(), args.candidate.resolve())
    encoded = canonical_bytes(result)
    if args.output:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
