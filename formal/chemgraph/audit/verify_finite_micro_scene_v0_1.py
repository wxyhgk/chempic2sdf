"""Independently verify graph-signature differences in collision certificates."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx


REVIEW_SCHEMA_VERSION = "finite-micro-scene-independent-review-0.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _graph(scene: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    for node in scene["nodes"]:
        graph.add_node(int(node["atom_id"]), color=str(node["color"]))
    for bond in scene["bonds"]:
        graph.add_edge(
            int(bond["begin_atom"]),
            int(bond["end_atom"]),
            order=int(bond["order"]),
            display_type=str(bond["display_type"]),
        )
    return graph


def verify_certificates(
    certificate_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    certificate_bytes = certificate_path.read_bytes()
    certificate_sha256 = _sha256(certificate_bytes)
    if expected_sha256 is not None and certificate_sha256 != expected_sha256:
        raise ValueError(
            "Collision certificate SHA256 mismatch: "
            f"expected {expected_sha256}, got {certificate_sha256}"
        )
    payload = json.loads(certificate_bytes)
    certificates = payload["certificates"]
    if int(payload["certificate_count"]) != len(certificates):
        raise ValueError("Collision certificate count does not match the payload")

    node_match = nx.algorithms.isomorphism.categorical_node_match("color", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        ["order", "display_type"],
        [None, None],
    )
    pair_count = 0
    false_differences = []
    for certificate in certificates:
        witnesses = certificate["witnesses"]
        signatures = {witness["graph_signature"] for witness in witnesses}
        if len(signatures) != int(certificate["graph_signature_count"]):
            raise ValueError(
                "Witness signature count mismatch for observation "
                f"{certificate['observation_sha256']}"
            )
        for left, right in combinations(witnesses, 2):
            pair_count += 1
            if nx.is_isomorphic(
                _graph(left["scene"]),
                _graph(right["scene"]),
                node_match=node_match,
                edge_match=edge_match,
            ):
                false_differences.append(
                    {
                        "left_scene_id": left["scene_id"],
                        "observation_sha256": certificate["observation_sha256"],
                        "right_scene_id": right["scene_id"],
                    }
                )

    script_path = Path(__file__).resolve()
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": "passed" if not false_differences else "failed",
        "training_authorized": False,
        "certificate_sha256": certificate_sha256,
        "certificate_count": len(certificates),
        "different_signature_pairs_checked": pair_count,
        "isomorphic_pairs_misclassified_as_different": len(false_differences),
        "false_differences": false_differences,
        "matcher": {
            "engine": "networkx.is_isomorphic",
            "networkx_version": nx.__version__,
            "node_attributes": ["color"],
            "edge_attributes": ["order", "display_type"],
        },
        "reviewer_script_sha256": _sha256(script_path.read_bytes()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_certificates(
        args.certificates,
        expected_sha256=args.expected_sha256,
    )
    output_bytes = _json_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    print(output_bytes.decode("utf-8"), end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
