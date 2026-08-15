#!/usr/bin/env python3
"""Produce the v0.6 constructive same-tuple ambiguity witness."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from formal.chemgraph.challenge_v0_5.domain_scene_v0_5 import (  # noqa: E402
    render_and_verify_padded_certificate,
)
from formal.chemgraph.challenge_v0_5.enumerate_domain_v0_5 import (  # noqa: E402
    AXES,
    TOTAL_TUPLES,
    decode_tuple_id,
)


SCHEMA_VERSION = "chemgraph-constructive-witness-v0.6"
PARENT_CONTRACT_SHA256 = (
    "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
)
PARENT_DOMAIN_CARDINALITY = 36_088_416


class ConstructiveWitnessError(ValueError):
    """Raised when the frozen constructive witness no longer verifies."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _graph_signature(
    labels: tuple[str, ...], bonds: tuple[tuple[int, int, int], ...]
) -> str:
    """Implement the parent contract's exact canonical graph byte protocol."""
    best: str | None = None
    for order in itertools.permutations(range(len(labels))):
        inverse = {old: new for new, old in enumerate(order)}
        value = {
            "edges": sorted(
                [
                    min(inverse[left], inverse[right]),
                    max(inverse[left], inverse[right]),
                    bond_order,
                ]
                for left, right, bond_order in bonds
            ),
            "nodes": [[labels[old], 0] for old in order],
        }
        encoded = json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        if best is None or encoded < best:
            best = encoded
    if best is None:
        raise ConstructiveWitnessError("empty_graph_not_allowed")
    return best


def _tuple_id(axis_values: dict[str, Any]) -> int:
    encoded = 0
    for name, allowed in AXES:
        try:
            index = allowed.index(axis_values[name])
        except (KeyError, ValueError) as error:
            raise ConstructiveWitnessError(f"axis_out_of_domain:{name}") from error
        encoded = encoded * len(allowed) + index
    if decode_tuple_id(encoded) != {
        "schema_version": "chemgraph-domain-tuple-v0.5",
        "tuple_id": encoded,
        **axis_values,
    }:
        raise ConstructiveWitnessError("tuple_mixed_radix_roundtrip_failed")
    return encoded


def build_witness(v0_6_contract_sha256: str) -> dict[str, Any]:
    if len(v0_6_contract_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in v0_6_contract_sha256
    ):
        raise ConstructiveWitnessError("invalid_v0_6_contract_sha256")
    if TOTAL_TUPLES != PARENT_DOMAIN_CARDINALITY:
        raise ConstructiveWitnessError("parent_domain_cardinality_changed")

    axes = {
        "half_length": 85,
        "motif_ordinal": 25,
        "width_px": 5,
        "cap": "round",
        "antialias_profile": "box_2x2",
        "reflection": "none",
        "rotation_index": 0,
        "scale_index": 3,
    }
    tuple_id = _tuple_id(axes)
    render_axes = {
        "half_length": axes["half_length"],
        "motif_ordinal": axes["motif_ordinal"],
        "width_px": axes["width_px"],
        "cap_index": 1,
        "antialias_index": 1,
        "reflection_index": 0,
        "rotation_index": axes["rotation_index"],
        "scale_index": axes["scale_index"],
    }
    first_gray, motif = render_and_verify_padded_certificate(**render_axes)
    second_gray, second_motif = render_and_verify_padded_certificate(**render_axes)
    if first_gray != second_gray:
        raise ConstructiveWitnessError("same_tuple_render_not_deterministic")
    if len(motif.interpretations) != 2 or motif != second_motif:
        raise ConstructiveWitnessError("c1_interpretation_domain_changed")

    interpretations = []
    signatures = []
    for index, interpretation in enumerate(motif.interpretations):
        labels = tuple(atom.label for atom in interpretation.atoms)
        bonds = tuple(
            (bond.begin, bond.end, bond.order) for bond in interpretation.bonds
        )
        signature = _graph_signature(labels, bonds)
        signatures.append(signature)
        interpretations.append(
            {
                "atom_count": len(labels),
                "bond_count": len(bonds),
                "decode_kind": interpretation.decode_kind,
                "graph_signature": signature,
                "graph_signature_sha256": sha256(signature.encode("ascii")),
                "interpretation_index": index,
            }
        )
    if signatures[0] == signatures[1]:
        raise ConstructiveWitnessError("interpretations_are_graph_equivalent")

    return {
        "checks": {
            "both_interpretations_in_same_frozen_tuple": True,
            "graph_signatures_distinct": True,
            "padded_clipping_certificate_passed": True,
            "render_is_interpretation_independent": True,
            "same_tuple_full_gray_bytes_equal": first_gray == second_gray,
        },
        "contract_binding": {
            "parent_contract_sha256": PARENT_CONTRACT_SHA256,
            "parent_domain_cardinality": PARENT_DOMAIN_CARDINALITY,
            "v0_6_contract_sha256": v0_6_contract_sha256,
        },
        "interpretations": interpretations,
        "raster": {
            "byte_length": len(first_gray),
            "format": "gray8-768x768",
            "sha256": sha256(first_gray),
        },
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "claim": "full_frozen_domain_contains_a_non_singleton_graph_fiber",
            "full_domain_collision_rate_claimed": False,
            "neural_generalization_claimed": False,
            "production_png_equivalence_claimed": False,
            "sampled_to_full_extrapolation_used": False,
            "training_authorized": False,
        },
        "status": "passed",
        "witness_tuple": {"tuple_id": tuple_id, **axes},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-6-contract-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = canonical_bytes(build_witness(args.v0_6_contract_sha256))
    if args.output is not None:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
