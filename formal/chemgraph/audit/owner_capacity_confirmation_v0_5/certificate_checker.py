"""Check frozen conditional bounds for R0 capacity axes that remain NO-GO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PATHS = {
    "build_observation": REPOSITORY_ROOT
    / "formal/chemgraph/audit/owner_candidate_domain_v0_5/build_observation.py",
    "candidate_constructor": REPOSITORY_ROOT
    / "formal/chemgraph/audit/owner_candidate_domain_v0_5/candidate_domain.py",
    "capacity_common": REPOSITORY_ROOT
    / "formal/chemgraph/audit/owner_candidate_domain_v0_5/common.py",
    "fragment_root": REPOSITORY_ROOT
    / "formal/chemgraph/audit/fragment_root_constructor_v0_4/fragment_root.py",
    "support_partition": REPOSITORY_ROOT
    / "formal/chemgraph/audit/fragment_support_partition_v0_4/support_partition.py",
    "lean_capacity": REPOSITORY_ROOT
    / "formal/chemgraph/ChemGraph/OwnerCapacityReachabilityV05.lean",
}
EXPECTED_SHA256 = {
    "build_observation": "5736e410270e06b5728f6285187156fd1549abd77ce495b57cccec97b1d9a748",
    "candidate_constructor": "5631ba0d2ca24ab324f77880ecc0ce9668bb7bc85de78e828325b2c04fb3b4cf",
    "capacity_common": "26b7ff36222e4b5239322052f03025fe4db7d98492f10425b82f489606ca68d8",
    "fragment_root": "69aa32043b87cc241d3fac83279e8e708f06c36c7c0771c8f623d97daad85106",
    "support_partition": "c36eef270ebee85a842e445e0048b214dec7382544766b7b29a30062d0240e36",
    "lean_capacity": "4001dcedb98d51eb5a04be9714d6adf9f3178e35701dbe111027c99263fec092",
}
LIMITS = {
    "max_adjacency_edges": 32_768,
    "max_ambiguous_variables": 20,
    "max_atomic_fragments": 64,
    "max_carriers": 8,
    "max_constraints": 32_768,
    "max_domain_classes": 64,
    "max_intersections": 8,
    "max_output_bytes": 16_777_216,
    "max_owner_options": 16,
    "max_pixel_variables": 4_096,
    "max_raw_assignments": 1_048_576,
    "max_total_rle_runs": 65_536,
}


class CapacityCertificateError(ValueError):
    """Frozen capacity certificate does not check."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _fixed_hashes() -> dict[str, str]:
    actual = {name: _sha256(path) for name, path in PATHS.items()}
    if actual != EXPECTED_SHA256:
        raise CapacityCertificateError("FROZEN_DEPENDENCY_SHA_MISMATCH")
    return actual


def _maximal_closed_artifact_template() -> dict[str, Any]:
    hash64 = "f" * 64
    variable_id = 9_999
    edge_id = 99_999
    carrier_id = 9
    fragment_id = 99
    option_id = 9
    variables = [variable_id] * 4_096
    edges = [edge_id] * 16_384
    evidence_item = {
        "carrier_ids": [carrier_id] * 8,
        "fragment_memberships": [
            {"atomic_fragment_id": fragment_id, "carrier_id": carrier_id}
            for _ in range(8)
        ],
        "id": variable_id,
        "pixel": [999, 999],
    }
    adjacency_item = {
        "id": edge_id,
        "left_variable_id": variable_id,
        "right_variable_id": variable_id,
    }
    constraints = [{
        "id": 9,
        "kind": "exact_one",
        "variable_ids": variables,
    }]
    for _ in range(2):
        constraints.extend([
            {
                "id": 9,
                "kind": "seed_owner",
                "owner_option_id": option_id,
                "variable_id": variable_id,
            },
            {
                "adjacency_edge_ids": edges,
                "eligible_variable_ids": variables,
                "id": 9,
                "kind": "connected_to_seed",
                "owner_option_id": option_id,
                "seed_variable_id": variable_id,
            },
            {
                "eligible_variable_ids": variables,
                "id": 9,
                "kind": "option_incidence",
                "owner_option_id": option_id,
            },
        ])
    counts = {
        "adjacency_edge_count": 99_999,
        "ambiguous_variable_count": 99,
        "atomic_fragment_count": 99,
        "carrier_count": 9,
        "constraint_count": 99_999,
        "domain_class_count": 99,
        "evidence_membership_count": 9_999,
        "owner_option_count": 99,
        "pixel_variable_count": 9_999,
        "residual_reason_count": 9,
    }
    metrics = {
        "baseline_id": "all_and_only_known_intersections_per_variable-v0.5-r0",
        "candidate_option_membership_count": 9_999,
        "domain_size_histogram": [
            {"domain_size": 9, "variable_count": 9_999},
            {"domain_size": 9, "variable_count": 9_999},
        ],
        "eligible_option_count": 9_999,
        "eligible_option_recall": [9_999_999, 9_999_999],
        "eliminated_option_membership_count": 9_999,
        "eliminated_option_ratio": [9_999_999, 9_999_999],
        "eliminated_ineligible_option_count": 9_999,
        "ineligible_option_count": 9_999,
        "ineligible_option_elimination_recall": [9_999_999, 9_999_999],
        "max_domain_size": 9,
        "mean_domain_size": [9_999_999, 9_999_999],
        "multi_intersection_ratio": [9_999_999, 9_999_999],
        "raw_cartesian_assignment_count": 9_999_999,
        "retained_eligible_option_count": 9_999,
        "residual_containing_ratio": [9_999_999, 9_999_999],
        "residual_only_ratio": [9_999_999, 9_999_999],
        "singleton_ratio": [9_999_999, 9_999_999],
        "universal_option_membership_count": 9_999,
    }
    section_names = [
        "source_binding", "limits", "counts", "owner_options",
        "evidence_memberships", "domain_classes", "pixel_variables",
        "adjacency_edges", "constraints", "residual_reasons", "metrics",
    ]
    stage_names = [
        "source_binding", "owner_options", "evidence_memberships", "domain_classes",
        "pixel_variables", "adjacency_edges", "constraints", "metrics",
    ]
    return {
        "adjacency_edges": [adjacency_item] * 16_384,
        "constraints": constraints,
        "counts": counts,
        "domain_classes": [
            {"id": 99, "owner_option_ids": [option_id, option_id]}
            for _ in range(3)
        ],
        "evidence_memberships": [evidence_item] * 4_096,
        "limits": LIMITS,
        "metrics": metrics,
        "owner_options": [{
            "anchor_pixel": [999, 999],
            "geometry_intersection_id": 9,
            "id": option_id,
            "incident_carrier_ids": [carrier_id] * 8,
            "kind": "intersection",
        }] * 2,
        "pixel_variables": [{
            "domain_class_id": 99,
            "evidence_membership_id": variable_id,
            "id": variable_id,
            "pixel": [999, 999],
        }] * 4_096,
        "residual_reasons": [],
        "schema_version": "chemgraph-owner-candidate-domain-v0.5-r0-research",
        "section_hashes": [
            {"item_count": 99_999, "section": name, "sha256": hash64}
            for name in section_names
        ],
        "source_binding": {
            "compiled_fragment_root_sha256": hash64,
            "compiled_support_partition_sha256": hash64,
            "constructor_protocol_commit": "fffffff",
            "geometry_root_sha256": hash64,
            "observation_sha256": hash64,
            "png_sha256": hash64,
            "renderer_sha256": hash64,
            "support_input_sha256": hash64,
        },
        "stage_trace": [{
            "component_sha256": hash64,
            "index": 9,
            "input_sha256": hash64,
            "output_sha256": hash64,
            "record_count": 99_999,
            "stage": name,
        } for name in stage_names],
        "status": "candidate_domain",
        "training_authorized": False,
    }


def check_certificates() -> dict[str, Any]:
    dependencies = _fixed_hashes()
    if LIMITS["max_owner_options"] <= 2:
        raise CapacityCertificateError("OWNER_OPTION_LIMIT_INVALID")
    if 1 + 3 * 2 != 7 or LIMITS["max_constraints"] <= 7:
        raise CapacityCertificateError("CONSTRAINT_BOUND_INVALID")
    if 2**2 - 1 != 3 or LIMITS["max_domain_classes"] <= 3:
        raise CapacityCertificateError("DOMAIN_CLASS_BOUND_INVALID")
    edge_bound = 4 * LIMITS["max_pixel_variables"]
    if not edge_bound < LIMITS["max_adjacency_edges"] + 1:
        raise CapacityCertificateError("ADJACENCY_BOUND_INVALID")
    fragment_bound = LIMITS["max_carriers"] ** 2
    if fragment_bound != LIMITS["max_atomic_fragments"]:
        raise CapacityCertificateError("FRAGMENT_BOUND_INVALID")
    raw_bound = 2 ** LIMITS["max_ambiguous_variables"]
    if raw_bound != LIMITS["max_raw_assignments"]:
        raise CapacityCertificateError("RAW_BOUND_INVALID")
    ambiguous_high_raw = 2 ** (LIMITS["max_ambiguous_variables"] + 1)
    if ambiguous_high_raw <= LIMITS["max_raw_assignments"]:
        raise CapacityCertificateError("AMBIGUOUS_ISOLATION_INVALID")
    output_bound = len(_canonical_bytes(_maximal_closed_artifact_template()))
    if output_bound >= LIMITS["max_output_bytes"]:
        raise CapacityCertificateError("OUTPUT_BYTE_BOUND_NOT_STRICT")
    return {
        "axes": {
            "max_adjacency_edges": {"upper_bound": edge_bound, "status": "conditional_unreachable"},
            "max_ambiguous_variables": {
                "high_implied_raw_assignments": ambiguous_high_raw,
                "status": "conditional_not_isolated",
            },
            "max_atomic_fragments": {
                "high_65_unreachable_if_premise": True,
                "upper_bound": fragment_bound,
                "status": "conditional_unreachable",
            },
            "max_constraints": {"upper_bound": 7, "status": "conditional_unreachable"},
            "max_domain_classes": {"upper_bound": 3, "status": "conditional_unreachable"},
            "max_output_bytes": {
                "upper_bound": output_bound,
                "status": "manual_schema_envelope_only",
            },
            "max_owner_options": {"upper_bound": 2, "status": "conditional_unreachable"},
            "max_raw_assignments": {"upper_bound": raw_bound, "status": "conditional_shadowed"},
        },
        "dependency_sha256": dependencies,
        "frozen_dependency_hashes_verified": True,
        "implementation_correctness_proved": False,
        "lean_module_hash_bound": True,
        "manual_schema_envelope_computed": True,
        "premises_recomputed": False,
        "python_arithmetic_mirrors_lean_statement": True,
        "schema_version": "owner-capacity-certificates-v0.5-preregistration",
        "status": "conditional_certificates_computed",
        "training_authorized": False,
        "universal_byte_bound_verified": False,
    }


def main() -> None:
    print(_canonical_bytes(check_certificates()).decode("ascii"), end="")


if __name__ == "__main__":
    main()
