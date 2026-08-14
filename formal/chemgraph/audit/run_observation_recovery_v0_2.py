"""Run the frozen Grammar v0.2 observation-recovery audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .observation_recovery_v0_2 import (
    AMBIGUITY_NONE,
    EXPECTED_SCENE_COUNT,
    FROZEN_CERTIFICATES_SHA256,
    FROZEN_MANIFEST_SHA256,
    SCHEMA_VERSION,
    STRATEGY_B_PRIOR_IDS,
    SUPPORTED_DOMAINS,
    ObservationClass,
    RecoveryDecision,
    build_observation_classes,
    fixture_observation_hashes,
    load_frozen_inputs,
    sha256_file,
    strategy_a_conservative,
    strategy_b_resolved,
)


EXPECTED_FULL_OBSERVATION_COUNTS = {
    "simple_domain": 19_823,
    "crossing_domain": 36_649,
}
EXPECTED_STRATEGY_A = {
    "simple_domain": {
        "Graph": 19_822,
        "Ambiguous": 1,
        "ambiguity_kinds": {"parallel_multiplicity": 1},
    },
    "crossing_domain": {
        "Graph": 36_641,
        "Ambiguous": 8,
        "ambiguity_kinds": {
            "crossing_junction": 7,
            "parallel_multiplicity": 1,
        },
    },
}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            data = (json.dumps(
                row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ) + "\n").encode("utf-8")
            output.write(data)
            digest.update(data)
    return digest.hexdigest()


def _decision_statistics(decisions: Sequence[RecoveryDecision]) -> dict[str, Any]:
    result_counts = Counter(decision.result_type for decision in decisions)
    ambiguity_kinds = Counter(
        decision.ambiguity_kind
        for decision in decisions
        if decision.ambiguity_kind != AMBIGUITY_NONE
    )
    field_strict = [decision for decision in decisions if decision.field_strict]
    return {
        "ambiguity_kind_counts": dict(sorted(ambiguity_kinds.items())),
        "coverage_fraction": (
            result_counts["Graph"] / len(decisions) if decisions else 0.0
        ),
        "decode_left_inverse_denominator": len(field_strict),
        "decode_left_inverse_numerator": sum(
            decision.decode_left_inverse is True for decision in field_strict
        ),
        "field_strict_observation_count": len(field_strict),
        "graph_output_count": result_counts["Graph"],
        "nonempty_survivor_observation_count": sum(
            decision.survivor_count > 0 for decision in decisions
        ),
        "reject_count": result_counts["Reject"],
        "result_counts": dict(sorted(result_counts.items())),
        "survivor_graph_collision_count": sum(
            decision.survivor_graph_count > 1 for decision in decisions
        ),
        "zero_survivor_observation_count": sum(
            decision.survivor_count == 0 for decision in decisions
        ),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: sha256_file(root / name)
        for name in (
            "observation_recovery_v0_2.py",
            "run_observation_recovery_v0_2.py",
        )
    }


def _select_smoke_classes(
    classes: Sequence[ObservationClass],
    fixture_hashes: frozenset[str],
    smoke_limit: int,
) -> tuple[ObservationClass, ...]:
    selected = list(classes[:smoke_limit])
    selected_hashes = {item.observation_sha256 for item in selected}
    selected.extend(
        item
        for item in classes
        if item.observation_sha256 in fixture_hashes
        and item.observation_sha256 not in selected_hashes
    )
    return tuple(sorted(selected, key=lambda item: item.observation_sha256))


def run_audit(
    manifest_path: Path,
    certificates_path: Path,
    output_dir: Path,
    *,
    smoke_limit: int | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_inputs(manifest_path, certificates_path)
    fixture_hashes = fixture_observation_hashes(frozen)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    certificate_records = []
    domain_summary: dict[str, Any] = {}
    decision_cache: dict[
        str, tuple[tuple[ObservationClass, RecoveryDecision, RecoveryDecision], ...]
    ] = {}

    for domain in SUPPORTED_DOMAINS:
        classes = build_observation_classes(frozen, domain)
        full_count = len(classes)
        if smoke_limit is not None:
            classes = _select_smoke_classes(classes, fixture_hashes, smoke_limit)
        triples = tuple(
            (
                observation,
                strategy_a_conservative(observation),
                strategy_b_resolved(observation),
            )
            for observation in classes
        )
        decision_cache[domain] = triples
        strategy_a = [item[1] for item in triples]
        strategy_b = [item[2] for item in triples]
        domain_summary[domain] = {
            "audited_observation_count": len(classes),
            "full_domain_observation_count": full_count,
            "strategy_a": _decision_statistics(strategy_a),
            "strategy_b": _decision_statistics(strategy_b),
        }

        for observation, decision_a, decision_b in triples:
            manifest_records.append(
                {
                    "domain": domain,
                    "observation_sha256": observation.observation_sha256,
                    "schema_version": SCHEMA_VERSION,
                    "strategy_a": decision_a.as_record(),
                    "strategy_b": decision_b.as_record(),
                }
            )
            if decision_a.result_type == "Ambiguous" or decision_b.result_type != "Graph":
                certificate_records.append(
                    {
                        "candidate_graph_signatures": sorted(
                            observation.graph_signatures
                        ),
                        "candidate_scene_ids": sorted(
                            candidate.scene_id for candidate in observation.candidates
                        ),
                        "domain": domain,
                        "observation_sha256": observation.observation_sha256,
                        "strategy_a_result": decision_a.result_type,
                        "strategy_a_ambiguity_kind": decision_a.ambiguity_kind,
                        "strategy_b_result": decision_b.result_type,
                        "strategy_b_survivor_count": decision_b.survivor_count,
                    }
                )

    is_full = smoke_limit is None
    full_count_match = all(
        domain_summary[domain]["full_domain_observation_count"]
        == EXPECTED_FULL_OBSERVATION_COUNTS[domain]
        for domain in SUPPORTED_DOMAINS
    )
    strategy_a_regression = is_full
    if is_full:
        for domain, expected in EXPECTED_STRATEGY_A.items():
            actual = domain_summary[domain]["strategy_a"]
            strategy_a_regression &= actual["result_counts"] == {
                key: value for key, value in expected.items() if key != "ambiguity_kinds"
            }
            strategy_a_regression &= (
                actual["ambiguity_kind_counts"] == expected["ambiguity_kinds"]
            )

    strategy_a_wrong_graph_outputs = sum(
        decision.result_type == "Graph"
        and decision.graph_signature not in observation.graph_signatures
        for triples in decision_cache.values()
        for observation, decision, _ in triples
    )
    strategy_a_ambiguous_graph_outputs = sum(
        len(observation.graph_signatures) > 1 and decision.result_type == "Graph"
        for triples in decision_cache.values()
        for observation, decision, _ in triples
    )
    strategy_b_zero_survivor = sum(
        decision.survivor_count == 0
        for triples in decision_cache.values()
        for _, _, decision in triples
    )
    strategy_b_survivor_collisions = sum(
        decision.survivor_graph_count > 1
        for triples in decision_cache.values()
        for _, _, decision in triples
    )
    left_inverse_failures = sum(
        decision.field_strict and decision.decode_left_inverse is not True
        for triples in decision_cache.values()
        for _, decision_a, decision_b in triples
        for decision in (decision_a, decision_b)
    )

    finite_policy_consistency = (
        is_full
        and full_count_match
        and strategy_a_regression
        and strategy_a_wrong_graph_outputs == 0
        and strategy_a_ambiguous_graph_outputs == 0
        and strategy_b_zero_survivor == 0
        and strategy_b_survivor_collisions == 0
    )
    observable_extractor_factorization = False
    strategy_b_zero_survivor_by_domain = {
        domain: domain_summary[domain]["strategy_b"][
            "zero_survivor_observation_count"
        ]
        for domain in SUPPORTED_DOMAINS
    }
    strategy_a_status = (
        "passed"
        if (
            is_full
            and strategy_a_regression
            and strategy_a_wrong_graph_outputs == 0
            and strategy_a_ambiguous_graph_outputs == 0
        )
        else ("not_evaluated_smoke" if not is_full else "no_go")
    )
    strategy_b_status = (
        "passed"
        if (
            is_full
            and strategy_b_zero_survivor == 0
            and strategy_b_survivor_collisions == 0
        )
        else "no_go"
    )

    summary = {
        "active_strategy_b_prior_ids": list(STRATEGY_B_PRIOR_IDS),
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "domains": domain_summary,
        "finite_domain_interpretation_completeness": (
            "complete_only_relative_to_frozen_v0_1_scene_domain"
        ),
        "input": {
            "certificates_sha256": frozen.certificates_sha256,
            "manifest_scene_count": len(frozen.manifest_rows),
            "manifest_sha256": frozen.manifest_sha256,
        },
        "mode": "full" if is_full else "smoke",
        "observable_extractor": {
            "implemented_from_opaque_raster_or_tensor": False,
            "lane_evidence_source": "finite_scene_physical_members",
            "same_observation_same_lane_evidence_checked": True,
            "status": "no_go_not_implemented",
        },
        "schema_version": SCHEMA_VERSION,
        "semantic_serialization_left_inverse": {
            "failure_count": left_inverse_failures,
            "scope": "field_strict_audit_oracle_fields_only",
            "status": "passed" if left_inverse_failures == 0 else "failed",
        },
        "strategy_a_finite_spec_status": strategy_a_status,
        "strategy_b_global_prior_status": {
            "status": strategy_b_status,
            "zero_survivor_by_domain": strategy_b_zero_survivor_by_domain,
        },
    }
    summary_sha = _write_json(output_dir / "summary.json", summary)

    manifest_records.sort(key=lambda item: (item["domain"], item["observation_sha256"]))
    manifest_sha = _write_jsonl(
        output_dir / "observation_recovery_manifest.jsonl", manifest_records
    )
    certificates = {
        "certificate_count": len(certificate_records),
        "certificates": sorted(
            certificate_records,
            key=lambda item: (item["domain"], item["observation_sha256"]),
        ),
        "schema_version": SCHEMA_VERSION,
    }
    certificate_sha = _write_json(
        output_dir / "recovery_certificates.json", certificates
    )

    checks = {
        "finite_domain_observation_counts_match_v0_1": full_count_match,
        "frozen_certificate_sha256_matches": (
            frozen.certificates_sha256 == FROZEN_CERTIFICATES_SHA256
        ),
        "frozen_manifest_has_66716_unique_scenes": (
            len(frozen.manifest_rows) == EXPECTED_SCENE_COUNT
        ),
        "frozen_manifest_sha256_matches": (
            frozen.manifest_sha256 == FROZEN_MANIFEST_SHA256
        ),
        "oracle_field_left_inverse_on_field_strict": left_inverse_failures == 0,
        "strategy_a_ambiguous_class_graph_output_count_is_zero": (
            strategy_a_ambiguous_graph_outputs == 0
        ),
        "strategy_a_regression_matches_preregistration": strategy_a_regression,
        "strategy_a_wrong_graph_output_count_is_zero": (
            strategy_a_wrong_graph_outputs == 0
        ),
        "strategy_b_coverage_is_100_percent": strategy_b_zero_survivor == 0,
        "strategy_b_survivor_graph_collision_count_is_zero": (
            strategy_b_survivor_collisions == 0
        ),
    }
    validation = {
        "artifacts": {
            "observation_recovery_manifest.jsonl": manifest_sha,
            "recovery_certificates.json": certificate_sha,
            "summary.json": summary_sha,
        },
        "audit_oracle_only": True,
        "checks": checks,
        "deployable_from_opaque_observation": False,
        "finite_policy_consistency": {
            "status": "passed" if finite_policy_consistency else "no_go",
            "strategy_b_zero_survivor_observation_count": strategy_b_zero_survivor,
        },
        "mode": "full" if is_full else "smoke",
        "observable_extractor_factorization": {
            "reason": "no extractor from opaque raster/tensor to lane evidence is implemented",
            "status": "no_go_not_implemented",
        },
        "overall_gate_status": "no_go",
        "schema_version": SCHEMA_VERSION,
        "semantic_serialization_left_inverse": {
            "failure_count": left_inverse_failures,
            "scope": "field_strict_only",
            "status": "passed" if left_inverse_failures == 0 else "failed",
        },
        "strategy_a_finite_spec_status": {
            "status": strategy_a_status,
            "wrong_graph_output_count": strategy_a_wrong_graph_outputs,
            "ambiguous_class_graph_output_count": (
                strategy_a_ambiguous_graph_outputs
            ),
        },
        "strategy_b_global_prior_status": {
            "status": strategy_b_status,
            "survivor_graph_collision_count": strategy_b_survivor_collisions,
            "zero_survivor_by_domain": strategy_b_zero_survivor_by_domain,
            "zero_survivor_observation_count": strategy_b_zero_survivor,
        },
        "source_sha256": _source_hashes(),
        "training_authorized": False,
    }
    validation_sha = _write_json(output_dir / "validation.json", validation)
    validation["validation_sha256_before_self_reference"] = validation_sha
    return validation


def _default_input_root() -> Path:
    return (
        Path(__file__).resolve().parent
        / "results"
        / "finite_geometry_grammar_v0_1"
        / "final_a"
    )


def main() -> None:
    input_root = _default_input_root()
    parser = argparse.ArgumentParser(
        description="Run the Grammar v0.2 observation-class recovery audit"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=input_root / "scene_manifest.jsonl",
    )
    parser.add_argument(
        "--certificates",
        type=Path,
        default=input_root / "collision_certificates.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="Audit the first N observations plus all fixed fixture observations",
    )
    args = parser.parse_args()
    if args.smoke_limit is not None and args.smoke_limit < 0:
        parser.error("--smoke-limit must be nonnegative")
    result = run_audit(
        args.manifest,
        args.certificates,
        args.output,
        smoke_limit=args.smoke_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
