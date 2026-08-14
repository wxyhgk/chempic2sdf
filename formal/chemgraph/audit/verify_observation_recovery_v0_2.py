"""Independently verify Grammar v0.2 observation-recovery artifacts.

The verifier treats frozen graph signatures as opaque labels.  It deliberately
does not reconstruct scenes, canonicalize graphs, or execute either recovery
strategy.  All expected candidate sets are regrouped directly from the frozen
v0.1 manifest with Python's standard library.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REVIEW_SCHEMA_VERSION = "observation-recovery-independent-review-v0.2"
RECOVERY_SCHEMA_VERSION = "observation-recovery-v0.2"

FROZEN_MANIFEST_SHA256 = (
    "f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7"
)
FROZEN_CERTIFICATES_SHA256 = (
    "f97262008a50edf26b263745b9058533820be8bb72fdffcc3f9c38e6f9839485"
)
EXPECTED_SCENE_COUNT = 66_716
EXPECTED_CERTIFICATE_COUNT = 3_518

SIMPLE_LAYER = "simple_line_angle_valid"
CROSSING_LAYER = "crossing_allowed"
DOMAINS = {
    "simple_domain": frozenset({SIMPLE_LAYER}),
    "crossing_domain": frozenset({SIMPLE_LAYER, CROSSING_LAYER}),
}
PRIOR_IDS = [
    "terminal_aligned_parallel_strokes_are_one_multiple_bond_v1",
    "unmarked_transversal_crossing_is_disconnected_v1",
]


class VerificationError(RuntimeError):
    """Raised when an independent invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise VerificationError(
                        f"Invalid JSONL at {path}:{line_number}: {error}"
                    ) from error
                _require(
                    isinstance(row, dict),
                    f"JSONL row is not an object at {path}:{line_number}",
                )
                rows.append(row)
    except OSError as error:
        raise VerificationError(f"Cannot read JSONL {path}: {error}") from error
    return tuple(rows)


def _candidate_classes(
    frozen_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    result: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for domain, allowed_layers in DOMAINS.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in frozen_rows:
            if row["exclusive_grammar_layer"] in allowed_layers:
                grouped[row["production_observation_sha256"]].append(row)
        result[domain] = {
            observation: tuple(sorted(rows, key=lambda item: item["scene_id"]))
            for observation, rows in grouped.items()
        }
    return result


def _graphs(rows: Sequence[dict[str, Any]]) -> frozenset[str]:
    return frozenset(row["canonical_graph_signature_structure"] for row in rows)


def _ambiguity_taxonomy(
    domain: str,
    observation: str,
    classes: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
) -> str:
    rows = classes[domain][observation]
    if len(_graphs(rows)) <= 1:
        return "none"

    layers = {row["exclusive_grammar_layer"] for row in rows}
    if layers == {SIMPLE_LAYER}:
        return "parallel_multiplicity"

    if domain == "crossing_domain" and layers == {SIMPLE_LAYER, CROSSING_LAYER}:
        simple_graphs = _graphs(
            [row for row in rows if row["exclusive_grammar_layer"] == SIMPLE_LAYER]
        )
        crossing_graphs = _graphs(
            [row for row in rows if row["exclusive_grammar_layer"] == CROSSING_LAYER]
        )
        _require(
            len(simple_graphs) == 1 and len(crossing_graphs) == 1,
            "Crossing taxonomy requires one graph meaning in each exclusive layer: "
            f"{domain}/{observation}",
        )
        _require(
            simple_graphs != crossing_graphs,
            "Crossing taxonomy requires a cross-layer graph disagreement: "
            f"{domain}/{observation}",
        )
        simple_only = classes["simple_domain"].get(observation, ())
        _require(
            len(_graphs(simple_only)) == 1,
            "Crossing taxonomy requires a strict simple-domain interpretation: "
            f"{observation}",
        )
        return "crossing_junction"

    raise VerificationError(
        "Ambiguity is not covered by the frozen parallel/crossing taxonomy: "
        f"{domain}/{observation}, layers={sorted(layers)}"
    )


def _verify_frozen_certificates(
    frozen_rows: Sequence[dict[str, Any]],
    certificate_document: dict[str, Any],
    expected_certificate_count: int | None,
) -> None:
    certificates = certificate_document.get("certificates")
    _require(isinstance(certificates, list), "Frozen certificates must be a list")
    _require(
        certificate_document.get("certificate_count") == len(certificates),
        "Frozen certificate_count does not match its list",
    )
    if expected_certificate_count is not None:
        _require(
            len(certificates) == expected_certificate_count,
            "Frozen certificate count does not match the commitment",
        )

    rows_by_scene = {row["scene_id"]: row for row in frozen_rows}
    all_classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        all_classes[row["production_observation_sha256"]].append(row)
    expected_collision_observations = {
        observation
        for observation, rows in all_classes.items()
        if len(_graphs(rows)) > 1
    }

    seen: set[str] = set()
    for certificate in certificates:
        observation = certificate.get("observation_sha256")
        _require(
            isinstance(observation, str) and observation not in seen,
            "Frozen certificate observations must be unique strings",
        )
        seen.add(observation)
        witnesses = certificate.get("witnesses")
        _require(isinstance(witnesses, list), f"Missing witnesses for {observation}")
        witness_graphs: set[str] = set()
        for witness in witnesses:
            scene_id = witness.get("scene_id")
            _require(scene_id in rows_by_scene, f"Unknown frozen witness {scene_id!r}")
            row = rows_by_scene[scene_id]
            _require(
                row["production_observation_sha256"] == observation,
                f"Frozen witness observation mismatch for {scene_id}",
            )
            graph = witness.get("graph_signature")
            _require(
                graph == row["canonical_graph_signature_structure"],
                f"Frozen witness graph mismatch for {scene_id}",
            )
            witness_graphs.add(graph)
        _require(
            witness_graphs == _graphs(all_classes[observation]),
            f"Frozen witnesses do not cover every graph for {observation}",
        )
        _require(
            certificate.get("graph_signature_count") == len(witness_graphs),
            f"Frozen graph_signature_count mismatch for {observation}",
        )
    _require(
        seen == expected_collision_observations,
        "Frozen certificates do not exactly cover all-domain graph collisions",
    )


def _verify_boundary_flags(decision: dict[str, Any], label: str) -> None:
    _require(decision.get("audit_oracle_only") is True, f"{label} is not audit-only")
    _require(
        decision.get("deployable_from_opaque_observation") is False,
        f"{label} incorrectly claims opaque-observation deployability",
    )
    _require(
        decision.get("observation_extractor_implemented") is False,
        f"{label} incorrectly claims an observation extractor",
    )
    field = decision.get("oracle_field")
    if field is not None:
        _require(isinstance(field, dict), f"{label} oracle_field is malformed")
        _require(field.get("audit_oracle_only") is True, f"{label} field is not audit-only")
        _require(
            field.get("deployable_from_opaque_observation") is False,
            f"{label} field incorrectly claims deployability",
        )
        _require(
            field.get("observation_extractor_implemented") is False,
            f"{label} field incorrectly claims an extractor",
        )


def _verify_strategy_a(
    decision: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    ambiguity_kind: str,
    label: str,
) -> None:
    graphs = _graphs(rows)
    _verify_boundary_flags(decision, label)
    _require(decision.get("candidate_count") == len(rows), f"{label} candidate_count mismatch")
    _require(decision.get("survivor_count") == len(rows), f"{label} survivor_count mismatch")
    _require(
        decision.get("survivor_graph_count") == len(graphs),
        f"{label} survivor_graph_count mismatch",
    )
    _require(decision.get("active_prior_ids") == [], f"{label} unexpectedly uses a prior")
    _require(
        decision.get("ambiguity_kind") == ambiguity_kind,
        f"{label} ambiguity taxonomy mismatch",
    )

    if len(graphs) == 1:
        graph = next(iter(graphs))
        _require(decision.get("result_type") == "Graph", f"{label} strict class is not Graph")
        _require(decision.get("graph_signature") == graph, f"{label} emitted the wrong Graph")
        _require(
            decision.get("protocol_choice_graph_signature") == graph,
            f"{label} strict protocol choice mismatch",
        )
        if decision.get("field_strict") is True:
            _require(
                decision.get("decode_left_inverse") is True,
                f"{label} claims field-strict without a left inverse",
            )
    else:
        _require(
            decision.get("result_type") == "Ambiguous",
            f"{label} ambiguous class was not rejected conservatively",
        )
        _require(decision.get("graph_signature") is None, f"{label} guessed an ambiguous Graph")
        _require(
            decision.get("protocol_choice_graph_signature") is None,
            f"{label} has a protocol choice without a prior",
        )
        _require(decision.get("field_strict") is False, f"{label} ambiguous field is strict")
        _require(decision.get("decode_left_inverse") is None, f"{label} ambiguous field decoded")


def _verify_strategy_b(
    decision: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    ambiguity_kind: str,
    label: str,
) -> None:
    graphs = _graphs(rows)
    _verify_boundary_flags(decision, label)
    _require(decision.get("candidate_count") == len(rows), f"{label} candidate_count mismatch")
    _require(decision.get("active_prior_ids") == PRIOR_IDS, f"{label} prior set mismatch")
    choice = decision.get("protocol_choice_graph_signature")
    _require(isinstance(choice, str) and choice, f"{label} lacks a protocol choice")
    survivor_count = sum(
        row["canonical_graph_signature_structure"] == choice for row in rows
    )
    _require(
        decision.get("survivor_count") == survivor_count,
        f"{label} survivor_count mismatch",
    )
    _require(
        decision.get("survivor_graph_count") == (1 if survivor_count else 0),
        f"{label} survivor_graph_count mismatch",
    )

    if choice in graphs:
        _require(decision.get("result_type") == "Graph", f"{label} choice in candidates is not Graph")
        _require(decision.get("graph_signature") == choice, f"{label} output differs from choice")
        _require(decision.get("ambiguity_kind") == "none", f"{label} resolved Graph is ambiguous")
        _require(decision.get("field_strict") is True, f"{label} resolved field is not strict")
        _require(decision.get("decode_left_inverse") is True, f"{label} resolved field did not decode")
    else:
        _require(decision.get("result_type") == "Reject", f"{label} out-of-set choice is not Reject")
        _require(decision.get("graph_signature") is None, f"{label} Reject emitted a Graph")
        _require(
            decision.get("ambiguity_kind") == ambiguity_kind,
            f"{label} Reject taxonomy mismatch",
        )
        _require(decision.get("field_strict") is False, f"{label} Reject field is strict")
        _require(decision.get("decode_left_inverse") is None, f"{label} Reject field decoded")


def _decision_statistics(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result_counts = Counter(decision["result_type"] for decision in decisions)
    ambiguity_kinds = Counter(
        decision["ambiguity_kind"]
        for decision in decisions
        if decision["ambiguity_kind"] != "none"
    )
    strict = [decision for decision in decisions if decision["field_strict"]]
    return {
        "ambiguity_kind_counts": dict(sorted(ambiguity_kinds.items())),
        "coverage_fraction": result_counts["Graph"] / len(decisions) if decisions else 0.0,
        "decode_left_inverse_denominator": len(strict),
        "decode_left_inverse_numerator": sum(
            decision["decode_left_inverse"] is True for decision in strict
        ),
        "field_strict_observation_count": len(strict),
        "graph_output_count": result_counts["Graph"],
        "nonempty_survivor_observation_count": sum(
            decision["survivor_count"] > 0 for decision in decisions
        ),
        "reject_count": result_counts["Reject"],
        "result_counts": dict(sorted(result_counts.items())),
        "survivor_graph_collision_count": sum(
            decision["survivor_graph_count"] > 1 for decision in decisions
        ),
        "zero_survivor_observation_count": sum(
            decision["survivor_count"] == 0 for decision in decisions
        ),
    }


def _verify_recovery_certificates(
    certificate_document: dict[str, Any],
    expected: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    certificates = certificate_document.get("certificates")
    _require(isinstance(certificates, list), "Recovery certificates must be a list")
    _require(
        certificate_document.get("certificate_count") == len(certificates),
        "Recovery certificate_count mismatch",
    )
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    for certificate in certificates:
        key = (certificate.get("domain"), certificate.get("observation_sha256"))
        _require(key not in actual, f"Duplicate recovery certificate {key}")
        actual[key] = certificate
    _require(set(actual) == set(expected), "Recovery certificate coverage mismatch")
    for key, expected_record in expected.items():
        _require(actual[key] == expected_record, f"Recovery certificate mismatch for {key}")


def _verify_artifact_hashes(result_dir: Path, validation: dict[str, Any]) -> dict[str, str]:
    artifacts = validation.get("artifacts")
    _require(isinstance(artifacts, dict), "validation.artifacts is missing")
    expected_names = {
        "observation_recovery_manifest.jsonl",
        "recovery_certificates.json",
        "summary.json",
    }
    _require(set(artifacts) == expected_names, "validation.artifacts has the wrong file set")
    actual = {name: _sha256(result_dir / name) for name in sorted(expected_names)}
    _require(actual == artifacts, "Recovery artifact SHA256 mismatch")
    return actual


def verify_observation_recovery(
    frozen_manifest_path: Path,
    frozen_certificates_path: Path,
    result_dir: Path,
    *,
    expected_frozen_manifest_sha256: str | None = FROZEN_MANIFEST_SHA256,
    expected_frozen_certificates_sha256: str | None = FROZEN_CERTIFICATES_SHA256,
    expected_scene_count: int | None = EXPECTED_SCENE_COUNT,
    expected_certificate_count: int | None = EXPECTED_CERTIFICATE_COUNT,
) -> dict[str, Any]:
    """Verify one smoke or full v0.2 artifact directory independently."""

    frozen_manifest_sha = _sha256(frozen_manifest_path)
    frozen_certificate_sha = _sha256(frozen_certificates_path)
    if expected_frozen_manifest_sha256 is not None:
        _require(
            frozen_manifest_sha == expected_frozen_manifest_sha256,
            "Frozen v0.1 manifest SHA256 mismatch",
        )
    if expected_frozen_certificates_sha256 is not None:
        _require(
            frozen_certificate_sha == expected_frozen_certificates_sha256,
            "Frozen v0.1 certificates SHA256 mismatch",
        )

    frozen_rows = _read_jsonl(frozen_manifest_path)
    if expected_scene_count is not None:
        _require(len(frozen_rows) == expected_scene_count, "Frozen scene count mismatch")
    scene_ids = [row.get("scene_id") for row in frozen_rows]
    _require(
        None not in scene_ids and len(set(scene_ids)) == len(scene_ids),
        "Frozen scene IDs are missing or duplicated",
    )
    frozen_certificates = _read_json(frozen_certificates_path)
    _verify_frozen_certificates(
        frozen_rows, frozen_certificates, expected_certificate_count
    )
    classes = _candidate_classes(frozen_rows)

    recovery_manifest_path = result_dir / "observation_recovery_manifest.jsonl"
    recovery_certificate_path = result_dir / "recovery_certificates.json"
    summary_path = result_dir / "summary.json"
    validation_path = result_dir / "validation.json"
    recovery_rows = _read_jsonl(recovery_manifest_path)
    recovery_certificates = _read_json(recovery_certificate_path)
    summary = _read_json(summary_path)
    validation = _read_json(validation_path)
    artifact_hashes = _verify_artifact_hashes(result_dir, validation)

    _require(summary.get("schema_version") == RECOVERY_SCHEMA_VERSION, "Summary schema mismatch")
    _require(validation.get("schema_version") == RECOVERY_SCHEMA_VERSION, "Validation schema mismatch")
    _require(summary.get("mode") in {"smoke", "full"}, "Unknown audit mode")
    _require(validation.get("mode") == summary["mode"], "Audit mode mismatch")
    _require(summary.get("active_strategy_b_prior_ids") == PRIOR_IDS, "Summary prior set mismatch")

    row_index: dict[tuple[str, str], dict[str, Any]] = {}
    domain_decisions: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {"strategy_a": [], "strategy_b": []} for domain in DOMAINS
    }
    expected_recovery_certificates: dict[tuple[str, str], dict[str, Any]] = {}
    strategy_a_truth_agreement = 0
    strategy_b_truth_agreement = 0
    strategy_b_truth_exclusion = 0
    ambiguity_taxonomy_counts: Counter[str] = Counter()

    for row in recovery_rows:
        _require(row.get("schema_version") == RECOVERY_SCHEMA_VERSION, "Manifest row schema mismatch")
        domain = row.get("domain")
        observation = row.get("observation_sha256")
        _require(domain in DOMAINS and isinstance(observation, str), "Malformed recovery row key")
        key = (domain, observation)
        _require(key not in row_index, f"Duplicate recovery row {key}")
        _require(observation in classes[domain], f"Recovery row is outside frozen domain: {key}")
        row_index[key] = row
        candidates = classes[domain][observation]
        kind = _ambiguity_taxonomy(domain, observation, classes)
        if kind != "none":
            ambiguity_taxonomy_counts[kind] += 1
        strategy_a = row.get("strategy_a")
        strategy_b = row.get("strategy_b")
        _require(isinstance(strategy_a, dict), f"Missing Strategy A for {key}")
        _require(isinstance(strategy_b, dict), f"Missing Strategy B for {key}")
        _verify_strategy_a(strategy_a, candidates, kind, f"{key}/strategy_a")
        _verify_strategy_b(strategy_b, candidates, kind, f"{key}/strategy_b")
        domain_decisions[domain]["strategy_a"].append(strategy_a)
        domain_decisions[domain]["strategy_b"].append(strategy_b)
        strategy_a_truth_agreement += strategy_a["survivor_count"]
        strategy_b_truth_agreement += strategy_b["survivor_count"]
        strategy_b_truth_exclusion += len(candidates) - strategy_b["survivor_count"]

        if strategy_a["result_type"] == "Ambiguous" or strategy_b["result_type"] != "Graph":
            expected_recovery_certificates[key] = {
                "candidate_graph_signatures": sorted(_graphs(candidates)),
                "candidate_scene_ids": sorted(candidate["scene_id"] for candidate in candidates),
                "domain": domain,
                "observation_sha256": observation,
                "strategy_a_ambiguity_kind": strategy_a["ambiguity_kind"],
                "strategy_a_result": strategy_a["result_type"],
                "strategy_b_result": strategy_b["result_type"],
                "strategy_b_survivor_count": strategy_b["survivor_count"],
            }

    _verify_recovery_certificates(recovery_certificates, expected_recovery_certificates)

    summary_domains = summary.get("domains")
    _require(isinstance(summary_domains, dict), "Summary domains are missing")
    for domain in DOMAINS:
        actual_observations = {
            observation for row_domain, observation in row_index if row_domain == domain
        }
        if summary["mode"] == "full":
            _require(
                actual_observations == set(classes[domain]),
                f"Full recovery manifest does not cover {domain}",
            )
        else:
            fixture_observations = {
                row["production_observation_sha256"]
                for row in frozen_rows
                if row.get("source") == "fixture"
                and row["exclusive_grammar_layer"] in DOMAINS[domain]
            }
            _require(
                fixture_observations <= actual_observations,
                f"Smoke recovery manifest omits a fixture in {domain}",
            )
        reported = summary_domains.get(domain)
        _require(isinstance(reported, dict), f"Missing summary for {domain}")
        _require(
            reported.get("audited_observation_count") == len(actual_observations),
            f"Audited observation count mismatch for {domain}",
        )
        _require(
            reported.get("full_domain_observation_count") == len(classes[domain]),
            f"Full-domain observation count mismatch for {domain}",
        )
        for strategy in ("strategy_a", "strategy_b"):
            expected_statistics = _decision_statistics(domain_decisions[domain][strategy])
            _require(
                reported.get(strategy) == expected_statistics,
                f"Summary statistics mismatch for {domain}/{strategy}",
            )

    _require(summary.get("audit_oracle_only") is True, "Summary is not audit-only")
    _require(summary.get("deployable_from_opaque_observation") is False, "Summary claims deployability")
    extractor = summary.get("observable_extractor")
    _require(isinstance(extractor, dict), "Summary extractor boundary is missing")
    _require(
        extractor.get("implemented_from_opaque_raster_or_tensor") is False
        and extractor.get("status") == "no_go_not_implemented",
        "Summary extractor boundary is not NO-GO",
    )
    input_record = summary.get("input")
    _require(isinstance(input_record, dict), "Summary input commitment is missing")
    _require(input_record.get("manifest_sha256") == frozen_manifest_sha, "Summary manifest SHA mismatch")
    _require(
        input_record.get("certificates_sha256") == frozen_certificate_sha,
        "Summary certificate SHA mismatch",
    )
    _require(input_record.get("manifest_scene_count") == len(frozen_rows), "Summary scene count mismatch")

    zero_survivor_by_domain = {
        domain: _decision_statistics(domain_decisions[domain]["strategy_b"])[
            "zero_survivor_observation_count"
        ]
        for domain in DOMAINS
    }
    total_zero_survivor = sum(zero_survivor_by_domain.values())
    left_inverse_failures = sum(
        decision["field_strict"] and decision["decode_left_inverse"] is not True
        for strategies in domain_decisions.values()
        for decisions in strategies.values()
        for decision in decisions
    )
    expected_strategy_a_status = (
        "passed" if summary["mode"] == "full" else "not_evaluated_smoke"
    )
    expected_strategy_b_status = (
        "passed"
        if summary["mode"] == "full" and total_zero_survivor == 0
        else "no_go"
    )
    _require(
        summary.get("strategy_a_finite_spec_status") == expected_strategy_a_status,
        "Summary Strategy A status mismatch",
    )
    summary_strategy_b = summary.get("strategy_b_global_prior_status")
    _require(isinstance(summary_strategy_b, dict), "Summary Strategy B status is missing")
    _require(
        summary_strategy_b.get("status") == expected_strategy_b_status
        and summary_strategy_b.get("zero_survivor_by_domain")
        == zero_survivor_by_domain,
        "Summary Strategy B status/count mismatch",
    )
    summary_left_inverse = summary.get("semantic_serialization_left_inverse")
    _require(isinstance(summary_left_inverse, dict), "Summary left-inverse result is missing")
    _require(
        summary_left_inverse.get("failure_count") == left_inverse_failures
        and summary_left_inverse.get("status")
        == ("passed" if left_inverse_failures == 0 else "failed"),
        "Summary left-inverse result mismatch",
    )

    _require(validation.get("training_authorized") is False, "Training was incorrectly authorized")
    _require(validation.get("overall_gate_status") == "no_go", "Overall gate is not NO-GO")
    _require(validation.get("audit_oracle_only") is True, "Validation is not audit-only")
    _require(
        validation.get("deployable_from_opaque_observation") is False,
        "Validation claims opaque-observation deployability",
    )
    validation_extractor = validation.get("observable_extractor_factorization")
    _require(isinstance(validation_extractor, dict), "Validation extractor boundary is missing")
    _require(
        validation_extractor.get("status") == "no_go_not_implemented",
        "Validation extractor factorization is not NO-GO",
    )
    checks = validation.get("checks")
    _require(isinstance(checks, dict), "Validation checks are missing")
    expected_full_counts = all(
        summary_domains[domain]["full_domain_observation_count"]
        == len(classes[domain])
        for domain in DOMAINS
    )
    expected_checks = {
        "finite_domain_observation_counts_match_v0_1": expected_full_counts,
        "frozen_certificate_sha256_matches": (
            expected_frozen_certificates_sha256 is None
            or frozen_certificate_sha == expected_frozen_certificates_sha256
        ),
        "frozen_manifest_has_66716_unique_scenes": (
            expected_scene_count is None or len(frozen_rows) == expected_scene_count
        ),
        "frozen_manifest_sha256_matches": (
            expected_frozen_manifest_sha256 is None
            or frozen_manifest_sha == expected_frozen_manifest_sha256
        ),
        "oracle_field_left_inverse_on_field_strict": left_inverse_failures == 0,
        "strategy_a_ambiguous_class_graph_output_count_is_zero": True,
        "strategy_a_regression_matches_preregistration": summary["mode"] == "full",
        "strategy_a_wrong_graph_output_count_is_zero": True,
        "strategy_b_coverage_is_100_percent": total_zero_survivor == 0,
        "strategy_b_survivor_graph_collision_count_is_zero": True,
    }
    _require(
        set(expected_checks) <= set(checks),
        "Validation omits one or more required independent checks",
    )
    for name, expected in expected_checks.items():
        _require(checks[name] is expected, f"Validation check mismatch: {name}")
    finite_policy = validation.get("finite_policy_consistency")
    if finite_policy is not None:
        _require(isinstance(finite_policy, dict), "Finite-policy result is malformed")
        expected_policy_status = (
            "passed"
            if summary["mode"] == "full" and total_zero_survivor == 0
            else "no_go"
        )
        _require(
            finite_policy.get("status") == expected_policy_status
            and finite_policy.get("strategy_b_zero_survivor_observation_count")
            == total_zero_survivor,
            "Finite-policy status/count mismatch",
        )
    validation_strategy_a = validation.get("strategy_a_finite_spec_status")
    if validation_strategy_a is not None:
        _require(
            isinstance(validation_strategy_a, dict)
            and validation_strategy_a.get("status") == expected_strategy_a_status
            and validation_strategy_a.get("wrong_graph_output_count") == 0
            and validation_strategy_a.get("ambiguous_class_graph_output_count") == 0,
            "Validation Strategy A status mismatch",
        )
    validation_strategy_b = validation.get("strategy_b_global_prior_status")
    if validation_strategy_b is not None:
        _require(
            isinstance(validation_strategy_b, dict)
            and validation_strategy_b.get("status") == expected_strategy_b_status
            and validation_strategy_b.get("survivor_graph_collision_count") == 0
            and validation_strategy_b.get("zero_survivor_by_domain")
            == zero_survivor_by_domain
            and validation_strategy_b.get("zero_survivor_observation_count")
            == total_zero_survivor,
            "Validation Strategy B status/count mismatch",
        )
    validation_left_inverse = validation.get("semantic_serialization_left_inverse")
    _require(
        isinstance(validation_left_inverse, dict)
        and validation_left_inverse.get("failure_count") == left_inverse_failures
        and validation_left_inverse.get("status")
        == ("passed" if left_inverse_failures == 0 else "failed"),
        "Validation left-inverse result mismatch",
    )

    total_candidate_truths = sum(
        len(classes[domain][observation]) for domain, observation in row_index
    )
    _require(
        strategy_b_truth_agreement + strategy_b_truth_exclusion == total_candidate_truths,
        "Strategy B truth agreement/exclusion accounting mismatch",
    )
    script_path = Path(__file__).resolve()
    return {
        "artifact_sha256": artifact_hashes,
        "ambiguity_taxonomy_counts": dict(sorted(ambiguity_taxonomy_counts.items())),
        "audit_mode": summary["mode"],
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "frozen_certificates_sha256": frozen_certificate_sha,
        "frozen_manifest_sha256": frozen_manifest_sha,
        "observation_extractor_status": "no_go_not_implemented",
        "recovery_manifest_row_count": len(recovery_rows),
        "reviewer_script_sha256": _sha256(script_path),
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": "passed",
        "strategy_a_truth_agreement_scene_count": strategy_a_truth_agreement,
        "strategy_b_truth_agreement_scene_count": strategy_b_truth_agreement,
        "strategy_b_truth_exclusion_scene_count": strategy_b_truth_exclusion,
        "strategy_b_zero_survivor_observation_count": total_zero_survivor,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    frozen_root = root / "results" / "finite_geometry_grammar_v0_1" / "final_a"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, default=frozen_root / "scene_manifest.jsonl")
    parser.add_argument(
        "--frozen-certificates",
        type=Path,
        default=frozen_root / "collision_certificates.json",
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_observation_recovery(
        args.frozen_manifest, args.frozen_certificates, args.result_dir
    )
    output = _json_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(output.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
