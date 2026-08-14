"""Smoke and tamper tests for the independent Grammar v0.2 verifier."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .verify_observation_recovery_v0_2 import (
    PRIOR_IDS,
    VerificationError,
    verify_observation_recovery,
)


def _json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value) -> str:
    data = _json_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows) -> str:
    data = b"".join(
        (json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _frozen_row(scene, observation, graph, layer, source="core"):
    return {
        "canonical_graph_signature_structure": graph,
        "exclusive_grammar_layer": layer,
        "production_observation_sha256": observation,
        "scene_id": scene,
        "source": source,
    }


def _field(ambiguity="none"):
    return {
        "ambiguity": ambiguity,
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "observation_extractor_implemented": False,
    }


def _decision_a(rows, kind):
    graphs = {row["canonical_graph_signature_structure"] for row in rows}
    if len(graphs) > 1:
        return {
            "active_prior_ids": [],
            "ambiguity_kind": kind,
            "audit_oracle_only": True,
            "candidate_count": len(rows),
            "decode_left_inverse": None,
            "deployable_from_opaque_observation": False,
            "evidence_factorizes_on_finite_domain": True,
            "field_strict": False,
            "graph_signature": None,
            "observation_extractor_implemented": False,
            "oracle_field": _field(kind),
            "protocol_choice_graph_signature": None,
            "result_type": "Ambiguous",
            "semantic_field_factorizes_through_evidence": False,
            "survivor_count": len(rows),
            "survivor_graph_count": len(graphs),
        }
    graph = next(iter(graphs))
    return {
        "active_prior_ids": [],
        "ambiguity_kind": "none",
        "audit_oracle_only": True,
        "candidate_count": len(rows),
        "decode_left_inverse": True,
        "deployable_from_opaque_observation": False,
        "evidence_factorizes_on_finite_domain": True,
        "field_strict": True,
        "graph_signature": graph,
        "observation_extractor_implemented": False,
        "oracle_field": _field(),
        "protocol_choice_graph_signature": graph,
        "result_type": "Graph",
        "semantic_field_factorizes_through_evidence": True,
        "survivor_count": len(rows),
        "survivor_graph_count": 1,
    }


def _decision_b(rows, choice, kind):
    survivor_count = sum(
        row["canonical_graph_signature_structure"] == choice for row in rows
    )
    if survivor_count:
        return {
            "active_prior_ids": PRIOR_IDS,
            "ambiguity_kind": "none",
            "audit_oracle_only": True,
            "candidate_count": len(rows),
            "decode_left_inverse": True,
            "deployable_from_opaque_observation": False,
            "evidence_factorizes_on_finite_domain": True,
            "field_strict": True,
            "graph_signature": choice,
            "observation_extractor_implemented": False,
            "oracle_field": _field(),
            "protocol_choice_graph_signature": choice,
            "result_type": "Graph",
            "semantic_field_factorizes_through_evidence": True,
            "survivor_count": survivor_count,
            "survivor_graph_count": 1,
        }
    return {
        "active_prior_ids": PRIOR_IDS,
        "ambiguity_kind": kind,
        "audit_oracle_only": True,
        "candidate_count": len(rows),
        "decode_left_inverse": None,
        "deployable_from_opaque_observation": False,
        "evidence_factorizes_on_finite_domain": True,
        "field_strict": False,
        "graph_signature": None,
        "observation_extractor_implemented": False,
        "oracle_field": None,
        "protocol_choice_graph_signature": choice,
        "result_type": "Reject",
        "semantic_field_factorizes_through_evidence": True,
        "survivor_count": 0,
        "survivor_graph_count": 0,
    }


def _statistics(decisions):
    result_counts = Counter(decision["result_type"] for decision in decisions)
    ambiguity = Counter(
        decision["ambiguity_kind"]
        for decision in decisions
        if decision["ambiguity_kind"] != "none"
    )
    strict = [decision for decision in decisions if decision["field_strict"]]
    return {
        "ambiguity_kind_counts": dict(sorted(ambiguity.items())),
        "coverage_fraction": result_counts["Graph"] / len(decisions),
        "decode_left_inverse_denominator": len(strict),
        "decode_left_inverse_numerator": len(strict),
        "field_strict_observation_count": len(strict),
        "graph_output_count": result_counts["Graph"],
        "nonempty_survivor_observation_count": sum(d["survivor_count"] > 0 for d in decisions),
        "reject_count": result_counts["Reject"],
        "result_counts": dict(sorted(result_counts.items())),
        "survivor_graph_collision_count": sum(d["survivor_graph_count"] > 1 for d in decisions),
        "zero_survivor_observation_count": sum(d["survivor_count"] == 0 for d in decisions),
    }


class IndependentRecoveryVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result_dir = self.root / "result"
        self.result_dir.mkdir()
        simple = "simple_line_angle_valid"
        crossing = "crossing_allowed"
        self.frozen_rows = [
            _frozen_row("strict", "obs-strict", "G-strict", simple),
            _frozen_row("parallel-double", "obs-parallel", "G-double", simple, "fixture"),
            _frozen_row("parallel-singles", "obs-parallel", "G-singles", simple, "fixture"),
            _frozen_row("junction", "obs-cross", "G-junction", simple, "fixture"),
            _frozen_row("crossing", "obs-cross", "G-crossing", crossing, "fixture"),
            _frozen_row("reject", "obs-reject", "G-original", simple),
        ]
        self.frozen_manifest = self.root / "scene_manifest.jsonl"
        _write_jsonl(self.frozen_manifest, self.frozen_rows)
        grouped = {}
        for row in self.frozen_rows:
            grouped.setdefault(row["production_observation_sha256"], []).append(row)
        frozen_certs = []
        for observation in ("obs-parallel", "obs-cross"):
            rows = grouped[observation]
            frozen_certs.append(
                {
                    "graph_signature_count": len({r["canonical_graph_signature_structure"] for r in rows}),
                    "observation_sha256": observation,
                    "witnesses": [
                        {
                            "graph_signature": row["canonical_graph_signature_structure"],
                            "scene_id": row["scene_id"],
                        }
                        for row in rows
                    ],
                }
            )
        self.frozen_certificates = self.root / "collision_certificates.json"
        _write_json(
            self.frozen_certificates,
            {"certificate_count": len(frozen_certs), "certificates": frozen_certs},
        )
        self._write_valid_recovery()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_valid_recovery(self):
        allowed = {
            "simple_domain": {"simple_line_angle_valid"},
            "crossing_domain": {"simple_line_angle_valid", "crossing_allowed"},
        }
        choices = {
            "obs-strict": "G-strict",
            "obs-parallel": "G-double",
            "obs-cross": "G-crossing",
            "obs-reject": "G-protocol-outside-domain",
        }
        rows = []
        recovery_certificates = []
        domain_records = {}
        for domain, layers in allowed.items():
            grouped = {}
            for frozen in self.frozen_rows:
                if frozen["exclusive_grammar_layer"] in layers:
                    grouped.setdefault(frozen["production_observation_sha256"], []).append(frozen)
            decisions_a = []
            decisions_b = []
            for observation, candidates in sorted(grouped.items()):
                graphs = {row["canonical_graph_signature_structure"] for row in candidates}
                if len(graphs) == 1:
                    kind = "none"
                elif {row["exclusive_grammar_layer"] for row in candidates} == {"simple_line_angle_valid"}:
                    kind = "parallel_multiplicity"
                else:
                    kind = "crossing_junction"
                decision_a = _decision_a(candidates, kind)
                decision_b = _decision_b(candidates, choices[observation], kind)
                decisions_a.append(decision_a)
                decisions_b.append(decision_b)
                rows.append(
                    {
                        "domain": domain,
                        "observation_sha256": observation,
                        "schema_version": "observation-recovery-v0.2",
                        "strategy_a": decision_a,
                        "strategy_b": decision_b,
                    }
                )
                if decision_a["result_type"] == "Ambiguous" or decision_b["result_type"] != "Graph":
                    recovery_certificates.append(
                        {
                            "candidate_graph_signatures": sorted(graphs),
                            "candidate_scene_ids": sorted(row["scene_id"] for row in candidates),
                            "domain": domain,
                            "observation_sha256": observation,
                            "strategy_a_ambiguity_kind": decision_a["ambiguity_kind"],
                            "strategy_a_result": decision_a["result_type"],
                            "strategy_b_result": decision_b["result_type"],
                            "strategy_b_survivor_count": decision_b["survivor_count"],
                        }
                    )
            domain_records[domain] = {
                "audited_observation_count": len(grouped),
                "full_domain_observation_count": len(grouped),
                "strategy_a": _statistics(decisions_a),
                "strategy_b": _statistics(decisions_b),
            }
        rows.sort(key=lambda row: (row["domain"], row["observation_sha256"]))
        _write_jsonl(self.result_dir / "observation_recovery_manifest.jsonl", rows)
        _write_json(
            self.result_dir / "recovery_certificates.json",
            {
                "certificate_count": len(recovery_certificates),
                "certificates": sorted(
                    recovery_certificates,
                    key=lambda item: (item["domain"], item["observation_sha256"]),
                ),
                "schema_version": "observation-recovery-v0.2",
            },
        )
        frozen_manifest_sha = hashlib.sha256(self.frozen_manifest.read_bytes()).hexdigest()
        frozen_certificate_sha = hashlib.sha256(self.frozen_certificates.read_bytes()).hexdigest()
        summary = {
            "active_strategy_b_prior_ids": PRIOR_IDS,
            "audit_oracle_only": True,
            "deployable_from_opaque_observation": False,
            "domains": domain_records,
            "input": {
                "certificates_sha256": frozen_certificate_sha,
                "manifest_scene_count": len(self.frozen_rows),
                "manifest_sha256": frozen_manifest_sha,
            },
            "mode": "smoke",
            "observable_extractor": {
                "implemented_from_opaque_raster_or_tensor": False,
                "status": "no_go_not_implemented",
            },
            "schema_version": "observation-recovery-v0.2",
            "semantic_serialization_left_inverse": {
                "failure_count": 0,
                "status": "passed",
            },
            "strategy_a_finite_spec_status": "not_evaluated_smoke",
            "strategy_b_global_prior_status": {
                "status": "no_go",
                "zero_survivor_by_domain": {
                    domain: domain_records[domain]["strategy_b"][
                        "zero_survivor_observation_count"
                    ]
                    for domain in allowed
                },
            },
        }
        summary_sha = _write_json(self.result_dir / "summary.json", summary)
        artifacts = {
            "observation_recovery_manifest.jsonl": hashlib.sha256(
                (self.result_dir / "observation_recovery_manifest.jsonl").read_bytes()
            ).hexdigest(),
            "recovery_certificates.json": hashlib.sha256(
                (self.result_dir / "recovery_certificates.json").read_bytes()
            ).hexdigest(),
            "summary.json": summary_sha,
        }
        zero_by_domain = {
            domain: domain_records[domain]["strategy_b"][
                "zero_survivor_observation_count"
            ]
            for domain in allowed
        }
        total_zero = sum(zero_by_domain.values())
        _write_json(
            self.result_dir / "validation.json",
            {
                "artifacts": artifacts,
                "audit_oracle_only": True,
                "checks": {
                    "finite_domain_observation_counts_match_v0_1": True,
                    "frozen_certificate_sha256_matches": True,
                    "frozen_manifest_has_66716_unique_scenes": True,
                    "frozen_manifest_sha256_matches": True,
                    "oracle_field_left_inverse_on_field_strict": True,
                    "strategy_a_ambiguous_class_graph_output_count_is_zero": True,
                    "strategy_a_regression_matches_preregistration": False,
                    "strategy_a_wrong_graph_output_count_is_zero": True,
                    "strategy_b_coverage_is_100_percent": False,
                    "strategy_b_survivor_graph_collision_count_is_zero": True,
                },
                "deployable_from_opaque_observation": False,
                "finite_policy_consistency": {
                    "status": "no_go",
                    "strategy_b_zero_survivor_observation_count": total_zero,
                },
                "mode": "smoke",
                "observable_extractor_factorization": {"status": "no_go_not_implemented"},
                "overall_gate_status": "no_go",
                "schema_version": "observation-recovery-v0.2",
                "semantic_serialization_left_inverse": {
                    "failure_count": 0,
                    "status": "passed",
                },
                "strategy_a_finite_spec_status": {
                    "ambiguous_class_graph_output_count": 0,
                    "status": "not_evaluated_smoke",
                    "wrong_graph_output_count": 0,
                },
                "strategy_b_global_prior_status": {
                    "status": "no_go",
                    "survivor_graph_collision_count": 0,
                    "zero_survivor_by_domain": zero_by_domain,
                    "zero_survivor_observation_count": total_zero,
                },
                "training_authorized": False,
            },
        )

    def _verify(self):
        return verify_observation_recovery(
            self.frozen_manifest,
            self.frozen_certificates,
            self.result_dir,
            expected_frozen_manifest_sha256=None,
            expected_frozen_certificates_sha256=None,
            expected_scene_count=None,
            expected_certificate_count=None,
        )

    def _mutate_manifest(self, mutate):
        path = self.result_dir / "observation_recovery_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        manifest_sha = _write_jsonl(path, rows)
        validation_path = self.result_dir / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["artifacts"]["observation_recovery_manifest.jsonl"] = manifest_sha
        _write_json(validation_path, validation)

    def test_valid_smoke_covers_fixtures_and_reject_accounting(self):
        review = self._verify()
        self.assertEqual(review["status"], "passed")
        self.assertEqual(review["audit_mode"], "smoke")
        self.assertGreater(review["strategy_b_truth_exclusion_scene_count"], 0)
        self.assertFalse(review["training_authorized"])

    def test_reject_choice_must_be_outside_candidate_graphs(self):
        def mutate(rows):
            row = next(
                item
                for item in rows
                if item["domain"] == "simple_domain"
                and item["observation_sha256"] == "obs-reject"
            )
            row["strategy_b"]["protocol_choice_graph_signature"] = "G-original"

        self._mutate_manifest(mutate)
        with self.assertRaisesRegex(VerificationError, "survivor_count mismatch"):
            self._verify()

    def test_strategy_a_cannot_guess_crossing_ambiguity(self):
        def mutate(rows):
            row = next(
                item
                for item in rows
                if item["domain"] == "crossing_domain"
                and item["observation_sha256"] == "obs-cross"
            )
            row["strategy_a"]["result_type"] = "Graph"
            row["strategy_a"]["graph_signature"] = "G-crossing"

        self._mutate_manifest(mutate)
        with self.assertRaisesRegex(VerificationError, "not rejected conservatively"):
            self._verify()

    def test_crossing_taxonomy_cannot_be_relabeled_parallel(self):
        def mutate(rows):
            row = next(
                item
                for item in rows
                if item["domain"] == "crossing_domain"
                and item["observation_sha256"] == "obs-cross"
            )
            row["strategy_a"]["ambiguity_kind"] = "parallel_multiplicity"

        self._mutate_manifest(mutate)
        with self.assertRaisesRegex(VerificationError, "taxonomy mismatch"):
            self._verify()

    def test_artifact_hash_tampering_is_detected(self):
        with (self.result_dir / "summary.json").open("ab") as output:
            output.write(b" ")
        with self.assertRaisesRegex(VerificationError, "artifact SHA256 mismatch"):
            self._verify()

    def test_training_and_extractor_boundary_is_hard(self):
        validation_path = self.result_dir / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["training_authorized"] = True
        _write_json(validation_path, validation)
        with self.assertRaisesRegex(VerificationError, "Training was incorrectly authorized"):
            self._verify()

    def test_summary_zero_survivor_count_tampering_is_detected(self):
        summary_path = self.result_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["strategy_b_global_prior_status"]["zero_survivor_by_domain"][
            "simple_domain"
        ] += 1
        summary_sha = _write_json(summary_path, summary)
        validation_path = self.result_dir / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["artifacts"]["summary.json"] = summary_sha
        _write_json(validation_path, validation)
        with self.assertRaisesRegex(VerificationError, "Summary Strategy B status/count mismatch"):
            self._verify()

    def test_validation_coverage_check_tampering_is_detected(self):
        validation_path = self.result_dir / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["checks"]["strategy_b_coverage_is_100_percent"] = True
        _write_json(validation_path, validation)
        with self.assertRaisesRegex(VerificationError, "strategy_b_coverage_is_100_percent"):
            self._verify()

    def test_validation_left_inverse_tampering_is_detected(self):
        validation_path = self.result_dir / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["semantic_serialization_left_inverse"]["failure_count"] = 1
        _write_json(validation_path, validation)
        with self.assertRaisesRegex(VerificationError, "Validation left-inverse result mismatch"):
            self._verify()

    def test_verifier_imports_only_standard_library(self):
        verifier_path = Path(__file__).with_name("verify_observation_recovery_v0_2.py")
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or "").split(".")[0])
        self.assertEqual(
            set(imports),
            {"__future__", "argparse", "collections", "hashlib", "json", "pathlib", "typing"},
        )


if __name__ == "__main__":
    unittest.main()
