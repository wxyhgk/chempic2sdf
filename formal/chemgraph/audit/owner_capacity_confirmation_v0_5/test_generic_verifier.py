from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path

from . import generic_verifier as verifier


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
R0_ROOT = REPOSITORY_ROOT / "formal/chemgraph/audit/owner_candidate_domain_v0_5"
R0_OBSERVATION = R0_ROOT / "r0_observation.json"
R0_CANDIDATE = R0_ROOT / "r0_candidate_domain.json"
R0_CONSTRUCTOR = R0_ROOT / "candidate_domain.py"
R0_ANNOTATION = (
    REPOSITORY_ROOT
    / "formal/chemgraph/audit/png_ownership_realization_v0_5/representative_witness.json"
)

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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class GenericVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.observation_raw = R0_OBSERVATION.read_bytes()
        cls.candidate_raw = R0_CANDIDATE.read_bytes()
        cls.observation = verifier.load_canonical_json(cls.observation_raw, "r0_observation")
        cls.candidate = verifier.load_canonical_json(cls.candidate_raw, "r0_candidate")
        cls.constructor_sha = _sha256(R0_CONSTRUCTOR.read_bytes())
        cls.expected, cls.context = verifier.rebuild_expected_artifact(
            cls.observation, LIMITS, cls.constructor_sha
        )

    def test_frozen_preregistration_anchors_match_committed_bytes(self) -> None:
        self.assertEqual(
            _sha256(ROOT.joinpath("axis_plan.json").read_bytes()),
            verifier.FROZEN_PLAN_SHA256,
        )
        self.assertEqual(
            _sha256(ROOT.joinpath("source_manifest.json").read_bytes()),
            verifier.FROZEN_MANIFEST_SHA256,
        )
        self.assertEqual(
            _sha256(ROOT.joinpath("EXPLORATORY_EXCLUSION.md").read_bytes()),
            verifier.FROZEN_EXCLUSION_SHA256,
        )

    def test_verifier_has_only_stdlib_imports_and_no_producer_imports(self) -> None:
        source = ROOT.joinpath("generic_verifier.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        allowed = {
            "__future__",
            "argparse",
            "binascii",
            "bisect",
            "hashlib",
            "json",
            "struct",
            "zlib",
            "collections",
            "fractions",
            "pathlib",
            "typing",
        }
        self.assertTrue(imported.issubset(allowed), imported - allowed)
        for forbidden in (
            "source_builder",
            "build_axis_plan",
            "candidate_domain",
            "build_observation",
            "fragment_root",
            "support_partition",
        ):
            self.assertNotIn(f"import {forbidden}", source)
            self.assertNotIn(f"from {forbidden}", source)
        self.assertNotIn("itertools", source)
        self.assertNotIn("product(", source)

    def test_independent_rebuild_matches_frozen_r0_artifact(self) -> None:
        self.assertEqual(self.expected, self.candidate)
        verifier.validate_candidate_envelope(
            self.candidate, self.candidate_raw, LIMITS, self.constructor_sha
        )

    def test_support_png_round_trip_is_exact(self) -> None:
        png = verifier.render_support_png(self.observation["parent_supports"])
        decoded = verifier.decode_support_png(png)
        self.assertEqual(
            decoded,
            verifier._support_union(self.observation["parent_supports"]),
        )
        corrupted = bytearray(png)
        corrupted[-8] ^= 1
        with self.assertRaises(verifier.GenericCapacityVerificationError):
            verifier.decode_support_png(bytes(corrupted))

    def test_candidate_section_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.candidate)
        mutated["counts"]["carrier_count"] += 1
        raw = verifier.canonical_bytes(mutated)
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError,
            "CANDIDATE_SECTION_HASH_MISMATCH",
        ):
            verifier.validate_candidate_envelope(
                mutated, raw, LIMITS, self.constructor_sha
            )

    def test_candidate_extra_field_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.candidate)
        mutated["graph"] = {}
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError, "CANDIDATE_NOT_CLOSED"
        ):
            verifier.validate_candidate_envelope(
                mutated, verifier.canonical_bytes(mutated), LIMITS, self.constructor_sha
            )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError, "JSON_DUPLICATE_KEY"
        ):
            verifier.load_canonical_json(b'{"a":1,"a":2}\n', "duplicate")

    def test_bool_identifier_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.observation)
        mutated["geometry_root"]["base_strokes"][0]["id"] = True
        with self.assertRaises(verifier.GenericCapacityVerificationError):
            verifier.rebuild_expected_artifact(mutated, LIMITS, self.constructor_sha)

    def _r0_feasible_witness(self) -> tuple[dict[str, object], bytes]:
        annotation = json.loads(R0_ANNOTATION.read_bytes())["configuration"]
        partition = annotation["partition_annotations"][0]
        owner_by_pixel = {}
        for owner_id, field in ((0, "left_region_runs"), (1, "right_region_runs")):
            for pixel in verifier._decode_runs(partition[field]):
                owner_by_pixel[pixel] = owner_id
        classes = {
            item["id"]: tuple(item["owner_option_ids"])
            for item in self.candidate["domain_classes"]
        }
        overrides = []
        for variable in self.candidate["pixel_variables"]:
            domain = classes[variable["domain_class_id"]]
            if len(domain) > 1:
                overrides.append(
                    {
                        "owner_option_id": owner_by_pixel[tuple(variable["pixel"])],
                        "variable_id": variable["id"],
                    }
                )
        source_sha = _sha256(self.observation_raw)
        witness = {
            "ambiguous_overrides": overrides,
            "artifact_sha256": _sha256(self.candidate_raw),
            "assignment_mode": "singleton_plus_overrides",
            "axis": "regression_r0",
            "observation_sha256": source_sha,
            "schema_version": verifier.WITNESS_SCHEMA,
            "side": "L",
            "source_sha256": source_sha,
            "training_authorized": False,
        }
        return witness, verifier.canonical_bytes(witness)

    def test_feasible_world_certificate_is_checked_without_enumeration(self) -> None:
        witness, raw = self._r0_feasible_witness()
        result = verifier.verify_feasible_witness(
            self.candidate,
            self.context["domains"],
            witness,
            raw,
            axis="regression_r0",
            observation_sha256=_sha256(self.observation_raw),
            source_sha256=_sha256(self.observation_raw),
            artifact_sha256=_sha256(self.candidate_raw),
        )
        self.assertEqual(result["ambiguous_override_count"], 10)
        self.assertEqual(result["assignment_materialized_count"], 58)

    def test_incomplete_feasible_world_certificate_is_rejected(self) -> None:
        witness, _ = self._r0_feasible_witness()
        witness["ambiguous_overrides"] = witness["ambiguous_overrides"][:-1]
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError,
            "FEASIBLE_WITNESS_OVERRIDE_COVERAGE_MISMATCH",
        ):
            verifier.verify_feasible_witness(
                self.candidate,
                self.context["domains"],
                witness,
                verifier.canonical_bytes(witness),
                axis="regression_r0",
                observation_sha256=_sha256(self.observation_raw),
                source_sha256=_sha256(self.observation_raw),
                artifact_sha256=_sha256(self.candidate_raw),
            )

    def test_preregistered_rejection_sources_have_independent_first_gates(self) -> None:
        plan_raw = ROOT.joinpath("axis_plan.json").read_bytes()
        manifest_raw = ROOT.joinpath("source_manifest.json").read_bytes()
        exclusion_raw = ROOT.joinpath("EXPLORATORY_EXCLUSION.md").read_bytes()
        renderer_raw = ROOT.joinpath("capacity_png.py").read_bytes()
        cases = (
            ("max_carriers", "H", "CAPACITY_EXCEEDED:max_carriers", "max_carriers", 0),
            ("max_intersections", "L", "R0_REQUIRES_TWO_INTERSECTIONS", "R0_REQUIRES_TWO_INTERSECTIONS", 4),
            ("max_intersections", "H", "CAPACITY_EXCEEDED:max_intersections", "max_intersections", 1),
            ("max_pixel_variables", "H", "CAPACITY_EXCEEDED:max_pixel_variables", "max_pixel_variables", 6),
            ("max_total_rle_runs", "H", "CAPACITY_EXCEEDED:max_total_rle_runs", "max_total_rle_runs", 3),
        )
        for axis, side, error, gate, index in cases:
            with self.subTest(axis=axis, side=side):
                result = verifier.verify_rejection_source(
                    axis=axis,
                    side=side,
                    expected_error=error,
                    plan_raw=plan_raw,
                    manifest_raw=manifest_raw,
                    exclusion_raw=exclusion_raw,
                    observation_raw=ROOT.joinpath(
                        f"witnesses/{axis}_{side}.observation.json"
                    ).read_bytes(),
                    renderer_raw=renderer_raw,
                    png_raw=ROOT.joinpath(
                        f"witnesses/{axis}_{side}.support.png"
                    ).read_bytes(),
                )
                self.assertEqual(result["first_failure_gate"], gate)
                self.assertEqual(result["first_failure_index"], index)
                self.assertTrue(result["target_count_recomputed_exact"])

    def test_rejection_verifier_does_not_accept_wrong_declared_error(self) -> None:
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError,
            "REJECTION_CASE_NOT_ALLOWLISTED",
        ):
            verifier.verify_rejection_source(
                axis="max_carriers",
                side="H",
                expected_error="CAPACITY_EXCEEDED:max_intersections",
                plan_raw=ROOT.joinpath("axis_plan.json").read_bytes(),
                manifest_raw=ROOT.joinpath("source_manifest.json").read_bytes(),
                exclusion_raw=ROOT.joinpath("EXPLORATORY_EXCLUSION.md").read_bytes(),
                observation_raw=ROOT.joinpath(
                    "witnesses/max_carriers_H.observation.json"
                ).read_bytes(),
                renderer_raw=ROOT.joinpath("capacity_png.py").read_bytes(),
                png_raw=ROOT.joinpath(
                    "witnesses/max_carriers_H.support.png"
                ).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
