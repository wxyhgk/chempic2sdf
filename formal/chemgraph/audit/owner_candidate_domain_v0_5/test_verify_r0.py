from __future__ import annotations

import ast
import copy
import itertools
import json
import unittest

from .verify_r0 import (
    CANDIDATE_PATH,
    OBSERVATION_PATH,
    ROOT,
    R0VerificationError,
    canonical_bytes,
    verify_frozen_r0,
    verify_payload,
)


class IndependentR0VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = json.loads(OBSERVATION_PATH.read_bytes())
        self.candidate = json.loads(CANDIDATE_PATH.read_bytes())

    def test_verifier_import_boundary(self) -> None:
        source = ROOT.joinpath("verify_r0.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden = (
            "owner_candidate_domain_v0_5.candidate_domain",
            "owner_candidate_domain_v0_5.build_observation",
            "owner_candidate_domain_v0_5.common",
        )
        self.assertFalse(any(any(token in name for token in forbidden) for name in imported))

    def test_frozen_verification_replays_twice_byte_exact(self) -> None:
        first = canonical_bytes(verify_frozen_r0())
        second = canonical_bytes(verify_frozen_r0())
        self.assertEqual(first, second)
        self.assertEqual(ROOT.joinpath("r0_verification.json").read_bytes(), first)

    def test_whole_world_truth_table_is_exact(self) -> None:
        result = verify_payload(self.observation, self.candidate)
        self.assertEqual(result["pointwise_assignment_count"], 1024)
        self.assertEqual(result["legal_assignment_count"], 873)
        self.assertEqual(result["candidate_assignment_count"], 873)
        self.assertEqual(result["sound_count"], 873)
        self.assertEqual(result["complete_count"], 873)
        self.assertTrue(result["candidate_nodup"])
        self.assertTrue(result["whole_world_exact"])
        self.assertEqual(result["false_singleton_count"], 0)
        self.assertEqual(result["false_multi_option_count"], 0)
        self.assertEqual(result["stored_annotation_coverage"], [2, 2])

    def test_candidate_mutations_are_closed_rejected(self) -> None:
        mutations = []

        dummy_owner = copy.deepcopy(self.candidate)
        dummy_owner["owner_options"].append({
            "anchor_pixel": [0, 0], "geometry_intersection_id": 2, "id": 2,
            "incident_carrier_ids": [1, 2], "kind": "intersection",
        })
        mutations.append(dummy_owner)

        false_singleton = copy.deepcopy(self.candidate)
        false_singleton["domain_classes"][1]["owner_option_ids"] = [0]
        mutations.append(false_singleton)

        missing_edge = copy.deepcopy(self.candidate)
        missing_edge["adjacency_edges"].pop()
        mutations.append(missing_edge)

        missing_constraint = copy.deepcopy(self.candidate)
        missing_constraint["constraints"].pop()
        mutations.append(missing_constraint)

        residual = copy.deepcopy(self.candidate)
        residual["residual_reasons"] = [{"code": "UNKNOWN"}]
        mutations.append(residual)

        false_count = copy.deepcopy(self.candidate)
        false_count["counts"]["pixel_variable_count"] = 57
        mutations.append(false_count)

        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(R0VerificationError):
                    verify_payload(self.observation, mutation)

    def test_oracle_field_in_observation_is_rejected(self) -> None:
        mutated = {**self.observation, "partition_annotations": []}
        with self.assertRaisesRegex(R0VerificationError, "OBSERVATION_REBUILD_MISMATCH"):
            verify_payload(mutated, self.candidate)

    def test_pointwise_product_is_not_the_legal_world_set(self) -> None:
        # Lseed-a-b-Rseed, with both middle variables locally allowing L/R.
        assignments = list(itertools.product((0, 1), repeat=2))

        def legal(middle: tuple[int, int]) -> bool:
            full = (0, middle[0], middle[1], 1)
            left = {index for index, owner in enumerate(full) if owner == 0}
            right = {index for index, owner in enumerate(full) if owner == 1}
            connected = lambda values: max(values) - min(values) + 1 == len(values)
            return connected(left) and connected(right)

        legal_assignments = [item for item in assignments if legal(item)]
        self.assertEqual(len(assignments), 4)
        self.assertEqual(len(legal_assignments), 3)
        self.assertNotIn((1, 0), legal_assignments)

    def test_known_owner_relabel_and_half_turn_are_equivariant(self) -> None:
        classes = {
            item["id"]: tuple(item["owner_option_ids"])
            for item in self.candidate["domain_classes"]
        }
        by_pixel = {
            tuple(item["pixel"]): classes[item["domain_class_id"]]
            for item in self.candidate["pixel_variables"]
        }
        for (row, column), domain in by_pixel.items():
            transformed_pixel = 39 - row, 39 - column
            transformed_domain = tuple(sorted(1 - owner for owner in domain))
            self.assertEqual(by_pixel[transformed_pixel], transformed_domain)

    def test_universal_baseline_alone_is_vacuous(self) -> None:
        raw = list(itertools.product((0, 1), repeat=4))
        self.assertEqual(len(raw), 16)
        # A universal pointwise domain covers every partition, including worlds
        # that violate any later connectivity or chemistry constraint.
        self.assertTrue(all(all(owner in (0, 1) for owner in item) for item in raw))


if __name__ == "__main__":
    unittest.main()
