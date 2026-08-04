"""Tests for render recoverability geometry."""

import unittest

import numpy as np
import torch

from chemvector.evaluation.audits.atoms.audit_g3_p2g_render_recoverability import (
    point_segment_distance,
    sample_ambiguities,
)
from chemvector.evaluation.audits.atoms.evaluate_g3_p2g_recoverability_buckets import (
    route_statuses,
    summarize_buckets,
)


class P2GRenderRecoverabilityTests(unittest.TestCase):
    @staticmethod
    def render(*, second_x: float = 20.0, crossing_bonds: int = 0) -> dict:
        bonds = []
        for bond_id in range(crossing_bonds):
            offset = float(bond_id)
            bonds.append({
                "bond_id": bond_id,
                "begin_image_x": 5.0,
                "begin_image_y": 9.5 + offset,
                "end_image_x": 15.0,
                "end_image_y": 10.5 - offset,
            })
        return {
            "atoms": [
                {"atom_id": 0, "image_x": 10.0, "image_y": 10.0},
                {"atom_id": 1, "image_x": second_x, "image_y": 10.0},
            ],
            "bonds": bonds,
            "render_parameters": {"bond_line_width": 2.0},
        }

    @staticmethod
    def layers(*, first_visible: bool = False, second_visible: bool = False) -> dict:
        return {
            "atoms": [
                {"atom_id": 0, "visible_text": first_visible},
                {"atom_id": 1, "visible_text": second_visible},
            ]
        }

    def test_point_segment_distance_reports_interior_projection(self) -> None:
        distance, projection = point_segment_distance(
            np.asarray([5.0, 2.0]), np.asarray([0.0, 0.0]), np.asarray([10.0, 0.0])
        )
        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(projection, 0.5)

    def test_visible_text_prevents_near_anchor_alias(self) -> None:
        result = sample_ambiguities(
            self.render(second_x=12.0),
            self.layers(first_visible=True, second_visible=True),
        )
        self.assertTrue(result["raw_geometry_flag"])
        self.assertFalse(result["ambiguous"])
        self.assertEqual(result["recoverability_status"], "eligible")

    def test_hidden_near_anchors_are_ambiguous(self) -> None:
        result = sample_ambiguities(
            self.render(second_x=12.0), self.layers()
        )
        self.assertEqual(len(result["hidden_alias_pairs"]), 1)
        self.assertEqual(result["recoverability_status"], "ambiguous")

    def test_single_crossing_is_only_raw_geometry_evidence(self) -> None:
        result = sample_ambiguities(
            self.render(crossing_bonds=1), self.layers()
        )
        self.assertTrue(result["raw_geometry_flag"])
        self.assertFalse(result["ambiguous"])

    def test_two_crossings_make_hidden_atom_ambiguous(self) -> None:
        result = sample_ambiguities(
            self.render(crossing_bonds=2), self.layers()
        )
        self.assertEqual(
            result["multiply_occluded_hidden_atoms"],
            [{"atom_id": 0, "nonincident_bond_count": 2}],
        )
        self.assertEqual(result["recoverability_status"], "ambiguous")

    def test_hidden_alias_and_crossings_are_severe(self) -> None:
        result = sample_ambiguities(
            self.render(second_x=12.0, crossing_bonds=2), self.layers()
        )
        self.assertEqual(result["recoverability_status"], "severe")

    def test_layer_atom_ids_must_match_render(self) -> None:
        with self.assertRaisesRegex(ValueError, "Atom IDs differ"):
            sample_ambiguities(
                self.render(), {"atoms": [{"atom_id": 0, "visible_text": False}]}
            )

    def test_route_statuses_default_to_eligible(self) -> None:
        audit = {
            "results": {"route100": {"flagged": [{
                "sample_key": "hard", "recoverability_status": "severe"
            }]}}
        }
        self.assertEqual(
            route_statuses(audit, {"easy", "hard"}),
            {"easy": "eligible", "hard": "severe"},
        )

    def test_bucket_summary_keeps_full_and_parallel_metrics(self) -> None:
        records = {
            "easy": {
                "expected": torch.tensor([[0.1, 0.1]]),
                "anchors": torch.tensor([[0.1, 0.1]]),
            },
            "hard": {
                "expected": torch.tensor([[0.2, 0.2], [0.8, 0.8]]),
                "anchors": torch.tensor([[0.2, 0.2], [0.5, 0.5]]),
            },
        }
        result = summarize_buckets(
            records, {"easy": "eligible", "hard": "severe"}, budgets=(2,)
        )["2"]
        self.assertEqual(result["all"]["matched_atoms"], 2)
        self.assertEqual(result["all"]["target_atoms"], 3)
        self.assertEqual(result["eligible"]["exact_images"], 1)
        self.assertEqual(result["severe"]["failures"][0]["sample_key"], "hard")


if __name__ == "__main__":
    unittest.main()
