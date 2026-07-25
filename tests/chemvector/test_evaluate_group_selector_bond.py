import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from chemvector.evaluate_group_selector_bond import (
    bond_length_bucket,
    ground_truth_edges,
    load_evaluation_selector,
    normalized_thresholds,
    selector_candidate_protocol,
    match_vertices_by_role,
    resolved_real_thresholds,
    role_candidate_coverage,
)
from chemvector.dual_graph_denoiser import (
    DualGraphDenoiserConfig,
    GroundedDualGraphDenoiser,
)
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)


class GroupSelectorBondEvaluationTests(unittest.TestCase):
    def test_selector_loader_supports_structured_and_graph_checkpoints(self) -> None:
        config = StructuredSelectorConfig(d_model=24, heads=4, ffn_dim=48, layers=1)
        structured_model = GroundedStructuredSelector(config)
        graph_denoiser_config = DualGraphDenoiserConfig(
            d_model=24, edge_dim=12, layers=1
        )
        graph_model = GroundedDualGraphDenoiser(
            GroundedStructuredSelector(config), graph_denoiser_config
        )
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base_checkpoint = directory / "base.pt"
            graph_checkpoint = directory / "graph.pt"
            torch.save(
                {
                    "structured_selector_config": asdict(config),
                    "dual_vertex_selector": structured_model.state_dict(),
                },
                base_checkpoint,
            )
            torch.save(
                {
                    "dual_graph_denoiser": graph_model.state_dict(),
                    "dual_graph_denoiser_config": asdict(graph_denoiser_config),
                    "base_selector_checkpoint": str(base_checkpoint),
                },
                graph_checkpoint,
            )
            structured, structured_config, _, structured_base = load_evaluation_selector(
                str(base_checkpoint), torch.device("cpu")
            )
            graph, graph_config, _, graph_base = load_evaluation_selector(
                str(graph_checkpoint), torch.device("cpu")
            )
        self.assertIsInstance(structured, GroundedStructuredSelector)
        self.assertIsNone(structured_base)
        self.assertIsInstance(graph, GroundedDualGraphDenoiser)
        self.assertEqual(structured_config.maximum_candidates, graph_config.maximum_candidates)
        self.assertIsNotNone(graph_base)

    def test_bond_length_buckets_have_inclusive_boundaries(self) -> None:
        self.assertEqual(bond_length_bucket(48.0), "short_le_48")
        self.assertEqual(bond_length_bucket(48.01), "medium_48_96")
        self.assertEqual(bond_length_bucket(96.0), "medium_48_96")
        self.assertEqual(bond_length_bucket(96.01), "long_gt_96")

    def test_threshold_grid_keeps_checkpoint_threshold(self) -> None:
        self.assertEqual(normalized_thresholds(0.85, [0.8, 0.9]), [0.8, 0.85, 0.9])
        with self.assertRaises(ValueError):
            normalized_thresholds(0.85, [1.0])

    def test_legacy_selector_checkpoint_uses_training_protocol(self) -> None:
        protocol = selector_candidate_protocol({})
        self.assertEqual(protocol["text_threshold"], 0.05)
        self.assertEqual(protocol["text_filter_threshold"], 0.3)
        self.assertEqual(protocol["suppression_radius_pixels"], 6.0)

    def test_role_aware_matching_rejects_cross_role_coordinate_match(self) -> None:
        assignment = match_vertices_by_role(
            torch.tensor([[0.1, 0.1], [0.5, 0.5]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.1, 0.1], [0.5, 0.5]]),
            torch.tensor([1, 1]),
        )
        self.assertEqual(assignment, {1: 1})

    def test_role_thresholds_fall_back_to_shared_value(self) -> None:
        self.assertEqual(
            resolved_real_thresholds(0.7, 0.35, None).tolist(),
            [0.3499999940395355, 0.699999988079071],
        )

    def test_role_candidate_coverage_separates_generation_from_selection(self) -> None:
        rows = role_candidate_coverage(
            torch.tensor([[0.2, 0.5], [0.8, 0.5], [0.5, 0.5]]),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([0, 1, 0]),
            torch.tensor([[0.2, 0.5], [0.8, 0.5], [0.5, 0.8]]),
            torch.tensor([0, 0, 0]),
            radius_pixels=6.0,
            image_size=100,
        )
        self.assertTrue(rows[0]["candidate_available"])
        self.assertFalse(rows[1]["candidate_available"])
        self.assertFalse(rows[2]["candidate_available"])

    def test_ground_truth_edges_reports_pixel_length_without_filtering(self) -> None:
        sample = {
            "layers": {
                "bonds": [
                    {
                        "begin_atom": 2,
                        "end_atom": 0,
                        "begin": [0.0, 0.0],
                        "end": [120.0, 0.0],
                        "display_type": "DOUBLE",
                        "chemical_type": "DOUBLE",
                        "direction": "NONE",
                    }
                ]
            }
        }
        edges, attributes = ground_truth_edges(sample)
        self.assertEqual(edges, {(0, 2)})
        self.assertEqual(attributes[(0, 2)]["length_bucket"], "long_gt_96")
        self.assertEqual(attributes[(0, 2)]["length_pixels"], 120.0)


if __name__ == "__main__":
    unittest.main()
