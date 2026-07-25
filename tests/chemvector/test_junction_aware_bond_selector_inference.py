import unittest
from unittest.mock import patch

import torch

from chemvector.junction_aware_bond_selector import (
    deployable_candidate_features,
    select_and_group_bonds,
)


class FakeSelector:
    class Config:
        input_dim = 363

    config = Config()

    def __call__(self, features, mask, display, chemical):
        del features, mask, display, chemical
        return {"keep_logits": torch.logit(torch.tensor(((0.6, 0.9),)))}


class FakeGrouper:
    def score_embeddings(self, geometry, first, second):
        del first, second
        return torch.zeros(len(geometry), device=geometry.device)


class JunctionAwareBondSelectorInferenceTest(unittest.TestCase):
    def test_deployable_candidate_feature_layout_matches_training(self) -> None:
        objects = [
            {
                "endpoints": torch.tensor(((0.1, 0.2), (0.8, 0.7))),
                "score": 0.9,
                "display": 1,
                "chemical": 1,
                "direction": 0,
                "feature": torch.arange(128),
            }
        ]
        features = deployable_candidate_features(objects, torch.zeros((1, 142)))
        self.assertEqual(features.shape, (1, 293))

    @patch(
        "chemvector.junction_aware_bond_selector.decode_endpoint_groups",
        return_value=torch.tensor((0, 1, 0, 1)),
    )
    @patch("chemvector.junction_aware_bond_selector.endpoint_visual_context")
    def test_selector_projection_matches_training_winner_and_points(
        self,
        visual_context,
        _decode_groups,
    ) -> None:
        visual_context.return_value = (
            torch.zeros((4, 32)),
            torch.zeros((4, 3)),
        )
        objects = [
            {
                "endpoints": torch.tensor(((0.1, 0.2), (0.9, 0.2))),
                "score": 0.95,
                "display": 0,
                "chemical": 0,
                "direction": 0,
                "feature": torch.zeros(128),
            },
            {
                "endpoints": torch.tensor(((0.2, 0.7), (0.8, 0.7))),
                "score": 0.20,
                "display": 2,
                "chemical": 2,
                "direction": 0,
                "feature": torch.zeros(128),
            },
        ]
        arguments = (
            torch.zeros((1, 3, 32, 32)),
            objects,
            torch.zeros((2, 142)),
            FakeSelector(),
            {"keep_threshold": 0.5},
            FakeGrouper(),
            {"threshold": 0.5, "linkage": "complete"},
        )

        legacy, _ = select_and_group_bonds(
            *arguments,
            projection_score_source="detector",
        )
        projected, diagnostics = select_and_group_bonds(
            *arguments,
            projection_score_source="selector",
        )
        line_band_features = torch.zeros((2, 142))
        line_band_features[1, 1 + 2] = 1.0
        line_band_features[1, 1 + 4 + 1] = 1.0
        semantic_arguments = (
            arguments[0],
            arguments[1],
            line_band_features,
            *arguments[3:],
        )
        semantic, semantic_diagnostics = select_and_group_bonds(
            *semantic_arguments,
            projection_score_source="selector",
            bond_type_source="line_band",
        )

        self.assertEqual(legacy.bond_objects[0]["chemical"], 0)
        self.assertEqual(projected.bond_objects[0]["chemical"], 2)
        self.assertEqual(semantic.bond_objects[0]["display"], 2)
        self.assertEqual(semantic.bond_objects[0]["chemical"], 1)
        self.assertTrue(
            torch.allclose(
                projected.points,
                torch.tensor(((0.2, 0.7), (0.8, 0.7))),
            )
        )
        self.assertEqual(diagnostics.duplicate_pair_objects, 1)
        self.assertEqual(diagnostics.projection_score_source, "selector")
        self.assertEqual(semantic_diagnostics.bond_type_source, "line_band")


if __name__ == "__main__":
    unittest.main()
