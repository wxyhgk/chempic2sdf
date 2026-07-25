import unittest
from unittest.mock import patch

import torch

from chemvector.pretrain_junction_aware_bond_selector import (
    deployable_base_features,
    group_selected,
    select_cached_examples,
    threshold_indices,
)


class FakeGrouper:
    def score_embeddings(self, geometry, first, second):
        del first, second
        return torch.zeros(len(geometry), device=geometry.device)


class JunctionAwareBondSelectorTest(unittest.TestCase):
    def test_deployable_features_remove_oracle_junction_columns(self) -> None:
        features = torch.arange(296, dtype=torch.float32).reshape(1, 296)
        output = deployable_base_features({"features": features})
        self.assertEqual(output.shape, (1, 293))
        self.assertEqual(output[0, 22].item(), 22.0)
        self.assertEqual(output[0, 23].item(), 26.0)

    def test_threshold_selection_respects_mask_and_score_order(self) -> None:
        logits = torch.logit(torch.tensor(((0.7, 0.9, 0.8, 0.2),)))
        mask = torch.tensor(((True, True, False, True),))
        self.assertEqual(threshold_indices(logits, mask, 0.5), [[1, 0]])

    def test_cached_examples_can_reuse_a_larger_nested_cache(self) -> None:
        samples = [{"key": "a"}, {"key": "b"}]
        cache = [{"key": "a"}, {"key": "b"}, {"key": "c"}]

        selected = select_cached_examples(cache, samples, 2, name="train")

        self.assertEqual(selected, cache[:2])

    def test_cached_examples_reject_wrong_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "cache order"):
            select_cached_examples(
                [{"key": "b"}], [{"key": "a"}], 1, name="train"
            )

    @patch(
        "chemvector.pretrain_junction_aware_bond_selector.decode_endpoint_groups",
        return_value=torch.tensor((0, 1, 0, 1)),
    )
    def test_training_projection_uses_selector_probability(self, _decode_groups) -> None:
        example = {
            "endpoints": torch.tensor(
                (
                    ((0.1, 0.2), (0.9, 0.2)),
                    ((0.2, 0.7), (0.8, 0.7)),
                )
            ),
            "scores": torch.tensor((0.95, 0.20)),
            "display_ids": torch.tensor((0, 2)),
            "chemical_ids": torch.tensor((0, 2)),
            "direction_ids": torch.tensor((0, 0)),
            "endpoint_embeddings": torch.zeros((4, 32)),
        }

        result = group_selected(
            example,
            [0, 1],
            torch.logit(torch.tensor((0.6, 0.9))),
            FakeGrouper(),
            {"threshold": 0.5, "linkage": "complete"},
            torch.device("cpu"),
        )

        self.assertEqual(len(result.bond_objects), 1)
        self.assertEqual(result.bond_objects[0]["chemical"], 2)
        self.assertTrue(
            torch.allclose(
                result.points,
                torch.tensor(((0.2, 0.7), (0.8, 0.7))),
            )
        )


if __name__ == "__main__":
    unittest.main()
