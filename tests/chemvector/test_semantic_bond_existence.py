import random
import unittest

import torch

from chemvector.pretrain_semantic_bond_existence import relation_candidates
from chemvector.semantic_bond_relation import SemanticBondExistence


class SemanticBondExistenceTest(unittest.TestCase):
    def test_candidates_keep_every_positive_and_match_negative_lengths(self) -> None:
        centers = {
            0: (0.0, 0.0),
            1: (10.0, 0.0),
            2: (20.0, 0.0),
            3: (0.0, 20.0),
            4: (20.0, 20.0),
        }
        positives = [(0, 1), (1, 2), (3, 4)]

        pairs, labels, kinds = relation_candidates(
            centers,
            positives,
            negative_ratio=1.0,
            rng=random.Random(7),
        )

        selected_positives = {
            pair for pair, label in zip(pairs, labels) if label
        }
        self.assertEqual(selected_positives, set(positives))
        self.assertEqual(sum(label == 0 for label in labels), len(positives))
        self.assertIn("length_matched", kinds)

    def test_existence_head_has_no_explicit_geometry_input(self) -> None:
        model = SemanticBondExistence()
        self.assertEqual(model.edge_visual[0].in_features, model.config.hidden_dim * 2)
        self.assertEqual(model.relation.shared[0].in_features, model.config.hidden_dim * 2 + 32)

        with torch.no_grad():
            model.relation.shared[0].weight.normal_()
            model.relation.shared[0].bias.normal_()
            model.relation.shared[2].weight.normal_()
            model.relation.shared[2].bias.normal_()
            model.initialize_edge_visual_from_relation()

        self.assertTrue(
            torch.equal(
                model.edge_visual[0].weight,
                model.relation.shared[0].weight[:, : model.config.hidden_dim * 2],
            )
        )

    def test_model_hard_negatives_are_selected_first(self) -> None:
        centers = {
            0: (0.0, 0.0),
            1: (10.0, 0.0),
            2: (20.0, 0.0),
            3: (30.0, 0.0),
        }
        pairs, labels, kinds = relation_candidates(
            centers,
            [(0, 1), (1, 2)],
            negative_ratio=1.0,
            hard_negative_pairs=[(0, 3), (1, 3)],
            rng=random.Random(3),
        )

        negatives = {
            pair: kind
            for pair, label, kind in zip(pairs, labels, kinds)
            if not label
        }
        self.assertEqual(negatives, {(0, 3): "model_hard", (1, 3): "model_hard"})


if __name__ == "__main__":
    unittest.main()
