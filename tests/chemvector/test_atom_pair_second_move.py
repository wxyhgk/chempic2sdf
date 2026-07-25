from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_one_move import (
    AtomPairAction,
    AtomPairOneMoveConfig,
    AtomPairOneMoveHead,
)
from chemvector.atom_pair_second_move import (
    AtomPairSecondMoveConfig,
    AtomPairSecondMoveHead,
    build_second_move_edge_features,
    candidate_second_action_indices,
    initialize_second_move_from_first,
    target_second_action_index,
)


class AtomPairSecondMoveTest(unittest.TestCase):
    def test_features_mark_current_selection_and_first_change(self) -> None:
        relation = torch.zeros((3, 18))
        logits = torch.tensor((1.0, -1.0, 0.5))
        selected = torch.tensor((True, True, False))

        features = build_second_move_edge_features(relation, logits, selected)

        self.assertEqual(tuple(features.shape), (3, 21))
        self.assertEqual(features[:, -2].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(features[:, -1].tolist(), [0.0, 1.0, 1.0])

    def test_candidates_follow_current_selection_not_logit_sign(self) -> None:
        config = AtomPairSecondMoveConfig(delete_candidates=2, add_candidates=2)
        logits = torch.tensor((2.0, 0.1, -0.2, -3.0, 1.0, -0.1))
        selected = torch.tensor((True, False, True, False, True, False))

        delete_indices, add_indices = candidate_second_action_indices(
            logits, selected, config=config
        )

        self.assertEqual(delete_indices.tolist(), [2, 4])
        self.assertEqual(add_indices.tolist(), [1, 5])

    def test_target_uses_current_selection(self) -> None:
        actions = [
            AtomPairAction("keep"),
            AtomPairAction("delete", delete_index=1),
            AtomPairAction("add", add_index=2),
            AtomPairAction("swap", delete_index=1, add_index=2),
        ]
        selected = torch.tensor((True, True, False))

        self.assertEqual(
            target_second_action_index(
                actions, selected, torch.tensor((True, False, True))
            ),
            3,
        )

    def test_zero_initialized_head_scores_all_actions_equally(self) -> None:
        config = AtomPairSecondMoveConfig(hidden_dim=16, maximum_groups=8)
        head = AtomPairSecondMoveHead(config)
        relation = torch.randn((3, 18))
        logits = torch.tensor((1.0, -0.5, 0.25))
        selected = torch.tensor((True, False, True))
        features = build_second_move_edge_features(relation, logits, selected)

        scores, actions = head.score_sample(
            features,
            relation,
            logits,
            torch.tensor(((0, 1), (1, 2), (2, 3))),
            selected,
        )

        self.assertEqual(len(scores), len(actions))
        self.assertTrue(bool((scores == 0).all()))

    def test_warm_start_matches_first_scores_before_any_edit(self) -> None:
        torch.manual_seed(7)
        first = AtomPairOneMoveHead(
            AtomPairOneMoveConfig(hidden_dim=16, maximum_groups=8)
        )
        for parameter in first.parameters():
            parameter.data.normal_(mean=0.0, std=0.1)
        second = AtomPairSecondMoveHead(
            AtomPairSecondMoveConfig(hidden_dim=16, maximum_groups=8)
        )
        initialize_second_move_from_first(second, first)
        relation = torch.randn((4, 18))
        logits = torch.tensor((1.0, -0.5, 0.25, -1.0))
        selected = logits >= 0.0
        pairs = torch.tensor(((0, 1), (1, 2), (2, 3), (0, 3)))

        first_scores, first_actions = first.score_sample(
            torch.cat((relation, logits[:, None]), dim=-1),
            relation,
            logits,
            pairs,
        )
        second_scores, second_actions = second.score_sample(
            build_second_move_edge_features(relation, logits, selected),
            relation,
            logits,
            pairs,
            selected,
        )

        self.assertEqual(first_actions, second_actions)
        self.assertTrue(torch.allclose(first_scores, second_scores, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
