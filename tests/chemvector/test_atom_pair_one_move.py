from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_one_move import (
    AtomPairAction,
    AtomPairOneMoveConfig,
    AtomPairOneMoveHead,
    apply_atom_pair_action,
    candidate_action_indices,
    choose_atom_pair_action,
    target_action_index,
)


class AtomPairOneMoveTest(unittest.TestCase):
    def test_candidate_indices_use_lowest_positive_and_highest_negative(self) -> None:
        config = AtomPairOneMoveConfig(delete_candidates=2, add_candidates=2)
        delete_indices, add_indices = candidate_action_indices(
            torch.tensor((2.0, 0.1, -0.2, -3.0, 1.0, -0.1)),
            config=config,
        )

        self.assertEqual(delete_indices.tolist(), [1, 4])
        self.assertEqual(add_indices.tolist(), [5, 2])

    def test_target_action_covers_keep_add_delete_and_swap(self) -> None:
        actions = [
            AtomPairAction("keep"),
            AtomPairAction("delete", delete_index=0),
            AtomPairAction("add", add_index=1),
            AtomPairAction("swap", delete_index=0, add_index=1),
        ]
        base = torch.tensor((1.0, -1.0))

        self.assertEqual(target_action_index(actions, base, torch.tensor((True, False))), 0)
        self.assertEqual(target_action_index(actions, base, torch.tensor((False, False))), 1)
        self.assertEqual(target_action_index(actions, base, torch.tensor((True, True))), 2)
        self.assertEqual(target_action_index(actions, base, torch.tensor((False, True))), 3)

    def test_apply_swap_changes_exactly_two_edges(self) -> None:
        selected = torch.tensor((True, True, False, False))
        output = apply_atom_pair_action(
            selected,
            AtomPairAction("swap", delete_index=1, add_index=3),
        )

        self.assertEqual(output.tolist(), [True, False, False, True])

    def test_margin_can_keep_or_apply_best_edit(self) -> None:
        actions = [AtomPairAction("keep"), AtomPairAction("delete", delete_index=0)]
        scores = torch.tensor((0.0, 0.75))

        kept, _ = choose_atom_pair_action(scores, actions, minimum_margin=1.0)
        edited, _ = choose_atom_pair_action(scores, actions, minimum_margin=0.5)

        self.assertEqual(kept.kind, "keep")
        self.assertEqual(edited.kind, "delete")

    def test_zero_initialized_model_prefers_keep_at_positive_margin(self) -> None:
        config = AtomPairOneMoveConfig(hidden_dim=16, maximum_groups=8)
        model = AtomPairOneMoveHead(config)
        relation = torch.randn(3, 18)
        base = torch.tensor((1.0, -0.5, 0.25))
        scores, actions = model.score_sample(
            torch.cat((relation, base[:, None]), dim=-1),
            relation,
            base,
            torch.tensor(((0, 1), (1, 2), (2, 3))),
        )
        action, _ = choose_atom_pair_action(scores, actions, minimum_margin=0.1)

        self.assertEqual(action.kind, "keep")


if __name__ == "__main__":
    unittest.main()
