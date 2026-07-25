from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_chemical_action import (
    AtomPairChemicalAction,
    AtomPairChemicalActionConfig,
    AtomPairChemicalActionHead,
    apply_chemical_action,
    build_chemical_action_features,
    initialize_chemical_action_from_one_move,
    target_chemical_action_index,
)
from chemvector.atom_pair_one_move import AtomPairOneMoveConfig, AtomPairOneMoveHead
from chemvector.audit_atom_pair_chemical_reachability import ABSENT_STATE


class AtomPairChemicalActionTest(unittest.TestCase):
    def test_target_actions_cover_keep_delete_add_and_retype(self) -> None:
        actions = [
            AtomPairChemicalAction("keep"),
            AtomPairChemicalAction("delete", 0, ABSENT_STATE),
            AtomPairChemicalAction("add", 1, 2),
            AtomPairChemicalAction("retype", 2, 3),
        ]

        self.assertEqual(
            target_chemical_action_index(
                actions, torch.tensor((0, ABSENT_STATE, 1)), torch.tensor((0, ABSENT_STATE, 1))
            ),
            0,
        )
        self.assertEqual(
            target_chemical_action_index(
                actions, torch.tensor((0, ABSENT_STATE, 1)), torch.tensor((ABSENT_STATE, ABSENT_STATE, 1))
            ),
            1,
        )
        self.assertEqual(
            target_chemical_action_index(
                actions, torch.tensor((0, ABSENT_STATE, 1)), torch.tensor((0, 2, 1))
            ),
            2,
        )
        self.assertEqual(
            target_chemical_action_index(
                actions, torch.tensor((0, ABSENT_STATE, 1)), torch.tensor((0, ABSENT_STATE, 3))
            ),
            3,
        )

    def test_apply_changes_exactly_one_typed_state(self) -> None:
        states = torch.tensor((0, 1, ABSENT_STATE))
        output = apply_chemical_action(
            states, AtomPairChemicalAction("retype", 1, 3)
        )
        self.assertEqual(output.tolist(), [0, 3, ABSENT_STATE])

    def test_feature_shape_includes_type_and_first_action_state(self) -> None:
        relation = torch.randn((3, 18))
        base = torch.tensor((1.0, -1.0, 0.5))
        chemical = torch.randn((3, 4))
        selected = torch.tensor((True, True, False))
        features = build_chemical_action_features(
            relation, base, chemical, selected
        )
        self.assertEqual(tuple(features.shape), (3, 26))

    def test_warm_start_preserves_keep_and_delete_scores(self) -> None:
        torch.manual_seed(11)
        first = AtomPairOneMoveHead(
            AtomPairOneMoveConfig(hidden_dim=16, maximum_groups=8)
        )
        for parameter in first.parameters():
            parameter.data.normal_(mean=0.0, std=0.1)
        chemical = AtomPairChemicalActionHead(
            AtomPairChemicalActionConfig(hidden_dim=16, maximum_groups=8)
        )
        initialize_chemical_action_from_one_move(chemical, first)
        relation = torch.randn((4, 18))
        base = torch.tensor((1.0, -0.5, 0.25, -1.0))
        type_logits = torch.randn((4, 4))
        selected = base >= 0.0

        first_scores, first_actions = first.score_sample(
            torch.cat((relation, base[:, None]), dim=-1),
            relation,
            base,
            torch.tensor(((0, 1), (1, 2), (2, 3), (0, 3))),
        )
        typed_scores, typed_actions = chemical.score_sample(
            build_chemical_action_features(relation, base, type_logits, selected),
            base,
            type_logits,
            selected,
        )

        self.assertTrue(torch.allclose(first_scores[0], typed_scores[0], atol=1e-6))
        first_delete = {
            action.delete_index: first_scores[index]
            for index, action in enumerate(first_actions)
            if action.kind == "delete"
        }
        typed_delete = {
            action.group_index: typed_scores[index]
            for index, action in enumerate(typed_actions)
            if action.kind == "delete"
        }
        self.assertEqual(set(first_delete), set(typed_delete))
        for index in first_delete:
            self.assertTrue(
                torch.allclose(first_delete[index], typed_delete[index], atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
