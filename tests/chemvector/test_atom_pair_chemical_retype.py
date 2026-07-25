from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_chemical_retype import (
    AtomPairChemicalRetypeConfig,
    AtomPairChemicalRetypeHead,
    AtomPairRetypeAction,
    apply_retype_action,
    apply_retype_to_chemical_logits,
    build_retype_features,
    chemical_retype_joint_loss,
    chemical_retype_loss,
    chemical_retype_safety_loss,
    choose_joint_retype_action,
    choose_retype_action,
    initialize_retype_encoder_from_one_move,
    joint_retype_logits,
    target_retype_action_index,
)
from chemvector.atom_pair_one_move import AtomPairOneMoveConfig, AtomPairOneMoveHead
from chemvector.audit_atom_pair_chemical_reachability import ABSENT_STATE


class AtomPairChemicalRetypeTest(unittest.TestCase):
    def test_target_distinguishes_keep_retype_and_unsupported_edit(self) -> None:
        actions = [AtomPairRetypeAction(1, 3)]
        current = torch.tensor((0, 1, ABSENT_STATE))
        self.assertEqual(target_retype_action_index(actions, current, current), (0, None))
        self.assertEqual(
            target_retype_action_index(
                actions, current, torch.tensor((0, 3, ABSENT_STATE))
            ),
            (1, 0),
        )
        self.assertIsNone(
            target_retype_action_index(
                actions, current, torch.tensor((0, 1, 2))
            )
        )

    def test_apply_retype_preserves_presence(self) -> None:
        states = torch.tensor((0, 1, ABSENT_STATE))
        output = apply_retype_action(states, AtomPairRetypeAction(1, 3))
        self.assertEqual(output.tolist(), [0, 3, ABSENT_STATE])
        self.assertEqual(
            (output != ABSENT_STATE).tolist(),
            (states != ABSENT_STATE).tolist(),
        )

    def test_apply_retype_logits_changes_only_target_group_type(self) -> None:
        logits = torch.tensor(((2.0, 1.0, 0.0, -1.0), (1.0, 0.0, -1.0, -2.0)))
        output = apply_retype_to_chemical_logits(
            logits, AtomPairRetypeAction(1, 2)
        )
        self.assertTrue(torch.equal(output[0], logits[0]))
        self.assertEqual(int(output[1].argmax()), 2)

    def test_zero_initialized_gate_keeps_at_zero_threshold(self) -> None:
        head = AtomPairChemicalRetypeHead(
            AtomPairChemicalRetypeConfig(hidden_dim=16, maximum_groups=8)
        )
        relation = torch.randn((3, 18))
        base = torch.tensor((1.0, -0.5, 0.25))
        chemical = torch.randn((3, 4))
        selected = base >= 0.0
        gate, scores, actions = head.score_sample(
            build_retype_features(relation, base, chemical, selected),
            chemical,
            selected,
        )
        self.assertIsNone(
            choose_retype_action(gate, scores, actions, gate_threshold=0.0)
        )

    def test_encoder_warm_start_copies_only_existing_feature_prefix(self) -> None:
        torch.manual_seed(3)
        first = AtomPairOneMoveHead(AtomPairOneMoveConfig(hidden_dim=16))
        first.edge_encoder[0].weight.data.normal_()
        retype = AtomPairChemicalRetypeHead(
            AtomPairChemicalRetypeConfig(hidden_dim=16)
        )
        initialize_retype_encoder_from_one_move(retype, first)
        self.assertTrue(
            torch.equal(
                retype.edge_encoder[0].weight[:, : first.config.input_dim],
                first.edge_encoder[0].weight,
            )
        )
        self.assertTrue(
            bool((retype.edge_encoder[0].weight[:, first.config.input_dim :] == 0).all())
        )

    def test_loss_separates_gate_and_action(self) -> None:
        values = chemical_retype_loss(
            torch.tensor(0.0, requires_grad=True),
            1,
            torch.tensor((0.0, 1.0), requires_grad=True),
            1,
            gate_positive_weight=4.0,
            action_weight=1.0,
        )
        values["loss"].backward()
        self.assertGreater(float(values["gate_loss"].detach()), 0.0)
        self.assertGreater(float(values["action_loss"].detach()), 0.0)

    def test_joint_loss_competes_with_keep(self) -> None:
        gate = torch.tensor(0.2, requires_grad=True)
        scores = torch.tensor((0.1, -0.4), requires_grad=True)
        values = chemical_retype_joint_loss(
            gate, 1, scores, 0, retype_weight=3.0
        )
        values["loss"].backward()
        self.assertGreater(float(values["joint_loss"].detach()), 0.0)
        self.assertIsNotNone(gate.grad)
        self.assertIsNotNone(scores.grad)

    def test_joint_decode_requires_action_to_beat_keep_margin(self) -> None:
        actions = [AtomPairRetypeAction(0, 1), AtomPairRetypeAction(1, 2)]
        logits = joint_retype_logits(
            torch.tensor(0.2), torch.tensor((0.1, -0.4))
        )
        self.assertTrue(
            torch.allclose(logits, torch.tensor((0.0, 0.3, -0.2)))
        )
        self.assertIsNone(
            choose_joint_retype_action(
                torch.tensor(-0.5),
                torch.tensor((0.1, 0.2)),
                actions,
                minimum_margin=0.0,
            )
        )
        self.assertEqual(
            choose_joint_retype_action(
                torch.tensor(0.1),
                torch.tensor((0.1, 0.4)),
                actions,
                minimum_margin=0.0,
            ),
            actions[1],
        )

    def test_safety_loss_calibrates_top_action_against_keep(self) -> None:
        gate = torch.tensor(-0.2, requires_grad=True)
        scores = torch.tensor((0.1, 0.4), requires_grad=True)
        values = chemical_retype_safety_loss(
            gate, scores, 1, positive_weight=5.0
        )
        values["loss"].backward()
        self.assertGreater(float(values["safety_loss"].detach()), 0.0)
        self.assertLess(float(gate.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
