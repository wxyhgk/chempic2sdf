from __future__ import annotations

import math
import unittest

import torch

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    SINGLE_LINE,
    SOLID_TRIANGLE,
    BondGroupProposalMembershipConfig,
    BondGroupProposalMembershipPredictor,
    bond_group_proposal_membership_assignments,
    bond_group_proposal_membership_loss,
    decode_bond_group_proposal_membership,
    group_normalized_membership_loss,
    proposal_slot_exclusivity_loss,
)
from chemvector.primitive_set import LINE, NO_OBJECT, TRIANGLE


def _config() -> BondGroupProposalMembershipConfig:
    return BondGroupProposalMembershipConfig(
        proposal_feature_dim=8,
        proposal_descriptor_dim=6,
        flat_type_classes=5,
        d_model=16,
        proposal_encoder_layers=1,
        group_decoder_layers=1,
        heads=4,
        ffn_dim=32,
        group_queries=4,
        max_members=8,
        dropout=0.0,
    )


def _inputs(batch: int = 2, proposals: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260721)
    mask = torch.ones((batch, proposals), dtype=torch.bool)
    if proposals >= 5:
        mask[0, -1] = False
        if batch > 1:
            mask[1, -2:] = False
    return {
        "proposal_features": torch.randn(
            batch,
            proposals,
            8,
            generator=generator,
        ),
        "proposal_descriptors": torch.randn(
            batch,
            proposals,
            6,
            generator=generator,
        ),
        "flat_type_logits": torch.randn(
            batch,
            proposals,
            5,
            generator=generator,
        ),
        "proposal_scores": torch.rand(
            batch,
            proposals,
            generator=generator,
        ),
        "proposal_mask": mask,
    }


def _manual_output(
    *,
    groups: int,
    proposals: int,
    max_members: int = 8,
) -> dict[str, torch.Tensor]:
    return {
        "proposal_keep_logits": torch.full((1, proposals), 8.0),
        "proposal_type_logits": torch.full((1, proposals, 4), -8.0),
        "group_type_logits": torch.full((1, groups, 7), -8.0),
        "member_count_logits": torch.full(
            (1, groups, max_members + 1),
            -8.0,
        ),
        "group_count_logits": torch.full((1, groups + 1), -8.0),
        "membership_logits": torch.full((1, groups, proposals), -8.0),
    }


class BondGroupProposalMembershipTest(unittest.TestCase):
    def test_config_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            BondGroupProposalMembershipConfig(d_model=15, heads=4)
        with self.assertRaisesRegex(ValueError, "at least three"):
            BondGroupProposalMembershipConfig(max_members=2)

    def test_forward_shapes_and_no_coordinate_output_head(self) -> None:
        model = BondGroupProposalMembershipPredictor(_config()).eval()
        inputs = _inputs()
        with torch.no_grad():
            output = model(**inputs)

        self.assertEqual(output["proposal_keep_logits"].shape, (2, 5))
        self.assertEqual(output["proposal_type_logits"].shape, (2, 5, 4))
        self.assertEqual(output["group_type_logits"].shape, (2, 4, 7))
        self.assertEqual(output["member_count_logits"].shape, (2, 4, 9))
        self.assertEqual(output["group_count_logits"].shape, (2, 5))
        self.assertEqual(output["membership_logits"].shape, (2, 4, 5))
        forbidden_outputs = {
            name
            for name in output
            if "geometry" in name or "coordinate" in name or "endpoint" in name
        }
        self.assertEqual(forbidden_outputs, set())
        forbidden_parameters = {
            name
            for name, _ in model.named_parameters()
            if "geometry_head" in name
            or "coordinate_head" in name
            or "endpoint_head" in name
        }
        self.assertEqual(forbidden_parameters, set())

    def test_padding_cannot_change_valid_group_predictions(self) -> None:
        model = BondGroupProposalMembershipPredictor(_config()).eval()
        inputs = _inputs(batch=1)
        changed = {name: value.clone() for name, value in inputs.items()}
        padded = ~inputs["proposal_mask"]
        changed["proposal_features"][padded] = 1000.0
        changed["proposal_descriptors"][padded] = -1000.0
        changed["flat_type_logits"][padded] = 500.0
        changed["proposal_scores"][padded] = 999.0
        with torch.no_grad():
            first = model(**inputs)
            second = model(**changed)

        for name in (
            "group_type_logits",
            "member_count_logits",
            "group_count_logits",
        ):
            torch.testing.assert_close(first[name], second[name], atol=1e-6, rtol=0)
        valid = inputs["proposal_mask"][0]
        torch.testing.assert_close(
            first["membership_logits"][0, :, valid],
            second["membership_logits"][0, :, valid],
            atol=1e-6,
            rtol=0,
        )

    def test_hungarian_assignment_uses_membership_not_target_order(self) -> None:
        output = _manual_output(groups=3, proposals=4)
        output["group_type_logits"][0, 0, PARALLEL_2] = 8.0
        output["group_type_logits"][0, 1, SINGLE_LINE] = 8.0
        output["group_type_logits"][0, 2, 0] = 8.0
        output["member_count_logits"][0, 0, 2] = 8.0
        output["member_count_logits"][0, 1, 1] = 8.0
        output["membership_logits"][0, 0] = torch.tensor([-8.0, 8.0, 8.0, -8.0])
        output["membership_logits"][0, 1] = torch.tensor([8.0, -8.0, -8.0, -8.0])
        targets = {
            "group_types": torch.tensor([[SINGLE_LINE, PARALLEL_2]]),
            "member_counts": torch.tensor([[1, 2]]),
            "group_mask": torch.tensor([[True, True]]),
            "group_membership": torch.tensor(
                [[[True, False, False, False], [False, True, True, False]]]
            ),
        }
        assignment = bond_group_proposal_membership_assignments(
            output,
            targets,
            torch.ones((1, 4), dtype=torch.bool),
        )[0]
        self.assertEqual(assignment[0].tolist(), [0, 1])
        self.assertEqual(assignment[1].tolist(), [1, 0])

    def test_membership_bce_is_normalized_per_group(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 2.0],
            ]
        )
        targets = torch.tensor(
            [
                [True, False, False, False],
                [False, True, True, True],
            ]
        )
        losses = group_normalized_membership_loss(
            logits,
            targets,
            torch.ones(4, dtype=torch.bool),
        )
        expected_positive = (
            F_softplus(0.0) + F_softplus(-2.0)
        ) / 2.0
        self.assertAlmostEqual(
            float(losses["positive_bce"]),
            expected_positive,
            places=6,
        )
        self.assertAlmostEqual(
            float(losses["negative_bce"]),
            F_softplus(0.0),
            places=6,
        )
        member_weighted = (
            F_softplus(0.0) + 3.0 * F_softplus(-2.0)
        ) / 4.0
        self.assertNotAlmostEqual(
            float(losses["positive_bce"]),
            member_weighted,
            places=4,
        )

    def test_membership_loss_downweights_shell_groups(self) -> None:
        logits = torch.tensor(
            [
                [4.0, -4.0],
                [-4.0, 4.0],
            ]
        )
        targets = torch.tensor(
            [
                [True, False],
                [True, False],
            ]
        )
        equal = group_normalized_membership_loss(
            logits,
            targets,
            torch.ones(2, dtype=torch.bool),
        )
        weighted = group_normalized_membership_loss(
            logits,
            targets,
            torch.ones(2, dtype=torch.bool),
            group_weights=torch.tensor([1.0, 0.25]),
        )

        self.assertLess(
            float(weighted["positive_bce"]),
            float(equal["positive_bce"]),
        )
        self.assertLess(float(weighted["dice"]), float(equal["dice"]))

    def test_exclusivity_penalizes_two_active_slots_claiming_one_proposal(self) -> None:
        membership = torch.tensor([[[4.0], [4.0]]])
        both_active = torch.full((1, 2, 7), -20.0)
        both_active[..., SINGLE_LINE] = 20.0
        second_inactive = both_active.clone()
        second_inactive[0, 1, SINGLE_LINE] = -20.0
        second_inactive[0, 1, 0] = 20.0
        mask = torch.ones((1, 1), dtype=torch.bool)
        high = proposal_slot_exclusivity_loss(membership, both_active, mask)
        low = proposal_slot_exclusivity_loss(membership, second_inactive, mask)
        self.assertGreater(float(high), 0.8)
        self.assertLess(float(low), 1e-5)

    def test_full_loss_is_finite_and_backpropagates(self) -> None:
        model = BondGroupProposalMembershipPredictor(_config())
        inputs = _inputs()
        output = model(**inputs)
        proposal_mask = inputs["proposal_mask"]
        group_types = torch.tensor(
            [
                [SINGLE_LINE, PARALLEL_2, 0],
                [SOLID_TRIANGLE, HASHED_WEDGE, 0],
            ]
        )
        member_counts = torch.tensor([[1, 2, 0], [1, 3, 0]])
        group_mask = group_types != 0
        membership = torch.zeros((2, 3, 5), dtype=torch.bool)
        membership[0, 0, 0] = True
        membership[0, 1, 1:3] = True
        membership[1, 0, 0] = True
        membership[1, 1, 1:3] = True
        proposal_keep = membership.any(dim=1)
        proposal_types = torch.full((2, 5), NO_OBJECT, dtype=torch.long)
        proposal_types[0, 0:3] = LINE
        proposal_types[1, 0] = TRIANGLE
        proposal_types[1, 1:3] = LINE
        targets = {
            "group_types": group_types,
            "member_counts": member_counts,
            "group_mask": group_mask,
            "group_membership": membership,
            "proposal_keep": proposal_keep,
            "proposal_types": proposal_types,
        }
        losses = bond_group_proposal_membership_loss(
            output,
            targets,
            proposal_mask,
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        self.assertIsNotNone(model.group_type_head.weight.grad)
        self.assertIsNotNone(model.membership_group_projection.weight.grad)
        self.assertIsNotNone(model.proposal_keep_head.weight.grad)

    def test_known_count_decode_is_exclusive_and_respects_fixed_counts(self) -> None:
        output = _manual_output(groups=3, proposals=4)
        output["group_type_logits"][0, 0, PARALLEL_2] = 10.0
        output["group_type_logits"][0, 1, SINGLE_LINE] = 9.0
        output["group_type_logits"][0, 2, 0] = 10.0
        output["member_count_logits"][0, 0, 2] = 10.0
        output["member_count_logits"][0, 1, 1] = 10.0
        output["proposal_type_logits"][0, :, LINE] = 10.0
        output["membership_logits"][0, 0] = torch.tensor([8.0, 6.0, 1.0, -8.0])
        output["membership_logits"][0, 1] = torch.tensor([7.0, -8.0, 7.5, -8.0])
        rows = decode_bond_group_proposal_membership(
            output,
            torch.ones((1, 4), dtype=torch.bool),
            mode="known_count",
            known_counts=torch.tensor([2]),
        )[0]
        self.assertEqual(len(rows), 2)
        by_kind = {row["kind"]: row for row in rows}
        self.assertEqual(by_kind["parallel_2"]["support_count"], 2)
        self.assertEqual(by_kind["single_line"]["support_count"], 1)
        used = [
            proposal
            for row in rows
            for proposal in row["proposal_indices"]
        ]
        self.assertEqual(len(used), len(set(used)))
        self.assertTrue(all(row["valid_membership"] for row in rows))

    def test_learned_count_decode_uses_group_count_head(self) -> None:
        output = _manual_output(groups=3, proposals=3)
        output["group_type_logits"][0, 0, SINGLE_LINE] = 10.0
        output["group_type_logits"][0, 1, SINGLE_LINE] = 9.0
        output["group_type_logits"][0, 2, 0] = 10.0
        output["member_count_logits"][0, :, 1] = 10.0
        output["proposal_type_logits"][0, :, LINE] = 10.0
        output["membership_logits"][0, 0, 0] = 10.0
        output["membership_logits"][0, 1, 1] = 10.0
        output["group_count_logits"][0, 1] = 10.0
        rows = decode_bond_group_proposal_membership(
            output,
            torch.ones((1, 3), dtype=torch.bool),
            mode="learned_count",
        )[0]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_index"], 0)

    def test_hashed_decode_can_fit_two_supports_and_expand_larger_count(self) -> None:
        output = _manual_output(groups=1, proposals=3)
        output["group_type_logits"][0, 0, HASHED_WEDGE] = 10.0
        output["member_count_logits"][0, 0, 5] = 10.0
        output["proposal_type_logits"][0, :, LINE] = 10.0
        output["membership_logits"][0, 0] = torch.tensor([10.0, 9.0, -10.0])
        rows = decode_bond_group_proposal_membership(
            output,
            torch.ones((1, 3), dtype=torch.bool),
            mode="known_count",
            known_counts=torch.tensor([1]),
        )[0]
        self.assertEqual(rows[0]["member_count"], 5)
        self.assertEqual(rows[0]["support_count"], 2)
        self.assertTrue(rows[0]["valid_membership"])

    def test_membership_targets_cannot_reference_padding(self) -> None:
        logits = torch.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, "padded"):
            group_normalized_membership_loss(
                logits,
                torch.tensor([[False, False, True]]),
                torch.tensor([True, True, False]),
            )

    def test_assignment_rejects_incomplete_fixed_group_target(self) -> None:
        output = _manual_output(groups=2, proposals=3)
        output["group_type_logits"][0, 0, PARALLEL_2] = 8.0
        output["member_count_logits"][0, 0, 2] = 8.0
        targets = {
            "group_types": torch.tensor([[PARALLEL_2]]),
            "member_counts": torch.tensor([[2]]),
            "group_mask": torch.tensor([[True]]),
            "group_membership": torch.tensor([[[True, False, False]]]),
        }
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            bond_group_proposal_membership_assignments(
                output,
                targets,
                torch.ones((1, 3), dtype=torch.bool),
            )


def F_softplus(value: float) -> float:
    return math.log1p(math.exp(value))


if __name__ == "__main__":
    unittest.main()
