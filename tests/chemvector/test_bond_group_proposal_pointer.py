from __future__ import annotations

import math
import unittest

import torch

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    SINGLE_LINE,
    SOLID_TRIANGLE,
)
from chemvector.bond_group_proposal_pointer import (
    BondGroupProposalPointerConfig,
    BondGroupProposalPointerPredictor,
    bond_group_proposal_pointer_loss,
    candidate_root_ownership_costs,
    decode_bond_group_proposal_pointer,
    type_macro_group_mean,
)


def _config(*, predict_group_count: bool = True) -> BondGroupProposalPointerConfig:
    return BondGroupProposalPointerConfig(
        proposal_feature_dim=8,
        proposal_descriptor_dim=6,
        flat_type_classes=5,
        d_model=16,
        proposal_encoder_layers=1,
        heads=4,
        ffn_dim=32,
        max_members=8,
        max_group_count=6,
        predict_group_count=predict_group_count,
        dropout=0.0,
    )


def _inputs(batch: int = 2, proposals: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260721)
    mask = torch.ones((batch, proposals), dtype=torch.bool)
    if proposals >= 6:
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
    proposals: int,
    max_members: int = 8,
    max_group_count: int = 6,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "keep_logits": torch.zeros(
            (1, proposals),
            requires_grad=requires_grad,
        ),
        "group_type_logits": torch.zeros(
            (1, proposals, 7),
            requires_grad=requires_grad,
        ),
        "member_count_logits": torch.zeros(
            (1, proposals, max_members + 1),
            requires_grad=requires_grad,
        ),
        "pointer_logits": torch.zeros(
            (1, proposals, proposals),
            requires_grad=requires_grad,
        ),
        "group_count_logits": torch.zeros(
            (1, max_group_count + 1),
            requires_grad=requires_grad,
        ),
    }


class BondGroupProposalPointerTest(unittest.TestCase):
    def test_config_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            BondGroupProposalPointerConfig(d_model=15, heads=4)
        with self.assertRaisesRegex(ValueError, "at least three"):
            BondGroupProposalPointerConfig(max_members=2)

    def test_forward_shapes_without_free_group_or_coordinate_heads(self) -> None:
        model = BondGroupProposalPointerPredictor(_config()).eval()
        with torch.no_grad():
            output = model(**_inputs())

        self.assertEqual(output["keep_logits"].shape, (2, 6))
        self.assertEqual(output["group_type_logits"].shape, (2, 6, 7))
        self.assertEqual(output["member_count_logits"].shape, (2, 6, 9))
        self.assertEqual(output["pointer_logits"].shape, (2, 6, 6))
        self.assertEqual(output["proposal_features"].shape, (2, 6, 16))
        self.assertEqual(output["group_count_logits"].shape, (2, 7))

        forbidden_fragments = (
            "group_query",
            "group_decoder",
            "geometry_head",
            "coordinate_head",
            "endpoint_head",
        )
        forbidden_parameters = {
            name
            for name, _ in model.named_parameters()
            if any(fragment in name for fragment in forbidden_fragments)
        }
        self.assertEqual(forbidden_parameters, set())
        forbidden_outputs = {
            name
            for name in output
            if any(
                fragment in name
                for fragment in ("geometry", "coordinate", "endpoint")
            )
        }
        self.assertEqual(forbidden_outputs, set())
        self.assertFalse(hasattr(model.config, "group_queries"))

    def test_pointer_matrix_is_strictly_symmetric(self) -> None:
        model = BondGroupProposalPointerPredictor(_config()).eval()
        inputs = _inputs()
        with torch.no_grad():
            pointer = model(**inputs)["pointer_logits"]
        torch.testing.assert_close(
            pointer,
            pointer.transpose(-1, -2),
            atol=0.0,
            rtol=0.0,
        )

    def test_proposal_permutation_is_equivariant_on_both_pointer_axes(self) -> None:
        model = BondGroupProposalPointerPredictor(_config()).eval()
        inputs = _inputs(batch=1)
        permutation = torch.tensor([3, 0, 4, 1, 5, 2])
        permuted = {
            name: (
                value[:, permutation]
                if value.ndim >= 2
                else value
            )
            for name, value in inputs.items()
        }
        with torch.no_grad():
            baseline = model(**inputs)
            changed = model(**permuted)

        for name in (
            "keep_logits",
            "group_type_logits",
            "member_count_logits",
            "proposal_features",
        ):
            torch.testing.assert_close(
                changed[name],
                baseline[name][:, permutation],
                atol=2e-6,
                rtol=1e-6,
            )
        expected_pointer = baseline["pointer_logits"][:, permutation][
            :, :, permutation
        ]
        torch.testing.assert_close(
            changed["pointer_logits"],
            expected_pointer,
            atol=2e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            changed["group_count_logits"],
            baseline["group_count_logits"],
            atol=2e-6,
            rtol=1e-6,
        )

    def test_padding_values_cannot_change_valid_predictions(self) -> None:
        model = BondGroupProposalPointerPredictor(_config()).eval()
        inputs = _inputs(batch=1)
        changed = {name: value.clone() for name, value in inputs.items()}
        padded = ~inputs["proposal_mask"]
        changed["proposal_features"][padded] = 1000.0
        changed["proposal_descriptors"][padded] = -1000.0
        changed["flat_type_logits"][padded] = 500.0
        changed["proposal_scores"][padded] = 999.0
        with torch.no_grad():
            baseline = model(**inputs)
            modified = model(**changed)

        valid = inputs["proposal_mask"][0]
        for name in (
            "keep_logits",
            "group_type_logits",
            "member_count_logits",
            "proposal_features",
        ):
            torch.testing.assert_close(
                baseline[name][0, valid],
                modified[name][0, valid],
                atol=1e-6,
                rtol=0.0,
            )
        torch.testing.assert_close(
            baseline["pointer_logits"][0][valid][:, valid],
            modified["pointer_logits"][0][valid][:, valid],
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            baseline["group_count_logits"],
            modified["group_count_logits"],
            atol=1e-6,
            rtol=0.0,
        )

    def test_any_group_member_can_be_the_latent_root(self) -> None:
        membership = torch.tensor([True, True, False])
        valid = torch.ones(3, dtype=torch.bool)
        root_zero_logits = torch.tensor(
            [
                [5.0, 4.0, -5.0],
                [4.0, 1.0, -5.0],
                [-5.0, -5.0, 5.0],
            ]
        )
        roots, costs = candidate_root_ownership_costs(
            root_zero_logits,
            membership,
            valid,
        )
        self.assertEqual(roots.tolist(), [0, 1])
        self.assertEqual(int(roots[costs.argmin()]), 0)

        permutation = torch.tensor([1, 0, 2])
        root_one_logits = root_zero_logits[permutation][:, permutation]
        roots, costs = candidate_root_ownership_costs(
            root_one_logits,
            membership,
            valid,
        )
        self.assertEqual(int(roots[costs.argmin()]), 1)

    def test_hard_and_softmin_ownership_match_their_definitions(self) -> None:
        output = _manual_output(proposals=3)
        pointer = torch.tensor(
            [
                [4.0, 2.0, -4.0],
                [3.0, 1.0, -4.0],
                [-4.0, -4.0, 4.0],
            ]
        )
        output["pointer_logits"] = pointer.unsqueeze(0)
        targets = {
            "group_types": torch.tensor([[PARALLEL_2]]),
            "member_counts": torch.tensor([[2]]),
            "group_mask": torch.tensor([[True]]),
            "group_membership": torch.tensor([[[True, True, False]]]),
        }
        proposal_mask = torch.ones((1, 3), dtype=torch.bool)
        _, costs = candidate_root_ownership_costs(
            pointer,
            targets["group_membership"][0, 0],
            proposal_mask[0],
        )
        hard = bond_group_proposal_pointer_loss(
            output,
            targets,
            proposal_mask,
            root_mode="hard",
        )
        temperature = 0.3
        soft = bond_group_proposal_pointer_loss(
            output,
            targets,
            proposal_mask,
            root_mode="softmin",
            softmin_temperature=temperature,
        )
        expected_soft = (
            -temperature * torch.logsumexp(-costs / temperature, dim=0)
            + temperature * math.log(len(costs))
        )
        torch.testing.assert_close(hard["pointer_loss"], costs.min())
        torch.testing.assert_close(soft["pointer_loss"], expected_soft)
        self.assertEqual(
            hard["selected_root_indices"].tolist(),
            soft["selected_root_indices"].tolist(),
        )

    def test_type_macro_mean_is_not_dominated_by_many_single_groups(self) -> None:
        values = torch.tensor([0.0] * 100 + [10.0])
        types = torch.tensor([SINGLE_LINE] * 100 + [PARALLEL_2])
        macro = type_macro_group_mean(values, types)
        self.assertAlmostEqual(float(macro), 5.0, places=6)
        self.assertLess(float(values.mean()), 0.1)

    def test_full_loss_is_finite_and_backpropagates(self) -> None:
        model = BondGroupProposalPointerPredictor(_config())
        inputs = _inputs()
        output = model(**inputs)
        group_types = torch.tensor(
            [
                [SINGLE_LINE, PARALLEL_2, 0],
                [SOLID_TRIANGLE, HASHED_WEDGE, 0],
            ]
        )
        member_counts = torch.tensor([[1, 2, 0], [1, 3, 0]])
        group_mask = group_types != 0
        membership = torch.zeros((2, 3, 6), dtype=torch.bool)
        membership[0, 0, 0] = True
        membership[0, 1, 1:3] = True
        membership[1, 0, 0] = True
        membership[1, 1, 1:3] = True
        targets = {
            "group_types": group_types,
            "member_counts": member_counts,
            "group_mask": group_mask,
            "group_membership": membership,
        }
        losses = bond_group_proposal_pointer_loss(
            output,
            targets,
            inputs["proposal_mask"],
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        for parameter in (
            model.keep_head.weight,
            model.group_type_head.weight,
            model.member_count_head[-1].weight,
            model.pointer_projection.weight,
            model.group_count_head[-1].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_optional_group_count_head_can_be_disabled(self) -> None:
        model = BondGroupProposalPointerPredictor(
            _config(predict_group_count=False)
        ).eval()
        inputs = _inputs(batch=1)
        with torch.no_grad():
            output = model(**inputs)
        self.assertNotIn("group_count_logits", output)
        targets = {
            "group_types": torch.tensor([[SINGLE_LINE]]),
            "member_counts": torch.tensor([[1]]),
            "group_mask": torch.tensor([[True]]),
            "group_membership": torch.tensor(
                [[[True, False, False, False, False, False]]]
            ),
        }
        losses = bond_group_proposal_pointer_loss(
            output,
            targets,
            inputs["proposal_mask"],
        )
        self.assertEqual(float(losses["group_count_loss"]), 0.0)

    def test_overlapping_groups_fail_hard(self) -> None:
        output = _manual_output(proposals=3)
        targets = {
            "group_types": torch.tensor([[SINGLE_LINE, SINGLE_LINE]]),
            "member_counts": torch.tensor([[1, 1]]),
            "group_mask": torch.tensor([[True, True]]),
            "group_membership": torch.tensor(
                [[[True, False, False], [True, False, False]]]
            ),
        }
        with self.assertRaisesRegex(ValueError, "multiple"):
            bond_group_proposal_pointer_loss(
                output,
                targets,
                torch.ones((1, 3), dtype=torch.bool),
            )

    def test_membership_cannot_reference_padding(self) -> None:
        output = _manual_output(proposals=3)
        targets = {
            "group_types": torch.tensor([[SINGLE_LINE]]),
            "member_counts": torch.tensor([[1]]),
            "group_mask": torch.tensor([[True]]),
            "group_membership": torch.tensor([[[False, False, True]]]),
        }
        with self.assertRaisesRegex(ValueError, "padded"):
            bond_group_proposal_pointer_loss(
                output,
                targets,
                torch.tensor([[True, True, False]]),
            )

    def test_fixed_group_requires_exact_cardinality(self) -> None:
        output = _manual_output(proposals=3)
        targets = {
            "group_types": torch.tensor([[PARALLEL_2]]),
            "member_counts": torch.tensor([[2]]),
            "group_mask": torch.tensor([[True]]),
            "group_membership": torch.tensor([[[True, False, False]]]),
        }
        with self.assertRaisesRegex(ValueError, "invalid support/count"):
            bond_group_proposal_pointer_loss(
                output,
                targets,
                torch.ones((1, 3), dtype=torch.bool),
            )

    def test_known_count_decode_self_anchors_and_partitions_proposals(self) -> None:
        output = _manual_output(proposals=5)
        output["keep_logits"][0] = torch.tensor([-4.0, 9.0, -3.0, 8.0, 20.0])
        output["group_type_logits"][0, 1, PARALLEL_2] = 10.0
        output["group_type_logits"][0, 3, PARALLEL_2] = 10.0
        output["member_count_logits"][0, 1, 2] = 10.0
        output["member_count_logits"][0, 3, 2] = 10.0
        pointer = torch.full((5, 5), -8.0)
        pointer[0, 1] = pointer[1, 0] = 8.0
        pointer[2, 3] = pointer[3, 2] = 8.0
        output["pointer_logits"][0] = pointer
        proposal_mask = torch.tensor([[True, True, True, True, False]])
        rows = decode_bond_group_proposal_pointer(
            output,
            proposal_mask,
            known_counts=torch.tensor([2]),
        )[0]

        self.assertEqual([row["root_index"] for row in rows], [1, 3])
        for row in rows:
            self.assertIn(row["root_index"], row["proposal_indices"])
            self.assertTrue(row["valid_membership"])
        assigned = [
            proposal
            for row in rows
            for proposal in row["proposal_indices"]
        ]
        self.assertEqual(sorted(assigned), [0, 1, 2, 3])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertNotIn(4, assigned)


if __name__ == "__main__":
    unittest.main()
