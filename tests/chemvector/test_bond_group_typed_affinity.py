from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
)
from chemvector.bond_group_typed_affinity import (
    PAIR_HASHED,
    PAIR_IGNORE_INDEX,
    PAIR_NONE,
    PAIR_PARALLEL_2,
    PAIR_PARALLEL_3,
    BondGroupTypedAffinityConfig,
    BondGroupTypedAffinityPredictor,
    bond_group_typed_affinity_loss,
    build_typed_affinity_targets,
    hard_negative_affinity_pair_loss,
    node_type_macro_loss,
    positive_affinity_pair_loss,
)


def _config() -> BondGroupTypedAffinityConfig:
    return BondGroupTypedAffinityConfig(
        proposal_feature_dim=8,
        proposal_descriptor_dim=6,
        flat_type_classes=5,
        d_model=16,
        proposal_encoder_layers=1,
        heads=4,
        ffn_dim=32,
        pair_dim=8,
        pair_hidden_dim=16,
        dropout=0.0,
    )


def _inputs(batch: int = 2, proposals: int = 9) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260722)
    mask = torch.ones((batch, proposals), dtype=torch.bool)
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


def _aggregate_targets() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    group_types = torch.tensor(
        [[PARALLEL_2, PARALLEL_3, HASHED_WEDGE, SINGLE_LINE, 0]]
    )
    membership = torch.zeros((1, 5, 9), dtype=torch.bool)
    membership[0, 0, 0:2] = True
    membership[0, 1, 2:5] = True
    membership[0, 2, 5:7] = True
    membership[0, 3, 7] = True
    group_mask = group_types != 0
    proposal_keep = membership.any(dim=1)
    proposal_mask = torch.tensor(
        [[True, True, True, True, True, True, True, True, False]]
    )
    return {
        "group_types": group_types,
        "group_mask": group_mask,
        "group_membership": membership,
        "proposal_keep": proposal_keep,
    }, proposal_mask


def _training_targets() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    group_types = torch.tensor(
        [
            [PARALLEL_2, SINGLE_LINE, 0],
            [PARALLEL_3, HASHED_WEDGE, 0],
        ]
    )
    membership = torch.zeros((2, 3, 6), dtype=torch.bool)
    membership[0, 0, 0:2] = True
    membership[0, 1, 2] = True
    membership[1, 0, 0:3] = True
    membership[1, 1, 3:5] = True
    proposal_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    return {
        "group_types": group_types,
        "group_mask": group_types != 0,
        "group_membership": membership,
        "proposal_keep": membership.any(dim=1),
        "group_weights": torch.tensor(
            [[1.0, 0.8, 0.0], [0.9, 1.0, 0.0]]
        ),
    }, proposal_mask


class BondGroupTypedAffinityTest(unittest.TestCase):
    def test_config_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            BondGroupTypedAffinityConfig(d_model=15, heads=4)
        with self.assertRaisesRegex(ValueError, "positive"):
            BondGroupTypedAffinityConfig(pair_dim=0)

    def test_forward_has_only_node_and_pair_structural_heads(self) -> None:
        model = BondGroupTypedAffinityPredictor(_config()).eval()
        with torch.no_grad():
            output = model(**_inputs())

        self.assertEqual(output["node_group_type_logits"].shape, (2, 9, 7))
        self.assertEqual(output["pair_logits"].shape, (2, 9, 9, 4))
        self.assertEqual(output["pair_mask"].shape, (2, 9, 9))
        self.assertEqual(output["proposal_features"].shape, (2, 9, 16))
        forbidden = (
            "group_query",
            "group_decoder",
            "root",
            "coordinate",
            "endpoint",
            "stroke",
            "geometry_head",
        )
        self.assertFalse(
            any(
                fragment in name
                for name, _ in model.named_parameters()
                for fragment in forbidden
            )
        )
        self.assertFalse(hasattr(model.config, "group_queries"))

    def test_pair_logits_are_strictly_symmetric_and_isolate_diagonal_padding(self) -> None:
        model = BondGroupTypedAffinityPredictor(_config()).eval()
        inputs = _inputs(batch=1)
        with torch.no_grad():
            output = model(**inputs)
        pair_logits = output["pair_logits"]
        torch.testing.assert_close(
            pair_logits,
            pair_logits.transpose(1, 2),
            atol=0.0,
            rtol=0.0,
        )

        invalid = ~output["pair_mask"]
        self.assertTrue(bool((pair_logits[..., PAIR_NONE][invalid] == 20.0).all()))
        self.assertTrue(bool((pair_logits[..., 1:][invalid] == -20.0).all()))
        diagonal = torch.arange(pair_logits.size(1))
        self.assertFalse(bool(output["pair_mask"][0, diagonal, diagonal].any()))

        padded = ~inputs["proposal_mask"]
        node_logits = output["node_group_type_logits"]
        self.assertTrue(bool((node_logits[..., 0][padded] == 20.0).all()))
        self.assertTrue(bool((node_logits[..., 1:][padded] == -20.0).all()))

    def test_proposal_permutation_is_equivariant_on_both_pair_axes(self) -> None:
        model = BondGroupTypedAffinityPredictor(_config()).eval()
        inputs = _inputs(batch=1)
        permutation = torch.tensor([3, 0, 7, 1, 8, 5, 2, 6, 4])
        changed_inputs = {
            name: value[:, permutation]
            for name, value in inputs.items()
        }
        with torch.no_grad():
            baseline = model(**inputs)
            changed = model(**changed_inputs)

        for name in ("node_group_type_logits", "proposal_features"):
            torch.testing.assert_close(
                changed[name],
                baseline[name][:, permutation],
                atol=2e-6,
                rtol=1e-6,
            )
        expected_pair = baseline["pair_logits"][:, permutation][
            :, :, permutation
        ]
        torch.testing.assert_close(
            changed["pair_logits"],
            expected_pair,
            atol=2e-6,
            rtol=1e-6,
        )
        self.assertTrue(
            torch.equal(
                changed["pair_mask"],
                baseline["pair_mask"][:, permutation][:, :, permutation],
            )
        )

    def test_padding_values_cannot_change_valid_outputs(self) -> None:
        model = BondGroupTypedAffinityPredictor(_config()).eval()
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
        torch.testing.assert_close(
            baseline["node_group_type_logits"][0, valid],
            modified["node_group_type_logits"][0, valid],
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            baseline["pair_logits"][0][valid][:, valid],
            modified["pair_logits"][0][valid][:, valid],
            atol=1e-6,
            rtol=0.0,
        )

    def test_aggregate_targets_become_symmetric_typed_pairs(self) -> None:
        targets, proposal_mask = _aggregate_targets()
        built = build_typed_affinity_targets(targets, proposal_mask)
        labels = built["pair_labels"][0]
        node_labels = built["node_group_type_labels"][0]

        self.assertEqual(node_labels.tolist(), [
            PARALLEL_2,
            PARALLEL_2,
            PARALLEL_3,
            PARALLEL_3,
            PARALLEL_3,
            HASHED_WEDGE,
            HASHED_WEDGE,
            SINGLE_LINE,
            0,
        ])
        self.assertEqual(int(labels[0, 1]), PAIR_PARALLEL_2)
        self.assertEqual(int(labels[2, 3]), PAIR_PARALLEL_3)
        self.assertEqual(int(labels[2, 4]), PAIR_PARALLEL_3)
        self.assertEqual(int(labels[5, 6]), PAIR_HASHED)
        self.assertEqual(int(labels[1, 5]), PAIR_NONE)
        self.assertEqual(int(labels[0, 0]), PAIR_IGNORE_INDEX)
        self.assertEqual(int(labels[8, 0]), PAIR_IGNORE_INDEX)
        self.assertTrue(torch.equal(labels, labels.transpose(0, 1)))
        self.assertTrue(
            torch.equal(
                built["pair_group_indices"],
                built["pair_group_indices"].transpose(1, 2),
            )
        )

    def test_target_builder_rejects_overlap_and_keep_union_mismatch(self) -> None:
        targets, proposal_mask = _aggregate_targets()
        overlap = {name: value.clone() for name, value in targets.items()}
        overlap["group_membership"][0, 3, 0] = True
        overlap["proposal_keep"] = overlap["group_membership"].any(dim=1)
        with self.assertRaisesRegex(ValueError, "multiple"):
            build_typed_affinity_targets(overlap, proposal_mask)

        broken_keep = {name: value.clone() for name, value in targets.items()}
        broken_keep["proposal_keep"][0, 7] = False
        with self.assertRaisesRegex(ValueError, "membership union"):
            build_typed_affinity_targets(broken_keep, proposal_mask)

    def test_hashed_positive_pairs_are_group_balanced_not_n_squared_weighted(self) -> None:
        group_types = torch.tensor([[HASHED_WEDGE, HASHED_WEDGE]])
        membership = torch.zeros((1, 2, 7), dtype=torch.bool)
        membership[0, 0, 0:2] = True
        membership[0, 1, 2:7] = True
        targets = {
            "group_types": group_types,
            "group_mask": torch.ones((1, 2), dtype=torch.bool),
            "group_membership": membership,
            "proposal_keep": membership.any(dim=1),
        }
        proposal_mask = torch.ones((1, 7), dtype=torch.bool)
        logits = torch.zeros((1, 7, 7, 4))
        logits[0, 0, 1, PAIR_HASHED] = 6.0
        logits[0, 1, 0] = logits[0, 0, 1]
        large_first = torch.tensor([2, 2, 2, 2, 3, 3, 3, 4, 4, 5])
        large_second = torch.tensor([3, 4, 5, 6, 4, 5, 6, 5, 6, 6])
        logits[0, large_first, large_second, PAIR_NONE] = 6.0
        logits[0, large_second, large_first] = logits[0, large_first, large_second]

        result = positive_affinity_pair_loss(logits, targets, proposal_mask)
        small = F.cross_entropy(
            logits[0, 0, 1].unsqueeze(0),
            torch.tensor([PAIR_HASHED]),
        )
        large = F.cross_entropy(
            logits[0, large_first, large_second],
            torch.full((10,), PAIR_HASHED),
        )
        expected = 0.5 * (small + large)
        pair_weighted = (small + 10.0 * large) / 11.0
        torch.testing.assert_close(result["loss"], expected)
        self.assertGreater(abs(float(result["loss"] - pair_weighted)), 1.0)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["pair_count"], 11)

    def test_hard_negative_top_k_is_per_image(self) -> None:
        proposals = 4
        labels = torch.full((1, proposals, proposals), PAIR_NONE, dtype=torch.long)
        diagonal = torch.arange(proposals)
        labels[0, diagonal, diagonal] = PAIR_IGNORE_INDEX
        pair_mask = labels != PAIR_IGNORE_INDEX
        logits = torch.zeros((1, proposals, proposals, 4))
        first = torch.tensor([0, 0, 0, 1, 1, 2])
        second = torch.tensor([1, 2, 3, 2, 3, 3])
        logits[0, first, second, PAIR_PARALLEL_2] = torch.arange(6.0)
        logits[0, second, first] = logits[0, first, second]

        result = hard_negative_affinity_pair_loss(
            logits,
            {"pair_labels": labels, "pair_mask": pair_mask},
            top_k=2,
        )
        all_losses = F.cross_entropy(
            logits[0, first, second],
            torch.zeros(6, dtype=torch.long),
            reduction="none",
        )
        torch.testing.assert_close(result["loss"], all_losses.topk(2).values.mean())
        self.assertEqual(result["selected_pair_count"], 2)

    def test_all_negative_images_form_one_macro_bucket(self) -> None:
        batch = 5
        proposals = 3
        labels = torch.full(
            (batch, proposals, proposals),
            PAIR_NONE,
            dtype=torch.long,
        )
        diagonal = torch.arange(proposals)
        labels[:, diagonal, diagonal] = PAIR_IGNORE_INDEX
        labels[0, 0, 1] = labels[0, 1, 0] = PAIR_PARALLEL_2
        pair_mask = labels != PAIR_IGNORE_INDEX
        logits = torch.zeros((batch, proposals, proposals, 4))
        logits[0, ..., PAIR_NONE] = 5.0
        logits[1:, ..., PAIR_PARALLEL_2] = 5.0
        logits = 0.5 * (logits + logits.transpose(1, 2))

        result = hard_negative_affinity_pair_loss(
            logits,
            {"pair_labels": labels, "pair_mask": pair_mask},
            top_k=None,
        )
        expected = 0.5 * (
            result["positive_image_loss"]
            + result["all_negative_image_loss"]
        )
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(result["positive_image_count"], 1)
        self.assertEqual(result["all_negative_image_count"], 4)

    def test_node_type_loss_is_group_then_type_macro_averaged(self) -> None:
        group_types = torch.tensor([[HASHED_WEDGE, PARALLEL_2]])
        membership = torch.zeros((1, 2, 7), dtype=torch.bool)
        membership[0, 0, 0:5] = True
        membership[0, 1, 5:7] = True
        targets = {
            "group_types": group_types,
            "group_mask": torch.ones((1, 2), dtype=torch.bool),
            "group_membership": membership,
            "proposal_keep": membership.any(dim=1),
        }
        proposal_mask = torch.ones((1, 7), dtype=torch.bool)
        logits = torch.zeros((1, 7, 7))
        logits[0, 0:5, HASHED_WEDGE] = 6.0
        logits[0, 5:7, 0] = 6.0

        result = node_type_macro_loss(logits, targets, proposal_mask)
        hashed = F.cross_entropy(
            logits[0, 0:5],
            torch.full((5,), HASHED_WEDGE),
        )
        parallel = F.cross_entropy(
            logits[0, 5:7],
            torch.full((2,), PARALLEL_2),
        )
        expected = 0.5 * (hashed + parallel)
        member_weighted = (5.0 * hashed + 2.0 * parallel) / 7.0
        torch.testing.assert_close(result["positive"], expected)
        self.assertGreater(abs(float(result["positive"] - member_weighted)), 1.0)
        self.assertEqual(float(result["background"]), 0.0)

    def test_full_loss_is_finite_and_backpropagates(self) -> None:
        model = BondGroupTypedAffinityPredictor(_config())
        inputs = _inputs(batch=2, proposals=6)
        targets, proposal_mask = _training_targets()
        inputs["proposal_mask"] = proposal_mask
        output = model(**inputs)
        losses = bond_group_typed_affinity_loss(
            output,
            targets,
            proposal_mask,
            negative_top_k=4,
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        for parameter in (
            model.node_group_type_head.weight,
            model.pair_node_encoder[0].weight,
            model.pair_head[-1].weight,
            model.proposal_encoder.fusion[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(losses["positive_group_count"], 0)
        self.assertGreater(losses["negative_pair_count"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_single_batch_backward_smoke(self) -> None:
        device = torch.device("cuda")
        model = BondGroupTypedAffinityPredictor(_config()).to(device)
        inputs = {
            name: value.to(device)
            for name, value in _inputs(batch=2, proposals=6).items()
        }
        targets, proposal_mask = _training_targets()
        proposal_mask = proposal_mask.to(device)
        inputs["proposal_mask"] = proposal_mask
        targets = {name: value.to(device) for name, value in targets.items()}
        output = model(**inputs)
        losses = bond_group_typed_affinity_loss(
            output,
            targets,
            proposal_mask,
            negative_top_k=4,
        )
        losses["loss"].backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertIsNotNone(model.pair_head[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
