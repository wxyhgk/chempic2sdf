"""Tests for the shadow-mode Atom Query detector."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    apply_atom_query_dense_head,
    apply_atom_query_vision_updates,
    atom_query_loss,
    decode_atom_queries,
    decode_atom_queries_by_cardinality,
    decode_atom_queries_by_count_head,
    decode_atom_queries_by_count_regression,
    decode_atom_queries_by_known_counts,
    decode_atom_queries_by_assignment_oracle,
    decode_atom_queries_by_selection_known_counts,
    decode_atom_queries_by_duplicate_penalty_known_counts,
    decode_atom_queries_by_set_count,
    hungarian_assignments,
    load_atom_query_state,
    poisson_binomial_log_probs,
    snap_queries_to_dense_candidates,
)
from chemvector.pretrain_atom_query import (
    checkpoint_is_eligible,
    metric_key,
    set_global_cardinality_trainable,
    set_count_decoder_trainable,
    set_selection_trainable,
    set_duplicate_trainable,
    trainable_vision_parameters,
    vision_updates,
)


class TinyVisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GroupNorm(1, 4))
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1))]
        )


class TinyBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = TinyVisionEncoder()


def tiny_base_model() -> TinyBaseModel:
    return TinyBaseModel()


class AtomQueryTests(unittest.TestCase):
    def test_forward_loss_and_decode_are_finite(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        detector = AtomQueryDetector(config)
        output = detector(torch.randn(2, 16, 8, 8))
        anchors = torch.tensor(
            [
                [[0.2, 0.3], [0.8, 0.7]],
                [[0.4, 0.5], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True], [True, False]])
        aromatic = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

        losses = atom_query_loss(output, anchors, mask, aromatic)
        decoded = decode_atom_queries(output, threshold=0.0)

        self.assertEqual(output["centers"].shape, (2, 8, 2))
        self.assertTrue(torch.equal(output["centers"], output["first_centers"]))
        self.assertEqual(output["count_logits"].shape, (2, 9))
        self.assertEqual(output["count_value"].shape, (2,))
        self.assertEqual(output["cardinality_hidden"].shape, (2, 32))
        self.assertEqual(output["role_logits"].shape, (2, 8, 2))
        self.assertEqual(output["selection_logits"].shape, (2, 8))
        self.assertEqual(output["base_presence_logits"].shape, (2, 8))
        self.assertEqual(output["ranking_presence_logits"].shape, (2, 8))
        self.assertEqual(output["local_presence_logits"].shape, (2, 8))
        self.assertEqual(output["selection_context_logits"].shape, (2, 8))
        self.assertTrue(
            torch.allclose(
                output["presence_logits"],
                output["base_presence_logits"],
            )
        )
        self.assertTrue(
            torch.allclose(
                output["selection_logits"],
                output["ranking_presence_logits"]
                + output["local_presence_logits"]
                + output["selection_context_logits"],
            )
        )
        self.assertEqual(output["duplicate_logits"].shape, (2, 8, 8))
        self.assertTrue(
            torch.allclose(output["centers"][0], detector.reference_points, atol=1e-6)
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual([len(items) for items in decoded], [8, 8])

    def test_each_nonfinal_decoder_layer_has_auxiliary_output(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=3,
            heads=4,
            ffn_dim=64,
        )

        output = AtomQueryDetector(config)(torch.randn(1, 16, 8, 8))

        self.assertEqual(len(output["auxiliary_outputs"]), config.layers - 1)
        self.assertTrue(
            all(
                item["centers"].shape == (1, config.queries, 2)
                for item in output["auxiliary_outputs"]
            )
        )

    def test_auxiliary_loss_weight_changes_total_loss(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=3,
            heads=4,
            ffn_dim=64,
        )
        output = AtomQueryDetector(config)(torch.randn(1, 16, 8, 8))
        anchors = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.zeros(1, 2)

        base = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            auxiliary_loss_weight=0.0,
        )
        supervised = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            auxiliary_loss_weight=0.5,
        )

        self.assertGreater(float(supervised["auxiliary_loss"].detach()), 0.0)
        self.assertGreater(
            float(supervised["loss"].detach()),
            float(base["loss"].detach()),
        )

    def test_auxiliary_coordinate_heads_do_not_update_final_coordinate_head(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=3,
            heads=4,
            ffn_dim=64,
        )
        detector = AtomQueryDetector(config)
        output = detector(torch.randn(1, 16, 8, 8))

        sum(
            item["centers"].sum() for item in output["auxiliary_outputs"]
        ).backward()

        self.assertIsNone(detector.center[-1].weight.grad)
        self.assertIsNotNone(detector.auxiliary_centers[0][-1].weight.grad)

    def test_invalid_head_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtomQueryConfig(d_model=190, heads=6)

    def test_dense_candidates_snap_one_to_one(self) -> None:
        queries = [
            {"x": 0.1, "y": 0.1, "score": 1.0, "aromatic": False},
            {"x": 0.9, "y": 0.9, "score": 1.0, "aromatic": False},
        ]
        candidates = [
            {"x": 0.12, "y": 0.1, "score": 0.8, "aromatic": False},
            {"x": 0.88, "y": 0.9, "score": 0.9, "aromatic": False},
        ]

        snapped = snap_queries_to_dense_candidates(
            queries,
            candidates,
            image_size=100,
            maximum_distance_pixels=5,
        )

        self.assertEqual(snapped[0]["x"], 0.12)
        self.assertEqual(snapped[1]["x"], 0.88)

    def test_checkpoint_metric_uses_f1_when_exact_rates_tie(self) -> None:
        weaker = {
            "vertex_exact_rate": 0.0,
            "f1": 0.1,
            "count_exact_rate": 0.2,
            "count_mae": 3.0,
        }
        stronger = dict(weaker, f1=0.2)

        self.assertGreater(metric_key(stronger), metric_key(weaker))

    def test_checkpoint_metric_can_prioritize_f1_for_selector_training(self) -> None:
        higher_exact = {
            "vertex_exact_rate": 0.001,
            "f1": 0.53,
            "count_exact_rate": 1.0,
            "count_mae": 0.0,
        }
        higher_f1 = dict(higher_exact, vertex_exact_rate=0.0, f1=0.55)

        self.assertGreater(
            metric_key(higher_f1, f1_first=True),
            metric_key(higher_exact, f1_first=True),
        )

    def test_checkpoint_f1_gate_rejects_coordinate_regression(self) -> None:
        self.assertFalse(checkpoint_is_eligible({"f1": 0.38}, 0.39))
        self.assertTrue(checkpoint_is_eligible({"f1": 0.39}, 0.39))

    def test_global_cardinality_mode_keeps_coordinate_heads_frozen(self) -> None:
        detector = AtomQueryDetector(
            AtomQueryConfig(
                input_channels=16,
                d_model=32,
                queries=8,
                layers=3,
                heads=4,
                ffn_dim=64,
            )
        )

        global_parameters, query_parameters = set_global_cardinality_trainable(
            detector,
            2,
        )

        self.assertTrue(global_parameters)
        self.assertTrue(query_parameters)
        self.assertTrue(detector.cardinality_token.weight.requires_grad)
        self.assertTrue(detector.decoder.layers[-1].self_attn.in_proj_weight.requires_grad)
        self.assertFalse(detector.center[-1].weight.requires_grad)
        self.assertFalse(detector.second_local_refinement[-1].weight.requires_grad)

    def test_cardinality_decode_uses_rounded_probability_sum(self) -> None:
        output = {
            "presence_logits": torch.logit(torch.tensor([[0.9, 0.8, 0.2, 0.1]])),
            "centers": torch.tensor(
                [[[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]]]
            ),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_cardinality(output)

        self.assertEqual(len(decoded[0]), 2)
        self.assertGreaterEqual(decoded[0][0]["score"], decoded[0][1]["score"])

    def test_cardinality_uses_presence_for_count_and_selection_for_ranking(self) -> None:
        output = {
            "presence_logits": torch.logit(torch.tensor([[0.9, 0.8, 0.2, 0.1]])),
            "selection_logits": torch.tensor([[0.0, 1.0, 4.0, 3.0]]),
            "centers": torch.tensor(
                [[[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]]]
            ),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_cardinality(output)

        self.assertEqual(len(decoded[0]), 2)
        self.assertEqual([round(item["x"], 1) for item in decoded[0]], [0.3, 0.4])

    def test_cardinality_loss_weight_changes_total_loss(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        output = AtomQueryDetector(config)(torch.randn(1, 16, 8, 8))
        anchors = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.zeros(1, 2)

        low = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            cardinality_loss_weight=0.0,
        )
        high = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            cardinality_loss_weight=1.0,
        )

        self.assertGreater(
            float(high["loss"].detach()),
            float(low["loss"].detach()),
        )

    def test_center_tail_loss_weight_changes_total_loss(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        output = AtomQueryDetector(config)(torch.randn(1, 16, 8, 8))
        anchors = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.zeros(1, 2)

        base = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            center_tail_loss_weight=0.0,
        )
        tail = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            center_tail_loss_weight=1.0,
        )

        self.assertGreater(float(tail["center_tail_loss"].detach()), 0.0)
        self.assertGreater(float(tail["loss"].detach()), float(base["loss"].detach()))

    def test_selection_loss_trains_local_query_ranking(self) -> None:
        detector = AtomQueryDetector(
            AtomQueryConfig(
                input_channels=16,
                d_model=32,
                queries=8,
                layers=1,
                heads=4,
                ffn_dim=64,
            )
        )
        output = detector(torch.randn(1, 16, 8, 8))
        anchors = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.zeros(1, 2)

        losses = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            selection_loss_weight=1.0,
            selection_ranking_loss_weight=1.0,
            duplicate_loss_weight=1.0,
        )
        losses["loss"].backward()

        self.assertGreater(float(losses["selection_loss"].detach()), 0.0)
        self.assertGreater(float(losses["selection_ranking_loss"].detach()), 0.0)
        self.assertGreater(float(losses["duplicate_loss"].detach()), 0.0)
        self.assertIsNotNone(detector.selection[-1].weight.grad)
        self.assertIsNotNone(detector.duplicate_projection.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in detector.parameters()
            )
        )

    def test_context_only_selection_keeps_local_score_frozen(self) -> None:
        detector = AtomQueryDetector(
            AtomQueryConfig(
                input_channels=16,
                d_model=32,
                queries=8,
                layers=1,
                heads=4,
                ffn_dim=64,
            )
        )

        trainable = set_selection_trainable(detector, context_only=True)

        self.assertTrue(trainable)
        self.assertFalse(detector.selection[-1].weight.requires_grad)
        self.assertTrue(detector.selection_context_score.weight.requires_grad)
        self.assertFalse(detector.center[-1].weight.requires_grad)

    def test_duplicate_only_mode_keeps_selection_and_coordinates_frozen(self) -> None:
        detector = AtomQueryDetector(
            AtomQueryConfig(
                input_channels=16,
                d_model=32,
                queries=8,
                layers=1,
                heads=4,
                ffn_dim=64,
            )
        )

        trainable = set_duplicate_trainable(detector)

        self.assertTrue(trainable)
        self.assertTrue(detector.duplicate_projection.weight.requires_grad)
        self.assertFalse(detector.selection[-1].weight.requires_grad)
        self.assertFalse(detector.center[-1].weight.requires_grad)

    def test_role_loss_trains_visible_vs_implicit_vertex_class(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        output = AtomQueryDetector(config)(torch.randn(1, 16, 8, 8))
        anchors = torch.tensor([[[0.2, 0.3], [0.8, 0.7]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.zeros(1, 2)
        roles = torch.tensor([[0, 1]])

        losses = atom_query_loss(
            output,
            anchors,
            mask,
            aromatic,
            role_targets=roles,
            role_loss_weight=1.0,
        )
        losses["loss"].backward()

        self.assertGreater(float(losses["role_loss"].detach()), 0.0)
        self.assertIsNotNone(output["role_logits"].grad_fn)

    def test_count_head_decode_uses_predicted_integer_count(self) -> None:
        output = {
            "presence_logits": torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
            "count_logits": torch.tensor([[0.0, 0.0, 4.0, 0.0, 0.0]]),
            "centers": torch.tensor(
                [[[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]]]
            ),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_count_head(output)

        self.assertEqual(len(decoded[0]), 2)

    def test_count_regression_decode_rounds_predicted_count(self) -> None:
        output = {
            "presence_logits": torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
            "count_value": torch.tensor([2.4]),
            "centers": torch.rand(1, 4, 2),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_count_regression(output)

        self.assertEqual(len(decoded[0]), 2)

    def test_known_count_decode_is_explicit_evaluation_oracle(self) -> None:
        output = {
            "presence_logits": torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
            "centers": torch.rand(1, 4, 2),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_known_counts(output, torch.tensor([3]))

        self.assertEqual(len(decoded[0]), 3)

    def test_assignment_oracle_selects_unique_queries_nearest_truth(self) -> None:
        output = {
            "presence_logits": torch.zeros(1, 4),
            "centers": torch.tensor(
                [[[0.1, 0.1], [0.9, 0.9], [0.5, 0.5], [0.52, 0.5]]]
            ),
            "aromatic_logits": torch.zeros(1, 4),
        }
        targets = torch.tensor([[[0.11, 0.1], [0.51, 0.5]]])
        mask = torch.tensor([[True, True]])

        decoded = decode_atom_queries_by_assignment_oracle(output, targets, mask)

        self.assertEqual(len(decoded[0]), 2)
        self.assertEqual({round(item["x"], 1) for item in decoded[0]}, {0.1, 0.5})

    def test_training_assignment_ignores_presence_confidence(self) -> None:
        output = {
            "presence_logits": torch.logit(torch.tensor([[0.01, 0.99]])),
            "centers": torch.tensor([[[0.11, 0.1], [0.2, 0.1]]]),
            "role_logits": torch.zeros(1, 2, 2),
        }
        targets = torch.tensor([[[0.1, 0.1]]])
        mask = torch.tensor([[True]])

        assignments = hungarian_assignments(output, targets, mask)

        self.assertEqual(assignments[0][0].tolist(), [0])

    def test_final_local_score_changes_ranking_without_changing_presence(self) -> None:
        detector = AtomQueryDetector(
            AtomQueryConfig(
                input_channels=16,
                d_model=32,
                queries=8,
                layers=1,
                heads=4,
                ffn_dim=64,
            )
        )
        features = torch.randn(1, 16, 8, 8)
        before = detector(features)
        with torch.no_grad():
            detector.selection[-1].bias.fill_(1.0)
        after = detector(features)

        self.assertTrue(
            torch.allclose(
                after["selection_logits"],
                before["selection_logits"] + 1.0,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.equal(after["presence_logits"], before["presence_logits"])
        )

    def test_selection_decode_uses_local_ranking_with_known_count(self) -> None:
        output = {
            "selection_logits": torch.tensor([[0.0, 3.0, 2.0, 1.0]]),
            "centers": torch.rand(1, 4, 2),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_selection_known_counts(
            output,
            torch.tensor([2]),
        )

        self.assertEqual(len(decoded[0]), 2)
        self.assertGreaterEqual(decoded[0][0]["score"], decoded[0][1]["score"])

    def test_duplicate_penalty_decode_preserves_known_count(self) -> None:
        output = {
            "selection_logits": torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
            "duplicate_logits": torch.tensor(
                [
                    [
                        [0.0, 8.0, -8.0, -8.0],
                        [8.0, 0.0, -8.0, -8.0],
                        [-8.0, -8.0, 0.0, -8.0],
                        [-8.0, -8.0, -8.0, 0.0],
                    ]
                ]
            ),
            "centers": torch.tensor(
                [[[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]]]
            ),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_duplicate_penalty_known_counts(
            output,
            torch.tensor([2]),
            duplicate_penalty=4.0,
        )

        self.assertEqual(len(decoded[0]), 2)
        self.assertAlmostEqual(decoded[0][0]["x"], 0.1)
        self.assertAlmostEqual(decoded[0][1]["x"], 0.3)

    def test_poisson_binomial_count_distribution_is_normalized(self) -> None:
        logits = torch.logit(torch.tensor([[0.9, 0.8, 0.1]]))

        log_probs = poisson_binomial_log_probs(logits)

        self.assertTrue(torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1)))
        self.assertEqual(int(log_probs.argmax(dim=-1)[0]), 2)

    def test_set_count_decode_uses_distribution_mode(self) -> None:
        output = {
            "presence_logits": torch.logit(torch.tensor([[0.9, 0.8, 0.1, 0.1]])),
            "centers": torch.rand(1, 4, 2),
            "aromatic_logits": torch.zeros(1, 4),
        }

        decoded = decode_atom_queries_by_set_count(output)

        self.assertEqual(len(decoded[0]), 2)

    def test_old_checkpoint_can_omit_count_head(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        source = AtomQueryDetector(config)
        old_state = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("count_head.")
            and not key.startswith("count_regression.")
            and not key.startswith("cardinality_token.")
            and not key.startswith("cardinality_decoder.")
            and not key.startswith("cardinality_norm.")
            and not key.startswith("selection.")
            and not key.startswith("selection_local_projection.")
            and not key.startswith("selection_context_projection.")
            and not key.startswith("selection_position_projection.")
            and not key.startswith("selection_context.")
            and not key.startswith("selection_context_score.")
            and not key.startswith("ranking_presence.")
            and not key.startswith("duplicate_projection.")
            and not key.startswith("duplicate_geometry.")
            and key != "duplicate_similarity_scale"
            and not key.startswith("auxiliary_")
            and not key.startswith("role.")
            and not key.startswith("second_local_refinement.")
        }
        target = AtomQueryDetector(config)

        load_atom_query_state(target, old_state)

        self.assertTrue(
            torch.equal(
                target.selection_local_projection.weight,
                target.local_projection.weight,
            )
        )
        self.assertTrue(
            torch.equal(target.ranking_presence.weight, target.presence.weight)
        )

    def test_count_head_can_be_the_only_trainable_module(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        detector = AtomQueryDetector(config)
        detector.requires_grad_(False)
        detector.count_head.requires_grad_(True)

        trainable_names = {
            name for name, parameter in detector.named_parameters() if parameter.requires_grad
        }

        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("count_head.") for name in trainable_names))

    def test_count_decoder_mode_opens_only_presence_and_last_decoder_layer(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=3,
            heads=4,
            ffn_dim=64,
        )
        detector = AtomQueryDetector(config)

        set_count_decoder_trainable(detector, 1)
        trainable_names = {
            name for name, parameter in detector.named_parameters() if parameter.requires_grad
        }

        self.assertIn("presence.weight", trainable_names)
        self.assertTrue(any(name.startswith("decoder.layers.2.") for name in trainable_names))
        self.assertTrue(any(name.startswith("decoder.norm.") for name in trainable_names))
        self.assertFalse(any(name.startswith("decoder.layers.1.") for name in trainable_names))
        self.assertFalse(any(name.startswith("center.") for name in trainable_names))

    def test_vision_modules_remain_frozen_when_not_selected(self) -> None:
        base = tiny_base_model()

        trainable = trainable_vision_parameters(
            base,
            train_stem=False,
            unfreeze_stroke_blocks=0,
        )

        self.assertEqual(trainable, [])
        self.assertFalse(
            any(parameter.requires_grad for parameter in base.vision_encoder.parameters())
        )

    def test_only_selected_vision_modules_are_trainable(self) -> None:
        base = tiny_base_model()

        trainable_vision_parameters(
            base,
            train_stem=True,
            unfreeze_stroke_blocks=1,
        )
        trainable_names = {
            name
            for name, parameter in base.vision_encoder.named_parameters()
            if parameter.requires_grad
        }

        self.assertTrue(any(name.startswith("stem.") for name in trainable_names))
        self.assertTrue(any(name.startswith("stages.0.1.") for name in trainable_names))
        self.assertFalse(any(name.startswith("stages.0.0.") for name in trainable_names))

    def test_vision_checkpoint_state_round_trips(self) -> None:
        source = tiny_base_model()
        target = tiny_base_model()
        with torch.no_grad():
            for parameter in source.vision_encoder.parameters():
                parameter.add_(1.0)

        payload = {"base_vision_updates": vision_updates(source)}
        apply_atom_query_vision_updates(target, payload)

        for source_value, target_value in zip(
            source.vision_encoder.state_dict().values(),
            target.vision_encoder.state_dict().values(),
        ):
            self.assertTrue(torch.equal(source_value, target_value))

    def test_packed_dense_head_state_is_applied(self) -> None:
        class DenseBase(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vertex_head = nn.Linear(4, 2)

        source = DenseBase()
        target = DenseBase()
        with torch.no_grad():
            source.vertex_head.weight.fill_(2.0)
            source.vertex_head.bias.fill_(3.0)

        apply_atom_query_dense_head(
            target,
            {"dense_vertex_head": source.vertex_head.state_dict()},
        )

        self.assertTrue(
            torch.equal(target.vertex_head.weight, source.vertex_head.weight)
        )
        self.assertTrue(torch.equal(target.vertex_head.bias, source.vertex_head.bias))


if __name__ == "__main__":
    unittest.main()
