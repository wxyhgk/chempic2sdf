import unittest

import torch

from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    LINE,
    NO_OBJECT,
    QUAD,
    TEXT_BOX,
    TRIANGLE,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    decode_primitive_set,
    geometry_shape,
    line_endpoint_aligned_losses,
    pairwise_geometry_cost,
    pairwise_line_geometry_cost,
    primitive_set_loss,
)


class PrimitiveSetGeometryTest(unittest.TestCase):
    def test_line_endpoint_reversal_has_zero_cost(self):
        predicted = torch.tensor([[0.1, 0.2, 0.8, 0.7, 0, 0, 0, 0]])
        target = torch.tensor([[0.8, 0.7, 0.1, 0.2, 0, 0, 0, 0]])

        coordinate, _ = pairwise_geometry_cost(
            predicted,
            target,
            torch.tensor([LINE]),
        )

        self.assertAlmostEqual(float(coordinate[0, 0]), 0.0, places=7)

    def test_endpoint_aligned_line_objective_is_swap_invariant(self):
        predicted = torch.tensor([[0.1, 0.2, 0.8, 0.7, 0, 0, 0, 0]])
        target = torch.tensor([[0.8, 0.7, 0.1, 0.2, 0, 0, 0, 0]])

        pairwise = pairwise_line_geometry_cost(predicted, target)
        losses = line_endpoint_aligned_losses(predicted, target)

        for cost in pairwise:
            self.assertAlmostEqual(float(cost[0, 0]), 0.0, places=7)
        for loss in losses.values():
            self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_line_orientation_gate_stops_zero_length_and_ramps_to_eight_pixels(self):
        target = torch.tensor([[0.4, 0.5, 0.6, 0.5, 0, 0, 0, 0]])
        zero_length = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]],
            requires_grad=True,
        )

        zero_losses = line_endpoint_aligned_losses(zero_length, target)
        zero_losses["line_orientation_loss"].backward()

        self.assertEqual(
            float(zero_losses["line_orientation_loss"].detach()),
            0.0,
        )
        torch.testing.assert_close(zero_length.grad, torch.zeros_like(zero_length))

        half_gate = torch.tensor(
            [
                [
                    0.5,
                    0.5 - 2.0 / 768.0,
                    0.5,
                    0.5 + 2.0 / 768.0,
                    0,
                    0,
                    0,
                    0,
                ]
            ]
        )
        half_losses = line_endpoint_aligned_losses(half_gate, target)
        short_reversed = half_gate.clone()
        short_reversed[:, :4] = half_gate[:, :4].reshape(-1, 2, 2).flip(1).flatten(1)
        exact_short_losses = line_endpoint_aligned_losses(
            half_gate,
            short_reversed,
        )

        self.assertAlmostEqual(
            float(half_losses["line_orientation_loss"]),
            0.5,
            places=5,
        )
        self.assertAlmostEqual(
            float(exact_short_losses["line_orientation_loss"]),
            0.0,
            places=7,
        )

    def test_polygon_rotation_and_reversal_have_zero_cost(self):
        predicted_triangle = torch.tensor(
            [[0.8, 0.2, 0.5, 0.8, 0.2, 0.2, 0.0, 0.0]]
        )
        reversed_rotated = torch.tensor(
            [[0.2, 0.2, 0.5, 0.8, 0.8, 0.2, 0.0, 0.0]]
        )

        coordinate, _ = pairwise_geometry_cost(
            predicted_triangle,
            reversed_rotated,
            torch.tensor([TRIANGLE]),
        )

        self.assertAlmostEqual(float(coordinate[0, 0]), 0.0, places=7)

    def test_text_shape_is_center_width_height(self):
        geometry = torch.tensor([[0.4, 0.6, 0.2, 0.1, 0, 0, 0, 0]])
        torch.testing.assert_close(
            geometry_shape(geometry, TEXT_BOX),
            geometry[:, :4],
        )


class PrimitiveSetModelTest(unittest.TestCase):
    def setUp(self):
        self.vision_config = ChemVectorConfig(
            vocab_size=31,
            pad_token_id=0,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16, 32, 64),
            vision_depths=(1, 1, 1, 1),
        )
        self.set_config = PrimitiveSetConfig(
            queries=6,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            dropout=0.0,
        )

    def test_forward_shapes_and_backward(self):
        model = PrimitiveSetPredictor(self.vision_config, self.set_config)
        pixels = torch.randn(2, 3, 64, 64)

        output = model(pixels)
        loss = output["type_logits"].square().mean() + output["geometry"].mean()
        loss.backward()

        self.assertEqual(output["type_logits"].shape, (2, 6, 5))
        self.assertEqual(output["count_logits"].shape, (2, 7))
        self.assertEqual(output["geometry"].shape, (2, 6, 8))
        self.assertEqual(output["text_geometry"].shape, (2, 6, 8))
        self.assertEqual(output["stroke"].shape, (2, 6))
        self.assertIsNotNone(model.query_embedding.weight.grad)

    def test_legacy_config_has_no_local_query_state_and_strictly_loads(self):
        legacy_payload = {
            "queries": 6,
            "d_model": 32,
            "decoder_layers": 2,
            "decoder_heads": 4,
            "decoder_ffn_dim": 64,
            "memory_grid_size": 4,
            "dropout": 0.0,
        }
        config = PrimitiveSetConfig(**legacy_payload)
        source = PrimitiveSetPredictor(self.vision_config, config)
        restored = PrimitiveSetPredictor(self.vision_config, config)

        restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(config.local_query_kernel, 0)
        self.assertEqual(config.endpoint_refine_kernel, 0)
        self.assertFalse(config.line_center_vector)
        self.assertFalse(
            any(name.startswith("local_query_projection.") for name in source.state_dict())
        )
        self.assertFalse(
            any(name.startswith("endpoint_") for name in source.state_dict())
        )
        self.assertFalse(
            any(name.startswith("line_geometry_head.") for name in source.state_dict())
        )
        pixels = torch.randn(1, 3, 64, 64)
        source.eval()
        restored.eval()
        with torch.inference_mode():
            source_output = source(pixels)
            restored_output = restored(pixels)
        self.assertNotIn("coarse_geometry", source_output)
        self.assertNotIn("line_geometry", source_output)
        self.assertEqual(source_output.keys(), restored_output.keys())
        for name in source_output:
            torch.testing.assert_close(
                source_output[name],
                restored_output[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_local_query_injection_responds_and_backpropagates(self):
        config = PrimitiveSetConfig(
            queries=6,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=8,
            dropout=0.0,
            local_query_kernel=3,
        )
        model = PrimitiveSetPredictor(self.vision_config, config).eval()
        images = torch.ones(2, 3, 64, 64)
        images[1, :, 4:20, 4:20] = -1.0
        captured = []

        def capture_queries(_module, inputs):
            captured.append(inputs[0])

        handle = model.decoder.register_forward_pre_hook(capture_queries)
        output = model(images)
        handle.remove()
        output["geometry"][1, 0].sum().backward()

        self.assertEqual(captured[0].shape, (2, 6, 32))
        self.assertGreater(
            float((captured[0][0] - captured[0][1]).abs().max().detach()),
            1e-6,
        )
        self.assertGreater(
            float(model.local_query_projection[0].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.local_query_projection[2].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(
                (output["geometry"][0] - output["geometry"][1])
                .abs()
                .max()
                .detach()
            ),
            1e-7,
        )

    def test_local_query_kernel_rejects_even_values(self):
        with self.assertRaisesRegex(ValueError, "zero or odd"):
            PrimitiveSetConfig(local_query_kernel=4)

    def test_endpoint_refinement_uses_stride_four_map_and_backpropagates(self):
        config = PrimitiveSetConfig(
            queries=6,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=8,
            dropout=0.0,
            endpoint_refine_kernel=3,
        )
        model = PrimitiveSetPredictor(self.vision_config, config).eval()
        images = torch.ones(2, 3, 64, 64)
        images[1, :, 2:24, 2:24] = -1.0
        refine_inputs = []
        stem_calls = []

        projection_handle = model.endpoint_refine_projection[0].register_forward_pre_hook(
            lambda _module, inputs: refine_inputs.append(inputs[0])
        )
        stem_handle = model.vision_encoder.stem.register_forward_hook(
            lambda _module, _inputs, _output: stem_calls.append(1)
        )
        output = model(images)
        projection_handle.remove()
        stem_handle.remove()

        self.assertEqual(len(stem_calls), 1)
        self.assertEqual(
            refine_inputs[0].shape,
            (2, self.vision_config.vision_dims[0] + 2, 16, 16),
        )
        self.assertEqual(output["coarse_geometry"].shape, (2, 6, 8))
        residual = output["geometry"] - output["coarse_geometry"]
        self.assertGreater(
            float((residual[0] - residual[1]).abs().max().detach()),
            1e-8,
        )

        output["geometry"][1, 0].sum().backward()

        for parameter in (
            model.endpoint_refine_projection[0].weight,
            model.endpoint_refine_depthwise[0].weight,
            model.endpoint_refine_depthwise[3].weight,
            model.endpoint_point_embedding.weight,
            model.endpoint_refine_mlp[-1].weight,
            model.vision_encoder.stem[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_endpoint_refinement_samples_feature_cell_centers(self):
        config = PrimitiveSetConfig(
            queries=1,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            dropout=0.0,
            endpoint_refine_kernel=3,
        )
        model = PrimitiveSetPredictor(self.vision_config, config).eval()
        height, width = 5, 6
        refine_map = torch.zeros(
            1,
            model.endpoint_refine_channels,
            height,
            width,
        )
        refine_map[0, 0] = torch.tensor(
            [[10.0 * y + x for x in range(width)] for y in range(height)]
        )
        center = torch.tensor((3.5 / width, 2.5 / height))
        coarse_geometry = center.repeat(4).reshape(1, 1, 8)
        object_features = torch.zeros(1, 1, config.d_model)
        captured = []
        handle = model.endpoint_refine_mlp.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0])
        )

        model._refine_endpoints(coarse_geometry, object_features, refine_map)
        handle.remove()

        sampled = captured[0][0, 0, :, config.d_model]
        torch.testing.assert_close(sampled, torch.full((4,), 23.0))

    def test_line_center_vector_geometry_is_bounded_symmetric_and_differentiable(self):
        config = PrimitiveSetConfig(
            queries=1,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            dropout=0.0,
            line_center_vector=True,
        )
        model = PrimitiveSetPredictor(self.vision_config, config)
        raw_outputs = []
        handle = model.line_geometry_head.register_forward_hook(
            lambda _module, _inputs, output: raw_outputs.append(output)
        )
        output = model(torch.randn(1, 3, 64, 64))
        handle.remove()

        self.assertEqual(output["line_geometry"].shape, (1, 1, 8))
        points = output["line_geometry"][..., :4].reshape(1, 1, 2, 2)
        self.assertTrue(bool((points >= 0.0).all()))
        self.assertTrue(bool((points <= 1.0).all()))
        torch.testing.assert_close(
            output["line_geometry"][..., 4:],
            torch.zeros_like(output["line_geometry"][..., 4:]),
        )
        references = model.reference_points.to(raw_outputs[0])
        expected_center = torch.sigmoid(
            model._inverse_sigmoid(references)[None] + raw_outputs[0][..., :2]
        )
        torch.testing.assert_close(points.mean(dim=-2), expected_center)

        targets = {
            "types": torch.tensor([[LINE]]),
            "geometry": torch.tensor(
                [[[0.1, 0.2, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0]]]
            ),
            "stroke": torch.zeros(1, 1),
            "has_stroke": torch.tensor([[False]]),
            "target_mask": torch.tensor([[True]]),
        }
        losses = primitive_set_loss(output, targets)
        losses["coordinate_loss"].backward()

        self.assertGreater(
            float(model.line_geometry_head[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(model.geometry_head[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_line_geometry_drives_matching_loss_and_decode_with_swapped_endpoints(self):
        target = torch.tensor(
            [[[0.8, 0.7, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]]]
        )
        generic = torch.zeros(1, 2, 8)
        generic[0, 0] = target[0, 0]
        specialized = torch.zeros(1, 2, 8)
        specialized[0, 1, :4] = torch.tensor([0.1, 0.2, 0.8, 0.7])
        output = {
            "type_logits": torch.tensor(
                [[[6.0, -6.0, -6.0, -6.0, -6.0], [-6.0, 6.0, -6.0, -6.0, -6.0]]]
            ),
            "count_logits": torch.tensor([[-6.0, 6.0, -6.0]]),
            "geometry": generic,
            "line_geometry": specialized,
            "stroke": torch.zeros(1, 2),
        }
        targets = {
            "types": torch.tensor([[LINE]]),
            "geometry": target,
            "stroke": torch.zeros(1, 1),
            "has_stroke": torch.tensor([[False]]),
            "target_mask": torch.tensor([[True]]),
        }

        losses = primitive_set_loss(output, targets, class_cost_weight=0.0)
        decoded = decode_primitive_set(output, threshold=0.5)

        self.assertEqual(losses["assignments"][0][0].tolist(), [1])
        self.assertAlmostEqual(float(losses["coordinate_loss"]), 0.0, places=7)
        self.assertEqual(len(decoded[0]), 1)
        torch.testing.assert_close(decoded[0][0]["geometry"], specialized[0, 1])

    def test_line_geometry_is_refined_independently_from_generic_geometry(self):
        config = PrimitiveSetConfig(
            queries=2,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            dropout=0.0,
            endpoint_refine_kernel=3,
            line_center_vector=True,
        )
        model = PrimitiveSetPredictor(self.vision_config, config)
        refine_calls = []
        handle = model.endpoint_refine_mlp.register_forward_hook(
            lambda _module, _inputs, output: refine_calls.append(output)
        )
        output = model(torch.randn(1, 3, 64, 64))
        handle.remove()

        self.assertEqual(len(refine_calls), 2)
        self.assertEqual(output["coarse_line_geometry"].shape, (1, 2, 8))
        self.assertEqual(output["line_geometry"].shape, (1, 2, 8))
        self.assertGreater(
            float(
                (
                    output["line_geometry"][..., :4]
                    - output["coarse_line_geometry"][..., :4]
                ).abs().max().detach()
            ),
            0.0,
        )
        torch.testing.assert_close(
            output["line_geometry"][..., 4:],
            torch.zeros_like(output["line_geometry"][..., 4:]),
        )

        output["line_geometry"][..., 0].sum().backward()
        self.assertGreater(
            float(model.line_geometry_head[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.endpoint_refine_mlp[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_endpoint_refine_kernel_rejects_even_values(self):
        with self.assertRaisesRegex(ValueError, "zero or odd"):
            PrimitiveSetConfig(endpoint_refine_kernel=24)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this smoke test")
    def test_endpoint_refinement_kernel_25_cuda_smoke(self):
        config = PrimitiveSetConfig(
            queries=6,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=8,
            dropout=0.0,
            endpoint_refine_kernel=25,
            line_center_vector=True,
        )
        model = PrimitiveSetPredictor(self.vision_config, config).cuda().train()
        refine_shapes = []
        handle = model.endpoint_refine_projection[0].register_forward_pre_hook(
            lambda _module, inputs: refine_shapes.append(tuple(inputs[0].shape))
        )
        images = torch.randn(1, 3, 768, 768, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = (
                output["geometry"].float().square().mean()
                + output["line_geometry"].float().square().mean()
            )
        handle.remove()
        loss.backward()

        self.assertEqual(
            refine_shapes,
            [(1, self.vision_config.vision_dims[0] + 2, 192, 192)],
        )
        self.assertTrue(torch.isfinite(output["geometry"]).all())
        self.assertTrue(torch.isfinite(output["line_geometry"]).all())
        self.assertGreater(
            float(model.endpoint_refine_depthwise[0].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.line_geometry_head[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_loss_matches_all_types_and_is_differentiable(self):
        type_logits = torch.full((1, 6, 5), -4.0)
        type_logits[..., NO_OBJECT] = 4.0
        type_logits[0, 0] = torch.tensor([-4.0, 4.0, -4.0, -4.0, -4.0])
        type_logits[0, 1] = torch.tensor([-4.0, -4.0, -4.0, -4.0, 4.0])
        geometry = torch.full((1, 6, 8), 0.5)
        geometry[0, 0, :4] = torch.tensor([0.1, 0.2, 0.8, 0.7])
        geometry[0, 1, :4] = torch.tensor([0.5, 0.5, 0.2, 0.1])
        stroke = torch.full((1, 6), 15.0 / 63.0)
        count_logits = torch.full((1, 7), -4.0)
        count_logits[0, 2] = 4.0
        output = {
            "type_logits": type_logits.requires_grad_(),
            "count_logits": count_logits.requires_grad_(),
            "geometry": geometry.requires_grad_(),
            "stroke": stroke.requires_grad_(),
        }
        targets = {
            "types": torch.tensor([[LINE, TEXT_BOX]]),
            "geometry": torch.tensor(
                [[[0.8, 0.7, 0.1, 0.2, 0, 0, 0, 0], [0.5, 0.5, 0.2, 0.1, 0, 0, 0, 0]]]
            ).float(),
            "stroke": torch.tensor([[15.0 / 63.0, 0.0]]),
            "has_stroke": torch.tensor([[True, False]]),
            "target_mask": torch.tensor([[True, True]]),
        }

        losses = primitive_set_loss(output, targets)
        without_count = primitive_set_loss(
            output,
            targets,
            count_classification_weight=0.0,
        )
        losses["loss"].backward()

        self.assertLess(float(losses["coordinate_loss"].detach()), 1e-6)
        self.assertLess(float(losses["stroke_loss"].detach()), 1e-6)
        self.assertLess(float(losses["count_classification_loss"].detach()), 0.01)
        torch.testing.assert_close(
            losses["loss"].detach() - without_count["loss"].detach(),
            0.25 * losses["count_loss"].detach(),
        )
        self.assertIsNotNone(output["geometry"].grad)
        self.assertIsNotNone(output["count_logits"].grad)
        self.assertEqual(len(losses["assignments"][0][0]), 2)

    def test_disabled_line_objective_is_bit_exact_with_explicit_legacy_options(self):
        torch.manual_seed(20260721)
        output = {
            "type_logits": torch.randn(1, 3, 5),
            "count_logits": torch.randn(1, 4),
            "geometry": torch.rand(1, 3, 8),
            "text_geometry": torch.rand(1, 3, 8),
            "stroke": torch.rand(1, 3),
        }
        targets = {
            "types": torch.tensor([[LINE, TEXT_BOX, NO_OBJECT]]),
            "geometry": torch.rand(1, 3, 8),
            "stroke": torch.rand(1, 3),
            "has_stroke": torch.tensor([[True, False, False]]),
            "target_mask": torch.tensor([[True, True, False]]),
        }

        implicit = primitive_set_loss(output, targets)
        explicit = primitive_set_loss(
            output,
            targets,
            line_geometry_objective=False,
            class_cost_weight=2.0,
        )

        for name in (
            "loss",
            "classification_loss",
            "coordinate_loss",
            "shape_loss",
            "stroke_loss",
            "cardinality_loss",
            "count_classification_loss",
            "line_endpoint_loss",
            "line_center_loss",
            "line_length_loss",
            "line_orientation_loss",
        ):
            self.assertTrue(torch.equal(implicit[name], explicit[name]), name)
        for left, right in zip(implicit["assignments"], explicit["assignments"]):
            torch.testing.assert_close(left[0], right[0])
            torch.testing.assert_close(left[1], right[1])
        for name in (
            "line_endpoint_loss",
            "line_center_loss",
            "line_length_loss",
            "line_orientation_loss",
        ):
            self.assertEqual(float(implicit[name]), 0.0)

    def test_enabled_line_objective_excludes_line_from_generic_geometry(self):
        type_logits = torch.tensor([[[-4.0, 4.0, -4.0, -4.0, -4.0]]])
        count_logits = torch.tensor([[-4.0, 4.0]])
        predicted_geometry = torch.tensor(
            [[[0.1, 0.1, 0.2, 0.1, 0, 0, 0, 0]]],
            requires_grad=True,
        )
        output = {
            "type_logits": type_logits,
            "count_logits": count_logits,
            "geometry": predicted_geometry,
            "stroke": torch.zeros(1, 1),
        }
        targets = {
            "types": torch.tensor([[LINE]]),
            "geometry": torch.tensor(
                [[[0.8, 0.8, 0.9, 0.8, 0, 0, 0, 0]]]
            ),
            "stroke": torch.zeros(1, 1),
            "has_stroke": torch.tensor([[False]]),
            "target_mask": torch.tensor([[True]]),
        }

        losses = primitive_set_loss(
            output,
            targets,
            line_geometry_objective=True,
            class_cost_weight=0.5,
        )
        losses["loss"].backward()

        self.assertEqual(float(losses["coordinate_loss"].detach()), 0.0)
        self.assertEqual(float(losses["shape_loss"].detach()), 0.0)
        self.assertGreater(float(losses["line_endpoint_loss"].detach()), 0.0)
        self.assertGreater(float(losses["line_center_loss"].detach()), 0.0)
        self.assertIsNotNone(predicted_geometry.grad)
        self.assertGreater(float(predicted_geometry.grad.abs().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this smoke test")
    def test_line_geometry_objective_cuda_backward(self):
        device = torch.device("cuda")
        geometry = torch.tensor(
            [[[0.2, 0.2, 0.7, 0.6, 0, 0, 0, 0]]],
            device=device,
            requires_grad=True,
        )
        output = {
            "type_logits": torch.tensor(
                [[[-4.0, 4.0, -4.0, -4.0, -4.0]]],
                device=device,
                requires_grad=True,
            ),
            "count_logits": torch.tensor(
                [[-4.0, 4.0]],
                device=device,
                requires_grad=True,
            ),
            "geometry": geometry,
            "stroke": torch.zeros(1, 1, device=device, requires_grad=True),
        }
        targets = {
            "types": torch.tensor([[LINE]], device=device),
            "geometry": torch.tensor(
                [[[0.1, 0.2, 0.8, 0.7, 0, 0, 0, 0]]],
                device=device,
            ),
            "stroke": torch.zeros(1, 1, device=device),
            "has_stroke": torch.tensor([[True]], device=device),
            "target_mask": torch.tensor([[True]], device=device),
        }

        losses = primitive_set_loss(
            output,
            targets,
            line_geometry_objective=True,
            class_cost_weight=0.5,
        )
        losses["loss"].backward()

        for name in (
            "loss",
            "line_endpoint_loss",
            "line_center_loss",
            "line_length_loss",
            "line_orientation_loss",
        ):
            self.assertTrue(torch.isfinite(losses[name]))
        self.assertGreater(float(geometry.grad.abs().sum()), 0.0)

    def test_decode_requires_object_to_beat_no_object(self):
        logits = torch.tensor(
            [[[0.0, 3.0, 0.0, 0.0, 0.0], [4.0, 3.0, 0.0, 0.0, 0.0]]]
        )
        output = {
            "type_logits": logits,
            "geometry": torch.zeros(1, 2, 8),
            "text_geometry": torch.zeros(1, 2, 8),
            "stroke": torch.zeros(1, 2),
        }

        decoded = decode_primitive_set(output, threshold=0.5)

        self.assertEqual(len(decoded[0]), 1)
        self.assertEqual(decoded[0][0]["type"], LINE)

    def test_count_head_decode_keeps_exact_predicted_top_k(self):
        logits = torch.tensor(
            [
                [
                    [4.0, 3.9, 0.0, 0.0, 0.0],
                    [4.0, 3.0, 3.8, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 3.7, 0.0],
                ]
            ]
        )
        count_logits = torch.full((1, 4), -5.0)
        count_logits[0, 2] = 5.0
        output = {
            "type_logits": logits,
            "count_logits": count_logits,
            "geometry": torch.zeros(1, 3, 8),
            "text_geometry": torch.zeros(1, 3, 8),
            "stroke": torch.zeros(1, 3),
        }

        threshold = decode_primitive_set(output, threshold=0.5)
        counted = decode_primitive_set(output, threshold=0.99, mode="count_head")

        self.assertEqual(len(threshold[0]), 0)
        self.assertEqual(len(counted[0]), 2)
        self.assertEqual(
            [row["query_index"] for row in counted[0]],
            [0, 1],
        )

    def test_first_geometry_loss_reaches_queries_decoder_and_vision(self):
        torch.manual_seed(20260721)
        model = PrimitiveSetPredictor(self.vision_config, self.set_config)
        output = model(torch.randn(1, 3, 64, 64))
        targets = {
            "types": torch.tensor([[LINE, 0, 0, 0, 0, 0]]),
            "geometry": torch.tensor(
                [[[0.1, 0.2, 0.8, 0.7, 0, 0, 0, 0]] * 6]
            ).float(),
            "stroke": torch.zeros(1, 6),
            "has_stroke": torch.zeros(1, 6, dtype=torch.bool),
            "target_mask": torch.tensor([[True, False, False, False, False, False]]),
        }

        losses = primitive_set_loss(output, targets)
        losses["coordinate_loss"].backward()

        self.assertGreater(float(model.geometry_head[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.query_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(
                model.decoder.layers[0]
                .multihead_attn.in_proj_weight.grad.abs()
                .sum()
            ),
            0.0,
        )
        self.assertGreater(float(model.memory_projection[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.vision_encoder.stem[0].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            int(torch.count_nonzero(model.text_geometry_head[-1].weight)),
            0,
        )

    def test_quad_equivalence_is_supported(self):
        square = torch.tensor(
            [[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]]
        )
        rotated = torch.tensor(
            [[0.8, 0.8, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1]]
        )
        coordinate, _ = pairwise_geometry_cost(square, rotated, torch.tensor([QUAD]))
        self.assertAlmostEqual(float(coordinate[0, 0]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
