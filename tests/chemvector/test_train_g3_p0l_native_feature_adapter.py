from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.atom_feature_adapter import (
    ATOM_FEATURE_ADAPTER_PARAMETERS,
    ATOM_FEATURE_DIM,
    AtomFeatureAdapter,
)
from chemvector.g3_hierarchical_graph import (
    G3AtomPairBondIdentityPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.train_g3_p0l_native_feature_adapter import (
    DEFAULT_OUTPUT,
    MODEL_TYPE,
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT_SHA256,
    build_parser,
    checkpoint_payload,
    evaluate_p0l,
    load_p0k_pair_overlay,
    masked_feature_distillation_loss,
    p0l_training_step,
    validate_adapter_optimizer,
    validate_args,
    validate_frozen_scope,
)


def tiny_g3() -> G3AtomPairBondIdentityPredictor:
    return G3AtomPairBondIdentityPredictor(
        G3HierarchicalGraphConfig(
            input_channels=8,
            hidden_channels=16,
            d_model=16,
            atom_slots_per_cell=2,
            maximum_atoms=4,
            bond_slots_per_cell=2,
            maximum_bonds=3,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_stride=1,
        )
    )


class G3P0LNativeFeatureAdapterTests(unittest.TestCase):
    def test_fixed_architecture_parameter_count_and_zero_init_identity(self) -> None:
        adapter = AtomFeatureAdapter()
        inputs = torch.randn(2, 5, ATOM_FEATURE_DIM)

        output = adapter(inputs)

        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            ATOM_FEATURE_ADAPTER_PARAMETERS,
        )
        self.assertFalse(adapter.norm.elementwise_affine)
        self.assertTrue(torch.equal(output, inputs))
        self.assertTrue(torch.count_nonzero(adapter.projection.weight) == 0)
        self.assertTrue(torch.count_nonzero(adapter.projection.bias) == 0)

    def test_loss_ignores_padding_and_detaches_teacher(self) -> None:
        adapter = AtomFeatureAdapter()
        native = torch.randn(1, 3, ATOM_FEATURE_DIM)
        teacher = torch.randn(1, 3, ATOM_FEATURE_DIM, requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        changed_padding = teacher.detach().clone()
        changed_padding[:, 2] = 1e6

        loss = masked_feature_distillation_loss(adapter, native, teacher, mask)
        changed = masked_feature_distillation_loss(
            adapter, native, changed_padding, mask
        )
        loss.backward()

        torch.testing.assert_close(loss.detach(), changed.detach())
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher.detach().grad_fn)

    def test_optimizer_and_training_step_update_only_adapter(self) -> None:
        adapter = AtomFeatureAdapter()
        frozen = torch.nn.Linear(8, 8).requires_grad_(False)
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=1e-3, weight_decay=0.0
        )
        validate_adapter_optimizer(adapter, optimizer)
        native = torch.randn(1, 3, ATOM_FEATURE_DIM)
        teacher = torch.randn(1, 3, ATOM_FEATURE_DIM)
        mask = torch.tensor([[True, True, False]])
        adapter_before = {
            name: value.detach().clone() for name, value in adapter.state_dict().items()
        }
        frozen_before = {
            name: value.detach().clone() for name, value in frozen.state_dict().items()
        }

        losses = p0l_training_step(
            adapter,
            optimizer,
            native,
            teacher,
            mask,
            precision="fp32",
        )

        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(adapter_before[name], value)
                for name, value in adapter.state_dict().items()
            )
        )
        for name, value in frozen.state_dict().items():
            self.assertTrue(torch.equal(frozen_before[name], value))
        invalid = torch.optim.AdamW(
            list(adapter.parameters()) + list(frozen.parameters()), lr=1e-3
        )
        with self.assertRaisesRegex(ValueError, "only adapter"):
            validate_adapter_optimizer(adapter, invalid)

    def test_frozen_scope_rejects_trainable_sources(self) -> None:
        p0i = torch.nn.Linear(2, 2).requires_grad_(False)
        p0g = torch.nn.Linear(2, 2).requires_grad_(False)
        adapter = AtomFeatureAdapter()
        validate_frozen_scope(p0i, p0g, adapter)

        p0g.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "must remain frozen"):
            validate_frozen_scope(p0i, p0g, adapter)

    def test_p0k_pair_state_is_strictly_overlaid_and_frozen(self) -> None:
        source = tiny_g3()
        target = tiny_g3()
        with torch.no_grad():
            for parameter in source.pair_bond.parameters():
                parameter.add_(0.25)
        payload = {
            "model_type": "g3-predicted-anchor-relation-alignment-p0k-v0.1",
            "global_step": 400,
            "source_p0i_checkpoint": {"sha256": P0I_CHECKPOINT_SHA256},
            "source_p0g_checkpoint": {"sha256": P0G_CHECKPOINT_SHA256},
            "pair_bond": source.pair_bond.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "p0k.pt"
            torch.save(payload, checkpoint)
            with patch(
                "chemvector.train_g3_p0l_native_feature_adapter.file_sha256",
                return_value="expected",
            ), patch(
                "chemvector.train_g3_p0l_native_feature_adapter.P0K_CHECKPOINT_SHA256",
                "expected",
            ):
                restored = load_p0k_pair_overlay(
                    target, checkpoint, torch.device("cpu")
                )

        self.assertEqual(restored["global_step"], 400)
        for name, value in source.pair_bond.state_dict().items():
            torch.testing.assert_close(target.pair_bond.state_dict()[name], value)
        self.assertFalse(any(parameter.requires_grad for parameter in target.parameters()))

    def test_evaluation_reports_adapted_control_and_oracle(self) -> None:
        branch = {
            "images": 32,
            "atom_targets": 1424,
            "atom_matched": 1424,
            "representable_bonds": 1508,
            "target_bonds": 1508,
            "pair_universe": 37380,
            "pair_identity_sha256": "a" * 64,
            "gt_bond_count": {"typed_graph_exact_images": 32},
            "deploy_bond_count": {"typed_graph_exact_images": 32},
        }
        oracle = {
            "gt_count": {"structural_graph_exact_images": 32},
            "deploy_count": {"structural_graph_exact_images": 32},
        }
        cache = {
            "native_features": torch.zeros(1, 1, ATOM_FEATURE_DIM),
            "teacher_features": torch.zeros(1, 1, ATOM_FEATURE_DIM),
            "atom_mask": torch.tensor([[True]]),
        }
        with patch(
            "chemvector.train_g3_p0l_native_feature_adapter.evaluate_adapted_composition",
            return_value=branch,
        ), patch(
            "chemvector.train_g3_p0l_native_feature_adapter.evaluate_composition_mode",
            return_value=dict(branch),
        ), patch(
            "chemvector.train_g3_p0l_native_feature_adapter.evaluate_g3_count_modes",
            return_value=oracle,
        ):
            result = evaluate_p0l(
                AtomFeatureAdapter(),
                object(),
                object(),
                torch.empty(1),
                {},
                ["sample"],
                cache,
                batch_size=1,
                precision="fp32",
                step0_smooth_l1=1.0,
            )

        self.assertEqual(result["p0l_gate"]["status"], "passed")
        self.assertLessEqual(
            result["feature_alignment"]["smooth_l1_ratio_to_step0"], 0.1
        )
        self.assertIn("cosine_similarity", result["feature_alignment"])
        self.assertIn("nrmse", result["feature_alignment"])
        self.assertIn("adapted_native_feature", result)
        self.assertIn("same_slot_p0k_control", result)
        self.assertIn("gt_atom_oracle_control", result)

    def test_cli_is_fixed_and_checkpoint_contains_no_frozen_models(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(Path(args.output_dir), DEFAULT_OUTPUT)
        invalid = build_parser().parse_args(["--max-steps", "401"])
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            validate_args(invalid)

        adapter = AtomFeatureAdapter()
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=1e-3, weight_decay=0.0
        )
        payload = checkpoint_payload(
            adapter=adapter,
            optimizer=optimizer,
            args=Namespace(batch_size=4),
            lineage={"p0i": {}, "p0g": {}, "p0k": {}, "vision": {}},
            step=100,
            losses={"loss": 1.0},
            metrics={"p0l_gate": {"status": "failed"}},
            step0_feature_alignment={"smooth_l1": 1.0},
        )

        self.assertEqual(payload["model_type"], MODEL_TYPE)
        self.assertEqual(
            set(payload),
            {
                "checkpoint_schema_version",
                "model_type",
                "adapter",
                "optimizer",
                "global_step",
                "samples_seen",
                "training_args",
                "last_losses",
                "train_only_metrics",
                "step0_feature_alignment",
                "lineage",
            },
        )
        self.assertNotIn("model", payload)
        self.assertNotIn("pair_bond", payload)


if __name__ == "__main__":
    unittest.main()
