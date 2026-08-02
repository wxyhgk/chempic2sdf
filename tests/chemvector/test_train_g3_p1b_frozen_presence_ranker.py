from argparse import Namespace
from pathlib import Path
import unittest

import torch
from torch import nn

from chemvector.components.atoms.g3_presence_ranker import (
    G3FrozenAtomPresenceRanker,
)
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms.train_g3_p1b_frozen_presence_ranker import (
    DEFAULT_OUTPUT,
    EXPECTED_RESIDUAL_PARAMETERS,
    MODEL_TYPE,
    P1A_CHECKPOINT,
    _checkpoint_payload,
    build_parser,
    load_p1a_ranker,
    p1b_training_step,
    selector_gate,
    validate_args,
    validate_optimizer,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    module_state_sha256,
)


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward_stroke_resolution(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels)


def tiny_config() -> G3HierarchicalGraphConfig:
    return G3HierarchicalGraphConfig(
        input_channels=4,
        hidden_channels=256,
        d_model=32,
        atom_slots_per_cell=3,
        maximum_atoms=4,
        bond_slots_per_cell=1,
        maximum_bonds=4,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=64,
        memory_stride=1,
        incidence_hidden_dim=16,
        incidence_pair_hidden_dim=8,
        dropout=0.0,
    )


class G3P1BFrozenPresenceRankerTrainerTests(unittest.TestCase):
    def test_real_p1a_fixed_final_loads_strictly_on_cpu(self) -> None:
        ranker, payload, config = load_p1a_ranker(torch.device("cpu"))

        self.assertEqual(payload["global_step"], 6_250)
        self.assertEqual(payload["p1a_gate"]["protocol_status"], "passed")
        self.assertEqual(payload["p1a_gate"]["decision"], "NO-GO")
        self.assertEqual(config.hidden_channels, 256)
        self.assertEqual(
            sum(parameter.numel() for parameter in ranker.presence_ranker.parameters()),
            EXPECTED_RESIDUAL_PARAMETERS,
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in ranker.atom_head.parameters())
        )

    def test_training_step_changes_only_presence_ranker(self) -> None:
        config = tiny_config()
        base = G3AtomOnlyPredictor(config)
        ranker = G3FrozenAtomPresenceRanker(base.atom_head, config).train()
        vision = TinyVision().requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(
            ranker.presence_ranker.parameters(), lr=3e-4, weight_decay=0.01
        )
        batch = {
            "pixel_values": torch.randn(2, 3, 8, 8),
            "atom_anchors": torch.tensor(
                [[[0.25, 0.25], [0.75, 0.75], [0.0, 0.0], [0.0, 0.0]]] * 2
            ),
            "atom_mask": torch.tensor([[True, True, False, False]] * 2),
            "keys": ["a", "b"],
        }
        atom_before = module_state_sha256(ranker.atom_head)
        vision_before = module_state_sha256(vision)
        residual_before = module_state_sha256(ranker.presence_ranker)

        losses = p1b_training_step(
            ranker,
            vision,
            optimizer,
            batch,
            device=torch.device("cpu"),
            precision="fp32",
            max_grad_norm=1.0,
        )

        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertNotEqual(
            residual_before, module_state_sha256(ranker.presence_ranker)
        )
        self.assertEqual(atom_before, module_state_sha256(ranker.atom_head))
        self.assertEqual(vision_before, module_state_sha256(vision))
        self.assertFalse(ranker.atom_head.training)
        self.assertFalse(
            any(parameter.grad is not None for parameter in ranker.atom_head.parameters())
        )

    def test_optimizer_rejects_any_frozen_atom_parameter(self) -> None:
        config = tiny_config()
        ranker = G3FrozenAtomPresenceRanker(
            G3AtomOnlyPredictor(config).atom_head, config
        )
        valid = torch.optim.AdamW(ranker.presence_ranker.parameters(), lr=3e-4)
        validate_optimizer(ranker, valid)
        invalid = torch.optim.AdamW(
            list(ranker.presence_ranker.parameters())
            + [next(ranker.atom_head.parameters())],
            lr=3e-4,
        )
        with self.assertRaisesRegex(ValueError, "only residual"):
            validate_optimizer(ranker, invalid)

    def test_gate_requires_all_selector_and_protocol_guards(self) -> None:
        metrics = {
            "known_count": {
                "precision_6px": 0.97,
                "recall_6px": 0.97,
                "f1_6px": 0.97,
                "whole_set_exact_images": 50,
            },
            "canonical_identity": {"recall": 0.95},
            "top128_coverage": {"matched": 3_072},
            "fixed_positive_localization": {"matched": 3_080, "total": 3_080},
        }

        self.assertEqual(selector_gate({"protocol": True}, metrics)["decision"], "GO")
        metrics["known_count"]["whole_set_exact_images"] = 49
        self.assertEqual(
            selector_gate({"protocol": True}, metrics)["decision"], "NO-GO"
        )
        metrics["known_count"]["whole_set_exact_images"] = 50
        self.assertEqual(
            selector_gate({"protocol": False}, metrics)["protocol_status"], "failed"
        )

    def test_cli_and_checkpoint_payload_are_residual_only(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(Path(args.output_dir), DEFAULT_OUTPUT)
        self.assertEqual(Path(args.p1a_checkpoint), P1A_CHECKPOINT)
        with self.assertRaisesRegex(ValueError, "fixed learning_rate"):
            validate_args(
                build_parser().parse_args(["--learning-rate", "0.0002"])
            )

        config = tiny_config()
        ranker = G3FrozenAtomPresenceRanker(
            G3AtomOnlyPredictor(config).atom_head, config
        )
        optimizer = torch.optim.AdamW(ranker.presence_ranker.parameters(), lr=3e-4)
        payload = _checkpoint_payload(
            ranker=ranker,
            optimizer=optimizer,
            args=Namespace(batch_size=4),
            config=config,
            lineage={},
            audit={},
            step=1_250,
            epoch=1,
            exposure=5_000,
            losses={"loss": 1.0},
            step0_metrics={},
            final_metrics=None,
            gate=None,
            epoch_audits=[],
        )

        self.assertEqual(payload["model_type"], MODEL_TYPE)
        self.assertIn("residual", payload)
        self.assertNotIn("atom_head", payload)
        self.assertNotIn("vision", payload)


if __name__ == "__main__":
    unittest.main()
