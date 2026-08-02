from argparse import Namespace
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from chemvector.components.atoms.g3_boundary_band_loss import (
    G3_BOUNDARY_BAND_SIZE,
    boundary_band_ranking_loss,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    G3Top128SetRanker,
)
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms.train_g3_p1d_boundary_band_set_ranker import (
    DEFAULT_OUTPUT,
    MODEL_TYPE,
    build_parser,
    checkpoint_payload,
    hybrid_topm_statistics,
    p1d_training_step,
    target_free_ranker_contract,
    validate_args,
    validate_optimizer,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    module_state_sha256,
)


def ranker_config():
    return SimpleNamespace(d_model=384, maximum_atoms=128, atom_slots_per_cell=3)


def tiny_config():
    return G3HierarchicalGraphConfig(
        input_channels=4,
        hidden_channels=16,
        d_model=384,
        atom_slots_per_cell=3,
        maximum_atoms=128,
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


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward_stroke_resolution(self, pixels):
        return self.projection(pixels)


class G3P1DBoundaryBandTests(unittest.TestCase):
    def test_loss_uses_only_weakest_and_strongest_eight(self) -> None:
        logits = torch.arange(20, dtype=torch.float32)[None]
        positive = torch.zeros(1, 20, dtype=torch.bool)
        positive[0, 10:] = True
        ranked = {"ranking_logits": logits}
        hybrid = {"positive_mask": positive}

        result = boundary_band_ranking_loss(ranked, hybrid)

        weakest = logits[0, 10:].topk(8, largest=False).values
        strongest = logits[0, :10].topk(8).values
        expected = torch.nn.functional.softplus(
            strongest[:, None] - weakest[None]
        ).mean()
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(float(result["weak_positive_count"]), 8.0)
        self.assertEqual(float(result["hard_negative_count"]), 8.0)
        self.assertEqual(float(result["boundary_exact_images"]), 1.0)

    def test_boundary_diagnostic_and_gradient(self) -> None:
        logits = torch.tensor(
            [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 3.0, 2.0]],
            requires_grad=True,
        )
        positive = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )

        result = boundary_band_ranking_loss(
            {"ranking_logits": logits},
            {"positive_mask": positive},
        )
        result["loss"].backward()

        self.assertEqual(float(result["boundary_exact_images"]), 1.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertEqual(float(result["positive_count"]), 4.0)
        self.assertEqual(float(result["negative_count"]), 4.0)

    def test_invalid_band_and_empty_class_are_rejected(self) -> None:
        ranked = {"ranking_logits": torch.zeros(1, 4)}
        with self.assertRaisesRegex(ValueError, "positive integer"):
            boundary_band_ranking_loss(
                ranked, {"positive_mask": torch.zeros(1, 4, dtype=torch.bool)}, 0
            )
        with self.assertRaisesRegex(ValueError, "representable positive"):
            boundary_band_ranking_loss(
                ranked, {"positive_mask": torch.zeros(1, 4, dtype=torch.bool)}
            )

    def test_p1d_ranker_starts_zero_initialized(self) -> None:
        ranker = G3Top128SetRanker(ranker_config())

        self.assertEqual(G3_BOUNDARY_BAND_SIZE, 8)
        self.assertEqual(int(torch.count_nonzero(ranker.residual_head.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(ranker.residual_head.bias)), 0)
        contract = target_free_ranker_contract()
        self.assertTrue(contract["passed"])
        self.assertEqual(contract["arguments"], ["self", "candidates"])
        self.assertEqual(contract["forbidden_identifiers"], [])

    def test_hybrid_topm_statistics_are_set_exact(self) -> None:
        ids = torch.arange(128)[None]
        positive = torch.zeros(1, 128, dtype=torch.bool)
        positive[0, [2, 7]] = True
        logits = torch.zeros(1, 128)
        logits[0, 2] = 3.0
        logits[0, 7] = 2.0
        ranked = {
            "candidate_ids": ids,
            "candidate_anchors": torch.zeros(1, 128, 2),
            "candidate_features": torch.zeros(1, 128, 384),
            "ranking_logits": logits,
        }

        result = hybrid_topm_statistics(ranked, {"positive_mask": positive})

        self.assertEqual(result["matched_positive_count"], 2)
        self.assertEqual(result["whole_set_exact_images"], 1)
        self.assertEqual(result["boundary_exact_images"], 1)

    def test_training_step_changes_only_new_ranker(self) -> None:
        config = tiny_config()
        atom = G3AtomOnlyPredictor(config).atom_head.requires_grad_(False).eval()
        with torch.no_grad():
            atom.presence.weight.zero_()
            atom.presence.bias.zero_()
            atom.anchor_offset.weight.zero_()
            atom.anchor_offset.bias.zero_()
        ranker = G3Top128SetRanker(config).train()
        vision = TinyVision().requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=0.01)
        batch = {
            "pixel_values": torch.randn(2, 3, 8, 8),
            "atom_anchors": torch.tensor(
                [[[0.1875, 0.1875], [0.6875, 0.6875]] + [[0.0, 0.0]] * 126]
                * 2
            ),
            "atom_mask": torch.tensor([[True, True] + [False] * 126] * 2),
            "keys": ["a", "b"],
        }
        atom_before = module_state_sha256(atom)
        vision_before = module_state_sha256(vision)
        ranker_before = module_state_sha256(ranker)

        losses, evidence = p1d_training_step(
            atom,
            ranker,
            vision,
            optimizer,
            batch,
            device=torch.device("cpu"),
            precision="fp32",
            max_grad_norm=1.0,
        )

        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertNotEqual(ranker_before, module_state_sha256(ranker))
        self.assertEqual(atom_before, module_state_sha256(atom))
        self.assertEqual(vision_before, module_state_sha256(vision))
        self.assertIn("positive_mask", evidence["hybrid"])
        validate_optimizer(atom, ranker, optimizer)

    def test_checkpoint_cli_and_lineage_are_p1d_only(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.output_dir, str(DEFAULT_OUTPUT))
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            validate_args(build_parser().parse_args(["--max-steps", "6251"]))
        config = tiny_config()
        ranker = G3Top128SetRanker(config)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4)
        payload = checkpoint_payload(
            ranker=ranker,
            optimizer=optimizer,
            args=Namespace(batch_size=4),
            config=config,
            lineage={"p1c_checkpoint_loaded": False},
            audit={},
            step=1_250,
            epoch=1,
            exposure=5_000,
            losses={"loss": 1.0},
            step0_metrics={},
            final_metrics=None,
            train_final_diagnostic=None,
            gate=None,
            epoch_audits=[],
        )

        self.assertEqual(payload["model_type"], MODEL_TYPE)
        self.assertIn("set_ranker", payload)
        self.assertNotIn("atom_head", payload)
        self.assertNotIn("vision", payload)
        self.assertNotIn("p1c_checkpoint", payload)


if __name__ == "__main__":
    unittest.main()
