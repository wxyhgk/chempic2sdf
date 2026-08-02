from argparse import Namespace
import inspect
import unittest

import torch
from torch import nn

from chemvector.components.atoms.g3_top128_set_ranker import G3Top128SetRanker
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms.train_g3_p1c_top128_set_ranker import (
    DEFAULT_OUTPUT,
    FrozenSetAuditAccumulator,
    _checkpoint_payload,
    build_parser,
    load_p1a_set_ranker,
    p1c_training_step,
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
        self.projection = nn.Conv2d(3, 4, 1)

    def forward_stroke_resolution(self, pixels):
        return self.projection(pixels)


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


class G3P1CTop128SetRankerTrainerTests(unittest.TestCase):
    def test_frozen_set_audit_hashes_identity_and_positive_membership(self) -> None:
        candidates = {
            "candidate_ids": torch.arange(128)[None],
            "candidate_anchors": torch.zeros(1, 128, 2),
            "candidate_features": torch.zeros(1, 128, 384),
        }
        positive = torch.zeros(1, 128, dtype=torch.bool)
        positive[0, :2] = True
        canonical = torch.zeros_like(positive)
        canonical[0, 0] = True
        hybrid = {
            "positive_mask": positive,
            "canonical_positive_mask": canonical,
            "representable_counts": torch.tensor([2]),
            "canonical_counts": torch.tensor([1]),
            "target_counts": torch.tensor([2]),
        }
        first = FrozenSetAuditAccumulator()
        second = FrozenSetAuditAccumulator()
        first.add(["sample"], candidates, hybrid)
        second.add(["sample"], candidates, hybrid)

        self.assertEqual(first.finalize(), second.finalize())
        changed = FrozenSetAuditAccumulator()
        changed_hybrid = dict(hybrid)
        changed_hybrid["positive_mask"] = positive.roll(1, dims=1)
        changed.add(["sample"], candidates, changed_hybrid)
        self.assertNotEqual(
            first.finalize()["positive_flat_ids_sha256"],
            changed.finalize()["positive_flat_ids_sha256"],
        )

    def test_ranker_forward_is_target_and_count_free(self) -> None:
        ranker = G3Top128SetRanker(tiny_config())

        self.assertEqual(
            tuple(inspect.signature(ranker.forward).parameters), ("candidates",)
        )

    def test_real_p1a_loads_with_only_set_ranker_trainable(self) -> None:
        atom, ranker, payload, config = load_p1a_set_ranker(torch.device("cpu"))

        self.assertEqual(payload["global_step"], 6_250)
        self.assertEqual(config.maximum_atoms, 128)
        self.assertFalse(any(value.requires_grad for value in atom.parameters()))
        self.assertTrue(all(value.requires_grad for value in ranker.parameters()))

    def test_training_step_changes_only_set_ranker(self) -> None:
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

        losses, evidence = p1c_training_step(
            atom,
            ranker,
            vision,
            optimizer,
            batch,
            device=torch.device("cpu"),
            precision="fp32",
            max_grad_norm=1.0,
        )

        self.assertGreater(losses["gradient_norm"], 0)
        self.assertNotEqual(ranker_before, module_state_sha256(ranker))
        self.assertEqual(atom_before, module_state_sha256(atom))
        self.assertEqual(vision_before, module_state_sha256(vision))
        accumulator = FrozenSetAuditAccumulator()
        accumulator.add(batch["keys"], evidence["candidates"], evidence["hybrid"])
        audit = accumulator.finalize()
        self.assertEqual(audit["candidate_count"], 256)
        self.assertGreater(audit["representable_positive_count"], 0)

    def test_optimizer_and_checkpoint_are_set_ranker_only(self) -> None:
        config = tiny_config()
        atom = G3AtomOnlyPredictor(config).atom_head.requires_grad_(False)
        ranker = G3Top128SetRanker(config)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4)
        validate_optimizer(atom, ranker, optimizer)
        invalid = torch.optim.AdamW(
            list(ranker.parameters()) + [next(atom.parameters())], lr=3e-4
        )
        with self.assertRaisesRegex(ValueError, "only set-ranker"):
            validate_optimizer(atom, ranker, invalid)
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
        self.assertIn("set_ranker", payload)
        self.assertNotIn("atom_head", payload)
        self.assertNotIn("vision", payload)

    def test_gate_and_fixed_cli(self) -> None:
        metrics = {
            "known_count": {
                "precision_6px": 0.97,
                "recall_6px": 0.97,
                "f1_6px": 0.97,
                "matched_atoms": 2_988,
                "whole_set_exact_images": 50,
            },
            "canonical_identity": {"recall": 0.95},
            "top128_coverage": {"matched": 3_072},
            "fixed_positive_localization": {"matched": 3_080, "total": 3_080},
        }
        self.assertEqual(selector_gate({"protocol": True}, metrics)["decision"], "GO")
        metrics["known_count"]["matched_atoms"] = 2_987
        self.assertEqual(selector_gate({"protocol": True}, metrics)["decision"], "NO-GO")
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.output_dir, str(DEFAULT_OUTPUT))
        with self.assertRaisesRegex(ValueError, "fixed learning_rate"):
            validate_args(build_parser().parse_args(["--learning-rate", "0.0002"]))


if __name__ == "__main__":
    unittest.main()
