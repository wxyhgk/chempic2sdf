from argparse import Namespace
import unittest

import torch
from torch import nn

from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms.train_g3_p1g_equivalent_set_matching import (
    DEFAULT_OUTPUT,
    MODEL_TYPE,
    _checkpoint_payload,
    _gate,
    build_parser,
    p1g_training_step,
    validate_args,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    module_state_sha256,
)


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


class G3P1GEquivalentSetTrainingTests(unittest.TestCase):
    def test_training_step_changes_only_ranker(self) -> None:
        from chemvector.components.atoms.g3_top128_set_ranker import (
            G3Top128SetRanker,
        )

        config = tiny_config()
        atom = G3AtomOnlyPredictor(config).atom_head.requires_grad_(False).eval()
        with torch.no_grad():
            atom.presence.weight.zero_()
            atom.presence.bias.zero_()
            atom.anchor_offset.weight.zero_()
            atom.anchor_offset.bias.zero_()
        ranker = G3Top128SetRanker(config).train()
        vision = TinyVision().requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=3e-4)
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

        losses, evidence = p1g_training_step(
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
        self.assertEqual(
            evidence["fixed"]["representable_counts"].tolist(),
            evidence["dynamic"]["representable_counts"].tolist(),
        )

    def test_gate_separates_causal_and_capability_decisions(self) -> None:
        route = {"known_count": {"f1_6px": 0.94}}
        causal_only = _gate(
            {"protocol": True},
            {"exact_rate_6px": 0.30, "atom_recall_6px": 0.97},
            {"exact_rate_6px": 0.18},
            route,
        )
        full = _gate(
            {"protocol": True},
            {"exact_rate_6px": 0.55, "atom_recall_6px": 0.99},
            {"exact_rate_6px": 0.18},
            route,
        )

        self.assertEqual(causal_only["decision"], "causal_go_capability_no_go")
        self.assertEqual(full["decision"], "causal_go_capability_go")

    def test_cli_and_checkpoint_identity_are_p1g(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.output_dir, str(DEFAULT_OUTPUT))
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            validate_args(build_parser().parse_args(["--max-steps", "1"]))
        ranker = nn.Linear(1, 1)
        payload = _checkpoint_payload(
            ranker,
            torch.optim.AdamW(ranker.parameters()),
            Namespace(batch_size=4),
            tiny_config(),
            {},
            {},
            1,
            1,
            4,
            {},
            {},
            None,
            None,
            None,
            None,
            [],
        )

        self.assertEqual(payload["model_type"], MODEL_TYPE)
        self.assertNotIn("atom_head", payload)
        self.assertNotIn("vision", payload)


if __name__ == "__main__":
    unittest.main()
