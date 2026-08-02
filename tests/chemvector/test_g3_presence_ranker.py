from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.atoms.g3_presence_ranker import (
    G3_PRESENCE_RANKER_PARAMETERS,
    G3FrozenAtomPresenceRanker,
    canonical_presence_ranking_loss,
)


class TinyFrozenAtomHead(nn.Module):
    def __init__(self, *, maximum_atoms: int = 4) -> None:
        super().__init__()
        self.hidden = nn.Conv2d(4, 256, kernel_size=1)
        self.presence = nn.Conv2d(256, 3, kernel_size=1)
        self.count = nn.Linear(256, maximum_atoms + 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.hidden(features)
        batch, _, height, width = hidden.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=hidden.device, dtype=hidden.dtype),
            torch.arange(width, device=hidden.device, dtype=hidden.dtype),
            indexing="ij",
        )
        anchors = torch.stack(
            (
                (x + 0.5) / float(width),
                (y + 0.5) / float(height),
            )
        )[None, None].expand(batch, 3, 2, height, width)
        slot_features = hidden[:, None, :4].expand(batch, 3, 4, height, width)
        offsets = hidden.new_full((batch, 3, 2, height, width), 0.5)
        return {
            "object_features": hidden,
            "presence_logits": self.presence(hidden),
            "anchor_offsets": offsets,
            "anchors": anchors,
            "slot_features": slot_features,
            "count_logits": self.count(hidden.mean(dim=(-2, -1))),
        }


def config(maximum_atoms: int = 4):
    return SimpleNamespace(
        hidden_channels=256,
        atom_slots_per_cell=3,
        maximum_atoms=maximum_atoms,
    )


class G3PresenceRankerTests(unittest.TestCase):
    def test_parameter_count_and_zero_init_dense_parity(self) -> None:
        model = G3FrozenAtomPresenceRanker(TinyFrozenAtomHead(), config())
        features = torch.randn(2, 4, 2, 3)

        dense = model.forward_dense(features)

        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, G3_PRESENCE_RANKER_PARAMETERS)
        self.assertEqual(trainable, 16_643)
        torch.testing.assert_close(
            dense["presence_logits"], dense["base_presence_logits"], rtol=0, atol=0
        )
        self.assertEqual(int(torch.count_nonzero(dense["presence_residual"])), 0)
        self.assertEqual(
            int(
                torch.count_nonzero(
                    model.presence_ranker.output_projection.weight
                )
            ),
            0,
        )
        self.assertEqual(
            int(
                torch.count_nonzero(model.presence_ranker.output_projection.bias)
            ),
            0,
        )

    def test_train_keeps_atom_head_frozen_and_eval(self) -> None:
        atom_head = TinyFrozenAtomHead()
        model = G3FrozenAtomPresenceRanker(atom_head, config())

        model.train()

        self.assertTrue(model.training)
        self.assertTrue(model.presence_ranker.training)
        self.assertFalse(model.atom_head.training)
        self.assertFalse(any(parameter.requires_grad for parameter in atom_head.parameters()))
        features = torch.randn(1, 4, 2, 2, requires_grad=True)
        dense = model.forward_dense(features)
        dense["presence_logits"].sum().backward()
        self.assertIsNone(features.grad)
        self.assertFalse(any(parameter.grad is not None for parameter in atom_head.parameters()))
        self.assertIsNotNone(model.presence_ranker.output_projection.weight.grad)

    def test_forward_reuses_frozen_count_and_supports_forced_count(self) -> None:
        model = G3FrozenAtomPresenceRanker(
            TinyFrozenAtomHead(maximum_atoms=4), config(maximum_atoms=4)
        ).eval()
        features = torch.randn(2, 4, 2, 2)

        deployed = model(features)
        forced = model(features, atom_counts=torch.tensor([1, 3]))

        torch.testing.assert_close(
            deployed["atom_counts"], deployed["atom_count_logits"].argmax(dim=-1)
        )
        self.assertEqual(forced["atom_mask"].sum(dim=1).tolist(), [1, 3])
        torch.testing.assert_close(
            forced["atom_dense_output"]["presence_logits"],
            forced["atom_dense_output"]["base_presence_logits"],
            rtol=0,
            atol=0,
        )

    def test_loss_uses_top_four_k_negatives_with_zero_margin(self) -> None:
        scores = torch.tensor([[[[0.0, 1.0]], [[0.5, 0.25]], [[0.0, -100.0]]]])
        positive = torch.zeros_like(scores, dtype=torch.bool)
        positive[0, 0, 0, 0] = True
        dense = {
            "presence_logits": scores,
            "anchors": torch.zeros(1, 3, 2, 1, 2),
        }
        targets = {
            "atom_anchors": torch.tensor([[[0.0, 0.0]]]),
            "atom_mask": torch.tensor([[True]]),
        }

        result = canonical_presence_ranking_loss(dense, targets)
        expected = F.softplus(torch.tensor([1.0, 0.5, 0.25, 0.0])).mean()

        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(float(result["positive_count"]), 1.0)
        self.assertEqual(float(result["hard_negative_count"]), 4.0)

    def test_loss_backpropagates_only_through_ranked_logits(self) -> None:
        model = G3FrozenAtomPresenceRanker(TinyFrozenAtomHead(), config())
        features = torch.randn(1, 4, 2, 2, requires_grad=True)
        dense = model.forward_dense(features)
        targets = {
            "atom_anchors": torch.tensor([[[0.25, 0.25], [0.75, 0.75]]]),
            "atom_mask": torch.tensor([[True, True]]),
        }

        result = canonical_presence_ranking_loss(dense, targets)
        result["loss"].backward()

        self.assertIsNone(features.grad)
        self.assertFalse(
            any(parameter.grad is not None for parameter in model.atom_head.parameters())
        )
        self.assertIsNotNone(model.presence_ranker.output_projection.weight.grad)

    def test_rejects_non_protocol_shape_and_invalid_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_channels=256"):
            G3FrozenAtomPresenceRanker(
                TinyFrozenAtomHead(),
                SimpleNamespace(
                    hidden_channels=128,
                    atom_slots_per_cell=3,
                    maximum_atoms=4,
                ),
            )
        with self.assertRaisesRegex(ValueError, "bottleneck_channels=64"):
            G3FrozenAtomPresenceRanker(
                TinyFrozenAtomHead(), config(), bottleneck_channels=32
            )
        dense = {
            "presence_logits": torch.zeros(1, 3, 1, 1),
            "anchors": torch.zeros(1, 3, 2, 1, 1),
        }
        empty_targets = {
            "atom_anchors": torch.zeros(1, 1, 2),
            "atom_mask": torch.tensor([[False]]),
        }
        with self.assertRaisesRegex(ValueError, "positive GT count"):
            canonical_presence_ranking_loss(dense, empty_targets)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            canonical_presence_ranking_loss(dense, empty_targets, negative_multiplier=0)


if __name__ == "__main__":
    unittest.main()
