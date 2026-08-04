"""Tests for permutation-invariant G3 Atom query sets."""

import unittest

import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    decode_g3_atom_query_set,
    g3_atom_query_set_loss,
    unique_pointer_targets,
)


class G3AtomQuerySetTests(unittest.TestCase):
    @staticmethod
    def candidates(batch: int = 2) -> dict[str, torch.Tensor]:
        return {
            "candidate_features": torch.randn(batch, 128, 32),
            "candidate_anchors": torch.rand(batch, 128, 2),
            "raw_scores": torch.rand(batch, 128),
            "normalized_scores": torch.randn(batch, 128),
            "scale_ranks": torch.arange(128).float()[None].repeat(batch, 1) / 127,
        }

    def test_forward_shapes(self) -> None:
        model = G3AtomQuerySet()
        candidates = self.candidates()
        output = model(candidates)
        self.assertEqual(output["existence_logits"].shape, (2, 128))
        self.assertEqual(output["count_logits"].shape, (2, 129))
        self.assertEqual(output["pointer_logits"].shape, (2, 128, 128))
        self.assertEqual(output["coordinates"].shape, (2, 128, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_heads_must_divide_model_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "heads must divide"):
            G3AtomQuerySetConfig(d_model=63, heads=4)

    def test_loss_accepts_equivalent_candidate_set(self) -> None:
        candidates = self.candidates(batch=1)
        candidates["candidate_anchors"][0, 0] = torch.tensor([0.25, 0.25])
        candidates["candidate_anchors"][0, 1] = torch.tensor([0.251, 0.25])
        target_anchors = torch.zeros(1, 4, 2)
        target_anchors[0, 0] = torch.tensor([0.25, 0.25])
        target_mask = torch.tensor([[True, False, False, False]])
        output = G3AtomQuerySet()(candidates)
        losses = g3_atom_query_set_loss(
            output, candidates["candidate_anchors"], target_anchors, target_mask
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()

    def test_decode_uses_distinct_candidate_pointers(self) -> None:
        output = {
            "existence_logits": torch.tensor([[5.0, 4.0, -1.0]]),
            "count_logits": torch.tensor([[0.0, 0.0, 5.0, 0.0]]),
            "pointer_logits": torch.tensor([[
                [9.0, 8.0, 0.0],
                [9.0, 7.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]),
        }
        anchors = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])
        decoded = decode_g3_atom_query_set(output, anchors)
        self.assertEqual(len(decoded[0]["candidate_indices"].unique()), 2)
        self.assertEqual(len(decoded[0]["anchors"]), 2)

    def test_unique_pointer_targets_resolve_competing_queries(self) -> None:
        pointer_logits = torch.tensor([[
            [9.0, 8.0, 0.0],
            [9.0, 7.0, 0.0],
        ]])
        candidates = torch.tensor([[
            [0.100, 0.100],
            [0.102, 0.100],
            [0.800, 0.800],
        ]])
        targets = torch.tensor([[
            [0.100, 0.100],
            [0.102, 0.100],
        ]])
        assignments = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        selected = unique_pointer_targets(
            pointer_logits, candidates, targets, assignments
        )[0]
        self.assertEqual(len(selected.unique()), 2)
        self.assertEqual(set(selected.tolist()), {0, 1})

    def test_unique_pointer_loss_backpropagates(self) -> None:
        candidates = self.candidates(batch=1)
        target_anchors = torch.zeros(1, 4, 2)
        target_anchors[0, :2] = candidates["candidate_anchors"][0, :2]
        target_mask = torch.tensor([[True, True, False, False]])
        model = G3AtomQuerySet()
        losses = g3_atom_query_set_loss(
            model(candidates),
            candidates["candidate_anchors"],
            target_anchors,
            target_mask,
            unique_pointers=True,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()


if __name__ == "__main__":
    unittest.main()
