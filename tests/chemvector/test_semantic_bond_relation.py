import unittest

import torch

from chemvector.semantic_bond_relation import (
    SemanticBondRelation,
    SemanticBondRelationConfig,
    apply_aromatic_ring_semantics,
)


class SemanticBondRelationTest(unittest.TestCase):
    def test_aromatic_semantics_promote_complete_ring_only(self) -> None:
        bonds = []
        for index in range(6):
            bonds.append(
                {
                    "junction_pair": (index, (index + 1) % 6),
                    "chemical": 0,
                }
            )
        bonds.append({"junction_pair": (0, 6), "chemical": 0})
        logits = torch.zeros((7, 4))
        logits[:6, 3] = 4.0
        logits[6, 3] = 6.0

        updated, diagnostics = apply_aromatic_ring_semantics(bonds, logits)

        self.assertTrue(all(item["chemical"] == 3 for item in updated[:6]))
        self.assertEqual(updated[6]["chemical"], 0)
        self.assertEqual(diagnostics["semantic_ring_accepted"], 1)
        self.assertEqual(diagnostics["semantic_ring_promoted_bonds"], 6)

    def test_output_is_invariant_to_endpoint_order(self) -> None:
        torch.manual_seed(3)
        model = SemanticBondRelation(
            SemanticBondRelationConfig(
                crop_width=48,
                local_height=24,
                context_height=32,
                hidden_dim=32,
            )
        ).eval()
        pixels = torch.ones((1, 3, 64, 64))
        pixels[:, :, 31:34, 10:54] = -1.0
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        pairs = torch.tensor([[[0, 1]]])
        mask = torch.tensor([[True]])

        direct = model(pixels, points, pairs, mask)
        reverse = model(pixels, points, pairs.flip(-1), mask)

        self.assertEqual(direct["display_logits"].shape, (1, 4))
        self.assertEqual(direct["chemical_logits"].shape, (1, 4))
        self.assertTrue(
            torch.allclose(direct["display_logits"], reverse["display_logits"])
        )
        self.assertTrue(
            torch.allclose(direct["chemical_logits"], reverse["chemical_logits"])
        )
        self.assertEqual(direct["local_views"].shape, (1, 6, 24, 48))


if __name__ == "__main__":
    unittest.main()
