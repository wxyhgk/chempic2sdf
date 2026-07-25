import unittest

import torch

from chemvector.junction_bond_assembler import (
    JunctionBondAssembler,
    JunctionBondAssemblerConfig,
    decode_junction_assignment_rescue,
    decode_junction_assignments,
    junction_bond_assembler_loss,
)


class JunctionBondAssemblerTest(unittest.TestCase):
    def test_forward_loss_and_decode_shapes(self) -> None:
        model = JunctionBondAssembler(JunctionBondAssemblerConfig(hidden_dim=16))
        logits = model(
            torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))),
            torch.tensor((0, 1, 0)),
            torch.tensor(
                (
                    ((0.1, 0.1), (0.5, 0.5)),
                    ((0.5, 0.5), (0.9, 0.9)),
                )
            ),
            torch.tensor((0.9, 0.8)),
        )
        self.assertEqual(logits.shape, (2, 2, 4))
        losses = junction_bond_assembler_loss(
            logits, torch.tensor(((0, 1), (1, 2)))
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_decode_rejects_null_same_and_duplicate_pairs(self) -> None:
        logits = torch.full((4, 2, 4), -5.0)
        logits[0, 0, 0] = logits[0, 1, 1] = 5.0
        logits[1, 0, 3] = logits[1, 1, 2] = 5.0
        logits[2, 0, 2] = logits[2, 1, 2] = 5.0
        logits[3, 0, 1] = logits[3, 1, 0] = 5.0
        assignments, diagnostics = decode_junction_assignments(
            logits, torch.tensor((0.9, 0.8, 0.7, 0.6))
        )
        self.assertEqual(assignments, [(0, 1), None, None, None])
        self.assertEqual(diagnostics["assembler_null_objects"], 1)
        self.assertEqual(diagnostics["assembler_same_junction_objects"], 1)
        self.assertEqual(diagnostics["assembler_duplicate_objects"], 1)

    def test_rescue_changes_only_nearest_conflicts(self) -> None:
        points = torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.1)))
        endpoints = torch.tensor(
            (
                ((0.1, 0.1), (0.5, 0.5)),
                ((0.12, 0.1), (0.48, 0.5)),
            )
        )
        logits = torch.full((2, 2, 4), -5.0)
        logits[0, 0, 0] = logits[0, 1, 2] = 5.0
        logits[1, 0, 1] = logits[1, 1, 2] = 5.0
        assignments, diagnostics = decode_junction_assignment_rescue(
            logits,
            torch.tensor((0.9, 0.8)),
            endpoints,
            points,
        )
        self.assertEqual(assignments, [(0, 1), (1, 2)])
        self.assertEqual(diagnostics["assembler_rescued_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
    decode_junction_assignment_rescue,
