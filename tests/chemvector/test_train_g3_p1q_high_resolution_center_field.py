"""Tests for the P1-Q high-resolution center-field capacity trainer."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_atom_center_field import (
    G3AtomCenterFieldHighResolutionResidual,
)
from chemvector.training.atoms.train_g3_p1q_high_resolution_center_field import (
    high_resolution_step0_parity,
)


class _AtomHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.presence = nn.Conv2d(256, 3, 1)
        self.anchor_offset = nn.Conv2d(256, 6, 1)


class P1QHighResolutionTrainerTests(unittest.TestCase):
    def test_step_zero_is_exact_upsampled_slot_zero(self) -> None:
        atom = _AtomHead()
        field = G3AtomCenterFieldHighResolutionResidual(atom)

        result = high_resolution_step0_parity(
            field, atom, torch.randn(1, 256, 192, 192)
        )

        self.assertTrue(result["exact"])
        self.assertEqual(result["center_max_abs_error"], 0.0)
        self.assertEqual(result["offset_zero_max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
