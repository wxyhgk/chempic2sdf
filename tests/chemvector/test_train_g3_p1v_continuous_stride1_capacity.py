"""Tests for the P1-V continuous stride-1 capacity protocol."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.training.atoms.train_g3_p1v_continuous_stride1_capacity import (
    FIELD_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    step0_parity,
)


class _AtomHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.presence = nn.Conv2d(256, 3, 1)
        self.anchor_offset = nn.Conv2d(256, 6, 1)


class P1VContinuousStride1CapacityTests(unittest.TestCase):
    def test_parameter_contract(self) -> None:
        self.assertEqual(FIELD_PARAMETER_COUNT, 17_379)
        self.assertEqual(TRAINABLE_PARAMETER_COUNT, 96_579)

    def test_step_zero_is_exact_upsampled_p1a_base(self) -> None:
        atom = _AtomHead()
        field = ContinuousStride1AtomCenterField(atom)
        result = step0_parity(
            field,
            atom,
            torch.randn(1, 256, 192, 192),
            torch.randn(1, 96, 192, 192),
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["center_max_abs_error"], 0.0)
        self.assertEqual(result["offset_zero_max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
