"""Tests for the P1-S phase-residual capacity trainer."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_atom_center_field import (
    VirtualStride2AtomCenterResidual,
)
from chemvector.training.atoms.train_g3_p1s_phase_residual_center_field import (
    capacity_gate,
    step0_parity,
)


class _AtomHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.presence = nn.Conv2d(256, 3, 1)
        self.anchor_offset = nn.Conv2d(256, 6, 1)


class P1SPhaseResidualTrainerTests(unittest.TestCase):
    def test_step_zero_is_exact_upsampled_p1a_base(self) -> None:
        atom = _AtomHead()
        field = VirtualStride2AtomCenterResidual(atom)

        result = step0_parity(
            field,
            atom,
            torch.randn(1, 256, 192, 192),
            torch.randn(1, 96, 192, 192),
        )

        self.assertTrue(result["exact"])
        self.assertEqual(result["center_max_abs_error"], 0.0)
        self.assertEqual(result["offset_zero_max_abs_error"], 0.0)

    def test_capacity_gate_keeps_strict_p1r_contract(self) -> None:
        metrics = {
            "known_full_count": {
                "matched_atoms": 1_424,
                "target_atoms": 1_424,
                "whole_set_exact_images": 32,
            },
            "top128_reachability": {
                "matched_targets": 1_424,
                "targets": 1_424,
                "exact_images": 32,
            },
            "regional_maxima": {"peak_shortfall_images": 0},
        }
        self.assertEqual(capacity_gate(metrics)[0], "overfit_capacity_go")
        metrics["top128_reachability"]["matched_targets"] = 1_423
        self.assertEqual(capacity_gate(metrics)[0], "phase_residual_capacity_no_go")


if __name__ == "__main__":
    unittest.main()
