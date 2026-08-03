"""Tests for the P1-O Atom center-field trainer contracts."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_atom_center_field import G3AtomCenterField
from chemvector.training.atoms.train_g3_p1o_atom_center_field import (
    _phase_gate,
    validate_optimizer,
    warmstart_parity,
)


class _AtomHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.presence = nn.Conv2d(256, 3, 1)
        self.anchor_offset = nn.Conv2d(256, 6, 1)


class P1OTrainerTests(unittest.TestCase):
    def test_optimizer_contract_accepts_only_771_field_parameters(self) -> None:
        field = G3AtomCenterField()
        atom = _AtomHead().requires_grad_(False)
        vision = nn.Conv2d(1, 1, 1).requires_grad_(False)
        optimizer = torch.optim.AdamW(field.parameters())

        validate_optimizer(field, atom, vision, optimizer)

        wrong = torch.optim.AdamW(list(field.parameters()) + list(atom.parameters()))
        with self.assertRaisesRegex(ValueError, "exactly 771"):
            validate_optimizer(field, atom, vision, wrong)

    def test_warmstart_parity_detects_exact_slot_zero_copy(self) -> None:
        atom = _AtomHead()
        field = G3AtomCenterField()
        with torch.no_grad():
            field.center.load_state_dict(
                {"weight": atom.presence.weight[0:1], "bias": atom.presence.bias[0:1]}
            )
            field.offset.load_state_dict(
                {
                    "weight": atom.anchor_offset.weight.reshape(3, 2, 256, 1, 1)[0],
                    "bias": atom.anchor_offset.bias.reshape(3, 2)[0],
                }
            )
        result = warmstart_parity(field, atom, torch.randn(2, 256, 4, 4))

        self.assertTrue(result["exact"])

    def test_overfit_gate_is_strict(self) -> None:
        metrics = {
            "known_full_count": {
                "matched_atoms": 1424,
                "target_atoms": 1424,
                "whole_set_exact_images": 32,
            },
            "top128_reachability": {
                "matched_targets": 1424,
                "targets": 1424,
                "exact_images": 32,
            },
            "regional_maxima": {"peak_shortfall_images": 0},
        }

        decision, checks = _phase_gate("overfit32", metrics)

        self.assertEqual(decision, "capacity_go")
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
