"""Tests for the P1-R virtual stride-2 capacity trainer."""

from __future__ import annotations

import unittest

from chemvector.training.atoms.train_g3_p1r_virtual_stride2_center_field import (
    capacity_gate,
)


class P1RVirtualStride2TrainerTests(unittest.TestCase):
    def test_capacity_gate_requires_every_atom_image_and_peak(self) -> None:
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

        decision, checks = capacity_gate(metrics)
        self.assertEqual(decision, "overfit_capacity_go")
        self.assertTrue(all(checks.values()))

        metrics["regional_maxima"]["peak_shortfall_images"] = 1
        decision, _ = capacity_gate(metrics)
        self.assertEqual(decision, "virtual_stride2_capacity_no_go")


if __name__ == "__main__":
    unittest.main()
