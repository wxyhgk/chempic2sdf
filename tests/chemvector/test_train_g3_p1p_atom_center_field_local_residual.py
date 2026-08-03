"""Tests for the P1-P local-residual capacity trainer."""

from __future__ import annotations

import unittest

from chemvector.training.atoms.train_g3_p1p_atom_center_field_local_residual import (
    capacity_gate,
)


class P1PLocalResidualTrainerTests(unittest.TestCase):
    def test_capacity_gate_requires_every_atom_and_image(self) -> None:
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

        decision, checks = capacity_gate(metrics)
        self.assertEqual(decision, "overfit_capacity_go")
        self.assertTrue(all(checks.values()))

        metrics["known_full_count"]["matched_atoms"] = 1423
        decision, _ = capacity_gate(metrics)
        self.assertEqual(decision, "local_residual_capacity_no_go")


if __name__ == "__main__":
    unittest.main()
