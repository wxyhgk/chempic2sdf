"""Tests for the P1-U dual-domain adjacent-Atom capacity protocol."""

from __future__ import annotations

import unittest

from chemvector.training.atoms.train_g3_p1u_adjacent_atom_capacity import (
    capacity_gate,
    domain_capacity_checks,
)


def _domain_metrics():
    return {
        "known_full_count": {
            "matched_atoms": 100,
            "target_atoms": 100,
            "whole_set_exact_images": 32,
        },
        "top128_reachability": {
            "matched_targets": 100,
            "targets": 100,
            "exact_images": 32,
        },
        "regional_maxima": {"peak_shortfall_images": 0},
    }


class P1UAdjacentAtomCapacityTests(unittest.TestCase):
    def test_domain_gate_requires_every_atom_image_and_peak(self) -> None:
        metrics = _domain_metrics()
        self.assertTrue(all(domain_capacity_checks(metrics, images=32).values()))
        metrics["top128_reachability"]["exact_images"] = 31
        self.assertFalse(all(domain_capacity_checks(metrics, images=32).values()))

    def test_dual_gate_requires_both_domains_and_all_collision_pairs(self) -> None:
        collision = {
            "collision_images": 7,
            "known_distinct_images": 7,
            "top128_distinct_images": 7,
        }
        decision, checks = capacity_gate(
            _domain_metrics(), _domain_metrics(), collision
        )
        self.assertEqual(decision, "dual_capacity_go")
        self.assertTrue(
            all(value for group in checks.values() for value in group.values())
        )
        collision["known_distinct_images"] = 6
        self.assertEqual(
            capacity_gate(_domain_metrics(), _domain_metrics(), collision)[0],
            "dual_capacity_no_go",
        )


if __name__ == "__main__":
    unittest.main()
