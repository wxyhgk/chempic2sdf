"""Tests for the P1-U one-center target resolution audit."""

from __future__ import annotations

import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1u_target_resolution import (
    occupancy_collisions,
)


class P1UTargetResolutionAuditTests(unittest.TestCase):
    def test_occupancy_reports_information_lost_by_one_center_target(self) -> None:
        anchors = torch.tensor(
            [[0.1000, 0.1000], [0.1005, 0.1005], [0.8, 0.8]]
        )
        result = occupancy_collisions(anchors, grid_size=384)
        self.assertEqual(result["collision_cells"], 1)
        self.assertEqual(result["lost_atoms"], 1)
        self.assertEqual(result["maximum_occupancy"], 2)

    def test_distinct_cells_are_lossless(self) -> None:
        result = occupancy_collisions(
            torch.tensor([[0.1, 0.1], [0.2, 0.2]]), grid_size=384
        )
        self.assertEqual(result["collision_cells"], 0)
        self.assertEqual(result["lost_atoms"], 0)


if __name__ == "__main__":
    unittest.main()
