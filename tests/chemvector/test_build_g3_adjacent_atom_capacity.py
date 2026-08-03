"""Tests for deterministic adjacent-Atom capacity selection."""

from __future__ import annotations

import unittest

import torch

from chemvector.evaluation.audits.atoms.build_g3_adjacent_atom_capacity import (
    sample_geometry,
    select_capacity_records,
)


class BuildG3AdjacentAtomCapacityTests(unittest.TestCase):
    def test_geometry_detects_stride4_collision_and_distance(self) -> None:
        geometry = sample_geometry(
            torch.tensor([[0.100, 0.100], [0.103, 0.103], [0.8, 0.8]])
        )
        self.assertTrue(geometry["same_stride4_cell"])
        self.assertEqual(geometry["nearest_atom_indices"], [0, 1])
        self.assertLess(geometry["minimum_distance_pixels"], 4.0)

    def test_selection_keeps_all_collisions_before_nearest_ordinary(self) -> None:
        records = [
            {"sample_key": "ordinary-close", "minimum_distance_pixels": 2.0, "same_stride4_cell": False},
            {"sample_key": "collision", "minimum_distance_pixels": 3.0, "same_stride4_cell": True},
            {"sample_key": "ordinary-far", "minimum_distance_pixels": 4.0, "same_stride4_cell": False},
        ]
        selected = select_capacity_records(records, count=2)
        self.assertEqual(
            [item["sample_key"] for item in selected],
            ["collision", "ordinary-close"],
        )


if __name__ == "__main__":
    unittest.main()
