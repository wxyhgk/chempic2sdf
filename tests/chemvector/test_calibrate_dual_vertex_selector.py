import unittest

from chemvector.calibrate_dual_vertex_selector import (
    atom_count_bucket,
    parse_thresholds,
    summarize_observations,
)
from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE


class DualVertexCalibrationTests(unittest.TestCase):
    def test_threshold_parser_sorts_and_rejects_boundaries(self) -> None:
        self.assertEqual(parse_thresholds([0.7, 0.5, 0.7], "test"), [0.5, 0.7])
        with self.assertRaises(ValueError):
            parse_thresholds([0.0], "test")

    def test_atom_count_buckets_have_stable_boundaries(self) -> None:
        self.assertEqual(atom_count_bucket(30), "le_30")
        self.assertEqual(atom_count_bucket(31), "31_70")
        self.assertEqual(atom_count_bucket(70), "31_70")
        self.assertEqual(atom_count_bucket(71), "gt_70")

    def test_summary_requires_each_role_to_match_for_strict_exact(self) -> None:
        observations = {
            IMPLICIT_ROLE: [
                {"predicted": 2, "expected": 2, "matches": 2},
                {"predicted": 1, "expected": 2, "matches": 1},
            ],
            TEXT_ROLE: [
                {"predicted": 1, "expected": 1, "matches": 1},
                {"predicted": 2, "expected": 1, "matches": 1},
            ],
        }
        result = summarize_observations(
            observations,
            [
                {"variant": "coordgen", "atom_count": 3},
                {"variant": "stress", "atom_count": 3},
            ],
        )

        self.assertEqual(result["exact_rate"], 0.5)
        self.assertEqual(result["total_count_exact_rate"], 1.0)
        self.assertEqual(result["role_count_exact_rate"], 0.5)
        self.assertEqual(result["by_layout"]["stress"]["exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
