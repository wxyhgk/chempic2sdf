from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import unittest

from formal.chemgraph.audit.audit_crossing_prior_risk_v0_2 import (
    audit_directory,
    is_strict_cross,
    matching_deviation_degrees,
)


class CrossingPriorRiskV02Tests(unittest.TestCase):
    def test_exact_cross_uses_rational_geometry(self) -> None:
        vectors = (
            (Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(-2)),
            (Fraction(0), Fraction(3)),
        )
        self.assertTrue(is_strict_cross(vectors))
        self.assertEqual(matching_deviation_degrees(vectors), 0.0)

    def test_non_cross_is_not_accepted(self) -> None:
        vectors = (
            (Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(1)),
            (Fraction(0), Fraction(-1)),
            (Fraction(0), Fraction(1)),
        )
        self.assertFalse(is_strict_cross(vectors))
        self.assertGreater(matching_deviation_degrees(vectors), 0.0)

    def test_mr_tadf_frozen_risk_counts_when_available(self) -> None:
        root = Path("Data/MR-TADF_SDF")
        if not root.is_dir():
            self.skipTest("MR-TADF external diagnostic directory is unavailable")
        result = audit_directory(root)
        self.assertEqual(result["sdf_file_count"], 1_120)
        self.assertEqual(result["parsed_file_count"], 1_120)
        self.assertEqual(result["strict_cross"]["atom_count"], 16)
        self.assertEqual(result["strict_cross"]["molecule_count"], 13)
        self.assertEqual(result["within_1_degree"]["atom_count"], 145)
        self.assertEqual(result["within_1_degree"]["molecule_count"], 94)
        self.assertEqual(result["within_5_degrees"]["atom_count"], 216)
        self.assertEqual(result["within_5_degrees"]["molecule_count"], 138)
        self.assertEqual(result["strict_non_single_or_non_carbon"], [])
        self.assertFalse(result["global_prior_safe"])


if __name__ == "__main__":
    unittest.main()
