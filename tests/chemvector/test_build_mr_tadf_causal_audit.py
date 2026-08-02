"""Tests for deterministic MR-TADF causal audit selection and bucketing."""

from __future__ import annotations

import unittest

from chemvector.build_mr_tadf_causal_audit import (
    automatic_bucket,
    stratified_sample,
)


def record(molecule_id: str, bucket: str = "atom_detection") -> dict:
    return {
        "molecule_id": molecule_id,
        "difficulty": "standard",
        "cohort": "error",
        "sdf_valid": True,
        "diagnostics": {
            "primary_error": bucket,
            "missing_atoms": 0,
            "extra_atoms": 0,
            "element_mismatches": 0,
            "missing_bonds": 0,
            "extra_bonds": 0,
            "crossing_extra_bonds": 0,
            "bond_type_mismatches": 0,
            "atom_property_mismatches": 0,
        },
    }


class CausalAuditTests(unittest.TestCase):
    def test_stratified_sample_is_deterministic_and_covers_strata(self) -> None:
        rows = [record(f"A{index}", "a") for index in range(10)] + [
            record(f"B{index}", "b") for index in range(2)
        ]

        first = stratified_sample(
            rows,
            6,
            seed="fixed",
            stratum=lambda row: (row["diagnostics"]["primary_error"],),
        )
        second = stratified_sample(
            rows,
            6,
            seed="fixed",
            stratum=lambda row: (row["diagnostics"]["primary_error"],),
        )

        self.assertEqual(
            [row["molecule_id"] for row in first],
            [row["molecule_id"] for row in second],
        )
        self.assertEqual({row["diagnostics"]["primary_error"] for row in first}, {"a", "b"})

    def test_automatic_bucket_prioritizes_count_before_ocr_and_bonds(self) -> None:
        row = record("X")
        row["diagnostics"].update(
            {"missing_atoms": 1, "element_mismatches": 1, "extra_bonds": 2}
        )
        self.assertEqual(automatic_bucket(row), "atom_under_detection")

    def test_crossing_extra_bond_is_specific_bucket(self) -> None:
        row = record("X")
        row["diagnostics"].update({"extra_bonds": 2, "crossing_extra_bonds": 1})
        self.assertEqual(automatic_bucket(row), "crossing_extra_bond")

    def test_control_has_dedicated_bucket(self) -> None:
        row = record("X")
        row["cohort"] = "control"
        self.assertEqual(automatic_bucket(row), "control_exact")


if __name__ == "__main__":
    unittest.main()
