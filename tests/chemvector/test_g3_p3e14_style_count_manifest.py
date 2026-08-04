"""Tests for the P3-E14 style Count manifest selection."""

import unittest

from chemvector.evaluation.audits.atoms.build_g3_p3e14_style_count_manifest import (
    EXPECTED_IMAGES,
    EXPECTED_MOLECULES,
    EXPECTED_VARIANTS,
    select_rows,
)


class G3P3E14StyleCountManifestTests(unittest.TestCase):
    def test_selection_excludes_molecules_and_scaffolds(self) -> None:
        rows = []
        for molecule, scaffold in (("m1", "s1"), ("m2", "s1"), ("m3", "s3")):
            for variant in EXPECTED_VARIANTS:
                rows.append({
                    "sample_key": f"{molecule}-{variant}",
                    "molecule_id": molecule,
                    "scaffold": scaffold,
                    "variant": variant,
                })
        selected = select_rows(rows, {"m1"}, {"s1"})
        self.assertEqual({row["molecule_id"] for row in selected}, {"m3"})
        self.assertEqual(len(selected), 4)

    def test_protocol_sizes_are_fixed(self) -> None:
        self.assertEqual(EXPECTED_IMAGES, 90_932)
        self.assertEqual(EXPECTED_MOLECULES, 22_733)
        self.assertEqual(len(EXPECTED_VARIANTS), 4)


if __name__ == "__main__":
    unittest.main()
