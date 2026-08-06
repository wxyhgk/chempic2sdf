"""Tests for the P3-E18 held-out-font manifest contract."""

import unittest

from chemvector.evaluation.audits.atoms.build_g3_p3e18_freshdev_manifest import (
    EXPECTED_FONT_GROUP,
    EXPECTED_IMAGES,
    EXPECTED_MOLECULES,
    EXPECTED_SPLIT,
    EXPECTED_VARIANTS,
    validate_fresh_rows,
)


def rows(molecules: int) -> list[dict]:
    return [
        {
            "sample_key": f"m{molecule}_{variant}",
            "molecule_id": f"m{molecule}",
            "split": EXPECTED_SPLIT,
            "font_family_group": EXPECTED_FONT_GROUP,
            "variant": variant,
        }
        for molecule in range(molecules)
        for variant in sorted(EXPECTED_VARIANTS)
    ]


class G3P3E18FreshDevManifestTests(unittest.TestCase):
    def test_protocol_sizes_are_frozen(self) -> None:
        self.assertEqual(EXPECTED_IMAGES, 8_000)
        self.assertEqual(EXPECTED_MOLECULES, 2_000)
        self.assertEqual(len(EXPECTED_VARIANTS), 4)

    def test_validation_rejects_non_heldout_font_group(self) -> None:
        values = rows(EXPECTED_MOLECULES)
        values[0]["font_family_group"] = "train"
        with self.assertRaisesRegex(ValueError, "non-held-out"):
            validate_fresh_rows(values)

    def test_validation_accepts_complete_protocol(self) -> None:
        audit = validate_fresh_rows(rows(EXPECTED_MOLECULES))
        self.assertEqual(audit["images"], EXPECTED_IMAGES)
        self.assertEqual(audit["molecules"], EXPECTED_MOLECULES)


if __name__ == "__main__":
    unittest.main()
