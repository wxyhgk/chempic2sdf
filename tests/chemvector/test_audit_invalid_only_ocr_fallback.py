import unittest

from chemvector.audit_invalid_only_ocr_fallback import audit_fallback


class InvalidOnlyOCRFallbackAuditTests(unittest.TestCase):
    @staticmethod
    def row(valid: bool, chemical: bool = False) -> dict:
        return {
            "sdf_valid": valid,
            "formula_match": chemical,
            "atom_count_match": chemical,
            "bond_count_match": chemical,
            "nonstereo_structure_match": chemical,
            "chemical_match": chemical,
        }

    def test_routes_only_invalid_primary_to_valid_expert(self) -> None:
        primary = {
            "a": self.row(False),
            "b": self.row(True, True),
            "c": self.row(False),
        }
        expert = {
            "a": self.row(True, True),
            "b": self.row(False),
            "c": self.row(False),
        }
        result = audit_fallback(primary, expert)
        self.assertEqual(result["routed_molecule_ids"], ["a"])
        self.assertEqual(result["metrics"]["sdf_valid"]["hybrid"], 2)
        self.assertEqual(result["metrics"]["chemical_match"]["rescue"], 1)
        self.assertEqual(result["metrics"]["chemical_match"]["break"], 0)
        self.assertTrue(result["gate"]["passed"])

    def test_rejects_unpaired_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            audit_fallback({"a": self.row(False)}, {"b": self.row(True)})


if __name__ == "__main__":
    unittest.main()
