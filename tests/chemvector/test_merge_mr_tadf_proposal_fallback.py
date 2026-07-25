import unittest

from chemvector.merge_mr_tadf_proposal_fallback import select_record


class MrTadfProposalFallbackTests(unittest.TestCase):
    def test_valid_primary_is_never_replaced(self) -> None:
        selected, source = select_record(
            {"sdf_valid": True, "chemical_match": False, "seconds": 1.0},
            {"sdf_valid": True, "chemical_match": True, "seconds": 2.0},
        )

        self.assertEqual(source, "single_scale_primary")
        self.assertFalse(selected["chemical_match"])

    def test_invalid_primary_uses_fallback_and_adds_runtime(self) -> None:
        selected, source = select_record(
            {"sdf_valid": False, "seconds": 1.0},
            {"sdf_valid": True, "decode_path": "primary_joint", "seconds": 2.0},
        )

        self.assertEqual(source, "multiscale_invalid_fallback")
        self.assertEqual(selected["decode_path"], "multiscale_primary_joint")
        self.assertEqual(selected["seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
