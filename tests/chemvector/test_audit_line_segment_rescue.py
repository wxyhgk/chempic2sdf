from __future__ import annotations

import unittest

from chemvector.audit_line_segment_rescue import _summarize


class AuditLineSegmentRescueTest(unittest.TestCase):
    def test_summary_reports_dense_and_union_coverage(self) -> None:
        mode = {
            "line_candidate_pairs": 3,
            "line_support_segments": 6,
            "line_mapped_true_pairs": 2,
            "line_mapped_false_pairs": 1,
            "line_unknown_pairs": 0,
            "line_rescued_target_pairs": 1,
            "union_covered_pairs": 4,
            "union_complete": True,
            "line_only_candidate_pairs": 1,
            "line_only_mapped_true_pairs": 1,
            "line_only_mapped_false_pairs": 0,
            "line_only_unknown_pairs": 0,
            "line_only_rescued_target_pairs": 1,
            "line_only_union_covered_pairs": 4,
            "line_only_union_complete": True,
        }
        summary = _summarize(
            [
                {
                    "difficulty": "test",
                    "target_bonds": 4,
                    "dense_covered_pairs": 3,
                    "dense_complete": False,
                    "proposal_atoms": mode,
                    "selected_atoms": mode,
                }
            ]
        )

        self.assertEqual(summary["dense_bond_recall"], 0.75)
        self.assertEqual(summary["selected_atoms_union_bond_recall"], 1.0)
        self.assertEqual(summary["selected_atoms_union_complete_rate"], 1.0)
        self.assertAlmostEqual(summary["selected_atoms_line_pair_precision"], 2 / 3)
        self.assertEqual(summary["selected_atoms_line_only_pair_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
