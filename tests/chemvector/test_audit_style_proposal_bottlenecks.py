from __future__ import annotations

import unittest

from chemvector.audit_style_proposal_bottlenecks import summarize_stage_rows


class AuditStyleProposalBottlenecksTest(unittest.TestCase):
    def test_summary_counts_paired_mixed_outcomes(self) -> None:
        rows = []
        for index, value in enumerate((True, True, False, False)):
            rows.append(
                {
                    "molecule_id": "m1",
                    "variant": str(index),
                    "style": {"layout": "rdkit"},
                    "atom_coverage": value,
                    "bond_coverage": True,
                    "incidence_eligibility": True,
                    "incidence_exact": True,
                    "group_coverage": True,
                    "type_coverage": True,
                    "representable": True,
                }
            )
        summary = summarize_stage_rows(rows)
        self.assertEqual(summary["overall"]["atom_coverage"], 0.5)
        self.assertEqual(summary["paired"]["atom_coverage"]["mixed"], 1)
        self.assertEqual(summary["paired"]["bond_coverage"]["all_renderings"], 1)


if __name__ == "__main__":
    unittest.main()
