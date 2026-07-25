from __future__ import annotations

import unittest

from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    ParsedVisualSVGSequenceV02,
    TextBoxRecordV02,
    canonicalize_visual_svg_v0_2,
)
from chemvector.audit_flat_proposal_carrier import assemble_oracle_carriers
from chemvector.evaluate_bond_group_set import decoded_rows_to_visual_svg_v0_2


def _line(member: LineMemberV02, *, dx: int = 0) -> dict:
    return {
        "kind": "line",
        "quantized_points": [
            [member.x0 + dx, member.y0],
            [member.x1 + dx, member.y1],
        ],
        "stroke_width_q": member.stroke_width,
    }


class AuditFlatProposalCarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.single = BondGroupRecordV02(
            "single_line", (LineMemberV02(100, 100, 300, 100, 21),)
        )
        self.parallel = BondGroupRecordV02(
            "parallel_2",
            (
                LineMemberV02(100, 200, 300, 200, 21),
                LineMemberV02(100, 220, 300, 220, 21),
            ),
        )
        self.hashed = BondGroupRecordV02(
            "hashed_wedge",
            tuple(
                LineMemberV02(
                    400 + 40 * index,
                    300 - (5 + 3 * index),
                    400 + 40 * index,
                    300 + (5 + 3 * index),
                    11,
                )
                for index in range(4)
            ),
        )
        self.text = TextBoxRecordV02(600, 500, 650, 530)
        self.expected = canonicalize_visual_svg_v0_2(
            ParsedVisualSVGSequenceV02(
                (self.single, self.parallel, self.hashed),
                (self.text,),
            )
        )

    def test_group_slots_keep_gt_count_but_missing_geometry_stays_invalid(self) -> None:
        predictions = [_line(self.single.members[0])]
        result = assemble_oracle_carriers(
            self.expected,
            predictions,
            tolerance_q=8.0,
        )
        converted = decoded_rows_to_visual_svg_v0_2(
            result.rows,
            max_members=143,
        )

        self.assertEqual(sum(converted.decoded_counts.values()), 4)
        self.assertEqual(converted.decoded_counts["single_line"], 1)
        self.assertEqual(converted.decoded_counts["parallel_2"], 1)
        self.assertEqual(converted.decoded_counts["hashed_wedge"], 1)
        self.assertEqual(converted.decoded_counts["text_box"], 1)
        self.assertEqual(len(converted.invalid_objects), 3)
        self.assertEqual(
            result.statistics["groups"]["parallel_2"]["eligible"], 0
        )

    def test_parallel_two_requires_both_members(self) -> None:
        predictions = [_line(self.parallel.members[0])]
        result = assemble_oracle_carriers(
            self.expected,
            predictions,
            tolerance_q=8.0,
        )
        self.assertEqual(
            result.statistics["groups"]["parallel_2"],
            {"expected": 1, "eligible": 0, "emitted": 0, "fit_failed": 0},
        )

    def test_hashed_wedge_uses_two_proposals_to_expand_gt_count(self) -> None:
        predictions = [
            _line(self.hashed.members[0]),
            _line(self.hashed.members[-1]),
        ]
        result = assemble_oracle_carriers(
            self.expected,
            predictions,
            tolerance_q=8.0,
        )
        hashed = next(row for row in result.rows if row["kind"] == "hashed_wedge")

        self.assertEqual(result.statistics["groups"]["hashed_wedge"]["eligible"], 1)
        self.assertEqual(result.statistics["groups"]["hashed_wedge"]["emitted"], 1)
        self.assertEqual(hashed["member_count"], 4)
        self.assertEqual(len(hashed["member_geometry"]), 4)

    def test_q_tolerance_controls_member_availability_without_gt_fill(self) -> None:
        predictions = [_line(self.single.members[0], dx=20)]
        strict = assemble_oracle_carriers(
            self.expected,
            predictions,
            tolerance_q=8.0,
        )
        loose = assemble_oracle_carriers(
            self.expected,
            predictions,
            tolerance_q=32.0,
        )

        self.assertEqual(strict.statistics["groups"]["single_line"]["eligible"], 0)
        self.assertEqual(loose.statistics["groups"]["single_line"]["eligible"], 1)
        loose_line = next(row for row in loose.rows if row["kind"] == "single_line")
        self.assertAlmostEqual(
            float(loose_line["member_geometry"][0][0]) * 2047.0,
            120.0,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
