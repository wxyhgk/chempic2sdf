from __future__ import annotations

import unittest

from chemvector.style_sensitivity import (
    style_dimensions,
    summarize_style_sensitivity,
)


def _row(
    molecule_id: str,
    *,
    font: str,
    size: int,
    line_width: float,
    layout: str,
    exact: bool,
) -> dict:
    return {
        "molecule_id": molecule_id,
        "variant": layout,
        "representable": True,
        "baseline_presence_exact": exact,
        "baseline_chemical_exact": exact,
        "baseline_false_positive": int(not exact),
        "baseline_false_negative": 0,
        "baseline_wrong_type": 0,
        "presence_exact": exact,
        "chemical_exact": exact,
        "false_positive": int(not exact),
        "false_negative": 0,
        "wrong_type": 0,
        "style": {
            "font_family_id": font,
            "font_size": size,
            "bond_line_width": line_width,
            "rotation_degrees": 9.0,
            "multiple_bond_offset": 0.12,
            "layout": layout,
            "kekulize": False,
            "stress_operation": None,
        },
    }


class StyleSensitivityTest(unittest.TestCase):
    def test_style_dimensions_bucket_numeric_parameters(self) -> None:
        dimensions = style_dimensions(
            _row(
                "m1",
                font="rasa",
                size=31,
                line_width=2.3,
                layout="stress",
                exact=True,
            )
        )
        self.assertEqual(dimensions["font_bucket"], "extreme")
        self.assertEqual(dimensions["line_width_bucket"], "thick")
        self.assertEqual(dimensions["rotation_bucket"], "high")
        self.assertEqual(dimensions["multiple_bond_offset_bucket"], "narrow")

    def test_summary_exposes_mixed_paired_renderings(self) -> None:
        rows = [
            _row(
                "m1",
                font="rasa",
                size=10,
                line_width=1.5,
                layout="rdkit",
                exact=True,
            ),
            _row(
                "m1",
                font="nimbus",
                size=22,
                line_width=2.4,
                layout="coordgen",
                exact=False,
            ),
            _row(
                "m1",
                font="rasa",
                size=16,
                line_width=2.0,
                layout="original",
                exact=True,
            ),
            _row(
                "m1",
                font="nimbus",
                size=31,
                line_width=1.6,
                layout="stress",
                exact=False,
            ),
            _row(
                "m2",
                font="rasa",
                size=10,
                line_width=1.5,
                layout="rdkit",
                exact=True,
            ),
            _row(
                "m2",
                font="nimbus",
                size=22,
                line_width=2.4,
                layout="coordgen",
                exact=True,
            ),
            _row(
                "m2",
                font="rasa",
                size=16,
                line_width=2.0,
                layout="original",
                exact=True,
            ),
            _row(
                "m2",
                font="nimbus",
                size=31,
                line_width=1.6,
                layout="stress",
                exact=True,
            ),
        ]
        summary = summarize_style_sensitivity(rows, baseline=False)
        self.assertEqual(summary["overall"]["chemical_exact"], 6)
        paired = summary["paired_renderings"]["chemical_exact"]
        self.assertEqual(paired["all_renderings_exact"], 1)
        self.assertEqual(paired["mixed_outcome"], 1)
        self.assertEqual(paired["mixed_outcome_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
