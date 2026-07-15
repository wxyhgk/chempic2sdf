"""Tests for MR-TADF Oracle isolation helpers."""

from __future__ import annotations

import unittest

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    VertexRecordV4,
    parse_chemsvg_v4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.oracle_mr_tadf import (
    apply_truth_ocr_properties,
    nearest_unused_coordinate,
)


def vertex(
    x: int,
    y: int,
    element: str,
    *,
    aromatic: bool,
) -> VertexRecordV4:
    return VertexRecordV4(
        x=x,
        y=y,
        element=element,
        formal_charge=0,
        isotope=0,
        aromatic=aromatic,
        explicit_h_count=0,
        no_implicit=False,
        radical_electrons=0,
        canonical_chiral_tag="CHI_UNSPECIFIED",
    )


class OracleMrTadfTests(unittest.TestCase):
    def test_truth_ocr_replacement_preserves_model_aromatic_flag(self) -> None:
        predicted = vertex(100, 200, "N", aromatic=False)
        truth = vertex(102, 199, "B", aromatic=True)
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4((predicted,), ())
        )

        replaced, matches = apply_truth_ocr_properties(sequence, (truth,))

        parsed = parse_chemsvg_v4(replaced)
        self.assertEqual(matches, 1)
        self.assertEqual(parsed.vertices[0].element, "B")
        self.assertFalse(parsed.vertices[0].aromatic)
        self.assertEqual((parsed.vertices[0].x, parsed.vertices[0].y), (100, 200))

    def test_colliding_truth_coordinate_moves_by_minimum_distance(self) -> None:
        used = {(500, 500)}

        coordinate = nearest_unused_coordinate(500, 500, used)

        self.assertNotIn(coordinate, used)
        self.assertEqual(
            max(abs(coordinate[0] - 500), abs(coordinate[1] - 500)),
            1,
        )


if __name__ == "__main__":
    unittest.main()
