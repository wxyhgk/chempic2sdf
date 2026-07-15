"""Tests for OCR-independent molecular skeleton metrics."""

from __future__ import annotations

import unittest

from PIL import Image
import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    StrokeRecordV4,
    VertexRecordV4,
)
from chemvector.evaluate_skeleton import compare_skeleton, skeleton_collate


def vertex(x: int, y: int) -> VertexRecordV4:
    return VertexRecordV4(
        x=x,
        y=y,
        element="C",
        formal_charge=0,
        isotope=0,
        aromatic=False,
        explicit_h_count=0,
        no_implicit=False,
        radical_electrons=0,
        canonical_chiral_tag="CHI_UNSPECIFIED",
    )


def stroke(begin: VertexRecordV4, end: VertexRecordV4, bond_type: str) -> StrokeRecordV4:
    return StrokeRecordV4(
        begin_x=begin.x,
        begin_y=begin.y,
        end_x=end.x,
        end_y=end.y,
        display_type=bond_type,
        bond_type=bond_type,
        direction="NONE",
        stereo="STEREONONE",
    )


class SkeletonComparisonTests(unittest.TestCase):
    def test_collator_does_not_tokenize_open_element_labels(self) -> None:
        batch = skeleton_collate(
            [
                {
                    "key": "explicit-h",
                    "image": Image.new("RGB", (32, 32), "white"),
                    "target_sequence": "<mol v=0.4>\n<v 10 10 H 0 0 0 0 0 0 CHI_UNSPECIFIED>\n<mol_end>\n",
                }
            ],
            64,
        )

        self.assertEqual(batch["keys"], ["explicit-h"])
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 64, 64))

    def test_ignores_element_labels_but_requires_bond_type(self) -> None:
        left = vertex(100, 200)
        right = vertex(300, 200)
        expected = ParsedChemSVGSequenceV4(
            (
                left,
                VertexRecordV4(
                    x=right.x,
                    y=right.y,
                    element="N",
                    formal_charge=0,
                    isotope=0,
                    aromatic=False,
                    explicit_h_count=0,
                    no_implicit=False,
                    radical_electrons=0,
                    canonical_chiral_tag="CHI_UNSPECIFIED",
                ),
            ),
            (stroke(left, right, "DOUBLE"),),
        )
        probabilities = torch.tensor([[0.01, 0.01, 0.97, 0.01, 0.0]])

        result = compare_skeleton(
            expected,
            [left, right],
            [(0, 1)],
            probabilities,
            edge_threshold=0.5,
        )

        self.assertTrue(result.vertex_exact)
        self.assertTrue(result.edge_set_exact)
        self.assertTrue(result.skeleton_exact)

    def test_extra_unmatched_vertex_and_edge_are_false_positive(self) -> None:
        left = vertex(100, 200)
        right = vertex(300, 200)
        extra = vertex(500, 500)
        expected = ParsedChemSVGSequenceV4(
            (left, right),
            (stroke(left, right, "SINGLE"),),
        )
        probabilities = torch.tensor(
            [
                [0.01, 0.97, 0.01, 0.01, 0.0],
                [0.01, 0.97, 0.01, 0.01, 0.0],
            ]
        )

        result = compare_skeleton(
            expected,
            [left, right, extra],
            [(0, 1), (1, 2)],
            probabilities,
            edge_threshold=0.5,
        )

        self.assertEqual(result.correct_edges, 1)
        self.assertEqual(result.false_edges, 1)
        self.assertFalse(result.vertex_exact)
        self.assertFalse(result.edge_set_exact)

    def test_aromatic_edges_require_connection_but_not_local_bond_type(self) -> None:
        left = vertex(100, 200)
        right = vertex(300, 200)
        expected = ParsedChemSVGSequenceV4(
            (left, right),
            (stroke(left, right, "AROMATIC"),),
        )

        result = compare_skeleton(
            expected,
            [left, right],
            [(0, 1)],
            torch.tensor([[0.01, 0.97, 0.01, 0.01, 0.0]]),
            edge_threshold=0.5,
        )

        self.assertEqual(result.expected_visible_bond_types, 0)
        self.assertTrue(result.skeleton_exact)


if __name__ == "__main__":
    unittest.main()
