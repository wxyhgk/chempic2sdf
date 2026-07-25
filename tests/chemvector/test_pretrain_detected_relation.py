"""Tests for detected-vertex relation supervision."""

from __future__ import annotations

import unittest

from Data.sdf.chemsvg_sequence_v4 import ParsedChemSVGSequenceV4, StrokeRecordV4, VertexRecordV4
from chemvector.pretrain_detected_relation import (
    detected_pair_labels,
    validation_score,
)


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


class DetectedRelationTests(unittest.TestCase):
    def test_labels_follow_hungarian_vertex_mapping(self) -> None:
        expected_vertices = (vertex(100, 100), vertex(500, 100), vertex(900, 100))
        expected = ParsedChemSVGSequenceV4(
            expected_vertices,
            (
                StrokeRecordV4(
                    begin_x=100,
                    begin_y=100,
                    end_x=500,
                    end_y=100,
                    display_type="DOUBLE",
                    bond_type="DOUBLE",
                    direction="NONE",
                    stereo="STEREONONE",
                ),
            ),
        )
        predicted = [vertex(902, 101), vertex(102, 99), vertex(498, 102)]
        pairs = [(0, 1), (0, 2), (1, 2)]

        labels, maximum_distance = detected_pair_labels(expected, predicted, pairs)

        self.assertEqual(labels.tolist(), [0, 0, 2])
        self.assertLess(maximum_distance, 3.0)

    def test_distant_hungarian_assignment_is_supervised_as_none(self) -> None:
        expected_vertices = (vertex(100, 100), vertex(500, 100))
        expected = ParsedChemSVGSequenceV4(
            expected_vertices,
            (
                StrokeRecordV4(
                    begin_x=100,
                    begin_y=100,
                    end_x=500,
                    end_y=100,
                    display_type="SINGLE",
                    bond_type="SINGLE",
                    direction="NONE",
                    stereo="STEREONONE",
                ),
            ),
        )
        predicted = [vertex(102, 100), vertex(900, 900)]

        labels, maximum_distance = detected_pair_labels(
            expected,
            predicted,
            [(0, 1)],
            maximum_mapping_distance_pixels=12.0,
        )

        self.assertEqual(labels.tolist(), [0])
        self.assertGreater(maximum_distance, 12.0)

    def test_validation_score_uses_edge_f1_before_precision(self) -> None:
        high_f1 = {
            "skeleton_exact_rate": 0.2,
            "edge_set_exact_rate": 0.3,
            "edge_precision": 0.8,
            "edge_recall": 0.8,
        }
        high_precision = {
            "skeleton_exact_rate": 0.2,
            "edge_set_exact_rate": 0.3,
            "edge_precision": 0.95,
            "edge_recall": 0.5,
        }

        self.assertGreater(validation_score(high_f1), validation_score(high_precision))


if __name__ == "__main__":
    unittest.main()
