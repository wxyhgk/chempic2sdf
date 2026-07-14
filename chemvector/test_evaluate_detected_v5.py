"""Tests for detected end-to-end graph metrics."""

from __future__ import annotations

from dataclasses import replace
import unittest

from Data.sdf.chemsvg_sequence_v4 import parse_chemsvg_v4, serialize_chemsvg_v4
from Data.sdf.test_chemmol_sequence import make_annotations
from chemvector.evaluation_metrics import compare_graphs, summarize


class DetectedEvaluationTests(unittest.TestCase):
    def test_identical_graph_has_perfect_metrics(self) -> None:
        parsed = parse_chemsvg_v4(serialize_chemsvg_v4(*make_annotations("c1ccccc1F")))
        result = compare_graphs(parsed, parsed)
        self.assertTrue(result.exact_topology)
        self.assertEqual(result.correct_edges, result.expected_edges)
        self.assertEqual(result.correct_bond_types, result.expected_edges)

    def test_element_error_does_not_hide_correct_geometry(self) -> None:
        parsed = parse_chemsvg_v4(serialize_chemsvg_v4(*make_annotations("CN")))
        predicted = replace(
            parsed,
            vertices=(replace(parsed.vertices[0], element="O"), *parsed.vertices[1:]),
        )
        result = compare_graphs(parsed, predicted)
        self.assertEqual(result.matched_vertices, 2)
        self.assertEqual(result.correct_elements, 1)
        self.assertFalse(result.exact_topology)

    def test_summary_uses_micro_averaged_graph_counts(self) -> None:
        summary = summarize(
            [
                {
                    "key": "validation_000001_rdkit",
                    "generation_completed": True,
                    "repair_success": True,
                    "sdf_valid": True,
                    "chemical_match": True,
                    "exact_topology": True,
                    "expected_vertices": 2,
                    "predicted_vertices": 2,
                    "matched_vertices": 2,
                    "correct_elements": 2,
                    "expected_edges": 1,
                    "predicted_edges": 1,
                    "correct_edges": 1,
                    "correct_bond_types": 1,
                    "seconds": 1.0,
                }
            ]
        )
        self.assertEqual(summary["chemical_match_rate"], 1.0)
        self.assertEqual(summary["edge_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
