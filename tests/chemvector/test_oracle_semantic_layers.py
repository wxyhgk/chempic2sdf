"""Tests for the semantic bond-layer Oracle experiment."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from chemvector.oracle_semantic_layers import (
    bond_alpha_to_ink_image,
    summarize_records,
)


def mode(*, exact: bool, false_edges: int, correct_edges: int = 2) -> dict:
    return {
        "repair_success": True,
        "sdf_valid": exact,
        "chemical_match": exact,
        "exact_topology": exact,
        "expected_vertices": 3,
        "predicted_vertices": 3,
        "matched_vertices": 3,
        "correct_elements": 3,
        "expected_edges": 2,
        "predicted_edges": correct_edges + false_edges,
        "correct_edges": correct_edges,
        "correct_bond_types": correct_edges,
        "false_edges": false_edges,
        "missed_edges": 2 - correct_edges,
    }


class OracleSemanticLayerTests(unittest.TestCase):
    def test_bond_alpha_is_inverted_to_black_ink(self) -> None:
        alpha = Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L")
        result = np.asarray(bond_alpha_to_ink_image(alpha).convert("L"))
        self.assertEqual(result.tolist(), [[255, 0]])

    def test_summary_reports_paired_false_edge_reduction(self) -> None:
        records = [
            {
                "font_bucket": "large",
                "font_family_id": "font-a",
                "variant": "rdkit",
                "split": "validation",
                "visible_text_count": 1,
                "baseline": mode(exact=False, false_edges=2),
                "oracle_bond": mode(exact=True, false_edges=0),
                "oracle_bbox_bond": mode(exact=True, false_edges=0),
            },
            {
                "font_bucket": "small",
                "font_family_id": "font-b",
                "variant": "stress",
                "split": "internal_test",
                "visible_text_count": 1,
                "baseline": mode(exact=True, false_edges=0),
                "oracle_bond": mode(exact=True, false_edges=0),
                "oracle_bbox_bond": mode(exact=True, false_edges=0),
            },
        ]

        summary = summarize_records(records, excluded_samples=2)

        self.assertEqual(summary["delta"]["false_edge_reduction"], 1.0)
        self.assertEqual(summary["delta"]["baseline_wrong_oracle_correct"], 1)
        self.assertEqual(summary["delta"]["baseline_correct_oracle_wrong"], 0)
        self.assertEqual(summary["large_text"]["samples"], 1)
        self.assertEqual(summary["excluded_samples"], 2)
        self.assertEqual(summary["bbox_delta"]["false_edge_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()
