from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from chemvector.evaluate_evidence_carrier_pair_coverage import (
    aggregate_pair_coverage_evaluations,
    evaluate_pair_coverage_sample,
    main,
)


def line(x0: float, x1: float, y: float) -> list[list[float]]:
    return [[x0, y], [x1, y]]


def record() -> dict:
    return {
        "anchor_segments_pixels": torch.tensor(
            (
                line(0, 5, -0.1),
                line(0, 5, 1.1),
                line(5, 10, -0.1),
                line(5, 10, 1.1),
                line(0, 10, 0.0),
                line(0, 10, 1.0),
            ),
            dtype=torch.float32,
        ),
        "decoded_segments_pixels": torch.tensor(
            (
                line(0, 5, 0),
                line(0, 5, 1),
                line(5, 10, 0),
                line(5, 10, 1),
                line(0, 10, 0),
                line(0, 10, 1),
            ),
            dtype=torch.float32,
        ),
        "anchor_target_line_ids": torch.tensor((0, 0, 1, 1, 0, 1)),
        "anchor_source_indices": torch.arange(6),
        "anchor_group_ids": torch.tensor((0, 0, 1, 1, 0, 1)),
        "anchor_group_types": torch.zeros(6, dtype=torch.long),
        "pair_indices": torch.tensor(((0, 1), (2, 3), (4, 5))),
        "pair_labels": torch.tensor((1, 1, 0)),
    }


class EvidenceCarrierPairCoverageEvaluationTest(unittest.TestCase):
    def test_main_reports_the_requested_output_path(self) -> None:
        output = Path("/tmp/custom-coverage-audit.json")
        report = {
            "status": "DEV-GO",
            "coverage_metrics": {
                "selected_cross_target_pairs": 0,
                "multiplicity_2_recall": 1.0,
                "pairable_sample_exact": 1.0,
                "coverage_vetoed_pair_count": 1,
            },
        }
        stdout = io.StringIO()
        with (
            patch(
                "chemvector.evaluate_evidence_carrier_pair_coverage.audit",
                return_value=report,
            ),
            patch.object(sys, "argv", ("coverage", "--output", str(output))),
            redirect_stdout(stdout),
        ):
            main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["output"], str(output.resolve()))

    def test_coverage_removes_only_the_bridge_cross_pair(self) -> None:
        result = evaluate_pair_coverage_sample(
            record(),
            torch.tensor((5.0, 4.0, 3.0)),
            threshold=0.5,
            verify_permutation=True,
        )

        self.assertEqual(result["initial_matching"]["selected_cross_target_pairs"], 1)
        self.assertEqual(result["selected_cross_target_pairs"], 0)
        self.assertEqual(result["coverage_vetoed_cross_target_pairs"], 1)
        self.assertEqual(result["coverage_vetoed_same_target_pairs"], 0)
        self.assertTrue(result["permutation_deterministic"])
        self.assertEqual(result["coverage"]["vetoed_candidate_indices"], [2])

    def test_aggregate_reports_coverage_latency_and_counts(self) -> None:
        first = evaluate_pair_coverage_sample(
            record(), torch.tensor((5.0, 4.0, 3.0)), threshold=0.5
        )
        second = evaluate_pair_coverage_sample(
            record(), torch.tensor((5.0, 4.0, 3.0)), threshold=0.5
        )
        metrics = aggregate_pair_coverage_evaluations((first, second))

        self.assertEqual(metrics["selected_cross_target_pairs"], 0)
        self.assertEqual(metrics["coverage_vetoed_pair_count"], 2)
        self.assertEqual(metrics["coverage_vetoed_cross_target_pairs"], 2)
        self.assertTrue(metrics["coverage_permutation_deterministic"])
        self.assertIsNotNone(metrics["coverage_latency_ms"]["p95"])


if __name__ == "__main__":
    unittest.main()
