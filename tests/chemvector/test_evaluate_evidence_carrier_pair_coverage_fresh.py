from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import chemvector.evaluate_evidence_carrier_pair_coverage_fresh as fresh_evaluator
from chemvector.evaluate_evidence_carrier_pair_coverage_fresh import (
    build_parser,
    evaluate_paired_sample,
    main,
    paired_gate,
)


def _pair_result() -> dict:
    return {
        "selected_candidate_indices": [0],
        "selected_anchor_pairs": [[0, 1]],
        "selected_cross_target_pairs": 0,
        "multiplicity_2_matched_count": 1,
        "parallel_2_violated_group_count": 0,
        "ordinary_support_lost_target_count": 0,
        "pairable_sample_eligible": True,
        "pairable_sample_exact": True,
    }


def _aggregate_metrics() -> dict:
    return {
        "selected_cross_target_pairs": 0,
        "selected_pair_precision": 1.0,
        "multiplicity_2_matched_count": 1,
        "multiplicity_2_recall": 1.0,
        "parallel_2_violated_group_count": 0,
        "parallel_2_distinct_preservation": 1.0,
        "ordinary_support_lost_target_count": 0,
        "ordinary_support_loss_pp": 0.0,
        "pairable_sample_exact": 1.0,
        "matching_optimal_sample_count": 1,
        "permutation_deterministic": True,
        "matching_latency_ms": {"p95": 1.0},
    }


def _paired_record(*, baseline_cross: int = 0, cross_delta: int = 0) -> dict:
    fingerprints = {
        "input_sha256": "a" * 64,
        "target_sha256": "b" * 64,
        "score_sha256": "c" * 64,
    }
    return {
        "identity": {"sample_key": "sample"},
        "target_builds": 1,
        "p0_forward_passes": 1,
        "same_initial_matching": True,
        "input_fingerprint": "a" * 64,
        "route_b_fingerprints": dict(fingerprints),
        "route_c_fingerprints": dict(fingerprints),
        "baseline": {"selected_cross_target_pairs": baseline_cross},
        "delta": {
            "cross_target_pairs": cross_delta,
            "multiplicity_2_matched": 0,
            "parallel_2_violations": 0,
            "ordinary_support_lost": 0,
            "pairable_exact_true_to_false": False,
        },
    }


class EvaluateFreshPairCoverageTest(unittest.TestCase):
    def test_one_target_and_one_score_feed_both_branches(self) -> None:
        target = {"anchor_segments": torch.zeros((2, 2, 2))}
        scores = torch.tensor((2.0,))
        baseline = _pair_result()
        coverage = {
            **_pair_result(),
            "initial_matching": dict(baseline),
        }
        sample = {
            "image": Image.new("RGB", (768, 768), "white"),
            "primitives": {},
            "layers": {},
            "input_fingerprint": "a" * 64,
            "identity": SimpleNamespace(
                as_dict=lambda: {
                    "sample_key": "sample",
                    "molecule_id": "molecule",
                }
            ),
        }
        pipeline = {
            "pair_model": object(),
            "p0_model": object(),
            "target_config": object(),
            "pairing_config": object(),
        }
        with patch(
            "chemvector.evaluate_evidence_carrier_pair_coverage_fresh."
            "build_conditional_ordinary_pair_target",
            return_value=target,
        ) as target_builder, patch(
            "chemvector.evaluate_evidence_carrier_pair_coverage_fresh."
            "score_frozen_p0_pair_target",
            return_value={
                "pair_scores": scores,
                "decoded_segments_pixels": torch.zeros((2, 2, 2)),
                "p0_forward_passes": 1,
            },
        ) as scorer, patch(
            "chemvector.evaluate_evidence_carrier_pair_coverage_fresh."
            "score_evidence_carrier_pair_record",
            return_value=baseline,
        ) as b_branch, patch(
            "chemvector.evaluate_evidence_carrier_pair_coverage_fresh."
            "evaluate_pair_coverage_sample",
            return_value=coverage,
        ) as c_branch:
            result = evaluate_paired_sample(
                pipeline,
                sample,
                device=torch.device("cpu"),
            )

        target_builder.assert_called_once()
        scorer.assert_called_once()
        self.assertIs(b_branch.call_args.args[1], scores)
        self.assertIs(c_branch.call_args.args[1], scores)
        self.assertEqual(result["target_builds"], 1)
        self.assertEqual(result["p0_forward_passes"], 1)
        self.assertTrue(result["same_initial_matching"])

    def test_smoke_gate_reports_only_execution_contract(self) -> None:
        package = SimpleNamespace(role="smoke", sample_count=1)
        record = _paired_record()
        baseline = _aggregate_metrics()
        coverage = {
            **_aggregate_metrics(),
            "coverage_vetoed_multiplicity_2_pairs": 0,
        }

        gate = paired_gate(package, [record], baseline, coverage)

        self.assertEqual(gate["status"], "SMOKE-PASS")
        self.assertEqual(gate["interpretation"], "execution-contract-only")
        self.assertIsNone(gate["quality_checks"])

    def test_dev_with_no_baseline_cross_is_safe_but_inconclusive(self) -> None:
        package = SimpleNamespace(
            role="dev",
            sample_count=1,
            independence_verified=True,
        )
        record = _paired_record()
        baseline = _aggregate_metrics()
        coverage = {
            **_aggregate_metrics(),
            "coverage_vetoed_multiplicity_2_pairs": 0,
        }

        gate = paired_gate(package, [record], baseline, coverage)

        self.assertEqual(gate["status"], "SAFE-BUT-INCONCLUSIVE")
        self.assertEqual(gate["mechanism_status"], "SAFE-BUT-INCONCLUSIVE")

    def test_dev_metrics_cannot_go_without_verified_independence_report(self) -> None:
        package = SimpleNamespace(
            role="dev",
            sample_count=1,
            independence_verified=False,
        )
        record = _paired_record(baseline_cross=1, cross_delta=-1)
        baseline = {**_aggregate_metrics(), "selected_cross_target_pairs": 1}
        coverage = {
            **_aggregate_metrics(),
            "selected_cross_target_pairs": 0,
            "coverage_vetoed_multiplicity_2_pairs": 0,
        }

        gate = paired_gate(package, [record], baseline, coverage)

        self.assertEqual(gate["status"], "FRESH-DEV-NO-GO")
        self.assertFalse(
            gate["technical_checks"]["independence_report_verified"]
        )

    def test_cli_exposes_no_threshold_checkpoint_or_sample_limit(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertEqual(
            destinations,
            {"help", "commitment", "output", "device"},
        )

    def test_registry_helper_rejects_unknown_formal_dev_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            fresh_evaluator,
            "COMMITMENT_REGISTRY",
            {},
        ):
            path = Path(directory) / "fresh_dev" / "commitment.dev.json"
            with self.assertRaisesRegex(ValueError, "registered|registry"):
                fresh_evaluator._registered_commitment(path)

    def test_unregistered_formal_dev_stops_before_loader_device_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                commitment=str(root / "fresh_dev" / "commitment.dev.json"),
                output=str(root / "paired.json"),
                device="cpu",
            )
            with (
                patch.object(
                    fresh_evaluator,
                    "_registered_commitment",
                    side_effect=ValueError("formal commitment is not registered"),
                ) as registry,
                patch.object(
                    fresh_evaluator,
                    "load_fresh_evidence_carrier_package",
                ) as loader,
                patch.object(fresh_evaluator, "_device") as device,
                patch.object(
                    fresh_evaluator,
                    "load_committed_v0911_pipeline",
                ) as model_loader,
            ):
                with self.assertRaisesRegex(ValueError, "not registered"):
                    fresh_evaluator.evaluate(args)

            registry.assert_called_once_with(args.commitment)
            loader.assert_not_called()
            device.assert_not_called()
            model_loader.assert_not_called()

    def test_registered_commitment_binds_loader_role_and_digest_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commitment = root / "fresh_dev" / "commitment.dev.json"
            args = SimpleNamespace(
                commitment=str(commitment),
                output=str(root / "paired.json"),
                device="cpu",
            )
            registration = {
                "role": "dev",
                "path": str(commitment.resolve()),
                "sha256": "a" * 64,
            }
            with (
                patch.object(
                    fresh_evaluator,
                    "_registered_commitment",
                    return_value=registration,
                ),
                patch.object(
                    fresh_evaluator,
                    "load_fresh_evidence_carrier_package",
                    side_effect=RuntimeError("stop after bound loader call"),
                ) as loader,
                patch.object(fresh_evaluator, "_device") as device,
                patch.object(
                    fresh_evaluator,
                    "load_committed_v0911_pipeline",
                ) as model_loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "bound loader"):
                    fresh_evaluator.evaluate(args)

            loader.assert_called_once_with(
                args.commitment,
                expected_role="dev",
                expected_commitment_sha256="a" * 64,
            )
            device.assert_not_called()
            model_loader.assert_not_called()

    def test_main_reports_requested_output(self) -> None:
        output = Path("/tmp/fresh-paired.json")
        report = {
            "status": "SMOKE-PASS",
            "role": "smoke",
            "baseline_metrics": {"selected_cross_target_pairs": 0},
            "coverage_metrics": {
                "selected_cross_target_pairs": 0,
                "coverage_vetoed_pair_count": 0,
            },
        }
        stdout = io.StringIO()
        with (
            patch(
                "chemvector.evaluate_evidence_carrier_pair_coverage_fresh.evaluate",
                return_value=report,
            ),
            patch.object(
                sys,
                "argv",
                ("fresh", "--output", str(output)),
            ),
            redirect_stdout(stdout),
        ):
            main()

        self.assertEqual(json.loads(stdout.getvalue())["output"], str(output.resolve()))


if __name__ == "__main__":
    unittest.main()
