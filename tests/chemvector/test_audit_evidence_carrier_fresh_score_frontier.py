from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import chemvector.audit_evidence_carrier_fresh_score_frontier as frontier


def _record(
    *,
    m2_matched: int = 1,
    m2_pairable: int = 1,
    exact: bool | None = True,
    cross: int = 0,
    vetoed: int = 0,
) -> dict:
    return {
        "baseline": {
            "multiplicity_2_matched_count": m2_matched,
            "multiplicity_2_target_count": m2_pairable,
            "pairable_multiplicity_2_target_count": m2_pairable,
            "pairable_sample_exact": exact,
            "selected_cross_target_pairs": cross,
        },
        "coverage": {"coverage": {"vetoed_pair_count": vetoed}},
    }


class FreshScoreFrontierAuditTest(unittest.TestCase):
    def test_relevant_sample_union_has_only_declared_failure_modes(self) -> None:
        self.assertFalse(frontier._is_related_record(_record()))
        self.assertTrue(
            frontier._is_related_record(_record(m2_matched=0, m2_pairable=1))
        )
        self.assertTrue(frontier._is_related_record(_record(exact=False)))
        self.assertTrue(frontier._is_related_record(_record(cross=1)))
        self.assertTrue(frontier._is_related_record(_record(vetoed=1)))

    def test_candidate_diagnostics_explain_positive_anchor_conflict_and_cross_rank(self) -> None:
        target = {
            "pair_indices": torch.tensor(((0, 1), (0, 2))),
            "pair_labels": torch.tensor((1, 0)),
            "anchor_target_line_ids": torch.tensor((0, 0, 1)),
            "decoded_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (5.0, 0.0)),
                    ((0.0, 1.0), (5.0, 1.0)),
                    ((0.0, 2.0), (5.0, 2.0)),
                )
            ),
        }
        scores = torch.tensor((0.8, 0.9))
        baseline = {"selected_candidate_indices": [1]}
        coverage = {
            "selected_candidate_indices": [1],
            "coverage": {
                "config": asdict(
                    frontier.EvidenceCarrierPairCoverageConfig()
                ),
                "initial_selected_candidate_indices": [1],
                "vetoed_candidate_indices": [],
                "veto_diagnostics": [],
            },
        }

        result = frontier._candidate_diagnostics(
            target, scores, baseline, coverage
        )

        positive = result["expected_positive_candidates"][0]
        self.assertFalse(positive["selected_by_baseline"])
        self.assertTrue(positive["conflicts_with_selected_pair_anchor"])
        self.assertEqual(positive["conflicting_selected_candidate_indices"], [1])
        self.assertEqual(
            positive["selection_state"], "blocked_by_selected_anchor_conflict"
        )
        self.assertEqual(
            positive["incident_rank_1based_by_anchor"], {"0": 2, "1": 1}
        )
        self.assertEqual(
            positive["best_incident_negative_candidate_index_by_anchor"],
            {"0": 1, "1": None},
        )
        self.assertAlmostEqual(
            positive["margin_to_best_incident_negative"], -0.1, places=6
        )
        self.assertTrue(positive["incident_negative_outranks_positive"])
        self.assertFalse(
            positive["positive_strictly_outscores_all_incident_negatives"]
        )
        cross = result["selected_cross_candidates"][0]
        self.assertEqual(cross["score_rank_1based"], 1)
        self.assertEqual(cross["candidate_count"], 2)
        self.assertEqual(cross["score_rank_normalized_0_best"], 0.0)
        self.assertTrue(cross["incident_top_both"])
        self.assertEqual(
            cross["target_anchor_multiplicity_by_anchor"], {"0": 2, "2": 1}
        )
        self.assertEqual(
            cross["same_target_positive_candidate_available_by_anchor"],
            {"0": True, "2": False},
        )
        self.assertEqual(
            cross["same_target_positive_candidate_indices_by_anchor"],
            {"0": [0], "2": []},
        )
        self.assertEqual(
            cross["incident_candidate_count_by_anchor"], {"0": 2, "2": 1}
        )
        self.assertEqual(
            cross["incident_rank_normalized_0_best_by_anchor"],
            {"0": 0.0, "2": 0.0},
        )
        self.assertEqual(
            cross["coverage_branch"]["reason"],
            "kept_insufficient_stronger_pairs",
        )
        sample = {
            "reasons": {
                "m2_miss": True,
                "pairable_exact_false": True,
                "selected_cross": True,
                "coverage_veto": False,
            },
            **result,
        }
        aggregate = frontier._aggregate_bucket([sample])
        self.assertEqual(aggregate["expected_positive_available_missed_count"], 1)
        self.assertEqual(
            aggregate["missed_available_quadrants"],
            {
                "at_or_below_threshold__positive_strictly_outscores": 0,
                "at_or_below_threshold__positive_does_not_strictly_outscore": 0,
                "above_threshold__positive_strictly_outscores": 0,
                "above_threshold__positive_does_not_strictly_outscore": 1,
            },
        )

    def test_below_threshold_local_top_is_separate_from_local_ranking_error(self) -> None:
        target = {
            "pair_indices": torch.tensor(((0, 1),)),
            "pair_labels": torch.tensor((1,)),
            "anchor_target_line_ids": torch.tensor((3, 3)),
            "decoded_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (5.0, 0.0)),
                    ((0.0, 1.0), (5.0, 1.0)),
                )
            ),
        }
        scores = torch.tensor((frontier.DEVELOPMENT_THRESHOLD - 0.1,))
        baseline = {"selected_candidate_indices": []}
        coverage = {
            "selected_candidate_indices": [],
            "coverage": {
                "config": asdict(frontier.EvidenceCarrierPairCoverageConfig()),
                "initial_selected_candidate_indices": [],
                "vetoed_candidate_indices": [],
                "veto_diagnostics": [],
            },
        }

        result = frontier._candidate_diagnostics(
            target, scores, baseline, coverage
        )

        positive = result["expected_positive_candidates"][0]
        self.assertEqual(
            positive["incident_rank_1based_by_anchor"], {"0": 1, "1": 1}
        )
        self.assertEqual(positive["incident_negative_count"], 0)
        self.assertEqual(
            positive["best_incident_negative_candidate_index_by_anchor"],
            {"0": None, "1": None},
        )
        self.assertIsNone(positive["best_incident_negative_score"])
        self.assertIsNone(positive["margin_to_best_incident_negative"])
        self.assertFalse(positive["incident_negative_outranks_positive"])
        self.assertTrue(
            positive["positive_strictly_outscores_all_incident_negatives"]
        )
        sample = {
            "reasons": {
                "m2_miss": True,
                "pairable_exact_false": True,
                "selected_cross": False,
                "coverage_veto": False,
            },
            **result,
        }
        aggregate = frontier._aggregate_bucket([sample])
        self.assertEqual(aggregate["expected_positive_available_missed_count"], 1)
        self.assertEqual(aggregate["expected_positive_absent_count"], 0)
        self.assertEqual(
            aggregate["missed_available_quadrants"]
            ["at_or_below_threshold__positive_strictly_outscores"],
            1,
        )

    def test_tied_incident_negative_is_not_counted_as_strict_local_win(self) -> None:
        target = {
            "pair_indices": torch.tensor(((0, 1), (0, 2))),
            "pair_labels": torch.tensor((1, 0)),
            "anchor_target_line_ids": torch.tensor((5, 5, 6)),
            "decoded_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (5.0, 0.0)),
                    ((0.0, 1.0), (5.0, 1.0)),
                    ((0.0, 2.0), (5.0, 2.0)),
                )
            ),
        }
        score = frontier.DEVELOPMENT_THRESHOLD + 0.1
        baseline = {"selected_candidate_indices": [1]}
        coverage = {
            "selected_candidate_indices": [1],
            "coverage": {
                "config": asdict(frontier.EvidenceCarrierPairCoverageConfig()),
                "initial_selected_candidate_indices": [1],
                "vetoed_candidate_indices": [],
                "veto_diagnostics": [],
            },
        }

        result = frontier._candidate_diagnostics(
            target, torch.tensor((score, score)), baseline, coverage
        )

        positive = result["expected_positive_candidates"][0]
        self.assertAlmostEqual(positive["margin_to_best_incident_negative"], 0.0)
        self.assertFalse(positive["incident_negative_outranks_positive"])
        self.assertFalse(
            positive["positive_strictly_outscores_all_incident_negatives"]
        )
        sample = {
            "reasons": {
                "m2_miss": True,
                "pairable_exact_false": True,
                "selected_cross": True,
                "coverage_veto": False,
            },
            **result,
        }
        aggregate = frontier._aggregate_bucket([sample])
        self.assertEqual(
            aggregate["missed_available_quadrants"]
            ["above_threshold__positive_does_not_strictly_outscore"],
            1,
        )

    def test_missing_m2_candidate_is_not_misreported_as_a_score_failure(self) -> None:
        target = {
            "pair_indices": torch.empty((0, 2), dtype=torch.long),
            "pair_labels": torch.empty((0,), dtype=torch.long),
            "anchor_target_line_ids": torch.tensor((7, 7)),
            "decoded_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (5.0, 0.0)),
                    ((0.0, 1.0), (5.0, 1.0)),
                )
            ),
        }
        baseline = {"selected_candidate_indices": []}
        coverage = {
            "selected_candidate_indices": [],
            "coverage": {
                "config": asdict(frontier.EvidenceCarrierPairCoverageConfig()),
                "initial_selected_candidate_indices": [],
                "vetoed_candidate_indices": [],
                "veto_diagnostics": [],
            },
        }

        result = frontier._candidate_diagnostics(
            target, torch.empty((0,)), baseline, coverage
        )

        candidate = result["expected_positive_candidates"][0]
        self.assertFalse(candidate["candidate_available"])
        self.assertIsNone(candidate["score"])
        self.assertIsNone(candidate["margin_to_frozen_threshold"])
        self.assertEqual(candidate["selection_state"], "candidate_absent")
        sample = {
            "reasons": {
                "m2_miss": True,
                "pairable_exact_false": False,
                "selected_cross": False,
                "coverage_veto": False,
            },
            **result,
        }
        aggregate = frontier._aggregate_bucket([sample])
        self.assertEqual(aggregate["expected_positive_absent_count"], 1)
        self.assertEqual(aggregate["expected_positive_available_missed_count"], 0)

    def test_vetoed_same_target_pair_reports_target_multiplicity(self) -> None:
        target = {
            "pair_indices": torch.tensor(((0, 1), (0, 2), (1, 2))),
            "pair_labels": torch.tensor((1, 1, 1)),
            "anchor_target_line_ids": torch.tensor((4, 4, 4)),
            "decoded_segments_pixels": torch.tensor(
                (
                    ((0.0, 0.0), (5.0, 0.0)),
                    ((0.0, 1.0), (5.0, 1.0)),
                    ((0.0, 2.0), (5.0, 2.0)),
                )
            ),
        }
        scores = torch.tensor((1.0, 0.9, 0.8))
        baseline = {"selected_candidate_indices": [0]}
        coverage = {
            "selected_candidate_indices": [],
            "coverage": {
                "config": asdict(frontier.EvidenceCarrierPairCoverageConfig()),
                "initial_selected_candidate_indices": [0],
                "vetoed_candidate_indices": [0],
                "veto_diagnostics": [{"candidate_index": 0}],
            },
        }

        result = frontier._candidate_diagnostics(
            target, scores, baseline, coverage
        )

        vetoed = result["vetoed_same_target_candidates"]
        self.assertEqual(len(vetoed), 1)
        self.assertEqual(vetoed[0]["target_line_id"], 4)
        self.assertEqual(vetoed[0]["target_anchor_multiplicity"], 3)
        sample = {
            "reasons": {
                "m2_miss": False,
                "pairable_exact_false": False,
                "selected_cross": False,
                "coverage_veto": True,
            },
            **result,
        }
        aggregate = frontier._aggregate_bucket([sample])
        self.assertEqual(aggregate["vetoed_same_target_candidate_count"], 1)
        self.assertEqual(
            aggregate["vetoed_same_target_target_multiplicity_counts"],
            {"3": 1},
        )

    def test_coverage_branch_recalculation_normalizes_geometry_to_float32(self) -> None:
        coverage = {
            "coverage": {
                "config": asdict(frontier.EvidenceCarrierPairCoverageConfig()),
                "initial_selected_candidate_indices": [0, 1, 2],
                "vetoed_candidate_indices": [],
                "veto_diagnostics": [],
            }
        }
        scores = np.asarray((3.0, 2.0, 1.0), dtype=np.float64)
        pairs = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int64)
        decoded = np.asarray(
            [
                ((0.0, float(index)), (5.0, float(index)))
                for index in range(6)
            ],
            dtype=np.float64,
        )

        def upper_bound(query, contributors, *, config):
            self.assertEqual(query.dtype, np.float32)
            self.assertEqual(contributors.dtype, np.float32)
            return np.zeros((2,), dtype=np.float64)

        with patch.object(frontier, "_coverage_upper_bound", side_effect=upper_bound):
            result = frontier._coverage_branch_reason(
                2,
                scores=scores,
                pairs=pairs,
                decoded=decoded,
                coverage_result=coverage,
            )

        self.assertEqual(result["reason"], "kept_union_upper_bound_below_minimum")

    def test_registered_paired_report_rejects_wrong_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired-run1.json"
            path.write_text("{}\n", encoding="utf-8")
            registry = {
                "fresh_dev_run1": {
                    "path": str(path),
                    "sha256": "0" * 64,
                }
            }
            with patch.object(frontier, "PAIRED_REPORT_REGISTRY", registry):
                registration = frontier._registered_paired_report(path)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    frontier._load_paired_report(
                        Path(registration["path"]),
                        expected_sha256=registration["sha256"],
                    )

    def test_registered_paired_report_rejects_post_commit_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired-run1.json"
            original = b'{"status":"frozen"}\n'
            digest = hashlib.sha256(original).hexdigest()
            path.write_bytes(original)
            registry = {
                "fresh_dev_run1": {
                    "path": str(path),
                    "sha256": digest,
                }
            }
            with patch.object(frontier, "PAIRED_REPORT_REGISTRY", registry):
                registration = frontier._registered_paired_report(path)
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    frontier._load_paired_report(
                        Path(registration["path"]),
                        expected_sha256=registration["sha256"],
                    )

    def test_implementation_map_includes_audit_script_sha256(self) -> None:
        pipeline = {
            "implementation": {
                "frozen.py": {"path": "/tmp/frozen.py", "sha256": "a" * 64}
            }
        }

        result = frontier._implementation_map(pipeline)

        audit_record = result[Path(frontier.__file__).name]
        self.assertEqual(audit_record["path"], str(Path(frontier.__file__).resolve()))
        self.assertEqual(
            audit_record["sha256"], frontier.file_sha256(Path(frontier.__file__))
        )

    def test_report_config_hash_is_checked_before_record_replay(self) -> None:
        package = SimpleNamespace(
            role="dev",
            commitment_sha256="a" * 64,
            input_fingerprint="b" * 64,
            fresh_audit_eligible=True,
            independence_verified=True,
            sample_count=0,
            molecule_count=0,
            manifest_sha256="c" * 64,
            freeze_sha256="d" * 64,
            isolation_sha256="e" * 64,
            identities=(),
        )
        registration = {
            "role": "dev",
            "path": "/tmp/commitment.dev.json",
            "sha256": "a" * 64,
        }
        route_b = {"route": "B"}
        route_c = {"route": "C"}
        report = {
            "schema_version": frontier.PAIRED_REPORT_SCHEMA_VERSION,
            "role": "dev",
            "training_steps": 0,
            "parameter_updates": 0,
            "independent_generalization": True,
            "protocol": {
                "training_steps": 0,
                "parameter_updates": 0,
                "single_dataset_pass": True,
                "shared_pair_scores": True,
                "threshold_recalibrated": False,
                "coverage_recalibrated": False,
                "sample_limit_exposed": False,
                "internal_test_read": False,
                "sealed_audit_read": False,
                "registered_commitment": True,
                "threshold": frontier.DEVELOPMENT_THRESHOLD,
            },
            "input": {
                "commitment_path": registration["path"],
                "commitment_sha256": "a" * 64,
                "package_input_fingerprint": "b" * 64,
                "fresh_audit_eligible": True,
                "independence_verified": True,
                "samples": 0,
                "molecules": 0,
                "manifest_sha256": "c" * 64,
                "freeze_sha256": "d" * 64,
                "isolation_sha256": "e" * 64,
            },
            "paired_execution": {
                "sample_count_b": 0,
                "sample_count_c": 0,
                "target_build_count": 0,
                "p0_forward_count": 0,
                "paired_any_mismatch_count": 0,
                "paired_input_mismatch_count": 0,
                "paired_target_mismatch_count": 0,
                "paired_score_mismatch_count": 0,
            },
            "pipeline": {
                "checkpoint_sha256": frontier.CHECKPOINT_SHA256,
                "calibration_report_sha256": frontier.CALIBRATION_REPORT_SHA256,
                "development_threshold": frontier.DEVELOPMENT_THRESHOLD,
                "route_b_config": route_b,
                "route_b_config_sha256": "f" * 64,
                "route_c_config": route_c,
                "route_c_config_sha256": frontier._canonical_sha256(route_c),
            },
            "sample_evaluations": [],
        }

        with self.assertRaisesRegex(ValueError, "route_b_config hash"):
            frontier._validate_report_before_pipeline(
                report, registration, package
            )

    def test_cli_exposes_no_threshold_or_sample_limit(self) -> None:
        destinations = {action.dest for action in frontier.build_parser()._actions}
        self.assertEqual(
            destinations,
            {"help", "commitment", "paired_report", "output", "device"},
        )

    def test_main_reports_diagnostic_output(self) -> None:
        output = Path("/tmp/frontier-audit.json")
        report = {
            "status": "DIAGNOSTIC-COMPLETE",
            "diagnostic_only": True,
            "input": {"replayed_sample_count": 7},
        }
        stdout = io.StringIO()
        with (
            patch.object(frontier, "audit", return_value=report),
            patch.object(
                sys,
                "argv",
                ("frontier", "--output", str(output)),
            ),
            redirect_stdout(stdout),
        ):
            frontier.main()

        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["diagnostic_only"])
        self.assertEqual(payload["replayed_sample_count"], 7)
        self.assertEqual(payload["output"], str(output.resolve()))

    def test_audit_rejects_smoke_registration_before_loading_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commitment = root / "commitment.smoke.json"
            paired = root / "paired-run1.json"
            output = root / "audit.json"
            args = SimpleNamespace(
                commitment=str(commitment),
                paired_report=str(paired),
                output=str(output),
                device="cpu",
            )
            with (
                patch.object(
                    frontier,
                    "_registered_commitment",
                    return_value={
                        "role": "smoke",
                        "path": str(commitment),
                        "sha256": "a" * 64,
                    },
                ),
                patch.object(
                    frontier, "load_fresh_evidence_carrier_package"
                ) as loader,
            ):
                with self.assertRaisesRegex(ValueError, "fresh dev"):
                    frontier.audit(args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
