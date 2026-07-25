from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import chemvector.build_evidence_carrier_sample_threshold_interval_manifest as interval_builder
import chemvector.train_evidence_carrier_sample_threshold_predictor as trainer

from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
    SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT,
    EvidenceCarrierSampleThresholdPredictor,
    robust_open_interval_loss,
    sample_threshold_global_count_features,
)
from chemvector.train_evidence_carrier_sample_threshold_predictor import (
    CHECKPOINT_SCHEMA_VERSION,
    INTERVAL_MANIFEST_SCHEMA_VERSION,
    SampleThresholdIntervalRecord,
    _acquire_training_output_lock,
    _begin_calibration_consumption,
    _complete_calibration_consumption,
    _complete_training_output_lock,
    collate_sample_threshold_interval_records,
    deterministic_fit_batch_indices,
    evaluate_sample_threshold_calibration_once,
    file_sha256,
    load_sample_threshold_interval_manifest,
    load_sample_threshold_predictor_checkpoint,
    predictor_implementation_sha256,
    train_sample_threshold_predictor_steps,
)


def _record(
    sample_key: str,
    *,
    target_interval: tuple[float, float] = (-0.2, 0.0),
    partition: str = "fit",
    pair_count: int = 3,
) -> SampleThresholdIntervalRecord:
    generator = torch.Generator().manual_seed(sum(map(ord, sample_key)))
    descriptors = torch.randn((pair_count, 20), generator=generator)
    gains = torch.linspace(0.1, 1.1, steps=pair_count)
    pair_tokens = torch.cat((descriptors, gains[:, None]), dim=-1)
    pair_indices = torch.stack(
        (torch.arange(pair_count), torch.arange(1, pair_count + 1)), dim=-1
    ).long()
    anchor_count = pair_count + 1
    lower, upper = target_interval
    return SampleThresholdIntervalRecord(
        identity={"sample_key": sample_key},
        partition=partition,
        anchor_count=anchor_count,
        pair_tokens=pair_tokens,
        pair_indices=pair_indices,
        global_count_features=sample_threshold_global_count_features(
            anchor_count, pair_indices, gains
        ),
        interval_lower=torch.tensor((lower,), dtype=torch.float32),
        interval_upper=torch.tensor((upper,), dtype=torch.float32),
        interval_normalization_scale=torch.tensor(
            (upper - lower,), dtype=torch.float32
        ),
        loss_eligible=True,
        exclusion_reason=None,
    )


def _interval(
    lower: float,
    upper: float,
) -> dict:
    if math.isfinite(lower) and math.isfinite(upper):
        scale = upper - lower
    elif math.isfinite(lower):
        scale = max(1.0, abs(lower - SAMPLE_THRESHOLD_BASELINE_TAU))
    elif math.isfinite(upper):
        scale = max(1.0, abs(upper - SAMPLE_THRESHOLD_BASELINE_TAU))
    else:
        scale = 1.0
    margin = 0.05 * scale
    return {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "normalization_scale": scale,
        "loss_lower_threshold": lower + margin if math.isfinite(lower) else lower,
        "loss_upper_threshold": upper - margin if math.isfinite(upper) else upper,
        "loss_boundary_inset_fraction": 0.05,
        "fp32_training_representable": True,
        "production_certified_open_interval": True,
        "permutation_deterministic": True,
    }


def _manifest_record(
    sample_key: str,
    *,
    loss_eligible: bool = True,
    partition: str = "fit",
) -> dict:
    record = _record(sample_key)
    return {
        "identity": dict(record.identity),
        "partition": partition,
        "target_free": {
            "pair_tokens": record.pair_tokens,
            "pair_indices": record.pair_indices,
            "canonical_to_source_candidate_indices": torch.arange(
                record.candidate_pair_count
            ),
            "source_to_canonical_candidate_indices": torch.arange(
                record.candidate_pair_count
            ),
            "raw_global_counts": (
                torch.expm1(record.global_count_features).round().long()
            ),
            "global_count_features": record.global_count_features,
        },
        "target_aware": {
            "loss_eligible": loss_eligible,
            "candidate_reachable": loss_eligible,
            "scalar_reachable_target_consistent": loss_eligible,
            "early_certificate_used": False,
            "parametric_hull_fully_enumerated": True,
            "exclusion_reason": (
                None if loss_eligible else "candidate_unreachable"
            ),
            "oracle_equivalent_robust_intervals": (
                [_interval(-0.2, 0.0)] if loss_eligible else []
            ),
        },
    }


class SampleThresholdPredictorTest(unittest.TestCase):
    def test_parameter_contract_is_exactly_1841(self) -> None:
        model = EvidenceCarrierSampleThresholdPredictor()

        self.assertEqual(model.parameter_count(), 1_841)
        self.assertEqual(
            model.parameter_count(), SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT
        )
        self.assertEqual(model.config_dict()["pair_encoder"], [21, 32, 16])
        self.assertEqual(model.config_dict()["threshold_head"], [36, 16, 1])

    def test_empty_pair_set_has_zero_mean_and_max_pool(self) -> None:
        model = EvidenceCarrierSampleThresholdPredictor().eval()
        output = model(
            torch.empty((2, 0, 21)),
            torch.empty((2, 0), dtype=torch.bool),
            torch.zeros((2, 4)),
        )

        self.assertTrue(torch.equal(output["pooled_pair_mean"], torch.zeros(2, 16)))
        self.assertTrue(torch.equal(output["pooled_pair_max"], torch.zeros(2, 16)))
        self.assertTrue(torch.isfinite(output["tau"]).all())

    def test_canonical_collation_is_bitwise_pair_permutation_invariant(self) -> None:
        torch.manual_seed(7)
        model = EvidenceCarrierSampleThresholdPredictor().eval()
        original = _record("permutation", pair_count=5)
        permutation = torch.tensor((4, 0, 3, 1, 2))
        permuted = replace(
            original,
            pair_tokens=original.pair_tokens[permutation],
            pair_indices=original.pair_indices[permutation],
        )
        first_batch = collate_sample_threshold_interval_records([original])
        second_batch = collate_sample_threshold_interval_records([permuted])

        self.assertTrue(
            torch.equal(first_batch["pair_tokens"], second_batch["pair_tokens"])
        )
        first = model(
            first_batch["pair_tokens"],
            first_batch["pair_mask"],
            first_batch["global_count_features"],
        )
        second = model(
            second_batch["pair_tokens"],
            second_batch["pair_mask"],
            second_batch["global_count_features"],
        )

        self.assertTrue(torch.equal(first["tau"], second["tau"]))
        self.assertTrue(torch.equal(first["delta"], second["delta"]))

    def test_global_counts_use_strict_tau0_and_unique_incident_anchors(self) -> None:
        pairs = torch.tensor(((0, 1), (1, 2), (3, 4), (0, 4)), dtype=torch.long)
        gains = torch.tensor(
            (
                SAMPLE_THRESHOLD_BASELINE_TAU,
                SAMPLE_THRESHOLD_BASELINE_TAU + 1e-4,
                SAMPLE_THRESHOLD_BASELINE_TAU + 0.2,
                SAMPLE_THRESHOLD_BASELINE_TAU - 0.1,
            ),
            dtype=torch.float64,
        )

        features = sample_threshold_global_count_features(5, pairs, gains)

        torch.testing.assert_close(
            features,
            torch.log1p(torch.tensor((5.0, 4.0, 2.0, 4.0), dtype=torch.float64)),
        )
        self.assertEqual(len(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES), 21)
        self.assertEqual(len(SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES), 4)

    def test_interval_loss_uses_inset_linear_hinge_and_best_interval(self) -> None:
        tau = torch.tensor((-1.0, 5.0, 2.0, 123.0), requires_grad=True)
        lower = torch.tensor(
            ((0.0, 4.0), (0.0, 4.0), (-torch.inf, 10.0), (-torch.inf, -torch.inf))
        )
        upper = torch.tensor(
            ((2.0, 6.0), (2.0, 6.0), (1.0, torch.inf), (torch.inf, torch.inf))
        )
        mask = torch.tensor(
            ((True, False), (True, True), (True, False), (True, False))
        )
        scale = torch.tensor(((2.0, 2.0), (2.0, 2.0), (1.0, 1.0), (1.0, 1.0)))

        output = robust_open_interval_loss(tau, lower, upper, mask, scale)

        torch.testing.assert_close(
            output["per_sample_loss"], torch.tensor((0.55, 0.0, 1.05, 0.0))
        )
        self.assertEqual(output["selected_interval_index"].tolist(), [0, 1, 0, 0])
        output["loss"].backward()
        self.assertTrue(torch.isfinite(tau.grad).all())

    def test_manifest_loader_rejects_fp32_collapsed_interval(self) -> None:
        with self.assertRaisesRegex(
            trainer.SampleThresholdManifestError,
            "FP32",
        ):
            trainer._validate_interval(
                _interval(1.00000001, 1.00000002),
                sample_label="collapsed",
            )

    def test_builder_loader_and_loss_share_fp32_interval_contract(self) -> None:
        interval = interval_builder.robust_loss_interval(
            1.9111892657945413,
            1.912528213945018,
        )
        lower, upper, scale = trainer._validate_interval(
            interval,
            sample_label="roundoff",
        )

        output = robust_open_interval_loss(
            torch.tensor([1.912], dtype=torch.float32),
            torch.tensor([[lower]], dtype=torch.float32),
            torch.tensor([[upper]], dtype=torch.float32),
            torch.tensor([[True]]),
            torch.tensor([[scale]], dtype=torch.float32),
        )

        self.assertTrue(torch.isfinite(output["loss"]))

    def test_manifest_loader_preserves_unreachable_records_outside_loss(self) -> None:
        payload = {
            "schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
            "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
            "formal_eligible": False,
            "early_shortcut_enabled": False,
            "feature_contract": {
                "pair_token_dim": 21,
                "pair_token_feature_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
                "normalized_descriptor_dim": 20,
                "normalized_descriptor_feature_names": list(
                    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES[:-1]
                ),
                "descriptor_source": (
                    "frozen_pair_scorer_transform_plus_checkpoint_mean_scale"
                ),
                "raw_pair_gain_index": 20,
                "pair_order": (
                    "lexicographic_first_anchor_id_then_second_anchor_id"
                ),
                "candidate_index_mapping": (
                    "target_free stores canonical_to_source and source_to_canonical"
                ),
                "global_count_feature_dim": 4,
                "raw_global_count_feature_names": [
                    "anchor_count",
                    "candidate_pair_count",
                    "gain_above_tau0_count",
                    "gain_above_tau0_incident_unique_anchor_count",
                ],
                "global_count_feature_names": list(
                    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
                ),
                "global_count_transform": "log1p(raw_count)",
                "gain_above_tau0_comparison": "strict_greater_than",
            },
            "frozen_descriptor_normalization": {
                "mean": torch.zeros(20),
                "scale": torch.ones(20),
                "source": "committed_pair_scorer_checkpoint",
            },
            "protocol": {
                "early_capability_certificate_allowed": False,
                "multiple_structurally_equivalent_intervals_retained": True,
                "candidate_unreachable_excluded_from_loss": True,
                "scalar_unreachable_excluded_from_loss": True,
                "loss_boundary_inset_fraction": 0.05,
                "finite_interval_normalization_scale": "upper-lower",
                "one_sided_interval_normalization_scale": (
                    "max(1,abs(finite_boundary-tau0))"
                ),
                "fully_unbounded_interval_normalization_scale": 1.0,
            },
            "fresh_read": False,
            "validation_read": False,
            "sealed_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
            "partitions": {
                "fit": {
                    "records": [
                        _manifest_record("fit-0"),
                        _manifest_record("fit-unreachable", loss_eligible=False),
                    ],
                    "loss_eligible_indices": torch.tensor((0,)),
                    "candidate_unreachable_indices": torch.tensor((1,)),
                    "scalar_unreachable_indices": torch.empty(0, dtype=torch.long),
                },
                "calibration": {
                    "records": [
                        _manifest_record("cal-0", partition="calibration")
                    ],
                    "loss_eligible_indices": torch.tensor((0,)),
                    "candidate_unreachable_indices": torch.empty(0, dtype=torch.long),
                    "scalar_unreachable_indices": torch.empty(0, dtype=torch.long),
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.pt"
            torch.save(payload, path)
            manifest = load_sample_threshold_interval_manifest(
                path, require_formal_counts=False
            )

        self.assertEqual(len(manifest.partitions["fit"].records), 2)
        self.assertEqual(len(manifest.partitions["fit"].loss_records), 1)
        self.assertEqual(len(manifest.partitions["fit"].excluded_records), 1)
        with self.assertRaisesRegex(ValueError, "loss-excluded"):
            collate_sample_threshold_interval_records(
                manifest.partitions["fit"].records
            )

    def test_builder_commitment_artifacts_feed_formal_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_paths = {
                "base_cache": root / "base.pt",
                "augmentation_cache": root / "augmentation.pt",
                "checkpoint": root / "pair-scorer.pt",
            }
            for name, path in source_paths.items():
                path.write_bytes(name.encode("ascii"))
            source_hashes = {
                name: file_sha256(path) for name, path in source_paths.items()
            }
            implementation = {
                "interval_manifest": "4" * 64,
                "reachability_audit": "5" * 64,
            }
            source_commitment = {
                **source_hashes,
                "manifest_implementation": implementation["interval_manifest"],
            }
            manifest_path = (root / "manifest.pt").resolve()
            audit_path = (root / "audit.json").resolve()
            lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
            feature_contract = {
                "pair_token_dim": 21,
                "pair_token_feature_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
                "normalized_descriptor_dim": 20,
                "normalized_descriptor_feature_names": list(
                    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES[:-1]
                ),
                "descriptor_source": (
                    "frozen_pair_scorer_transform_plus_checkpoint_mean_scale"
                ),
                "raw_pair_gain_index": 20,
                "pair_order": (
                    "lexicographic_first_anchor_id_then_second_anchor_id"
                ),
                "candidate_index_mapping": (
                    "target_free stores canonical_to_source and source_to_canonical"
                ),
                "global_count_feature_dim": 4,
                "raw_global_count_feature_names": [
                    "anchor_count",
                    "candidate_pair_count",
                    "gain_above_tau0_count",
                    "gain_above_tau0_incident_unique_anchor_count",
                ],
                "global_count_feature_names": list(
                    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
                ),
                "global_count_transform": "log1p(raw_count)",
                "gain_above_tau0_comparison": "strict_greater_than",
            }
            protocol = {
                "complete_parametric_upper_hull_required": True,
                "early_capability_certificate_allowed": False,
                "production_interval_witnesses": "strict_25_50_75_percent",
                "multiple_structurally_equivalent_intervals_retained": True,
                "candidate_unreachable_excluded_from_loss": True,
                "scalar_unreachable_excluded_from_loss": True,
                "loss_boundary_inset_fraction": 0.05,
                "finite_interval_normalization_scale": "upper-lower",
                "one_sided_interval_normalization_scale": (
                    "max(1,abs(finite_boundary-tau0))"
                ),
                "fully_unbounded_interval_normalization_scale": 1.0,
            }
            fit_records = [
                _manifest_record("fit-0"),
                _manifest_record("fit-1"),
            ]
            calibration_records = [
                _manifest_record("cal-0", partition="calibration")
            ]

            def partition(records: list[dict]) -> dict:
                return {
                    "sample_count": len(records),
                    "loss_eligible_sample_count": len(records),
                    "candidate_unreachable_sample_count": 0,
                    "candidate_unreachable_target_count": 0,
                    "scalar_unreachable_sample_count": 0,
                    "oracle_interval_count": len(records),
                    "multi_interval_loss_sample_count": 0,
                    "finite_interval_count": len(records),
                    "lower_unbounded_interval_count": 0,
                    "upper_unbounded_interval_count": 0,
                    "full_hull_sample_count": len(records),
                    "early_certificate_sample_count": 0,
                    "expected_samples": len(records),
                    "complete_partition": True,
                    "records": records,
                    "loss_eligible_indices": torch.arange(len(records)),
                    "candidate_unreachable_indices": torch.empty(
                        0, dtype=torch.long
                    ),
                    "scalar_unreachable_indices": torch.empty(0, dtype=torch.long),
                }

            partitions = {
                "fit": partition(fit_records),
                "calibration": partition(calibration_records),
            }
            manifest_input = {
                "base_cache_path": str(source_paths["base_cache"]),
                "augmentation_cache_path": str(source_paths["augmentation_cache"]),
                "checkpoint_path": str(source_paths["checkpoint"]),
                "sha256": source_hashes,
                "partitions": ["fit", "calibration"],
                "limit_per_partition": None,
            }
            manifest_payload = {
                "schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
                "status": "TRAIN-CAL-INTERVAL-MANIFEST-COMPLETE",
                "formal_eligible": True,
                "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
                "source_sha256": source_commitment,
                "early_shortcut_enabled": False,
                "commitment_protocol": {
                    "manifest_path": str(manifest_path),
                    "summary_path": str(audit_path),
                    "completed_build_lock_path": str(lock_path),
                    "summary_binds_manifest_sha256": True,
                    "completed_build_lock_required_for_training": True,
                    "no_overwrite": True,
                },
                "target_free_predictor_inputs": True,
                "target_aware_fields_are_loss_labels_only": True,
                "validation_read": False,
                "fresh_read": False,
                "sealed_read": False,
                "internal_test_read": False,
                "mr_tadf_read": False,
                "protocol": protocol,
                "feature_contract": feature_contract,
                "frozen_descriptor_normalization": {
                    "mean": torch.zeros(20),
                    "scale": torch.ones(20),
                    "source": "committed_pair_scorer_checkpoint",
                },
                "input": manifest_input,
                "implementation_sha256": implementation,
                "partitions": partitions,
            }
            audit_payload = {
                "schema_version": interval_builder.SUMMARY_SCHEMA_VERSION,
                "manifest_schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
                "status": "TRAIN-CAL-INTERVAL-MANIFEST-COMPLETE",
                "formal_eligible": True,
                "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
                "source_sha256": source_commitment,
                "validation_read": False,
                "fresh_read": False,
                "sealed_read": False,
                "internal_test_read": False,
                "mr_tadf_read": False,
                "input": manifest_input,
                "feature_contract": feature_contract,
                "implementation_sha256": implementation,
                "partitions": {
                    name: {
                        key: value
                        for key, value in value.items()
                        if key not in {
                            "records",
                            "loss_eligible_indices",
                            "candidate_unreachable_indices",
                            "scalar_unreachable_indices",
                        }
                    }
                    for name, value in partitions.items()
                },
            }
            final_audit = interval_builder._write_outputs(
                manifest_path,
                audit_path,
                manifest_payload,
                audit_payload,
            )
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "evidence-carrier-interval-manifest-build-lock-0.1"
                        ),
                        "status": "completed",
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": final_audit["manifest_sha256"],
                        "summary_path": str(audit_path),
                        "summary_sha256": file_sha256(audit_path),
                        "formal_eligible": True,
                        "source_sha256": source_commitment,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    interval_builder,
                    "BASE_CACHE_SHA256",
                    source_hashes["base_cache"],
                ),
                patch.object(
                    interval_builder,
                    "AUGMENTATION_CACHE_SHA256",
                    source_hashes["augmentation_cache"],
                ),
                patch.object(
                    interval_builder,
                    "CHECKPOINT_SHA256",
                    source_hashes["checkpoint"],
                ),
                patch.object(
                    interval_builder,
                    "_implementation_sha256",
                    return_value=implementation,
                ),
                patch.dict(
                    trainer.EXPECTED_PARTITION_SAMPLE_COUNTS,
                    {"fit": 2, "calibration": 1},
                    clear=True,
                ),
            ):
                loaded = load_sample_threshold_interval_manifest(
                    manifest_path,
                    require_formal_counts=True,
                    audit_path=audit_path,
                )

        self.assertEqual(loaded.sha256, final_audit["manifest_sha256"])
        self.assertEqual(len(loaded.partitions["fit"].records), 2)
        self.assertEqual(loaded.commitment["status"], "completed")

    def test_deterministic_batch_stream_cycles_and_repeats_exactly(self) -> None:
        first = list(
            deterministic_fit_batch_indices(5, steps=4, batch_size=3, seed=11)
        )
        second = list(
            deterministic_fit_batch_indices(5, steps=4, batch_size=3, seed=11)
        )

        self.assertEqual(first, second)
        self.assertEqual([len(batch) for batch in first], [3, 3, 3, 3])
        self.assertNotEqual(first, list(
            deterministic_fit_batch_indices(5, steps=4, batch_size=3, seed=12)
        ))

    def test_synthetic_training_is_weight_deterministic(self) -> None:
        records = [
            _record("fit-0", target_interval=(-1.0, -0.8)),
            _record("fit-1", target_interval=(1.4, 1.6)),
            _record("fit-2", target_interval=(-0.5, -0.3)),
        ]

        def run() -> tuple[dict[str, torch.Tensor], dict]:
            torch.manual_seed(123)
            model = EvidenceCarrierSampleThresholdPredictor()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4
            )
            metrics = train_sample_threshold_predictor_steps(
                model,
                optimizer,
                records,
                steps=5,
                batch_size=2,
                seed=17,
                max_grad_norm=1.0,
                device=torch.device("cpu"),
            )
            return (
                {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                },
                metrics,
            )

        first_state, first_metrics = run()
        second_state, second_metrics = run()

        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(first_metrics["optimizer_steps"], 5)
        for name in first_state:
            torch.testing.assert_close(
                first_state[name], second_state[name], rtol=0.0, atol=0.0
            )

    def test_calibration_consumption_marker_is_exclusive_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "consumption.json"
            checkpoint = root / "last.pt"
            calibration = root / "calibration.json"
            checkpoint.write_bytes(b"sealed-checkpoint")
            calibration.write_text("{}\n", encoding="utf-8")
            checkpoint_sha = file_sha256(checkpoint)
            started = _begin_calibration_consumption(
                marker,
                manifest_path=root / "manifest.pt",
                manifest_sha256="1" * 64,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha,
            )

            with self.assertRaises(FileExistsError):
                _begin_calibration_consumption(
                    marker,
                    manifest_path=root / "manifest.pt",
                    manifest_sha256="1" * 64,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                )
            marker_sha = _complete_calibration_consumption(
                started,
                calibration_path=calibration,
                calibration_sha256=file_sha256(calibration),
                checkpoint_sha256_after_calibration=checkpoint_sha,
                calibration_status="CALIBRATION-TAU-PREDICTOR-GO",
            )
            completed = json.loads(marker.read_text(encoding="utf-8"))

            self.assertEqual(completed["status"], "complete")
            self.assertTrue(completed["checkpoint_sha256_unchanged"])
            self.assertEqual(marker_sha, file_sha256(marker))
            with self.assertRaises(FileExistsError):
                _begin_calibration_consumption(
                    marker,
                    manifest_path=root / "manifest.pt",
                    manifest_sha256="1" * 64,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                )

    def test_training_output_lock_blocks_same_directory_and_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = _acquire_training_output_lock(root)
            with self.assertRaises(FileExistsError):
                _acquire_training_output_lock(root)
            artifacts = {}
            for name in (
                "final.pt",
                "calibration.json",
                "summary.json",
                "calibration-consumption.json",
            ):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                artifacts[name] = path
            lock_sha = _complete_training_output_lock(
                lock,
                checkpoint_path=artifacts["final.pt"],
                checkpoint_sha256=file_sha256(artifacts["final.pt"]),
                calibration_path=artifacts["calibration.json"],
                calibration_sha256=file_sha256(artifacts["calibration.json"]),
                summary_path=artifacts["summary.json"],
                summary_sha256=file_sha256(artifacts["summary.json"]),
                calibration_consumption_marker=artifacts[
                    "calibration-consumption.json"
                ],
                calibration_consumption_marker_sha256=file_sha256(
                    artifacts["calibration-consumption.json"]
                ),
            )
            committed = json.loads(lock.read_text(encoding="utf-8"))

            self.assertEqual(committed["status"], "complete")
            self.assertEqual(lock_sha, file_sha256(lock))
            with self.assertRaises(FileExistsError):
                _acquire_training_output_lock(root)

    def test_sealed_checkpoint_loads_into_a_fresh_exact_model(self) -> None:
        torch.manual_seed(31)
        expected_model = EvidenceCarrierSampleThresholdPredictor()
        implementation = predictor_implementation_sha256()
        manifest_implementation = interval_builder._implementation_sha256()
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": expected_model.model_type,
            "model": expected_model.state_dict(),
            "model_config": expected_model.config_dict(),
            "parameter_count": expected_model.parameter_count(),
            "optimizer_steps": 1_500,
            "checkpoint_selection": "last_step_only",
            "implementation_sha256": implementation,
            "manifest_implementation_sha256": manifest_implementation,
            "manifest_path": "manifest.pt",
            "manifest_sha256": "1" * 64,
            "manifest_audit_path": "audit.json",
            "manifest_audit_sha256": "2" * 64,
            "manifest_commitment_path": "manifest.pt.lock",
            "manifest_commitment_sha256": "3" * 64,
            "base_cache_path": "base.pt",
            "base_cache_sha256": interval_builder.BASE_CACHE_SHA256,
            "augmentation_cache_path": "augmentation.pt",
            "augmentation_cache_sha256": interval_builder.AUGMENTATION_CACHE_SHA256,
            "frozen_pair_scorer_checkpoint_path": "pair-scorer.pt",
            "frozen_pair_scorer_checkpoint_sha256": (
                interval_builder.CHECKPOINT_SHA256
            ),
            "calibration_used_for_training": False,
            "calibration_used_for_checkpoint_selection": False,
            "calibration_evaluated": False,
            "validation_read": False,
            "fresh_read": False,
            "sealed_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(payload, path)
            loaded, checkpoint = load_sample_threshold_predictor_checkpoint(
                path, verify_provenance_hashes=False
            )

        self.assertEqual(checkpoint["implementation_sha256"], implementation)
        self.assertIsNot(loaded, expected_model)
        for name, expected in expected_model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], expected))

    def test_calibration_path_evaluates_each_sample_exactly_once(self) -> None:
        manifest_records = [
            _record(
                f"cal-{index}",
                target_interval=(0.0, 2.0),
                partition="calibration",
                pair_count=1,
            )
            for index in range(2)
        ]
        source_records = []
        for index, manifest_record in enumerate(manifest_records):
            source_records.append(
                {
                    "identity": {
                        "sample_key": f"cal-{index}",
                    },
                    "anchor_segments_pixels": torch.tensor(
                        (((0.0, 0.0), (10.0, 0.0)), ((0.0, 1.0), (10.0, 1.0)))
                    ),
                    "anchor_target_line_ids": torch.tensor((0, 0), dtype=torch.long),
                    "anchor_source_indices": torch.tensor((0, 1), dtype=torch.long),
                    "anchor_group_ids": torch.tensor((0, 0), dtype=torch.long),
                    "anchor_group_types": torch.tensor((1, 1), dtype=torch.long),
                    "pair_indices": manifest_record.pair_indices,
                    "pair_labels": torch.tensor((1,), dtype=torch.long),
                }
            )
        torch.manual_seed(5)
        model = EvidenceCarrierSampleThresholdPredictor()

        with patch.object(
            trainer,
            "evaluate_evidence_carrier_pair_sample",
            wraps=trainer.evaluate_evidence_carrier_pair_sample,
        ) as evaluator:
            result = evaluate_sample_threshold_calibration_once(
                model,
                manifest_records,
                source_records,
                calibration_component_count=1,
                batch_size=2,
                device=torch.device("cpu"),
            )

        self.assertEqual(evaluator.call_count, 2)
        self.assertTrue(result["one_shot"])
        self.assertFalse(result["threshold_search"])
        self.assertFalse(result["checkpoint_selection"])
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(len(result["sample_evaluations"]), 2)
        self.assertIn(
            result["status"],
            {
                "CALIBRATION-TAU-PREDICTOR-GO",
                "CALIBRATION-TAU-PREDICTOR-NO-GO",
            },
        )
        self.assertNotIn("matching_p95_below_20ms", result["gate"]["checks"])
        self.assertIn(
            "matching_p95_below_20ms", result["gate"]["diagnostic_checks"]
        )
        self.assertGreaterEqual(
            result["predictor_latency_ms_per_sample"]["mean"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
