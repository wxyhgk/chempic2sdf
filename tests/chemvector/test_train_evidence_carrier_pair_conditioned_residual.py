from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import chemvector.build_evidence_carrier_pair_residual_fit_dev as builder
import chemvector.train_evidence_carrier_pair_conditioned_residual as trainer
from chemvector.evidence_carrier_pair_conditioned_residual import (
    EvidenceCarrierPairConditionedResidual,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)
from chemvector.train_evidence_carrier_pair_conditioned_residual import (
    PairResidualTrainingError,
    audit_fit_model_finite,
    collate_pair_residual_records,
    deterministic_fit_batch_indices,
    evaluate_dev_once,
    load_dev_sidecar_after_consumption,
    load_pair_residual_manifest,
    load_pair_residual_checkpoint,
    pair_residual_capability_gate,
    train_pair_residual_steps,
)
from tests.chemvector.test_build_evidence_carrier_pair_residual_fit_dev import (
    _TinyContract,
    _fixture_payloads,
)


def _record(sample_key: str, *, partition: str = "fit") -> dict:
    pair_indices = torch.tensor(((0, 1), (0, 2), (3, 4)), dtype=torch.long)
    pair_tokens = torch.zeros((3, 21), dtype=torch.float32)
    pair_tokens[:, -1] = torch.tensor((2.0, 1.5, -0.5))
    raw_counts = torch.tensor((5, 3, 2, 3), dtype=torch.long)
    return {
        "identity": {
            "sample_key": sample_key,
            "molecule_id": f"molecule-{sample_key}",
            "source_structure_id": f"source-{sample_key}",
        },
        "partition": partition,
        "target_free": {
            "pair_indices": pair_indices,
            "pair_tokens": pair_tokens,
            "canonical_to_source_candidate_indices": torch.arange(3),
            "source_to_canonical_candidate_indices": torch.arange(3),
            "raw_global_counts": raw_counts,
            "global_count_features": torch.log1p(raw_counts.float()),
        },
        "target_aware": {"pair_labels": torch.tensor((1, 0, 0))},
    }


def _metrics(**overrides: object) -> dict:
    result = {
        "sample_count": 900,
        "selected_cross_target_pairs": 0,
        "selected_pair_count": 100,
        "selected_pair_precision": 1.0,
        "multiplicity_2_target_count": 100,
        "multiplicity_2_matched_count": 100,
        "multiplicity_2_recall": 1.0,
        "pairable_sample_count": 90,
        "pairable_sample_exact_count": 90,
        "pairable_sample_exact": 1.0,
        "parallel_2_group_count": 10,
        "parallel_2_violated_group_count": 0,
        "parallel_2_distinct_preservation": 1.0,
        "ordinary_supported_target_count_before": 100,
        "ordinary_support_lost_target_count": 0,
        "ordinary_support_loss_pp": 0.0,
        "matching_optimal_sample_count": 900,
        "permutation_deterministic": True,
    }
    result.update(overrides)
    return result


class PairResidualTrainerTest(unittest.TestCase):
    def test_collate_separates_target_labels_from_forward_inputs(self) -> None:
        batch = collate_pair_residual_records([_record("a"), _record("b")])

        self.assertEqual(batch["pair_tokens"].shape, (2, 3, 21))
        self.assertEqual(batch["global_count_features"].shape, (2, 4))
        self.assertEqual(batch["pair_indices"].shape, (2, 3, 2))
        self.assertEqual(batch["pair_labels"].shape, (2, 3))
        self.assertNotIn("target_aware", batch)

        model = EvidenceCarrierPairConditionedResidual()
        output = model(
            batch["pair_tokens"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        torch.testing.assert_close(
            output["margin"][batch["pair_mask"]],
            batch["pair_tokens"][..., -1][batch["pair_mask"]]
            - SAMPLE_THRESHOLD_BASELINE_TAU,
        )

    def test_deterministic_batches_fill_across_shuffle_cycles(self) -> None:
        first = list(
            deterministic_fit_batch_indices(3, steps=4, batch_size=5, seed=17)
        )
        second = list(
            deterministic_fit_batch_indices(3, steps=4, batch_size=5, seed=17)
        )

        self.assertEqual(first, second)
        self.assertEqual([len(batch) for batch in first], [5, 5, 5, 5])
        self.assertTrue(all(0 <= index < 3 for batch in first for index in batch))

    def test_short_training_is_reproducible_and_moves_from_zero_residual(self) -> None:
        records = [_record(str(index)) for index in range(4)]

        def run() -> tuple[dict[str, torch.Tensor], dict]:
            torch.manual_seed(123)
            model = EvidenceCarrierPairConditionedResidual()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            metrics = train_pair_residual_steps(
                model,
                optimizer,
                records,
                steps=4,
                batch_size=3,
                seed=99,
                device=torch.device("cpu"),
            )
            return deepcopy(model.state_dict()), metrics

        first_state, first_metrics = run()
        second_state, second_metrics = run()

        for name in first_state:
            torch.testing.assert_close(first_state[name], second_state[name])
        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(first_metrics["optimizer_steps"], 4)
        self.assertGreater(first_metrics["final_batch_residual"]["maximum_absolute"], 0.0)

    def test_fit_finite_audit_covers_valid_fit_records(self) -> None:
        records = [_record("a"), _record("b")]
        model = EvidenceCarrierPairConditionedResidual()
        audit = audit_fit_model_finite(
            model,
            records,
            {
                "mean_loss": 1.0,
                "final_batch_residual": {"maximum_absolute": 0.0},
            },
            batch_size=1,
        )

        self.assertEqual(audit["status"], "FIT-FINITE-AUDIT-PASSED")
        self.assertEqual(audit["sample_count"], 2)
        self.assertEqual(audit["pair_count"], 6)
        self.assertFalse(audit["dev_sidecar_loaded"])

    def test_fit_finite_audit_rejects_nonfinite_state(self) -> None:
        records = [_record("a")]
        model = EvidenceCarrierPairConditionedResidual()
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = torch.nan
        with self.assertRaisesRegex(PairResidualTrainingError, "parameter.*non-finite"):
            audit_fit_model_finite(model, records, {"mean_loss": 1.0})

        model = EvidenceCarrierPairConditionedResidual()
        with self.assertRaisesRegex(PairResidualTrainingError, "mean_loss is non-finite"):
            audit_fit_model_finite(model, records, {"mean_loss": float("nan")})

    def test_gate_requires_absolute_safety_and_strict_baseline_improvement(self) -> None:
        baseline = _metrics(multiplicity_2_matched_count=99)
        candidate = _metrics(multiplicity_2_matched_count=100)
        passed = pair_residual_capability_gate(baseline, candidate)

        self.assertEqual(passed["status"], "GO")
        self.assertTrue(passed["checks"]["strictly_better_than_frozen_baseline"])

        no_op = pair_residual_capability_gate(candidate, candidate)
        self.assertEqual(no_op["status"], "NO-GO")
        self.assertFalse(no_op["checks"]["strictly_better_than_frozen_baseline"])

        unsafe = pair_residual_capability_gate(
            baseline,
            _metrics(selected_cross_target_pairs=1),
        )
        self.assertEqual(unsafe["status"], "NO-GO")
        self.assertFalse(unsafe["checks"]["selected_cross_target_equal_0"])

        changed_checkpoint = pair_residual_capability_gate(
            baseline,
            candidate,
            checkpoint_sha_unchanged=False,
        )
        self.assertEqual(changed_checkpoint["status"], "NO-GO")
        self.assertFalse(
            changed_checkpoint["checks"]["checkpoint_sha_unchanged"]
        )

    def test_real_schema_index_loader_never_hashes_or_loads_dev_sidecar(self) -> None:
        contract = _TinyContract()
        with tempfile.TemporaryDirectory() as directory, contract.patches():
            root = Path(directory)
            index, fit_sidecar, _ = _fixture_payloads(contract)
            index_path = root / "index.pt"
            fit_path = root / "fit.pt"
            dev_path = root / "dev.pt"
            audit_path = root / "audit.json"
            lock_path = index_path.with_suffix(".pt.lock")
            source_paths = {
                "base_cache": root / "base.pt",
                "augmentation_cache": root / "augmentation.pt",
                "frozen_pair_scorer_checkpoint": root / "pair.pt",
                "frozen_p0_checkpoint": root / "p0.pt",
            }
            for name, path in source_paths.items():
                path.write_bytes(name.encode("ascii"))
                index["input"][name]["path"] = str(path)

            fit_sha = "1" * 64
            dev_sha = "2" * 64
            index_sha = "3" * 64
            audit_sha = "4" * 64
            lock_sha = "5" * 64
            index["partitions"]["fit"]["records_path"] = str(fit_path)
            index["partitions"]["fit"]["records_sha256"] = fit_sha
            index["partitions"]["dev"]["records_path"] = str(dev_path)
            index["partitions"]["dev"]["records_sha256"] = dev_sha
            torch.save(fit_sidecar, fit_path)
            torch.save(index, index_path)
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": builder.AUDIT_SCHEMA_VERSION,
                        "status": builder.STATUS_COMPLETE,
                        "formal_eligible": True,
                        "index_path": str(index_path),
                        "index_sha256": index_sha,
                        "fit_sidecar_path": str(fit_path),
                        "fit_sidecar_sha256": fit_sha,
                        "dev_sidecar_path": str(dev_path),
                        "dev_sidecar_sha256": dev_sha,
                        "source_sha256": index["source_sha256"],
                        "implementation_sha256": index["implementation_sha256"],
                    }
                ),
                encoding="utf-8",
            )
            lock_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "source_sha256": index["source_sha256"],
                        "implementation_sha256": index["implementation_sha256"],
                        "artifacts": {
                            "index": {"path": str(index_path), "sha256": index_sha},
                            "fit_sidecar": {"path": str(fit_path), "sha256": fit_sha},
                            "dev_sidecar": {"path": str(dev_path), "sha256": dev_sha},
                            "audit": {"path": str(audit_path), "sha256": audit_sha},
                        },
                    }
                ),
                encoding="utf-8",
            )
            hash_values = {
                str(index_path): index_sha,
                str(audit_path): audit_sha,
                str(lock_path): lock_sha,
                str(fit_path): fit_sha,
                **{
                    str(path): index["source_sha256"][name]
                    for name, path in source_paths.items()
                },
            }
            touched: list[tuple[str, str]] = []
            real_torch_load = torch.load

            def fake_sha(path: str | Path) -> str:
                resolved = str(Path(path).resolve())
                touched.append(("hash", resolved))
                if resolved == str(dev_path):
                    raise AssertionError("dev sidecar was hashed before marker")
                return hash_values[resolved]

            def spy_load(path: str | Path, *args, **kwargs):
                resolved = str(Path(path).resolve())
                touched.append(("load", resolved))
                if resolved == str(dev_path):
                    raise AssertionError("dev sidecar was loaded before marker")
                return real_torch_load(path, *args, **kwargs)

            with (
                patch.object(builder, "DEFAULT_FIT_SIDECAR", str(fit_path)),
                patch.object(builder, "DEFAULT_DEV_SIDECAR", str(dev_path)),
                patch.object(trainer, "file_sha256", side_effect=fake_sha),
                patch.object(torch, "load", side_effect=spy_load),
            ):
                loaded_index, fit_records, metadata = load_pair_residual_manifest(
                    index_path,
                    audit_path=audit_path,
                )

            self.assertEqual(len(fit_records), 4)
            self.assertEqual(loaded_index["partitions"]["dev"]["records_path"], str(dev_path))
            self.assertEqual(metadata["dev_sidecar_sha256"], dev_sha)
            self.assertNotIn(("hash", str(dev_path)), touched)
            self.assertNotIn(("load", str(dev_path)), touched)

    def test_dev_loader_rejects_before_any_dev_io_without_marker(self) -> None:
        dev_path = Path("/tmp/v0917-missing-marker-dev.pt")
        metadata = {
            "sha256": "3" * 64,
            "dev_sidecar_path": str(dev_path),
            "dev_sidecar_sha256": "2" * 64,
        }
        index = {
            "implementation_sha256": {},
            "source_sha256": {},
            "split_contract": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "missing-marker.json"
            with (
                patch.object(
                    trainer,
                    "file_sha256",
                    side_effect=AssertionError("dev hash ran before marker"),
                ),
                patch.object(
                    torch,
                    "load",
                    side_effect=AssertionError("dev load ran before marker"),
                ),
                self.assertRaisesRegex(
                    PairResidualTrainingError,
                    "marker is unreadable",
                ),
            ):
                load_dev_sidecar_after_consumption(index, metadata, marker)

    def test_dev_evaluator_reconstructs_canonical_evaluation_target(self) -> None:
        contract = _TinyContract()
        with contract.patches():
            _, _, dev_sidecar = _fixture_payloads(contract)
            model = EvidenceCarrierPairConditionedResidual().eval()
            with patch.object(trainer, "EXPECTED_DEV_SAMPLE_COUNT", 4):
                result = evaluate_dev_once(
                    model,
                    dev_sidecar["records"],
                    device=torch.device("cpu"),
                )

        self.assertEqual(result["baseline_selected_metrics"]["sample_count"], 4)
        self.assertEqual(result["candidate_selected_metrics"]["sample_count"], 4)
        self.assertEqual(result["changed_selection_sample_count"], 0)
        self.assertEqual(result["status"], "DEV900-PAIR-RESIDUAL-NO-GO")

    def test_dev_sidecar_load_and_completion_require_exclusive_marker(self) -> None:
        contract = _TinyContract()
        with tempfile.TemporaryDirectory() as directory, contract.patches():
            root = Path(directory)
            index, _, dev_sidecar = _fixture_payloads(contract)
            dev_path = root / "dev.pt"
            marker_path = root / "dev-consumption.json"
            checkpoint_path = root / "final.pt"
            report_path = root / "dev900.json"
            torch.save(dev_sidecar, dev_path)
            checkpoint_path.write_bytes(b"checkpoint")
            report_path.write_bytes(b"report")
            dev_sha = trainer.file_sha256(dev_path)
            report_sha = trainer.file_sha256(report_path)
            index["partitions"]["dev"]["records_path"] = str(dev_path)
            index["partitions"]["dev"]["records_sha256"] = dev_sha
            metadata = {
                "path": str(root / "index.pt"),
                "sha256": "3" * 64,
                "dev_sidecar_path": str(dev_path),
                "dev_sidecar_sha256": dev_sha,
            }

            trainer._begin_dev_consumption(
                marker_path,
                manifest_metadata=metadata,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256="4" * 64,
            )
            records = load_dev_sidecar_after_consumption(
                index,
                metadata,
                marker_path,
            )
            marker_sha = trainer._complete_dev_consumption(
                marker_path,
                dev_sidecar_path=dev_path,
                dev_sidecar_sha256=dev_sha,
                report_path=report_path,
                report_sha256=report_sha,
                checkpoint_sha256_after="4" * 64,
                status="DEV900-PAIR-RESIDUAL-NO-GO",
            )

            self.assertEqual(len(records), 4)
            self.assertEqual(len(marker_sha), 64)
            completed = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "complete")
            self.assertTrue(completed["checkpoint_sha256_unchanged"])
            with self.assertRaises(FileExistsError):
                trainer._begin_dev_consumption(
                    marker_path,
                    manifest_metadata=metadata,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256="4" * 64,
                )

    def test_sealed_checkpoint_loader_preserves_fit_only_contract(self) -> None:
        model = EvidenceCarrierPairConditionedResidual()
        checkpoint = {
            "checkpoint_schema_version": trainer.CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "model_config": model.config_dict(),
            "parameter_count": model.parameter_count(),
            "optimizer_steps": trainer.TRAINING_STEPS,
            "checkpoint_selection": "last_step_only",
            "implementation_sha256": trainer.implementation_sha256(),
            "source_sha256": builder._expected_source_sha256(),
            "dev_sidecar_loaded": False,
            "dev_used_for_training": False,
            "dev_used_for_checkpoint_selection": False,
            "dev_evaluated": False,
            "original_calibration_records_accessed": False,
            "validation_read": False,
            "fresh_read": False,
            "sealed_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
            "dev_sidecar_path": "/tmp/v0917-declared-dev.pt",
            "dev_sidecar_sha256": "2" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            torch.save(checkpoint, path)
            loaded_model, loaded = load_pair_residual_checkpoint(
                path,
                verify_provenance=False,
            )

        self.assertEqual(loaded["checkpoint_selection"], "last_step_only")
        self.assertFalse(loaded["dev_sidecar_loaded"])
        self.assertEqual(loaded_model.parameter_count(), 2_841)


if __name__ == "__main__":
    unittest.main()
