from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from chemvector.bond_group_set import PARALLEL_2, SINGLE_LINE
from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalPairSampleIdentity,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_cache import CACHE_SCHEMA_VERSION
from chemvector.evaluate_evidence_carrier_pair_scorer import (
    evaluate,
    pair_scorer_validation_gate,
)
from chemvector.train_evidence_carrier_pair_scorer import file_sha256


class FakePairScorer(torch.nn.Module):
    def forward(self, pair_features: torch.Tensor, pair_mask: torch.Tensor):
        gains = pair_features[..., 0]
        return {"pair_gains": torch.where(pair_mask, gains, torch.zeros_like(gains))}


def identity(
    key: str,
    molecule: str,
    source: str,
    *,
    split: str,
    variant: str,
) -> ConditionalPairSampleIdentity:
    return ConditionalPairSampleIdentity(
        sample_key=key,
        molecule_id=molecule,
        source_structure_id=f"pilot|{split}|{source}",
        source_pilot="pilot",
        source_split=split,
        source_record=source,
        render_variant=variant,
    )


def validation_sample(current: ConditionalPairSampleIdentity) -> dict:
    return {
        "key": current.sample_key,
        "identity": current,
        "image": Image.new("RGB", (100, 100), "white"),
        "primitives": {},
        "layers": {},
        "meta": current.as_dict(),
    }


def target(
    target_ids: list[int],
    pairs: list[tuple[int, int]],
    scores: list[float],
    *,
    group_ids: list[int],
    group_types: list[int],
) -> dict:
    anchor_count = len(target_ids)
    segments = torch.tensor(
        [
            ((10.0, 10.0 + index * 3), (90.0, 10.0 + index * 3))
            for index in range(anchor_count)
        ],
        dtype=torch.float32,
    )
    pair_indices = torch.tensor(pairs, dtype=torch.long)
    if not pairs:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    features = torch.zeros((len(pairs), 8), dtype=torch.float32)
    if scores:
        features[:, 0] = torch.tensor(scores)
    pair_labels = torch.tensor(
        [int(target_ids[first] == target_ids[second]) for first, second in pairs],
        dtype=torch.long,
    )
    pair_target_ids = torch.tensor(
        [[target_ids[first], target_ids[second]] for first, second in pairs],
        dtype=torch.long,
    )
    if not pairs:
        pair_target_ids = torch.empty((0, 2), dtype=torch.long)
    return {
        "anchor_segments_pixels": segments,
        "anchor_target_line_ids": torch.tensor(target_ids, dtype=torch.long),
        "anchor_source_indices": torch.arange(anchor_count),
        "pair_indices": pair_indices,
        "pair_features": features,
        "pair_labels": pair_labels,
        "pair_target_line_ids": pair_target_ids,
        "raw_keep_labels": torch.ones(anchor_count, dtype=torch.long),
        "raw_target_line_ids": torch.tensor(target_ids, dtype=torch.long),
        "raw_group_ids": torch.tensor(group_ids, dtype=torch.long),
        "raw_group_types": torch.tensor(group_types, dtype=torch.long),
    }


def args(
    checkpoint: Path,
    validation_shard: Path,
    output: Path,
    *,
    limit: int = 2,
) -> Namespace:
    return Namespace(
        checkpoint=str(checkpoint),
        validation_shards=[str(validation_shard)],
        output=str(output),
        limit=limit,
        device="cpu",
    )


class EvaluateEvidenceCarrierPairScorerTest(unittest.TestCase):
    def test_one_pass_evaluation_uses_frozen_threshold_and_writes_validation_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"frozen-checkpoint")
            cache_path = root / "train-cache.pt"
            cache_path.write_bytes(b"frozen-train-cache")
            shard = root / "validation-000000.tar"
            shard.write_bytes(b"synthetic-validation-shard")
            output_dir = root / "evaluation"

            train_identity = identity(
                "train-1", "mol-train", "source-train", split="train", variant="original"
            )
            validation_identities = [
                identity(
                    "validation-1",
                    "mol-validation-1",
                    "source-validation-1",
                    split="validation",
                    variant="original",
                ),
                identity(
                    "validation-2",
                    "mol-validation-2",
                    "source-validation-2",
                    split="validation",
                    variant="stress",
                ),
            ]
            samples = [validation_sample(current) for current in validation_identities]
            targets = [
                target(
                    [0, 0],
                    [(0, 1)],
                    [2.0],
                    group_ids=[10, 10],
                    group_types=[SINGLE_LINE, SINGLE_LINE],
                ),
                target(
                    [1, 2],
                    [],
                    [],
                    group_ids=[20, 20],
                    group_types=[PARALLEL_2, PARALLEL_2],
                ),
            ]
            checkpoint = {
                "model_type": "fake-pair-scorer",
                "parameter_count": 833,
                "best_epoch": 4,
                "gain_threshold": 1.0,
                "cache_path": str(cache_path),
                "cache_sha256": file_sha256(cache_path),
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
                "pairing_config": asdict(
                    conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
                ),
                "train_cal_gate": {"status": "GO"},
                "uses_internal_test": False,
                "validation_read": False,
            }
            cache = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "fit_records": [
                    {"identity": train_identity.as_dict(), "partition": "fit"}
                ],
                "calibration_records": [],
            }

            with (
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "load_pair_scorer_checkpoint",
                    return_value=(FakePairScorer(), checkpoint),
                ) as checkpoint_loader,
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "load_evidence_carrier_pair_cache",
                    return_value=cache,
                ) as cache_loader,
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "ConditionalOrdinaryPairWebDataset",
                    return_value=samples,
                ) as dataset_loader,
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "build_conditional_ordinary_pair_target",
                    side_effect=targets,
                ) as target_builder,
            ):
                payload = evaluate(args(checkpoint_path, shard, output_dir))

            output = output_dir / "validation.json"
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["schema_version"], payload["schema_version"])
            self.assertEqual(payload["protocol"]["validation_passes"], 1)
            self.assertEqual(payload["protocol"]["target_build_count"], 2)
            self.assertFalse(payload["protocol"]["threshold_recalibrated"])
            self.assertEqual(payload["checkpoint"]["gain_threshold"], 1.0)
            self.assertEqual(payload["checkpoint"]["sha256"], file_sha256(checkpoint_path))
            self.assertEqual(payload["checkpoint"]["cache_sha256"], file_sha256(cache_path))
            self.assertEqual(payload["identity"]["validation_sample_count"], 2)
            self.assertEqual(
                payload["identity"]["train_validation_zero_overlap"]["molecule_overlap"],
                0,
            )
            self.assertEqual(payload["metrics"]["selected_same_target_pairs"], 1)
            self.assertEqual(payload["metrics"]["selected_cross_target_pairs"], 0)
            self.assertEqual(payload["metrics"]["parallel_2_group_count"], 1)
            self.assertEqual(target_builder.call_count, 2)
            checkpoint_loader.assert_called_once()
            cache_loader.assert_called_once()
            dataset_kwargs = dataset_loader.call_args.kwargs
            self.assertIsNone(dataset_kwargs["render_variant"])
            self.assertEqual(dataset_kwargs["split"], "validation")
            self.assertEqual(dataset_kwargs["limit"], 2)

    def test_identity_overlap_aborts_instead_of_publishing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            cache_path = root / "cache.pt"
            cache_path.write_bytes(b"cache")
            shard = root / "validation.tar"
            shard.write_bytes(b"validation")
            output = root / "result"
            train_identity = identity(
                "train", "shared-molecule", "train-source", split="train", variant="original"
            )
            validation_identity = identity(
                "validation",
                "shared-molecule",
                "validation-source",
                split="validation",
                variant="original",
            )
            checkpoint = {
                "gain_threshold": 0.0,
                "cache_path": str(cache_path),
                "cache_sha256": file_sha256(cache_path),
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
                "pairing_config": asdict(
                    conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
                ),
                "train_cal_gate": {"status": "GO"},
                "uses_internal_test": False,
                "validation_read": False,
            }
            cache = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "fit_records": [{"identity": train_identity.as_dict()}],
                "calibration_records": [],
            }
            empty_target = target(
                [], [], [], group_ids=[], group_types=[]
            )
            with (
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "load_pair_scorer_checkpoint",
                    return_value=(FakePairScorer(), checkpoint),
                ),
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "load_evidence_carrier_pair_cache",
                    return_value=cache,
                ),
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "ConditionalOrdinaryPairWebDataset",
                    return_value=[validation_sample(validation_identity)],
                ),
                patch(
                    "chemvector.evaluate_evidence_carrier_pair_scorer."
                    "build_conditional_ordinary_pair_target",
                    return_value=empty_target,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "identity leakage"):
                    evaluate(args(checkpoint_path, shard, output, limit=1))
            self.assertFalse((output / "validation.json").exists())

    def test_internal_test_path_is_rejected_before_checkpoint_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "internal_test" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"sealed")
            shard = root / "validation.tar"
            shard.write_bytes(b"validation")
            with patch(
                "chemvector.evaluate_evidence_carrier_pair_scorer."
                "load_pair_scorer_checkpoint"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "internal_test"):
                    evaluate(args(checkpoint, shard, root / "out", limit=1))
                loader.assert_not_called()

    def test_formal_gate_requires_all_topology_and_identity_checks(self) -> None:
        metrics = {
            "selected_cross_target_pairs": 0,
            "parallel_2_group_count": 10,
            "parallel_2_distinct_preservation": 1.0,
            "selected_pair_precision": 1.0,
            "multiplicity_2_recall": 0.999,
            "pairable_sample_count": 100,
            "pairable_sample_exact": 0.98,
            "ordinary_support_loss_pp": 0.0,
            "matching_optimal_sample_count": 500,
            "permutation_deterministic": True,
            "matching_latency_ms": {"p95": 5.0},
        }
        identity_audit = {"molecule_overlap": 0, "source_structure_overlap": 0}
        gate = pair_scorer_validation_gate(
            metrics,
            sample_count=500,
            expected_sample_count=500,
            render_variants={"coordgen", "original", "rdkit", "stress"},
            identity_audit=identity_audit,
            checkpoint={"train_cal_gate": {"status": "GO"}},
        )

        self.assertEqual(gate["status"], "GO")
        metrics["selected_cross_target_pairs"] = 1
        failed = pair_scorer_validation_gate(
            metrics,
            sample_count=500,
            expected_sample_count=500,
            render_variants={"coordgen", "original", "rdkit", "stress"},
            identity_audit=identity_audit,
            checkpoint={"train_cal_gate": {"status": "GO"}},
        )
        self.assertEqual(failed["status"], "NO-GO")
        self.assertFalse(failed["checks"]["selected_cross_target_equal_0"])


if __name__ == "__main__":
    unittest.main()
