from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_NAMES,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_p0_pair_cache import (
    CACHE_SCHEMA_VERSION as P0_PAIR_CACHE_SCHEMA_VERSION,
    DEFAULT_OUTPUT as DEFAULT_P0_PAIR_AUGMENTATION_CACHE,
)
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    EvidenceCarrierP0PairScorer,
)
from chemvector.evidence_carrier_pair_cache import (
    CACHE_SCHEMA_VERSION as BASE_PAIR_CACHE_SCHEMA_VERSION,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_AUGMENTATION_CACHE,
    collate_p0_pair_training_records,
    evaluate_p0_pair_scorer_loss,
    fit_p0_pair_feature_normalization,
    load_p0_pair_scorer_checkpoint,
    load_train_only_p0_pair_caches,
    p0_pair_scorer_loss,
    save_p0_pair_scorer_checkpoint,
    train_p0_pair_scorer_epoch,
    verify_frozen_p0_provenance,
)
from chemvector.train_evidence_carrier_extent import (
    CHECKPOINT_SCHEMA_VERSION as P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION as P0_EXTENT_VALIDATION_SCHEMA_VERSION,
)


def record(
    sample_key: str = "sample-0",
    *,
    anchor_count: int = 3,
    one_pair: bool = False,
) -> dict:
    pair_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pair_labels = torch.tensor([1, 0], dtype=torch.long)
    pair_features = torch.tensor(
        [
            [0.2, 2.0, 40.0, 0.95, 0.90, 0.95, 0.80, 0.90],
            [6.0, 7.0, 8.0, 0.40, 0.20, 0.30, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )
    consistency = torch.tensor(
        [
            [0.1 * index for index in range(12)],
            [1.0 + 0.1 * index for index in range(12)],
        ],
        dtype=torch.float32,
    )
    if one_pair:
        pair_indices = pair_indices[:1]
        pair_labels = pair_labels[:1]
        pair_features = pair_features[:1]
        consistency = consistency[:1]
    generator = torch.Generator().manual_seed(sum(map(ord, sample_key)))
    return {
        "identity": {
            "sample_key": sample_key,
            "molecule_id": f"mol-{sample_key}",
        },
        "partition": "fit",
        "anchor_features": torch.randn(
            (anchor_count, 128), generator=generator, dtype=torch.float32
        ),
        "pair_indices": pair_indices,
        "pair_features": pair_features,
        "pair_consistency_features": consistency,
        "pair_labels": pair_labels,
    }


def fitted_model(records: list[dict]) -> tuple[EvidenceCarrierP0PairScorer, dict]:
    model = EvidenceCarrierP0PairScorer()
    normalization = fit_p0_pair_feature_normalization(model, records)
    return model, normalization


class P0PairScorerTrainerTest(unittest.TestCase):
    def test_default_augmentation_path_tracks_the_v02_fp32_sidecar(self) -> None:
        self.assertEqual(DEFAULT_AUGMENTATION_CACHE, DEFAULT_P0_PAIR_AUGMENTATION_CACHE)
        self.assertIn("step1000-fp32", DEFAULT_AUGMENTATION_CACHE)

    def test_collator_preserves_float32_anchors_and_pads_variable_rows(self) -> None:
        second = record("sample-1", one_pair=True)
        batch = collate_p0_pair_training_records(
            [record(), second], anchor_feature_dim=128
        )

        self.assertEqual(batch["anchor_features"].dtype, torch.float32)
        self.assertEqual(batch["anchor_features"].shape, (2, 3, 128))
        self.assertEqual(batch["pair_features"].shape, (2, 2, 8))
        self.assertEqual(batch["pair_count"].tolist(), [2, 1])
        self.assertEqual(batch["pair_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["pair_labels"][1].tolist(), [1, -1])

    def test_collator_rejects_index_valid_only_against_another_samples_padding(self) -> None:
        short = record("short", anchor_count=2, one_pair=True)
        short["pair_indices"] = torch.tensor([[0, 2]], dtype=torch.long)
        long = record("long", anchor_count=4)

        with self.assertRaisesRegex(ValueError, "this sample's anchors"):
            collate_p0_pair_training_records(
                [short, long], anchor_feature_dim=128
            )

    def test_hard_negative_loss_uses_the_strongest_negative(self) -> None:
        gains = torch.tensor([[2.0, -2.0, 4.0]], requires_grad=True)
        indices = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
        labels = torch.tensor([[0, 1, 0]])
        mask = torch.ones((1, 3), dtype=torch.bool)

        losses = p0_pair_scorer_loss(
            gains,
            indices,
            labels,
            mask,
            hard_negative_top_k=1,
            hard_negative_weight=0.5,
        )

        torch.testing.assert_close(
            losses["hard_negative_loss"], torch.nn.functional.softplus(gains[0, 2])
        )
        self.assertEqual(losses["images_with_hard_negatives"], 1)
        losses["loss"].backward()
        self.assertGreater(float(gains.grad[0, 2]), 0.0)

    def test_one_epoch_updates_the_p0_conditioned_model(self) -> None:
        records = [record(f"sample-{index}") for index in range(4)]
        model, _ = fitted_model(records)
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        metrics = train_p0_pair_scorer_epoch(
            model,
            optimizer,
            records,
            anchor_feature_dim=128,
            batch_size=2,
            device=torch.device("cpu"),
            rng=random.Random(7),
            max_grad_norm=5.0,
            hard_negative_top_k=1,
            hard_negative_weight=0.5,
        )

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertTrue(
            any(
                not torch.equal(before[name], value.detach())
                for name, value in model.named_parameters()
            )
        )

    def test_calibration_loss_is_invariant_to_batch_partition(self) -> None:
        records = [record(f"sample-{index}") for index in range(3)]
        model, _ = fitted_model(records)

        individual = evaluate_p0_pair_scorer_loss(
            model,
            records,
            anchor_feature_dim=128,
            batch_size=1,
            device=torch.device("cpu"),
            hard_negative_top_k=1,
            hard_negative_weight=0.5,
        )
        combined = evaluate_p0_pair_scorer_loss(
            model,
            records,
            anchor_feature_dim=128,
            batch_size=2,
            device=torch.device("cpu"),
            hard_negative_top_k=1,
            hard_negative_weight=0.5,
        )

        for name in (
            "loss",
            "base_loss",
            "balanced_bce",
            "ranking_loss",
            "hard_negative_loss",
        ):
            self.assertAlmostEqual(individual[name], combined[name], places=6)

    def test_checkpoint_roundtrip_preserves_three_provenance_links(self) -> None:
        records = [record(), record("sample-1")]
        model, normalization = fitted_model(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.pt"
            augmentation_path = root / "augmentation.pt"
            p0_path = root / "step-001000.pt"
            base_path.write_bytes(b"base-cache")
            augmentation_path.write_bytes(b"p0-augmentation-cache")
            p0_path.write_bytes(b"p0-checkpoint")
            from chemvector.train_evidence_carrier_p0_pair_scorer import file_sha256

            payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_type": model.model_type,
                "model": model.state_dict(),
                "model_config": model.config_dict(),
                "parameter_count": model.parameter_count(),
                "pair_feature_names": list(PAIR_FEATURE_NAMES),
                "pair_consistency_feature_names": list(
                    P0_PAIR_CONSISTENCY_FEATURE_NAMES
                ),
                "normalization": normalization,
                "loss_config": {
                    "hard_negative_top_k": 4,
                    "hard_negative_weight": 0.5,
                },
                "gain_threshold": 0.75,
                "base_cache_path": str(base_path),
                "base_cache_sha256": file_sha256(base_path),
                "base_cache_schema_version": BASE_PAIR_CACHE_SCHEMA_VERSION,
                "augmentation_cache_path": str(augmentation_path),
                "augmentation_cache_sha256": file_sha256(augmentation_path),
                "augmentation_cache_schema_version": P0_PAIR_CACHE_SCHEMA_VERSION,
                "augmentation_cache_complete": True,
                "frozen_p0_checkpoint_path": str(p0_path),
                "frozen_p0_checkpoint_sha256": file_sha256(p0_path),
                "p0_frozen": True,
                "p0_checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                "p0_model_type": "evidence-carrier-extent-v0.1",
                "p0_global_step": 1000,
                "p0_checkpoint_contains_validation_metrics": True,
                "p0_checkpoint_selected_by_validation": False,
                "upstream_p0_validation_read": True,
                "upstream_p0_checkpoint_selected_by_validation": False,
                "anchor_feature_dim": 128,
                "pairing_profile": "training_wide",
                "pairing_config": asdict(
                    conditional_pair_pairing_config("training_wide")
                ),
                "train_cal_gate": {"status": "GO"},
                "checkpoint_selection": {"candidate_count": 1},
                "uses_internal_test": False,
                "validation_read": False,
            }
            path = root / "best.pt"
            save_p0_pair_scorer_checkpoint(path, payload)
            loaded, checkpoint = load_p0_pair_scorer_checkpoint(path)

        self.assertEqual(checkpoint["gain_threshold"], 0.75)
        self.assertFalse(checkpoint["p0_checkpoint_selected_by_validation"])
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    def test_checkpoint_loader_rehashes_each_live_provenance_file_by_default(self) -> None:
        records = [record(), record("sample-1")]
        model, normalization = fitted_model(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "base_cache": root / "base.pt",
                "augmentation_cache": root / "augmentation.pt",
                "frozen_p0_checkpoint": root / "step-001000.pt",
            }
            originals = {
                "base_cache": b"base",
                "augmentation_cache": b"augmentation",
                "frozen_p0_checkpoint": b"p0",
            }
            for name, source in paths.items():
                source.write_bytes(originals[name])
            from chemvector.train_evidence_carrier_p0_pair_scorer import file_sha256

            payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_type": model.model_type,
                "model": model.state_dict(),
                "model_config": model.config_dict(),
                "parameter_count": model.parameter_count(),
                "pair_feature_names": list(PAIR_FEATURE_NAMES),
                "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
                "normalization": normalization,
                "loss_config": {},
                "gain_threshold": 0.0,
                "base_cache_path": str(paths["base_cache"]),
                "base_cache_sha256": file_sha256(paths["base_cache"]),
                "base_cache_schema_version": BASE_PAIR_CACHE_SCHEMA_VERSION,
                "augmentation_cache_path": str(paths["augmentation_cache"]),
                "augmentation_cache_sha256": file_sha256(paths["augmentation_cache"]),
                "augmentation_cache_schema_version": P0_PAIR_CACHE_SCHEMA_VERSION,
                "augmentation_cache_complete": True,
                "frozen_p0_checkpoint_path": str(paths["frozen_p0_checkpoint"]),
                "frozen_p0_checkpoint_sha256": file_sha256(
                    paths["frozen_p0_checkpoint"]
                ),
                "p0_frozen": True,
                "p0_checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                "p0_model_type": "evidence-carrier-extent-v0.1",
                "p0_global_step": 1000,
                "p0_checkpoint_contains_validation_metrics": True,
                "p0_checkpoint_selected_by_validation": False,
                "upstream_p0_validation_read": True,
                "upstream_p0_checkpoint_selected_by_validation": False,
                "anchor_feature_dim": 128,
                "pairing_profile": "training_wide",
                "pairing_config": asdict(
                    conditional_pair_pairing_config("training_wide")
                ),
                "train_cal_gate": {},
                "checkpoint_selection": {},
                "uses_internal_test": False,
                "validation_read": False,
            }
            checkpoint_path = root / "best.pt"
            save_p0_pair_scorer_checkpoint(checkpoint_path, payload)
            for name, source in paths.items():
                with self.subTest(name=name):
                    source.write_bytes(originals[name] + b"-tampered")
                    with self.assertRaisesRegex(ValueError, f"{name}_sha256"):
                        load_p0_pair_scorer_checkpoint(checkpoint_path)
                    source.write_bytes(originals[name])

    def test_cache_paths_reject_internal_test_before_any_load(self) -> None:
        with (
            patch(
                "chemvector.train_evidence_carrier_p0_pair_scorer."
                "load_train_only_cache"
            ) as base_loader,
            patch(
                "chemvector.train_evidence_carrier_p0_pair_scorer."
                "load_evidence_carrier_p0_pair_cache"
            ) as augmentation_loader,
        ):
            with self.assertRaisesRegex(ValueError, "internal_test"):
                load_train_only_p0_pair_caches(
                    "/tmp/internal_test/base.pt", "/tmp/augmentation.pt"
                )
        base_loader.assert_not_called()
        augmentation_loader.assert_not_called()

    def test_checkpoint_path_rejects_internal_test_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal_test" / "best.pt"
            with self.assertRaisesRegex(ValueError, "internal_test"):
                load_p0_pair_scorer_checkpoint(path)

    def test_frozen_p0_provenance_cross_checks_explicit_step_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-001000.pt"
            torch.save(
                {
                    "checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                    "model_type": "evidence-carrier-extent-v0.1",
                    "global_step": 1000,
                    "validation": {
                        "schema_version": P0_EXTENT_VALIDATION_SCHEMA_VERSION,
                        "loss": 1.0,
                    },
                    "uses_internal_test": False,
                },
                path,
            )
            from chemvector.train_evidence_carrier_p0_pair_scorer import file_sha256

            provenance = verify_frozen_p0_provenance(
                {
                    "p0_checkpoint_path": str(path),
                    "p0_checkpoint_sha256": file_sha256(path),
                    "p0_checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                    "p0_model_type": "evidence-carrier-extent-v0.1",
                    "p0_global_step": 1000,
                    "p0_checkpoint_contains_validation_metrics": True,
                    "p0_validation_metrics_schema_version": (
                        P0_EXTENT_VALIDATION_SCHEMA_VERSION
                    ),
                    "p0_checkpoint_selected_by_validation": False,
                }
            )

        self.assertTrue(provenance["upstream_validation_read"])
        self.assertTrue(provenance["contains_validation_metrics"])
        self.assertFalse(provenance["checkpoint_selected_by_validation"])

    def test_frozen_p0_provenance_rejects_sidecar_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-001000.pt"
            torch.save(
                {
                    "checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                    "model_type": "evidence-carrier-extent-v0.1",
                    "global_step": 1000,
                    "validation": {
                        "schema_version": P0_EXTENT_VALIDATION_SCHEMA_VERSION
                    },
                    "uses_internal_test": False,
                },
                path,
            )
            from chemvector.train_evidence_carrier_p0_pair_scorer import file_sha256

            base = {
                "p0_checkpoint_path": str(path),
                "p0_checkpoint_sha256": file_sha256(path),
                "p0_checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
                "p0_model_type": "evidence-carrier-extent-v0.1",
                "p0_global_step": 1000,
                "p0_checkpoint_contains_validation_metrics": True,
                "p0_validation_metrics_schema_version": (
                    P0_EXTENT_VALIDATION_SCHEMA_VERSION
                ),
                "p0_checkpoint_selected_by_validation": False,
            }
            mutations = {
                "selected": {"p0_checkpoint_selected_by_validation": True},
                "contains": {"p0_checkpoint_contains_validation_metrics": False},
                "step": {"p0_global_step": 999},
                "model": {"p0_model_type": "different-model"},
                "sha": {"p0_checkpoint_sha256": "0" * 64},
            }
            for name, mutation in mutations.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verify_frozen_p0_provenance({**base, **mutation})


if __name__ == "__main__":
    unittest.main()
