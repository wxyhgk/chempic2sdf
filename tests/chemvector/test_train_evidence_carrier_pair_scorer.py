from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest

import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_cache import CACHE_SCHEMA_VERSION
from chemvector.evidence_carrier_pair_scorer import EvidenceCarrierPairScorer
from chemvector.train_evidence_carrier_pair_scorer import (
    CHECKPOINT_SCHEMA_VERSION,
    collate_pair_training_records,
    evaluate_pair_scorer_loss,
    fit_pair_feature_normalization,
    load_pair_scorer_checkpoint,
    load_train_only_cache,
    save_pair_scorer_checkpoint,
    train_pair_scorer_epoch,
)


def record(sample_key: str = "sample-0") -> dict:
    return {
        "sample_key": sample_key,
        "identity": {"molecule_id": f"mol-{sample_key}"},
        "pair_features": torch.tensor(
            [
                [0.2, 2.0, 40.0, 0.95, 0.90, 0.95, 0.80, 0.90],
                [6.0, 7.0, 8.0, 0.40, 0.20, 0.30, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
        "pair_indices": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "pair_labels": torch.tensor([1, 0], dtype=torch.long),
    }


class PairScorerTrainerTest(unittest.TestCase):
    def test_collator_pads_variable_pair_rows(self) -> None:
        second = record("sample-1")
        second["pair_features"] = second["pair_features"][:1]
        second["pair_indices"] = second["pair_indices"][:1]
        second["pair_labels"] = second["pair_labels"][:1]

        batch = collate_pair_training_records([record(), second])

        self.assertEqual(batch["pair_features"].shape, (2, 2, 8))
        self.assertEqual(batch["pair_count"].tolist(), [2, 1])
        self.assertEqual(batch["pair_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["pair_labels"][1].tolist(), [1, -1])

    def test_one_train_epoch_updates_the_833_parameter_model(self) -> None:
        records = [record(f"sample-{index}") for index in range(4)]
        model = EvidenceCarrierPairScorer()
        fit_pair_feature_normalization(model, records)
        before = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        metrics = train_pair_scorer_epoch(
            model,
            optimizer,
            records,
            batch_size=2,
            device=torch.device("cpu"),
            rng=random.Random(7),
            max_grad_norm=5.0,
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
        model = EvidenceCarrierPairScorer()
        fit_pair_feature_normalization(model, records)

        individual = evaluate_pair_scorer_loss(
            model,
            records,
            batch_size=1,
            device=torch.device("cpu"),
        )
        combined = evaluate_pair_scorer_loss(
            model,
            records,
            batch_size=2,
            device=torch.device("cpu"),
        )

        for name in ("loss", "balanced_bce", "ranking_loss"):
            self.assertAlmostEqual(individual[name], combined[name], places=6)

    def test_checkpoint_roundtrip_preserves_threshold_and_normalization(self) -> None:
        model = EvidenceCarrierPairScorer()
        normalization = fit_pair_feature_normalization(
            model, [record(), record("sample-1")]
        )
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "model_config": model.config_dict(),
            "parameter_count": model.parameter_count(),
            "feature_transform_config": {
                "maximum_candidate_angle_degrees": 8.0,
                "maximum_candidate_normal_separation_pixels": 8.0,
                "reference_image_size_pixels": 768,
                "minimum_normalization_scale": 1e-6,
            },
            "pair_feature_names": [
                "angle_degrees",
                "normal_separation_pixels",
                "overlap_pixels",
                "shorter_interval_overlap",
                "interval_iou",
                "interior_ink_occupancy",
                "interior_mean_darkness",
                "center_mean_darkness",
            ],
            "gain_threshold": 0.75,
            "cache_path": "/tmp/train-cache.pt",
            "cache_sha256": "0" * 64,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "pairing_profile": "training_wide",
            "pairing_config": asdict(
                conditional_pair_pairing_config("training_wide")
            ),
            "normalization": normalization,
            "train_cal_gate": {"status": "GO"},
            "checkpoint_selection": {"candidate_count": 1},
            "uses_internal_test": False,
            "validation_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_pair_scorer_checkpoint(path, payload)
            loaded, checkpoint = load_pair_scorer_checkpoint(path)

        self.assertEqual(checkpoint["gain_threshold"], 0.75)
        self.assertTrue(bool(loaded.normalization_fitted))
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    def test_train_cache_rejects_internal_test_path_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal_test_cache.pt"
            with self.assertRaisesRegex(ValueError, "internal_test"):
                load_train_only_cache(path)


if __name__ == "__main__":
    unittest.main()
