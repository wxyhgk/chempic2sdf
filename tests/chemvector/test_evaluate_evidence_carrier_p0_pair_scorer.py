from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from chemvector.bond_group_set import SINGLE_LINE
from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalPairSampleIdentity,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.evidence_carrier_p0_pair_scorer import (
    EvidenceCarrierP0PairScorerConfig,
)
from chemvector.evaluate_evidence_carrier_p0_pair_scorer import (
    VALIDATION_INTERPRETATION,
    build_parser,
    evaluate_frozen_p0_pair_sample,
    load_frozen_p0_pair_pipeline,
    score_frozen_p0_pair_target,
)
from chemvector.train_evidence_carrier_extent import file_sha256


def identity() -> ConditionalPairSampleIdentity:
    return ConditionalPairSampleIdentity(
        sample_key="validation-1",
        molecule_id="validation-molecule",
        source_structure_id="pilot|validation|source-1",
        source_pilot="pilot",
        source_split="validation",
        source_record="source-1",
        render_variant="original",
    )


def pair_target() -> dict[str, torch.Tensor]:
    anchors_pixels = torch.tensor(
        [
            ((10.0, 50.0), (90.0, 50.0)),
            ((10.0, 54.0), (90.0, 54.0)),
        ],
        dtype=torch.float32,
    )
    anchors = anchors_pixels / 99.0
    pair_indices = torch.tensor([[0, 1]], dtype=torch.long)
    target_ids = torch.tensor([4, 4], dtype=torch.long)
    return {
        "anchor_segments": anchors,
        "anchor_segments_pixels": anchors_pixels,
        "anchor_source_indices": torch.tensor([0, 1], dtype=torch.long),
        "anchor_target_line_ids": target_ids,
        "anchor_mask": torch.ones(2, dtype=torch.bool),
        "pair_indices": pair_indices,
        "pair_features": torch.tensor(
            [[3.0, 4.0, 80.0, 1.0, 0.95, 1.0, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "pair_labels": torch.ones(1, dtype=torch.long),
        "pair_target_line_ids": torch.tensor([[4, 4]], dtype=torch.long),
        "pair_overlap_centerline_pixels": torch.tensor(
            [[((10.0, 52.0), (90.0, 52.0))]], dtype=torch.float32
        ).reshape(1, 2, 2),
        "pair_mask": torch.ones(1, dtype=torch.bool),
        "raw_keep_labels": torch.ones(2, dtype=torch.long),
        "raw_target_line_ids": target_ids.clone(),
        "raw_group_ids": torch.tensor([9, 9], dtype=torch.long),
        "raw_group_types": torch.tensor(
            [SINGLE_LINE, SINGLE_LINE], dtype=torch.long
        ),
    }


class FakePairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = EvidenceCarrierP0PairScorerConfig()

    def forward(
        self,
        anchor_features,
        pair_indices,
        pair_features,
        consistency_features,
        pair_mask,
    ):
        self.inputs = {
            "anchor_features": anchor_features,
            "pair_indices": pair_indices,
            "pair_features": pair_features,
            "consistency_features": consistency_features,
            "pair_mask": pair_mask,
        }
        return {
            "pair_gains": torch.where(
                pair_mask, pair_features[..., 0], torch.zeros_like(pair_features[..., 0])
            )
        }


class FakeP0Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_dim=128)


class EvaluateFrozenP0PairScorerTest(unittest.TestCase):
    def test_live_p0_features_and_12d_consistency_feed_b_scorer(self) -> None:
        target = pair_target()
        pair_model = FakePairModel().eval()
        p0_model = FakeP0Model().eval().requires_grad_(False)
        decoded = target["anchor_segments_pixels"].unsqueeze(0)
        evidence = {
            "anchor_features": torch.zeros((1, 2, 128), dtype=torch.float32),
            "decoded_segments_pixels": decoded,
            "anchor_mask": torch.ones((1, 2), dtype=torch.bool),
        }
        with patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
            "extract_frozen_p0_anchor_evidence",
            return_value=evidence,
        ) as extractor:
            result = score_frozen_p0_pair_target(
                pair_model,
                p0_model,
                Image.new("RGB", (100, 100), "white"),
                target,
                device=torch.device("cpu"),
                expected_image_size=100,
            )

        self.assertEqual(result["pair_scores"].tolist(), [3.0])
        self.assertTrue(
            torch.equal(result["decoded_segments_pixels"], decoded[0])
        )
        self.assertEqual(result["p0_forward_passes"], 1)
        self.assertEqual(result["consistency_feature_count"], 12)
        self.assertEqual(
            tuple(pair_model.inputs["consistency_features"].shape), (1, 1, 12)
        )
        self.assertEqual(
            tuple(pair_model.inputs["anchor_features"].shape), (1, 2, 128)
        )
        self.assertEqual(extractor.call_args.kwargs["coordinate_mode"], "normalized")

    def test_sample_builds_target_once_and_uses_supplied_frozen_threshold(self) -> None:
        current_identity = identity()
        sample = {
            "identity": current_identity,
            "image": Image.new("RGB", (100, 100), "white"),
            "primitives": {},
            "layers": {"width": 100, "height": 100},
        }
        target = pair_target()
        with patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
            "build_conditional_ordinary_pair_target",
            return_value=target,
        ) as builder, patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
            "score_frozen_p0_pair_target",
            return_value={
                "pair_scores": torch.tensor([2.0]),
                "anchor_count": 2,
                "pair_count": 1,
                "p0_forward_passes": 1,
                "consistency_feature_count": 12,
            },
        ):
            result = evaluate_frozen_p0_pair_sample(
                FakePairModel(),
                FakeP0Model().requires_grad_(False),
                sample,
                threshold=1.0,
                pairing_config=object(),
                target_config=EvidenceCarrierExtentTargetConfig(),
                device=torch.device("cpu"),
                expected_image_size=100,
            )

        builder.assert_called_once()
        self.assertEqual(result["target_builds"], 1)
        self.assertEqual(result["pair_evaluation"]["selected_same_target_pairs"], 1)
        self.assertEqual(result["pair_evaluation"]["selected_cross_target_pairs"], 0)

    def test_pipeline_checks_all_three_live_sha_links_and_p0_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            base_path = root / "base.pt"
            augmentation_path = root / "augmentation.pt"
            p0_path = root / "p0.pt"
            for path, payload in (
                (checkpoint_path, b"B"),
                (base_path, b"base"),
                (augmentation_path, b"augmentation"),
                (p0_path, b"P0"),
            ):
                path.write_bytes(payload)
            checkpoint = {
                "gain_threshold": 0.75,
                "train_cal_gate": {"status": "GO"},
                "base_cache_path": str(base_path),
                "base_cache_sha256": file_sha256(base_path),
                "augmentation_cache_path": str(augmentation_path),
                "augmentation_cache_sha256": file_sha256(augmentation_path),
                "frozen_p0_checkpoint_path": str(p0_path),
                "frozen_p0_checkpoint_sha256": file_sha256(p0_path),
                "anchor_feature_dim": 128,
                "p0_global_step": 900,
                "p0_model_type": "fake-p0",
                "p0_checkpoint_schema_version": "fake-p0-checkpoint",
                "p0_checkpoint_contains_validation_metrics": True,
                "p0_checkpoint_selected_by_validation": False,
                "upstream_p0_validation_read": True,
                "upstream_p0_checkpoint_selected_by_validation": False,
                "pairing_config": asdict(
                    conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
                ),
            }
            base_cache = {"cache_sha256": file_sha256(base_path)}
            augmentation_cache = {
                "cache_sha256": file_sha256(augmentation_path),
                "base_cache_path": str(base_path),
                "base_cache_sha256": file_sha256(base_path),
                "p0_checkpoint_path": str(p0_path),
                "p0_checkpoint_sha256": file_sha256(p0_path),
                "p0_model_type": "fake-p0",
                "p0_global_step": 900,
                "p0_checkpoint_schema_version": "fake-p0-checkpoint",
                "p0_checkpoint_contains_validation_metrics": True,
                "p0_checkpoint_selected_by_validation": False,
                "complete": True,
                "pair_consistency_config": asdict(
                    EvidenceCarrierP0PairScorerConfig()
                ),
            }
            p0_provenance = {
                "path": str(p0_path.resolve()),
                "sha256": file_sha256(p0_path),
                "model_type": "fake-p0",
                "global_step": 900,
                "upstream_validation_read": True,
                "checkpoint_selected_by_validation": False,
                "uses_internal_test": False,
            }
            pair_model = FakePairModel()
            p0_model = FakeP0Model()
            with patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_p0_pair_scorer_checkpoint",
                return_value=(pair_model, checkpoint),
            ), patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_train_only_cache",
                return_value=base_cache,
            ) as base_loader, patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_evidence_carrier_p0_pair_cache",
                return_value=augmentation_cache,
            ) as augmentation_loader, patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "verify_frozen_p0_provenance",
                return_value=p0_provenance,
            ), patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer.load_checkpoint",
                return_value=(
                    p0_model,
                    EvidenceCarrierExtentTargetConfig(),
                    {
                        "checkpoint_schema_version": "fake-p0-checkpoint",
                        "model_type": "fake-p0",
                        "global_step": 900,
                    },
                ),
            ):
                pipeline = load_frozen_p0_pair_pipeline(
                    checkpoint_path, device=torch.device("cpu")
                )

            self.assertEqual(pipeline["pair_threshold"], 0.75)
            self.assertEqual(pipeline["p0_checkpoint_sha256"], file_sha256(p0_path))
            self.assertTrue(pipeline["p0_provenance"]["upstream_validation_read"])
            self.assertFalse(next(pair_model.parameters(), torch.tensor(0.0)).requires_grad)
            self.assertFalse(p0_model.weight.requires_grad)
            base_loader.assert_called_once_with(base_path.resolve())
            augmentation_loader.assert_called_once_with(
                augmentation_path.resolve(), base_cache=base_cache, require_complete=True
            )

    def test_explicit_development_loader_can_accept_a_no_go_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            path.write_bytes(b"B")
            with patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_p0_pair_scorer_checkpoint",
                return_value=(
                    FakePairModel(),
                    {
                        "gain_threshold": 0.0,
                        "train_cal_gate": {"status": "NO-GO"},
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "lacks base_cache_path"):
                    load_frozen_p0_pair_pipeline(
                        path,
                        device=torch.device("cpu"),
                        require_train_cal_go=False,
                    )

    def test_bad_live_sha_aborts_before_any_cache_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            base_path = root / "base.pt"
            augmentation_path = root / "augmentation.pt"
            p0_path = root / "p0.pt"
            for path in (checkpoint_path, base_path, augmentation_path, p0_path):
                path.write_bytes(path.name.encode("ascii"))
            checkpoint = {
                "gain_threshold": 0.0,
                "train_cal_gate": {"status": "GO"},
                "base_cache_path": str(base_path),
                "base_cache_sha256": file_sha256(base_path),
                "augmentation_cache_path": str(augmentation_path),
                "augmentation_cache_sha256": "0" * 64,
                "frozen_p0_checkpoint_path": str(p0_path),
                "frozen_p0_checkpoint_sha256": file_sha256(p0_path),
            }
            with patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_p0_pair_scorer_checkpoint",
                return_value=(FakePairModel(), checkpoint),
            ), patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_train_only_cache"
            ) as cache_loader:
                with self.assertRaisesRegex(ValueError, "augmentation_cache_sha256"):
                    load_frozen_p0_pair_pipeline(
                        checkpoint_path, device=torch.device("cpu")
                    )
            cache_loader.assert_not_called()

    def test_internal_test_is_rejected_before_checkpoint_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal_test" / "best.pt"
            path.parent.mkdir()
            path.write_bytes(b"sealed")
            with patch(
                "chemvector.evaluate_evidence_carrier_p0_pair_scorer."
                "load_p0_pair_scorer_checkpoint"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "internal_test"):
                    load_frozen_p0_pair_pipeline(path, device=torch.device("cpu"))
            loader.assert_not_called()

    def test_cli_exposes_no_validation_calibration_or_p0_override(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("threshold", destinations)
        self.assertNotIn("gain_threshold", destinations)
        self.assertNotIn("calibration", destinations)
        self.assertNotIn("p0_checkpoint", destinations)
        self.assertEqual(
            VALIDATION_INTERPRETATION, "component-conditioned pipeline regression"
        )


if __name__ == "__main__":
    unittest.main()
