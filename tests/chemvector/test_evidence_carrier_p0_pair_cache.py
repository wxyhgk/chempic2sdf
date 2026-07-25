from __future__ import annotations

from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from chemvector.evidence_carrier_p0_pair_cache import (
    CACHE_FORMAT,
    CACHE_SCHEMA_VERSION,
    DEFAULT_OUTPUT,
    DEFAULT_P0_CHECKPOINT,
    EvidenceCarrierP0PairCacheValidationError,
    _augment_batch,
    _bucket_partition_metadata,
    build_evidence_carrier_p0_pair_cache,
    joined_p0_pair_records,
    load_evidence_carrier_p0_pair_cache,
    validate_evidence_carrier_p0_pair_cache,
)
from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalPairSampleIdentity,
)
from chemvector.evidence_carrier_extent import EvidenceCarrierExtentConfig
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_DIM,
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    EvidenceCarrierP0PairScorerConfig,
)
from chemvector.train_evidence_carrier_extent import (
    CHECKPOINT_SCHEMA_VERSION as P0_CHECKPOINT_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION as P0_VALIDATION_SCHEMA_VERSION,
    file_sha256,
)


def base_record(key: str, partition: str) -> dict:
    return {
        "identity": {
            "sample_key": key,
            "molecule_id": f"mol-{key}",
            "source_structure_id": f"pilot|train|source-{key}",
            "source_pilot": "pilot",
            "source_split": "train",
            "source_record": f"source-{key}",
            "render_variant": "original",
        },
        "partition": partition,
        "canvas_size": torch.tensor([768, 768]),
        "anchor_segments": torch.tensor(
            [[[0.1, 0.5], [0.9, 0.5]], [[0.1, 0.51], [0.9, 0.51]]]
        ),
        "anchor_segments_pixels": torch.tensor(
            [[[76.7, 383.5], [690.3, 383.5]], [[76.7, 391.2], [690.3, 391.2]]]
        ),
        "anchor_source_indices": torch.tensor([0, 1]),
        "anchor_target_line_ids": torch.tensor([0, 0]),
        "anchor_group_ids": torch.tensor([0, 0]),
        "anchor_group_types": torch.tensor([1, 1]),
        "pair_indices": torch.tensor([[0, 1]]),
        "pair_features": torch.zeros((1, 8)),
        "pair_labels": torch.tensor([1]),
        "pair_target_line_ids": torch.tensor([[0, 0]]),
        "pair_overlap_centerline": torch.tensor([[[0.1, 0.505], [0.9, 0.505]]]),
        "pair_overlap_centerline_pixels": torch.tensor(
            [[[76.7, 387.35], [690.3, 387.35]]]
        ),
    }


def augmentation_record(base: dict) -> dict:
    from chemvector.evidence_carrier_p0_pair_cache import _tensor_sha256

    return {
        "identity": dict(base["identity"]),
        "partition": base["partition"],
        "anchor_segments_sha256": _tensor_sha256(base["anchor_segments"]),
        "pair_indices_sha256": _tensor_sha256(base["pair_indices"]),
        "anchor_features": torch.zeros((2, 128), dtype=torch.float32),
        "decoded_segments_pixels": base["anchor_segments_pixels"].float().clone(),
        "pair_consistency_features": torch.zeros(
            (1, P0_PAIR_CONSISTENCY_FEATURE_DIM), dtype=torch.float32
        ),
    }


def caches(
    *,
    complete: bool = True,
    p0_checkpoint_path: str = "/tmp/p0/step-001000.pt",
    p0_checkpoint_sha256: str = "b" * 64,
) -> tuple[dict, dict]:
    fit = base_record("fit", "fit")
    calibration = base_record("cal", "calibration")
    base = {
        "fit_records": [fit],
        "calibration_records": [calibration],
        "cache_path": "/tmp/base.pt",
        "cache_sha256": "a" * 64,
    }
    augmentation = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "scope": "train-only",
        "base_cache_path": "/tmp/base.pt",
        "base_cache_sha256": "a" * 64,
        "p0_checkpoint_path": p0_checkpoint_path,
        "p0_checkpoint_sha256": p0_checkpoint_sha256,
        "p0_checkpoint_schema_version": P0_CHECKPOINT_SCHEMA_VERSION,
        "p0_model_type": "evidence-carrier-extent-v0.1",
        "p0_global_step": 1000,
        "p0_model_config": asdict(EvidenceCarrierExtentConfig()),
        "p0_target_config": asdict(EvidenceCarrierExtentTargetConfig()),
        "p0_loss_protocol": {
            "keep_weight": 0.0,
            "normal_offset_weight": 0.25,
            "extent_weight": 1.0,
            "equivalence_weight": "target_line_id_balanced",
        },
        "p0_raster_source": "full_png",
        "p0_oracle_bond_layer": False,
        "p0_parameter_count": 337620,
        "p0_trainable_parameter_count": 337491,
        "p0_trainer_sha256": "c" * 64,
        "p0_checkpoint_contains_validation_metrics": True,
        "p0_validation_metrics_schema_version": P0_VALIDATION_SCHEMA_VERSION,
        "p0_checkpoint_selected_by_validation": False,
        "p0_checkpoint_selection": {
            "method": "explicit-step-filename",
            "filename": "step-001000.pt",
            "filename_global_step": 1000,
            "uses_validation_metrics": False,
        },
        "p0_frozen": True,
        "anchor_feature_dim": 128,
        "anchor_feature_dtype": "float32",
        "decoded_segment_dtype": "float32",
        "pair_consistency_dtype": "float32",
        "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
        "pair_consistency_config": asdict(EvidenceCarrierP0PairScorerConfig()),
        "p0_extraction_provenance": {
            "device": "cpu",
            "device_type": "cpu",
            "compute_dtype": "float32",
            "autocast_enabled": False,
            "inference_mode_enabled": True,
        },
        "record_count": 2,
        "complete": complete,
        "requested_limit": None if complete else 2,
        "records": {
            "fit": [augmentation_record(fit)],
            "calibration": [augmentation_record(calibration)],
        },
        "bucket_partition": {},
        "uses_validation": False,
        "sidecar_validation_data_read": False,
        "uses_internal_test": False,
    }
    augmentation["bucket_partition"] = _bucket_partition_metadata(
        augmentation["records"]
    )
    return base, augmentation


class EvidenceCarrierP0PairCacheTest(unittest.TestCase):
    def test_defaults_use_fixed_step_and_new_fp32_output(self) -> None:
        self.assertTrue(DEFAULT_P0_CHECKPOINT.endswith("/step-001000.pt"))
        self.assertIn("step1000-fp32", DEFAULT_OUTPUT)
        self.assertEqual(
            CACHE_SCHEMA_VERSION,
            "evidence-carrier-p0-pair-augmentation-cache-0.2",
        )

    def test_build_rejects_internal_test_checkpoint_before_other_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                build_evidence_carrier_p0_pair_cache(
                    Path(directory) / "missing-base.pt",
                    Path(directory) / "internal_test" / "best.pt",
                    Path(directory) / "output.pt",
                    device=torch.device("cpu"),
                    limit=1,
                )

    def test_build_rejects_nonexplicit_checkpoint_alias_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "explicit step-NNNNNN"):
                build_evidence_carrier_p0_pair_cache(
                    Path(directory) / "missing-base.pt",
                    Path(directory) / "best.pt",
                    Path(directory) / "output.pt",
                    device=torch.device("cpu"),
                    limit=1,
                )

    def test_validation_binds_records_to_base_geometry_and_partition(self) -> None:
        base, augmentation = caches()
        validated = validate_evidence_carrier_p0_pair_cache(
            augmentation, base_cache=base
        )
        self.assertEqual(validated["record_count"], 2)

        broken = {**augmentation, "records": {**augmentation["records"]}}
        broken["records"]["fit"] = [
            {**augmentation["records"]["fit"][0], "pair_indices_sha256": "c" * 64}
        ]
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "geometry hashes"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

        wrong_partition = {
            **augmentation,
            "records": {
                "fit": augmentation["records"]["calibration"],
                "calibration": augmentation["records"]["fit"],
            },
        }
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "wrong partition"
        ):
            validate_evidence_carrier_p0_pair_cache(
                wrong_partition, base_cache=None
            )

    def test_join_retains_base_fields_and_adds_independent_p0_fields(self) -> None:
        base, augmentation = caches()
        joined = joined_p0_pair_records(base, augmentation, "fit")
        self.assertEqual(len(joined), 1)
        self.assertIn("pair_features", joined[0])
        self.assertIn("anchor_features", joined[0])
        self.assertIn("decoded_segments_pixels", joined[0])
        self.assertIn("pair_consistency_features", joined[0])
        self.assertEqual(joined[0]["anchor_features"].dtype, torch.float32)

    def test_augment_batch_disables_autocast_and_retains_float32(self) -> None:
        base = base_record("fit", "fit")
        identity = ConditionalPairSampleIdentity(**base["identity"])
        sample = {
            "identity": identity,
            "image": Image.new("RGB", (768, 768), "white"),
        }

        def fake_extract(
            _model,
            pixels,
            anchors,
            anchor_mask,
            *,
            coordinate_mode,
        ):
            self.assertEqual(pixels.dtype, torch.float32)
            self.assertEqual(anchors.dtype, torch.float32)
            self.assertEqual(coordinate_mode, "normalized")
            scale = anchors.new_tensor((767.0, 767.0))
            return {
                "anchor_features": torch.ones(
                    (*anchors.shape[:2], 128),
                    dtype=torch.float32,
                    device=anchors.device,
                ),
                "decoded_segments_pixels": anchors * scale,
                "anchor_mask": anchor_mask,
            }

        real_autocast = torch.autocast
        with (
            patch(
                "chemvector.evidence_carrier_p0_pair_cache."
                "extract_frozen_p0_anchor_evidence",
                side_effect=fake_extract,
            ),
            patch(
                "chemvector.evidence_carrier_p0_pair_cache.torch.autocast",
                wraps=real_autocast,
            ) as autocast,
        ):
            result = _augment_batch(
                [sample],
                [base],
                p0_model=torch.nn.Identity(),
                device=torch.device("cpu"),
                consistency_config=EvidenceCarrierP0PairScorerConfig(),
            )

        autocast.assert_called_once_with(device_type="cpu", enabled=False)
        self.assertEqual(result[0]["anchor_features"].dtype, torch.float32)
        self.assertEqual(result[0]["decoded_segments_pixels"].dtype, torch.float32)
        self.assertEqual(result[0]["pair_consistency_features"].dtype, torch.float32)

    def test_training_loader_rejects_incomplete_smoke_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step-001000.pt"
            torch.save({"fixed_step": 1000}, checkpoint)
            base, augmentation = caches(
                complete=False,
                p0_checkpoint_path=str(checkpoint),
                p0_checkpoint_sha256=file_sha256(checkpoint),
            )
            path = Path(directory) / "smoke.pt"
            torch.save(augmentation, path)
            loaded = load_evidence_carrier_p0_pair_cache(
                path, base_cache=base, require_complete=False
            )
            self.assertFalse(loaded["complete"])
            with self.assertRaisesRegex(ValueError, "complete"):
                load_evidence_carrier_p0_pair_cache(
                    path, base_cache=base, require_complete=True
                )

    def test_default_loader_rehashes_frozen_p0_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step-001000.pt"
            torch.save({"fixed_step": 1000}, checkpoint)
            base, augmentation = caches(
                p0_checkpoint_path=str(checkpoint),
                p0_checkpoint_sha256=file_sha256(checkpoint),
            )
            path = Path(directory) / "sidecar.pt"
            torch.save(augmentation, path)
            load_evidence_carrier_p0_pair_cache(path, base_cache=base)

            torch.save({"fixed_step": 1000, "tampered": True}, checkpoint)
            with self.assertRaisesRegex(
                EvidenceCarrierP0PairCacheValidationError, "SHA differs"
            ):
                load_evidence_carrier_p0_pair_cache(path, base_cache=base)

    def test_validation_requires_complete_consistency_and_p0_provenance(self) -> None:
        base, augmentation = caches()
        broken = dict(augmentation)
        broken["pair_consistency_config"] = dict(
            augmentation["pair_consistency_config"]
        )
        broken["pair_consistency_config"].pop("coordinate_max")
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "config.*incomplete"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

        broken = dict(augmentation)
        broken.pop("p0_model_config")
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "model/target config"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

        broken = dict(augmentation)
        broken["p0_checkpoint_selected_by_validation"] = True
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "explicit step"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

    def test_validation_rejects_nonexplicit_or_mismatched_step_provenance(self) -> None:
        base, augmentation = caches()
        broken = dict(augmentation)
        broken["p0_checkpoint_path"] = "/tmp/p0/best.pt"
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "explicit step-NNNNNN"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

        broken = dict(augmentation)
        broken["p0_global_step"] = 900
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "filename and global_step"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)

    def test_validation_enforces_bucket_identity_and_selection_closure(self) -> None:
        _, augmentation = caches()
        crossed = dict(augmentation)
        crossed["records"] = {
            "fit": list(augmentation["records"]["fit"]),
            "calibration": [dict(augmentation["records"]["calibration"][0])],
        }
        crossed["records"]["calibration"][0]["identity"] = dict(
            crossed["records"]["calibration"][0]["identity"]
        )
        crossed["records"]["calibration"][0]["identity"]["molecule_id"] = (
            augmentation["records"]["fit"][0]["identity"]["molecule_id"]
        )
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "not closed over molecule"
        ):
            validate_evidence_carrier_p0_pair_cache(crossed, base_cache=None)

        base, augmentation = caches()
        wrong_selection = dict(augmentation)
        wrong_selection["complete"] = False
        wrong_selection["requested_limit"] = 1
        wrong_selection["records"] = {
            "fit": list(augmentation["records"]["fit"]),
            "calibration": [],
        }
        wrong_selection["record_count"] = 1
        wrong_selection["bucket_partition"] = _bucket_partition_metadata(
            wrong_selection["records"]
        )
        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "selection is not closed"
        ):
            validate_evidence_carrier_p0_pair_cache(
                wrong_selection, base_cache=base
            )

    def test_validation_rejects_negative_consistency_measurements(self) -> None:
        base, augmentation = caches()
        broken = {**augmentation, "records": {**augmentation["records"]}}
        current = dict(augmentation["records"]["fit"][0])
        current["pair_consistency_features"] = current[
            "pair_consistency_features"
        ].clone()
        current["pair_consistency_features"][0, 0] = -1.0
        broken["records"]["fit"] = [current]

        with self.assertRaisesRegex(
            EvidenceCarrierP0PairCacheValidationError, "nonnegative"
        ):
            validate_evidence_carrier_p0_pair_cache(broken, base_cache=base)


if __name__ == "__main__":
    unittest.main()
