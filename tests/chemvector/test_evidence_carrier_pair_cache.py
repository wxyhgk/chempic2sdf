from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_NAMES,
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalPairSampleIdentity,
)
from chemvector.evidence_carrier_pair_cache import (
    CACHE_SCHEMA_VERSION,
    EvidenceCarrierPairCacheIntegrityError,
    EvidenceCarrierPairCacheValidationError,
    build_evidence_carrier_pair_train_cache,
    collate_evidence_carrier_pair_records,
    compact_evidence_carrier_pair_record,
    expand_evidence_carrier_pair_train_shards,
    load_evidence_carrier_pair_cache,
    scan_evidence_carrier_pair_train_identities,
    validate_evidence_carrier_pair_record,
)


WIDTH = HEIGHT = 100


def identity(
    key: str,
    molecule: str,
    source: str,
    *,
    variant: str = "original",
    split: str = "train",
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


def pair_target(*, with_pair: bool = True) -> dict[str, torch.Tensor]:
    anchor_pixels = torch.tensor(
        [
            [[10.0, 49.0], [90.0, 49.0]],
            [[10.0, 51.0], [90.0, 51.0]],
        ],
        dtype=torch.float32,
    )
    anchor_segments = anchor_pixels / torch.tensor(
        [WIDTH - 1, HEIGHT - 1], dtype=torch.float32
    )
    if with_pair:
        pair_indices = torch.tensor([[0, 1]], dtype=torch.long)
        pair_features = torch.tensor(
            [[0.0, 2.0, 80.0, 1.0, 1.0, 0.9, 0.8, 0.95]],
            dtype=torch.float32,
        )
        pair_labels = torch.tensor([1], dtype=torch.long)
        pair_targets = torch.tensor([[7, 7]], dtype=torch.long)
        centerline_pixels = torch.tensor(
            [[[10.0, 50.0], [90.0, 50.0]]], dtype=torch.float32
        )
    else:
        anchor_pixels = anchor_pixels[:1]
        anchor_segments = anchor_segments[:1]
        pair_indices = torch.empty((0, 2), dtype=torch.long)
        pair_features = torch.empty((0, 8), dtype=torch.float32)
        pair_labels = torch.empty((0,), dtype=torch.long)
        pair_targets = torch.empty((0, 2), dtype=torch.long)
        centerline_pixels = torch.empty((0, 2, 2), dtype=torch.float32)
    centerline = centerline_pixels / torch.tensor(
        [WIDTH - 1, HEIGHT - 1], dtype=torch.float32
    )
    return {
        "anchor_segments": anchor_segments,
        "anchor_segments_pixels": anchor_pixels,
        "anchor_source_indices": torch.tensor(
            [1, 3] if with_pair else [1], dtype=torch.long
        ),
        "anchor_target_line_ids": torch.tensor(
            [7, 7] if with_pair else [7], dtype=torch.long
        ),
        "pair_indices": pair_indices,
        "pair_features": pair_features,
        "pair_labels": pair_labels,
        "pair_target_line_ids": pair_targets,
        "pair_overlap_centerline": centerline,
        "pair_overlap_centerline_pixels": centerline_pixels,
        "raw_group_ids": torch.tensor([2, 11, 3, 11], dtype=torch.long),
        "raw_group_types": torch.tensor([1, 4, 1, 4], dtype=torch.long),
    }


def compact_record(
    key: str = "sample",
    *,
    with_pair: bool = True,
    partition: str = "fit",
) -> dict:
    return compact_evidence_carrier_pair_record(
        identity(key, f"mol-{key}", f"source-{key}"),
        pair_target(with_pair=with_pair),
        partition=partition,
        canvas_size=(WIDTH, HEIGHT),
    )


def build_fixture_identities() -> tuple[ConditionalPairSampleIdentity, ...]:
    return (
        identity("a-original", "mol-a", "source-a", variant="original"),
        identity("a-stress", "mol-a", "source-a", variant="stress"),
        identity("b-original", "mol-b", "source-b"),
        identity("c-original", "mol-c", "source-c"),
    )


def build_fixture_samples(
    identities: tuple[ConditionalPairSampleIdentity, ...],
) -> list[dict]:
    return [
        {
            "identity": current,
            "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
            "primitives": {},
            "layers": {},
        }
        for current in identities
    ]


class EvidenceCarrierPairRecordTest(unittest.TestCase):
    def test_compact_record_derives_anchor_group_fields_from_raw_indices(self) -> None:
        record = compact_record()

        self.assertEqual(record["anchor_source_indices"].tolist(), [1, 3])
        self.assertEqual(record["anchor_group_ids"].tolist(), [11, 11])
        self.assertEqual(record["anchor_group_types"].tolist(), [4, 4])
        self.assertEqual(record["pair_indices"].tolist(), [[0, 1]])
        self.assertEqual(record["pair_labels"].tolist(), [1])
        torch.testing.assert_close(
            record["pair_overlap_centerline_pixels"][0],
            torch.tensor([[10.0, 50.0], [90.0, 50.0]]),
        )
        validate_evidence_carrier_pair_record(record)

    def test_record_validation_rejects_target_label_inconsistency(self) -> None:
        record = compact_record()
        record["pair_labels"] = torch.tensor([0], dtype=torch.long)

        with self.assertRaisesRegex(
            EvidenceCarrierPairCacheValidationError,
            "labels disagree",
        ):
            validate_evidence_carrier_pair_record(record)

    def test_record_validation_rejects_duplicate_pair_rows(self) -> None:
        record = compact_record()
        for name in (
            "pair_indices",
            "pair_features",
            "pair_labels",
            "pair_target_line_ids",
            "pair_overlap_centerline",
            "pair_overlap_centerline_pixels",
        ):
            record[name] = torch.cat((record[name], record[name]), dim=0)

        with self.assertRaisesRegex(
            EvidenceCarrierPairCacheValidationError,
            "duplicate rows",
        ):
            validate_evidence_carrier_pair_record(record)

    def test_record_validation_enforces_pair_feature_domain(self) -> None:
        for column in range(3):
            record = compact_record()
            record["pair_features"][0, column] = -0.01
            with self.assertRaisesRegex(
                EvidenceCarrierPairCacheValidationError,
                "must be nonnegative",
            ):
                validate_evidence_carrier_pair_record(record)

        for column in range(3, 8):
            for invalid in (-2e-5, 1.0 + 2e-5):
                record = compact_record()
                record["pair_features"][0, column] = invalid
                with self.assertRaisesRegex(
                    EvidenceCarrierPairCacheValidationError,
                    r"must lie in \[0,1\]",
                ):
                    validate_evidence_carrier_pair_record(record)

    def test_record_validation_uses_scorer_fraction_tolerance(self) -> None:
        record = compact_record()
        record["pair_features"][0, 3:] = torch.tensor(
            (-5e-6, 1.0 + 5e-6, 0.0, 1.0, 0.5),
            dtype=torch.float32,
        )

        validate_evidence_carrier_pair_record(record)

    def test_collator_pads_anchor_pair_and_group_fields(self) -> None:
        first = compact_record("first")
        second = compact_record("second", with_pair=False, partition="calibration")

        batch = collate_evidence_carrier_pair_records((first, second))

        self.assertEqual(batch["cache_schema_version"], CACHE_SCHEMA_VERSION)
        self.assertEqual(batch["pair_feature_names"], PAIR_FEATURE_NAMES)
        self.assertEqual(batch["anchor_count"].tolist(), [2, 1])
        self.assertEqual(batch["pair_count"].tolist(), [1, 0])
        self.assertEqual(batch["anchor_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["pair_mask"].tolist(), [[True], [False]])
        self.assertEqual(batch["pair_labels"].tolist(), [[1], [-1]])
        self.assertEqual(batch["anchor_group_ids"][1].tolist(), [11, -1])
        self.assertEqual(batch["pair_features"].shape, (2, 1, 8))


class EvidenceCarrierPairCacheBuildTest(unittest.TestCase):
    def _build(self, directory: str):
        root = Path(directory)
        shard = root / "train-000000.tar"
        shard.write_bytes(b"synthetic-train-shard")
        output = root / "pair-cache.pt"
        identities = build_fixture_identities()
        samples = build_fixture_samples(identities)
        with (
            patch(
                "chemvector.evidence_carrier_pair_cache."
                "iter_conditional_pair_identities",
                return_value=iter(identities),
            ),
            patch(
                "chemvector.evidence_carrier_pair_cache."
                "ConditionalOrdinaryPairWebDataset",
                return_value=samples,
            ),
            patch(
                "chemvector.evidence_carrier_pair_cache."
                "build_conditional_ordinary_pair_target",
                side_effect=[pair_target() for _ in identities],
            ) as target_builder,
        ):
            payload = build_evidence_carrier_pair_train_cache(
                [str(shard)],
                output,
                calibration_fraction=0.5,
                partition_seed="fixed",
                progress_every=None,
            )
        for call in target_builder.call_args_list:
            pairing_config = call.kwargs["pairing_config"]
            self.assertEqual(pairing_config.maximum_angle_degrees, 8.0)
            self.assertEqual(pairing_config.maximum_normal_separation_pixels, 8.0)
            self.assertEqual(pairing_config.minimum_overlap_pixels, 4.0)
        return shard, output, identities, payload

    def test_build_load_is_atomic_train_only_and_keeps_variants_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard, output, identities, payload = self._build(directory)

            self.assertTrue(output.is_file())
            self.assertEqual(list(Path(directory).glob(".pair-cache.pt.tmp-*")), [])
            self.assertEqual(payload["pairing_profile"], PAIRING_PROFILE_TRAINING_WIDE)
            self.assertEqual(payload["config"]["render_mode"], "all_variants")
            self.assertEqual(payload["config"]["source_split"], "train")
            self.assertEqual(payload["pair_feature_names"], list(PAIR_FEATURE_NAMES))
            self.assertEqual(payload["record_count"], len(identities))
            self.assertEqual(len(payload["shard_sha256"][str(shard.resolve())]), 64)

            side_by_key = {
                record["identity"]["sample_key"]: side
                for side, records in (
                    ("fit", payload["fit_records"]),
                    ("calibration", payload["calibration_records"]),
                )
                for record in records
            }
            self.assertEqual(side_by_key["a-original"], side_by_key["a-stress"])
            loaded = load_evidence_carrier_pair_cache(output)
            self.assertEqual(
                loaded["configuration_sha256"],
                payload["configuration_sha256"],
            )
            self.assertFalse(loaded["uses_validation"])
            self.assertFalse(loaded["uses_internal_test"])

    def test_source_shard_tamper_can_be_explicitly_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard, output, _, _ = self._build(directory)
            shard.write_bytes(b"changed")

            with self.assertRaisesRegex(
                EvidenceCarrierPairCacheIntegrityError,
                "size changed|SHA256 mismatch",
            ):
                load_evidence_carrier_pair_cache(output)
            loaded = load_evidence_carrier_pair_cache(
                output,
                validate_source_hashes=False,
            )
            self.assertEqual(loaded["scope"], "train-only")

    def test_load_rejects_schema_and_cross_partition_variant_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, output, _, _ = self._build(directory)
            payload = torch.load(output, map_location="cpu", weights_only=False)
            payload["pair_feature_names"][0] = "changed"
            changed = Path(directory) / "changed.pt"
            torch.save(payload, changed)
            with self.assertRaisesRegex(
                EvidenceCarrierPairCacheValidationError,
                "feature names",
            ):
                load_evidence_carrier_pair_cache(
                    changed,
                    validate_source_hashes=False,
                )

            payload = torch.load(output, map_location="cpu", weights_only=False)
            payload["uses_validation"] = True
            validation_flag = Path(directory) / "validation-flag.pt"
            torch.save(payload, validation_flag)
            with self.assertRaisesRegex(
                EvidenceCarrierPairCacheValidationError,
                "uses_validation=False",
            ):
                load_evidence_carrier_pair_cache(
                    validation_flag,
                    validate_source_hashes=False,
                )

            payload = torch.load(output, map_location="cpu", weights_only=False)
            fit = payload["fit_records"]
            calibration = payload["calibration_records"]
            source_side = fit if any(
                record["identity"]["sample_key"] == "a-original" for record in fit
            ) else calibration
            target_side = calibration if source_side is fit else fit
            moved_index = next(
                index
                for index, record in enumerate(source_side)
                if record["identity"]["sample_key"] == "a-stress"
            )
            moved = source_side.pop(moved_index)
            moved["partition"] = "calibration" if target_side is calibration else "fit"
            target_side.append(moved)
            leaked = Path(directory) / "leaked.pt"
            torch.save(payload, leaked)
            with self.assertRaisesRegex(
                EvidenceCarrierPairCacheValidationError,
                "variants cross|leakage",
            ):
                load_evidence_carrier_pair_cache(
                    leaked,
                    validate_source_hashes=False,
                )

    def test_atomic_write_failure_never_publishes_partial_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "train-000000.tar"
            shard.write_bytes(b"synthetic")
            output = root / "cache.pt"
            identities = build_fixture_identities()
            samples = build_fixture_samples(identities)
            with (
                patch(
                    "chemvector.evidence_carrier_pair_cache."
                    "iter_conditional_pair_identities",
                    return_value=iter(identities),
                ),
                patch(
                    "chemvector.evidence_carrier_pair_cache."
                    "ConditionalOrdinaryPairWebDataset",
                    return_value=samples,
                ),
                patch(
                    "chemvector.evidence_carrier_pair_cache."
                    "build_conditional_ordinary_pair_target",
                    side_effect=[pair_target() for _ in identities],
                ),
                patch(
                    "chemvector.evidence_carrier_pair_cache.torch.save",
                    side_effect=OSError("write failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "write failed"):
                    build_evidence_carrier_pair_train_cache(
                        str(shard),
                        output,
                        calibration_fraction=0.5,
                        progress_every=None,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".cache.pt.tmp-*")), [])


class EvidenceCarrierPairCacheBoundaryTest(unittest.TestCase):
    def test_validation_internal_test_and_resolved_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation-000000.tar"
            validation.touch()
            internal = root / "internal_test-000000.tar"
            internal.touch()
            alias = root / "train-alias.tar"
            alias.symlink_to(validation)

            for source, message in (
                (validation, "validation"),
                (internal, "internal_test"),
                (alias, "validation"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    expand_evidence_carrier_pair_train_shards(str(source))

    def test_identity_scanner_rejects_non_train_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "train-000000.tar"
            shard.touch()
            validation_identity = identity(
                "validation-sample",
                "mol",
                "source",
                split="validation",
            )
            with patch(
                "chemvector.evidence_carrier_pair_cache."
                "iter_conditional_pair_identities",
                return_value=iter((validation_identity,)),
            ) as scan:
                with self.assertRaisesRegex(ValueError, "non-train"):
                    scan_evidence_carrier_pair_train_identities(str(shard))
            self.assertEqual(scan.call_args.kwargs["split"], "train")
            self.assertIsNone(scan.call_args.kwargs["render_variant"])


if __name__ == "__main__":
    unittest.main()
