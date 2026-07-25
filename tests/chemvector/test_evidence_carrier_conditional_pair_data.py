from __future__ import annotations

from argparse import Namespace
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_NAMES,
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalOrdinaryPairCollator,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    ConditionalPairSummary,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    conditional_pair_sample_identity,
    generate_target_free_compatible_pairs,
    iter_conditional_pair_identities,
    label_conditional_pairs,
    partition_fit_calibration_identities,
    validate_molecule_disjoint_splits,
)
from chemvector.audit_evidence_carrier_conditional_pair_data import _pairing_config
from chemvector.evidence_carrier_stroke_pairing import StrokeBoundaryPairingConfig


WIDTH = HEIGHT = 100


def qpoint(x: float, y: float) -> list[float]:
    return [x / (WIDTH - 1) * 2047.0, y / (HEIGHT - 1) * 2047.0]


def single_line_sidecars() -> tuple[dict, dict]:
    primitives = {
        "schema_version": "visual-primitives-0.1",
        "primitives": [
            {
                "primitive_id": 0,
                "kind": "line",
                "quantized_points": [qpoint(10, 50), qpoint(90, 50)],
                "stroke_width_q": 4,
                "owner_kind": "bond",
                "owner_id": 0,
            }
        ],
        "bond_groups": [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0]}
        ],
        "text_instances": [],
    }
    layers = {
        "width": WIDTH,
        "height": HEIGHT,
        "atoms": [],
        "bonds": [{"bond_id": 0, "begin": [10, 50], "end": [90, 50]}],
    }
    return primitives, layers


def stroke_image(*, width: int = 3) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    ImageDraw.Draw(image).line((10, 50, 90, 50), fill="black", width=width)
    return image


def sample_meta(
    key: str,
    molecule_id: str,
    *,
    split: str = "train",
    variant: str = "original",
    source_record: str | None = None,
) -> dict:
    source_record = source_record or f"{split}_{key}"
    return {
        "sample_key": key,
        "molecule_id": molecule_id,
        "source_pilot": "pilot",
        "source_split": split,
        "source_sample_key": f"{source_record}_{variant}",
        "source_variant": variant,
        "split": split,
        "variant": variant,
    }


def identity(
    key: str,
    molecule: str,
    source: str,
    *,
    split: str = "train",
    variant: str = "original",
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


def pairing_args(profile: str, **overrides: float | None) -> Namespace:
    values = {
        "pairing_profile": profile,
        "maximum_angle_degrees": None,
        "minimum_normal_separation_pixels": None,
        "maximum_normal_separation_pixels": None,
        "minimum_overlap_pixels": None,
        "minimum_shorter_interval_overlap": None,
        "minimum_interior_ink_occupancy": None,
    }
    values.update(overrides)
    return Namespace(**values)


class ConditionalPairCandidateTest(unittest.TestCase):
    def test_named_training_profile_is_wide_but_does_not_change_strict_default(self) -> None:
        strict = conditional_pair_pairing_config()
        wide = conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)

        self.assertEqual(strict, StrokeBoundaryPairingConfig())
        self.assertEqual(wide.maximum_angle_degrees, 8.0)
        self.assertEqual(wide.minimum_normal_separation_pixels, 0.5)
        self.assertEqual(wide.maximum_normal_separation_pixels, 8.0)
        self.assertEqual(wide.minimum_overlap_pixels, 4.0)
        self.assertEqual(wide.minimum_shorter_interval_overlap, 0.35)
        self.assertEqual(wide.minimum_interior_ink_occupancy, 0.25)

    def test_cli_forbids_named_profile_override_ambiguity(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _pairing_config(
                pairing_args(
                    PAIRING_PROFILE_TRAINING_WIDE,
                    maximum_angle_degrees=9.0,
                )
            )
        custom = _pairing_config(
            pairing_args("custom", maximum_angle_degrees=9.0)
        )
        self.assertEqual(custom.maximum_angle_degrees, 9.0)
        self.assertEqual(
            custom.maximum_normal_separation_pixels,
            StrokeBoundaryPairingConfig().maximum_normal_separation_pixels,
        )

    def test_target_free_candidates_are_symmetric_and_keep_fixed_feature_order(self) -> None:
        segments = np.asarray(
            [
                ((10.0, 48.5), (90.0, 48.5)),
                ((10.0, 51.5), (90.0, 51.5)),
            ]
        )

        forward = generate_target_free_compatible_pairs(segments, stroke_image())
        reversed_members = generate_target_free_compatible_pairs(
            segments[::-1].copy(), stroke_image()
        )

        self.assertEqual(PAIR_FEATURE_NAMES, (
            "angle_degrees",
            "normal_separation_pixels",
            "overlap_pixels",
            "shorter_interval_overlap",
            "interval_iou",
            "interior_ink_occupancy",
            "interior_mean_darkness",
            "center_mean_darkness",
        ))
        np.testing.assert_array_equal(forward["pair_indices"], [[0, 1]])
        np.testing.assert_array_equal(reversed_members["pair_indices"], [[0, 1]])
        np.testing.assert_allclose(
            forward["pair_features"], reversed_members["pair_features"], atol=1e-12
        )
        np.testing.assert_allclose(
            forward["pair_overlap_centerline_pixels"],
            reversed_members["pair_overlap_centerline_pixels"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            forward["pair_overlap_centerline_pixels"][0],
            ((10.0, 50.0), (90.0, 50.0)),
            atol=1e-6,
        )

    def test_all_distinct_target_compatible_pairs_are_hard_negatives(self) -> None:
        labels, target_ids = label_conditional_pairs(
            np.asarray(((0, 1), (1, 2), (0, 2))),
            np.asarray((7, 7, 8)),
        )

        np.testing.assert_array_equal(labels, (1, 0, 0))
        np.testing.assert_array_equal(
            target_ids, ((7, 7), (7, 8), (7, 8))
        )

    def test_candidate_pool_can_be_widened_without_target_dependent_logic(self) -> None:
        image = stroke_image(width=7)
        segments = np.asarray(
            [
                ((10.0, 47.0), (90.0, 47.0)),
                ((10.0, 53.0), (90.0, 53.0)),
            ]
        )

        strict = generate_target_free_compatible_pairs(segments, image)
        wide = generate_target_free_compatible_pairs(
            segments,
            image,
            pairing_config=StrokeBoundaryPairingConfig(
                maximum_normal_separation_pixels=8.0
            ),
        )

        self.assertEqual(len(strict["pair_indices"]), 0)
        self.assertEqual(wide["pair_indices"].tolist(), [[0, 1]])

    def test_builder_routes_only_positive_ordinary_anchors_and_retains_raw_labels(self) -> None:
        primitives, layers = single_line_sidecars()
        carriers = np.asarray(
            [
                ((20.0, 49.0), (70.0, 49.0)),
                ((20.0, 51.0), (70.0, 51.0)),
                ((5.0, 5.0), (5.0, 25.0)),
            ]
        )

        target = build_conditional_ordinary_pair_target(
            stroke_image(),
            primitives,
            layers,
            carrier_segments_pixels=carriers,
        )

        self.assertEqual(target["anchor_source_indices"].tolist(), [0, 1])
        self.assertEqual(target["anchor_target_line_ids"].tolist(), [0, 0])
        self.assertEqual(target["pair_indices"].tolist(), [[0, 1]])
        self.assertEqual(target["pair_labels"].tolist(), [1])
        self.assertEqual(target["pair_target_line_ids"].tolist(), [[0, 0]])
        self.assertEqual(target["raw_keep_labels"].tolist(), [1, 1, 0])
        self.assertEqual(target["raw_target_line_ids"].tolist(), [0, 0, -1])
        self.assertEqual(target["raw_group_ids"].tolist(), [0, 0, -1])
        self.assertEqual(target["pair_features"].shape, (1, 8))
        torch.testing.assert_close(
            target["pair_overlap_centerline_pixels"][0],
            torch.tensor(((20.0, 50.0), (70.0, 50.0))),
            atol=1e-5,
            rtol=0,
        )


class ConditionalPairIdentityTest(unittest.TestCase):
    def test_identity_strips_variant_and_validates_declared_split(self) -> None:
        sample = {
            "key": "0001",
            "meta": sample_meta(
                "0001",
                "chembl:1",
                split="validation",
                variant="stress",
                source_record="validation_0042",
            ),
        }

        result = conditional_pair_sample_identity(
            sample,
            expected_split="validation",
            expected_render_variant="stress",
        )

        self.assertEqual(result.source_record, "validation_0042")
        self.assertEqual(
            result.source_structure_id, "pilot|validation|validation_0042"
        )
        with self.assertRaisesRegex(ValueError, "expected"):
            conditional_pair_sample_identity(sample, expected_split="train")

    def test_fit_calibration_keeps_all_variants_and_transitive_identities_together(self) -> None:
        identities = [
            identity("a0", "mol-a", "source-a", variant="original"),
            identity("a1", "mol-a", "source-a", variant="stress"),
            identity("b0", "mol-b", "source-b"),
            identity("c0", "mol-c", "source-c"),
            identity("d0", "mol-d", "shared"),
            identity("e0", "mol-e", "shared"),
        ]

        first = partition_fit_calibration_identities(
            identities, calibration_fraction=0.5, seed="fixed"
        )
        second = partition_fit_calibration_identities(
            list(reversed(identities)), calibration_fraction=0.5, seed="fixed"
        )

        self.assertEqual(first.fit_sample_keys, second.fit_sample_keys)
        self.assertEqual(
            first.calibration_sample_keys, second.calibration_sample_keys
        )
        self.assertTrue(
            {"a0", "a1"}.issubset(first.fit_sample_keys)
            or {"a0", "a1"}.issubset(first.calibration_sample_keys)
        )
        self.assertTrue(
            {"d0", "e0"}.issubset(first.fit_sample_keys)
            or {"d0", "e0"}.issubset(first.calibration_sample_keys)
        )
        audit = validate_molecule_disjoint_splits(
            first.fit,
            first.calibration,
            left_name="fit",
            right_name="calibration",
        )
        self.assertEqual(audit["molecule_overlap"], 0)
        self.assertEqual(audit["source_structure_overlap"], 0)

    def test_train_validation_requires_zero_overlap_for_both_identity_axes(self) -> None:
        train = [identity("train", "mol-a", "source-a")]
        clean_validation = [
            identity(
                "validation",
                "mol-b",
                "source-b",
                split="validation",
            )
        ]
        audit = validate_molecule_disjoint_splits(train, clean_validation)
        self.assertEqual(audit["molecule_overlap"], 0)
        self.assertEqual(audit["source_structure_overlap"], 0)

        leaked = [
            identity(
                "validation-leak",
                "mol-a",
                "source-z",
                split="validation",
            )
        ]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_molecule_disjoint_splits(train, leaked)

    def test_internal_test_is_rejected_before_shard_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            ConditionalOrdinaryPairWebDataset(
                "shards/internal-test-000000.tar",
                split="validation",
                render_variant="original",
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "internal_test-000000.tar"
            sealed.touch()
            alias = root / "validation-alias.tar"
            alias.symlink_to(sealed)
            with self.assertRaisesRegex(ValueError, "internal_test"):
                ConditionalOrdinaryPairWebDataset(
                    str(alias),
                    split="validation",
                    render_variant="original",
                )

    def test_single_variant_capacity_and_all_variant_modes_are_distinct(self) -> None:
        raw_samples = []
        for index, variant in enumerate(("original", "stress")):
            key = f"render-{index}"
            raw_samples.append(
                {
                    "__key__": key,
                    "meta.json": json.dumps(
                        sample_meta(
                            key,
                            "mol-shared",
                            variant=variant,
                            source_record="train_shared",
                        )
                    ).encode("utf-8"),
                }
            )

        with patch(
            "chemvector.evidence_carrier_conditional_pair_data.wds.WebDataset",
            return_value=raw_samples,
        ):
            single = list(
                iter_conditional_pair_identities(
                    "train.tar",
                    split="train",
                    render_variant="original",
                )
            )
        with patch(
            "chemvector.evidence_carrier_conditional_pair_data.wds.WebDataset",
            return_value=raw_samples,
        ):
            all_variants = list(
                iter_conditional_pair_identities(
                    "train.tar",
                    split="train",
                    render_variant=None,
                )
            )

        self.assertEqual([item.render_variant for item in single], ["original"])
        self.assertEqual(
            {item.render_variant for item in all_variants}, {"original", "stress"}
        )
        self.assertEqual(
            {item.source_structure_id for item in all_variants},
            {"pilot|train|train_shared"},
        )


class ConditionalPairCollatorAndSummaryTest(unittest.TestCase):
    def test_collator_pads_pair_and_raw_fields_and_exposes_schema(self) -> None:
        primitives, layers = single_line_sidecars()
        samples = []
        for index in range(2):
            key = f"sample-{index}"
            samples.append(
                {
                    "key": key,
                    "image": stroke_image(),
                    "bond_layer": stroke_image().convert("L"),
                    "primitives": primitives,
                    "layers": layers,
                    "meta": sample_meta(key, f"mol-{index}"),
                }
            )
        detections = [
            np.asarray(
                [
                    ((20.0, 49.0), (70.0, 49.0)),
                    ((20.0, 51.0), (70.0, 51.0)),
                ]
            ),
            np.asarray([((20.0, 49.0), (70.0, 49.0))]),
        ]
        collator = ConditionalOrdinaryPairCollator(image_size=WIDTH)

        with patch(
            "chemvector.evidence_carrier_conditional_pair_data.detect_evidence_carriers",
            side_effect=detections,
        ):
            batch = collator(samples)

        self.assertEqual(batch["anchor_count"].tolist(), [2, 1])
        self.assertEqual(batch["pair_count"].tolist(), [1, 0])
        self.assertEqual(batch["pair_features"].shape, (2, 1, 8))
        self.assertEqual(batch["pair_mask"].tolist(), [[True], [False]])
        self.assertEqual(batch["pair_labels"].tolist(), [[1], [-1]])
        self.assertEqual(batch["raw_group_ids"].shape, (2, 2))
        self.assertEqual(batch["raw_group_ids"][1].tolist(), [0, -1])
        self.assertEqual(batch["pair_feature_names"], PAIR_FEATURE_NAMES)
        self.assertEqual(batch["pixel_values"].shape, (2, 3, HEIGHT, WIDTH))
        self.assertEqual(batch["identity"][0]["molecule_id"], "mol-0")

    def test_summary_reports_distribution_and_exact_oracle_ceiling(self) -> None:
        target = {
            "anchor_target_line_ids": torch.tensor([0, 0, 1, 1]),
            "pair_indices": torch.tensor([[0, 1], [1, 2], [2, 3]]),
            "pair_labels": torch.tensor([1, 0, 1]),
            "pair_features": torch.tensor(
                [
                    [0, 2, 50, 1, 1, 1, 1, 1],
                    [0, 2, 50, 1, 1, 1, 1, 1],
                    [0, 2, 50, 1, 1, 1, 1, 1],
                ],
                dtype=torch.float32,
            ),
        }
        summary = ConditionalPairSummary()
        summary.update(target)
        result = summary.result()

        distribution = result["class_distribution"]
        self.assertEqual(distribution["same_target_positive"], 2)
        self.assertEqual(distribution["distinct_target_hard_negative"], 1)
        ceiling = result["candidate_learnability_ceiling"]
        self.assertEqual(ceiling["oracle_max_disjoint_positive_pairs"], 2)
        self.assertEqual(ceiling["oracle_paired_anchor_fraction"], 1.0)
        collision = result["empirical_feature_collision_ceiling"]
        self.assertEqual(collision["conflicting_feature_rows"], 1)
        self.assertAlmostEqual(collision["majority_lookup_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
