"""Tests for equivalence-aware evidence-carrier extent targets."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from chemvector.bond_group_set import PARALLEL_2, SINGLE_LINE
from chemvector.evidence_carrier_extent import (
    canonicalize_carrier_segments,
    clip_carrier_lines_to_image,
)
from chemvector.evidence_carrier_extent_data import (
    IGNORE_LABEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    EvidenceCarrierExtentCollator,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
    associate_evidence_carriers,
)


WIDTH = HEIGHT = 100


def qpoint(x: float, y: float) -> list[float]:
    return [x / (WIDTH - 1) * 2047.0, y / (HEIGHT - 1) * 2047.0]


def primitive(
    primitive_id: int,
    kind: str,
    points: list[tuple[float, float]],
) -> dict:
    return {
        "primitive_id": primitive_id,
        "kind": kind,
        "quantized_points": [qpoint(x, y) for x, y in points],
        "stroke_width_q": 4,
        "owner_kind": "bond",
        "owner_id": primitive_id,
    }


def payload(
    primitives: list[dict],
    groups: list[dict],
    *,
    text_instances: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "visual-primitives-0.1",
        "primitives": primitives,
        "bond_groups": groups,
        "text_instances": text_instances or [],
    }


def layers(groups: list[dict], *, text_bbox: list[float] | None = None) -> dict:
    atoms = []
    if text_bbox is not None:
        atoms.append(
            {
                "atom_id": 99,
                "visible_text": True,
                "bbox": text_bbox,
            }
        )
    bonds = []
    for index, group in enumerate(groups):
        bonds.append(
            {
                "bond_id": int(group.get("bond_id", index)),
                "begin": [10.0, 50.0],
                "end": [90.0, 50.0],
            }
        )
    return {"width": WIDTH, "height": HEIGHT, "atoms": atoms, "bonds": bonds}


def single_line_fixture() -> tuple[dict, dict]:
    primitives = [primitive(0, "line", [(10, 50), (90, 50)])]
    groups = [
        {
            "bond_id": 0,
            "direction": "NONE",
            "primitive_ids": [0],
        }
    ]
    return payload(primitives, groups), layers(groups)


class EvidenceCarrierAssociationTest(unittest.TestCase):
    def test_equivalent_raster_edges_are_all_positive_for_one_physical_line(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        carriers = np.asarray(
            [
                [[55.0, 49.0], [30.0, 49.0]],
                [[40.0, 51.0], [70.0, 51.0]],
            ]
        )
        target_config = EvidenceCarrierExtentTargetConfig(
            extent_bins=10, normal_offset_bins=9
        )

        target = associate_evidence_carriers(
            carriers,
            primitives,
            semantic_layers,
            target_config=target_config,
        )

        self.assertEqual(target["keep_labels"].tolist(), [POSITIVE_LABEL] * 2)
        self.assertEqual(target["target_line_id"].tolist(), [0, 0])
        self.assertEqual(target["group_id"].tolist(), [0, 0])
        self.assertEqual(target["group_type"].tolist(), [SINGLE_LINE] * 2)
        torch.testing.assert_close(
            target["equivalence_weight"], torch.tensor([0.5, 0.5])
        )
        self.assertAlmostEqual(float(target["equivalence_weight"].sum()), 1.0)
        torch.testing.assert_close(
            target["carrier_segments"][:, :, 0],
            torch.tensor(
                [[30 / 99, 55 / 99], [40 / 99, 70 / 99]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            target["normal_offset_pixels"], torch.tensor([1.0, -1.0]), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            target["extent_left_fraction"],
            torch.tensor([10 / 99, 10 / 99]),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(
            target["extent_right_fraction"],
            torch.tensor([90 / 99, 90 / 99]),
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(target["extent_left_bin"].tolist(), [1, 1])
        self.assertEqual(target["extent_right_bin"].tolist(), [8, 8])
        self.assertTrue(bool(target["geometry_loss_mask"].all()))

    def test_parallel_members_choose_nearest_physical_line_without_one_to_one_limit(self) -> None:
        primitives = [
            primitive(0, "line", [(10, 40), (90, 40)]),
            primitive(1, "line", [(10, 46), (90, 46)]),
        ]
        groups = [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0, 1]}
        ]
        carriers = np.asarray(
            [
                [[20.0, 39.0], [60.0, 39.0]],
                [[25.0, 41.0], [55.0, 41.0]],
                [[30.0, 47.0], [70.0, 47.0]],
            ]
        )

        target = associate_evidence_carriers(
            carriers, payload(primitives, groups), layers(groups)
        )

        self.assertEqual(target["keep_labels"].tolist(), [1, 1, 1])
        self.assertEqual(target["target_line_id"].tolist(), [0, 0, 1])
        self.assertEqual(target["group_id"].tolist(), [0, 0, 0])
        self.assertEqual(target["group_type"].tolist(), [PARALLEL_2] * 3)
        torch.testing.assert_close(
            target["equivalence_weight"], torch.tensor([0.5, 0.5, 1.0])
        )

    def test_equidistant_parallel_candidate_is_ignored_as_ambiguous(self) -> None:
        primitives = [
            primitive(0, "line", [(10, 40), (90, 40)]),
            primitive(1, "line", [(10, 46), (90, 46)]),
        ]
        groups = [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0, 1]}
        ]

        target = associate_evidence_carriers(
            np.asarray([[[20.0, 43.0], [70.0, 43.0]]]),
            payload(primitives, groups),
            layers(groups),
        )

        self.assertEqual(target["keep_labels"].tolist(), [IGNORE_LABEL])
        self.assertEqual(target["keep_loss_mask"].tolist(), [False])
        self.assertEqual(target["geometry_loss_mask"].tolist(), [False])
        self.assertEqual(target["target_line_id"].tolist(), [-1])

    def test_metric_broadcast_handles_more_than_two_target_lines(self) -> None:
        primitives = [
            primitive(0, "line", [(10, 30), (90, 30)]),
            primitive(1, "line", [(10, 50), (90, 50)]),
            primitive(2, "line", [(10, 70), (90, 70)]),
        ]
        groups = [
            {"bond_id": index, "direction": "NONE", "primitive_ids": [index]}
            for index in range(3)
        ]
        target = associate_evidence_carriers(
            np.asarray([[[20.0, 69.0], [70.0, 69.0]]]),
            payload(primitives, groups),
            layers(groups),
        )

        self.assertEqual(target["target_line_id"].tolist(), [2])
        self.assertEqual(target["group_id"].tolist(), [2])

    def test_near_miss_is_ignored_and_only_far_incompatible_anchor_is_negative(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        carriers = np.asarray(
            [
                [[20.0, 56.0], [70.0, 56.0]],
                [[5.0, 5.0], [5.0, 25.0]],
            ]
        )

        target = associate_evidence_carriers(carriers, primitives, semantic_layers)

        self.assertEqual(
            target["keep_labels"].tolist(), [IGNORE_LABEL, NEGATIVE_LABEL]
        )
        self.assertEqual(target["keep_loss_mask"].tolist(), [False, True])
        self.assertEqual(target["equivalence_weight"].tolist(), [0.0, 1.0])
        self.assertFalse(bool(target["geometry_loss_mask"].any()))

    def test_short_line_uses_quarter_length_positive_overlap(self) -> None:
        primitives = [primitive(0, "line", [(20, 50), (24, 50)])]
        groups = [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0]}
        ]

        target = associate_evidence_carriers(
            np.asarray([[[20.0, 50.0], [21.2, 50.0]]]),
            payload(primitives, groups),
            layers(groups),
        )

        self.assertEqual(target["keep_labels"].tolist(), [POSITIVE_LABEL])
        self.assertTrue(bool(target["geometry_loss_mask"][0]))

    def test_extent_uses_reclipped_carrier_after_normal_shift(self) -> None:
        primitives = [primitive(0, "line", [(20, 30), (70, 80)])]
        groups = [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0]}
        ]
        target_config = EvidenceCarrierExtentTargetConfig(
            positive_perpendicular_pixels=8.0,
            compatibility_perpendicular_pixels=10.0,
            normal_offset_limit_pixels=8.0,
            normal_offset_bins=17,
        )

        target = associate_evidence_carriers(
            np.asarray([[[30.0, 30.0], [60.0, 60.0]]]),
            payload(primitives, groups),
            layers(groups),
            target_config=target_config,
        )

        self.assertEqual(target["keep_labels"].tolist(), [POSITIVE_LABEL])
        self.assertAlmostEqual(
            float(target["extent_left_fraction"][0]), 20 / 89, places=5
        )
        self.assertAlmostEqual(
            float(target["extent_right_fraction"][0]), 70 / 89, places=5
        )

    def test_negative_slope_endpoint_order_matches_model_canonical_axis(self) -> None:
        primitives = [primitive(0, "line", [(20, 80), (80, 20)])]
        groups = [
            {"bond_id": 0, "direction": "NONE", "primitive_ids": [0]}
        ]
        carriers = np.asarray(
            [
                [[30.0, 70.0], [60.0, 40.0]],
                [[60.0, 40.0], [30.0, 70.0]],
            ]
        )

        target = associate_evidence_carriers(
            carriers,
            payload(primitives, groups),
            layers(groups),
        )

        self.assertEqual(target["keep_labels"].tolist(), [1, 1])
        torch.testing.assert_close(
            target["carrier_segments"][0], target["carrier_segments"][1]
        )
        torch.testing.assert_close(
            target["normal_offset_pixels"], torch.zeros(2), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            target["extent_left_fraction"][:1], target["extent_left_fraction"][1:]
        )
        torch.testing.assert_close(
            target["extent_right_fraction"][:1], target["extent_right_fraction"][1:]
        )

        mask = target["carrier_mask"].unsqueeze(0)
        scale = torch.tensor((WIDTH - 1, HEIGHT - 1), dtype=torch.float32)
        segments_pixels = target["carrier_segments"].unsqueeze(0) * scale
        _, centers, axes, normals = canonicalize_carrier_segments(
            segments_pixels, mask
        )
        shifted_centers = centers + (
            normals * target["normal_offset_pixels"].unsqueeze(0).unsqueeze(-1)
        )
        clipped, intersects = clip_carrier_lines_to_image(
            shifted_centers,
            axes,
            mask,
            image_height=HEIGHT,
            image_width=WIDTH,
        )
        interval = clipped[..., 1, :] - clipped[..., 0, :]
        reconstructed = torch.stack(
            (
                clipped[..., 0, :]
                + interval * target["extent_left_fraction"].unsqueeze(0).unsqueeze(-1),
                clipped[..., 0, :]
                + interval * target["extent_right_fraction"].unsqueeze(0).unsqueeze(-1),
            ),
            dim=-2,
        )
        expected = torch.tensor([[[[20.0, 80.0], [80.0, 20.0]]] * 2])
        self.assertTrue(bool(intersects.all()))
        torch.testing.assert_close(reconstructed, expected, atol=1e-4, rtol=0)

    def test_hashed_polygon_and_text_neighborhoods_are_never_negatives(self) -> None:
        primitives = [
            primitive(0, "line", [(10, 10), (30, 10)]),
            primitive(1, "line", [(10, 14), (30, 14)]),
            primitive(2, "triangle", [(40, 10), (55, 20), (40, 30)]),
            primitive(3, "quad", [(65, 10), (80, 10), (80, 25), (65, 25)]),
        ]
        groups = [
            {"bond_id": 0, "direction": "BEGINDASH", "primitive_ids": [0, 1]},
            {"bond_id": 1, "direction": "BEGINWEDGE", "primitive_ids": [2]},
            {"bond_id": 2, "direction": "BEGINWEDGE", "primitive_ids": [3]},
        ]
        carriers = np.asarray(
            [
                [[12.0, 10.0], [28.0, 10.0]],
                [[41.0, 11.0], [53.0, 19.0]],
                [[67.0, 11.0], [78.0, 11.0]],
                [[42.0, 72.0], [58.0, 72.0]],
                [[5.0, 90.0], [25.0, 90.0]],
            ]
        )
        semantic_layers = layers(groups, text_bbox=[40.0, 68.0, 60.0, 76.0])

        target = associate_evidence_carriers(
            carriers,
            payload(primitives, groups),
            semantic_layers,
        )

        self.assertEqual(
            target["keep_labels"].tolist(),
            [IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, NEGATIVE_LABEL],
        )
        self.assertEqual(target["equivalence_weight"].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(target["target_line_id"].tolist(), [-1] * 5)


class EvidenceCarrierDataAdapterTest(unittest.TestCase):
    def test_collator_dynamically_pads_lsd_carriers(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        samples = [
            {
                "key": "sample-a",
                "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
                "bond_layer": Image.new("L", (WIDTH, HEIGHT), 0),
                "primitives": primitives,
                "layers": semantic_layers,
                "meta": {"split": "train"},
            },
            {
                "key": "sample-b",
                "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
                "bond_layer": Image.new("L", (WIDTH, HEIGHT), 0),
                "primitives": primitives,
                "layers": semantic_layers,
                "meta": {"split": "validation"},
            },
        ]
        detected = [
            np.asarray([[[20.0, 49.0], [70.0, 49.0]]]),
            np.asarray(
                [
                    [[20.0, 49.0], [70.0, 49.0]],
                    [[5.0, 5.0], [5.0, 25.0]],
                ]
            ),
        ]
        collator = EvidenceCarrierExtentCollator(image_size=WIDTH)

        with patch(
            "chemvector.evidence_carrier_extent_data.detect_lsd_lines",
            side_effect=detected,
        ):
            batch = collator(samples)

        self.assertEqual(batch["carrier_segments"].shape, (2, 2, 2, 2))
        self.assertEqual(batch["carrier_count"].tolist(), [1, 2])
        self.assertEqual(
            batch["carrier_mask"].tolist(), [[True, False], [True, True]]
        )
        self.assertEqual(
            batch["keep_labels"].tolist(), [[POSITIVE_LABEL, -1], [POSITIVE_LABEL, 0]]
        )
        self.assertEqual(batch["pixel_values"].shape, (2, 3, HEIGHT, WIDTH))

    def test_collator_rejects_implicit_image_resize(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        sample = {
            "key": "wrong-size",
            "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
            "bond_layer": Image.new("L", (WIDTH, HEIGHT), 0),
            "primitives": primitives,
            "layers": semantic_layers,
            "meta": {"split": "train"},
        }
        with self.assertRaisesRegex(ValueError, "native-size"):
            EvidenceCarrierExtentCollator(image_size=32)([sample])

    def test_rejects_internal_test_before_detection_or_shard_open(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        sample = {
            "key": "sealed",
            "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
            "bond_layer": Image.new("L", (WIDTH, HEIGHT), 0),
            "primitives": primitives,
            "layers": semantic_layers,
            "meta": {"source_split": "internal-test"},
        }
        with patch("chemvector.evidence_carrier_extent_data.detect_lsd_lines") as detector:
            with self.assertRaisesRegex(ValueError, "internal_test"):
                EvidenceCarrierExtentCollator()([sample])
            detector.assert_not_called()
        with self.assertRaisesRegex(ValueError, "internal_test"):
            EvidenceCarrierExtentWebDataset("shards/internal-test-000000.tar")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "internal_test-000000.tar"
            sealed.touch()
            alias = root / "validation-alias.tar"
            alias.symlink_to(sealed)
            with self.assertRaisesRegex(ValueError, "internal_test"):
                EvidenceCarrierExtentWebDataset(str(alias))

    def test_oracle_bond_layer_requires_a_strict_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean"):
            EvidenceCarrierExtentCollator(oracle_bond_layer="false")

    def test_bin_counts_are_configuration_not_model_width_constants(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        carrier = np.asarray([[[20.0, 49.0], [70.0, 49.0]]])
        small = associate_evidence_carriers(
            carrier,
            primitives,
            semantic_layers,
            target_config=EvidenceCarrierExtentTargetConfig(
                extent_bins=5, normal_offset_bins=7
            ),
        )
        large = associate_evidence_carriers(
            carrier,
            primitives,
            semantic_layers,
            target_config=EvidenceCarrierExtentTargetConfig(
                extent_bins=20, normal_offset_bins=21
            ),
        )
        self.assertLess(int(small["extent_right_bin"][0]), 5)
        self.assertLess(int(large["extent_right_bin"][0]), 20)
        self.assertNotEqual(
            int(small["extent_right_bin"][0]), int(large["extent_right_bin"][0])
        )

    def test_default_bins_match_model_linspace_nearest_indices(self) -> None:
        primitives, semantic_layers = single_line_fixture()
        target = associate_evidence_carriers(
            np.asarray([[[20.0, 49.0], [70.0, 49.0]]]),
            primitives,
            semantic_layers,
        )
        defaults = EvidenceCarrierExtentTargetConfig()

        self.assertEqual(defaults.extent_bins, 385)
        self.assertEqual(defaults.normal_offset_bins, 17)
        self.assertEqual(int(target["extent_left_bin"][0]), round(10 / 99 * 384))
        self.assertEqual(int(target["extent_right_bin"][0]), round(90 / 99 * 384))
        self.assertEqual(int(target["normal_offset_bin"][0]), 10)


if __name__ == "__main__":
    unittest.main()
