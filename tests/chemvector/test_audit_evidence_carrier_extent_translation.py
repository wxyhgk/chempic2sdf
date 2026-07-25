from __future__ import annotations

import argparse
import copy
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from chemvector.audit_evidence_carrier_extent_translation import (
    compare_scenarios,
    evaluate_scenario,
    foreground_bounds,
    parse_translation,
    select_paired_samples,
    summarize_records,
    translate_image,
    translate_sample,
    translate_sidecars,
    translation_name,
    translation_preserves_foreground,
)
from chemvector.audit_evidence_carrier_proposals import COORDINATE_MAX
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
)


WIDTH = 11
HEIGHT = 21


def qpoint(x: float, y: float) -> list[float]:
    return [
        x * COORDINATE_MAX / (WIDTH - 1),
        y * COORDINATE_MAX / (HEIGHT - 1),
    ]


def sidecars() -> tuple[dict, dict]:
    primitives = {
        "width": WIDTH,
        "height": HEIGHT,
        "primitives": [
            {
                "primitive_id": 0,
                "kind": "line",
                "points": [[2.0, 3.0], [8.0, 9.0]],
                "quantized_points": [qpoint(2.0, 3.0), qpoint(8.0, 9.0)],
                "owner_kind": "bond",
                "owner_id": 0,
            },
            {
                "primitive_id": 1,
                "kind": "text_box",
                "points": [[1.0, 2.0], [3.0, 4.0]],
                "quantized_points": [qpoint(1.0, 2.0), qpoint(3.0, 4.0)],
                "owner_kind": "atom",
                "owner_id": 0,
            },
        ],
        "bond_groups": [
            {
                "bond_id": 0,
                "begin_atom": 0,
                "end_atom": 1,
                "direction": "NONE",
                "primitive_ids": [0],
            }
        ],
        "text_instances": [
            {"atom_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "primitive_ids": [1]}
        ],
    }
    layers = {
        "width": WIDTH,
        "height": HEIGHT,
        "atoms": [
            {
                "atom_id": 0,
                "visible_text": True,
                "center": [2.0, 3.0],
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            {
                "atom_id": 1,
                "visible_text": False,
                "center": [8.0, 9.0],
                "bbox": None,
            },
        ],
        "bonds": [
            {
                "bond_id": 0,
                "begin": [2.0, 3.0],
                "end": [8.0, 9.0],
                "bbox": [1.0, 2.0, 9.0, 10.0],
            }
        ],
    }
    return primitives, layers


def sample(*, foreground_x: int = 2) -> dict:
    primitives, layers = sidecars()
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    image.putpixel((foreground_x, 3), (0, 0, 0))
    bond = Image.new("L", (WIDTH, HEIGHT), 0)
    bond.putpixel((foreground_x, 3), 255)
    return {
        "key": f"sample-{foreground_x}",
        "meta": {"split": "validation"},
        "image": image,
        "bond_layer": bond,
        "primitives": primitives,
        "layers": layers,
    }


def record(
    *,
    target_lines: int,
    matched: int,
    exact: bool,
    single_matched: int = 0,
    single_expected: int = 0,
) -> dict:
    tolerance = {
        "ordinary_members": {"matched": matched, "expected": target_lines},
        "groups": {
            "single": {"matched": single_matched, "expected": single_expected},
            "p2": {"matched": 0, "expected": 0},
            "p3": {"matched": 0, "expected": 0},
        },
        "long_groups": {"matched": 0, "expected": 0},
        "structural_sample_exact": exact,
    }
    return {
        "ordinary_target_lines": target_lines,
        "tolerances": {name: copy.deepcopy(tolerance) for name in ("q8", "q16", "q32")},
    }


class EvidenceCarrierExtentTranslationAuditTest(unittest.TestCase):
    def test_translation_cli_and_names_support_signed_offsets(self) -> None:
        self.assertEqual(parse_translation("16,0"), (16, 0))
        self.assertEqual(parse_translation("-32,5"), (-32, 5))
        self.assertEqual(translation_name(-32, 5), "x-32_y+5")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_translation("16")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_translation("0,0")

    def test_integer_translation_has_no_interpolation_and_supports_negative_y(self) -> None:
        image = Image.new("RGB", (6, 5), "white")
        image.putpixel((1, 2), (5, 10, 15))
        translated = translate_image(
            image,
            dx=2,
            dy=-1,
            mode="RGB",
            fill=(255, 255, 255),
        )
        self.assertEqual(translated.getpixel((3, 1)), (5, 10, 15))
        self.assertEqual(foreground_bounds(translated), (3, 1, 3, 1))

    def test_foreground_crop_policy_is_raster_defined(self) -> None:
        safe = Image.new("RGB", (6, 5), "white")
        safe.putpixel((2, 2), (0, 0, 0))
        self.assertTrue(translation_preserves_foreground(safe, dx=2, dy=1))

        cropped = Image.new("RGB", (6, 5), "white")
        cropped.putpixel((5, 2), (0, 0, 0))
        self.assertFalse(translation_preserves_foreground(cropped, dx=1, dy=0))

    def test_sidecars_use_pixel_center_q_translation_without_mutation(self) -> None:
        primitives, layers = sidecars()
        original_primitives = copy.deepcopy(primitives)
        original_layers = copy.deepcopy(layers)
        shifted_primitives, shifted_layers = translate_sidecars(
            primitives, layers, dx=2, dy=-1
        )

        q_shift = np.asarray(
            (
                2.0 * COORDINATE_MAX / (WIDTH - 1),
                -COORDINATE_MAX / (HEIGHT - 1),
            )
        )
        difference = np.asarray(
            shifted_primitives["primitives"][0]["quantized_points"]
        ) - np.asarray(primitives["primitives"][0]["quantized_points"])
        np.testing.assert_allclose(
            difference,
            np.broadcast_to(q_shift, difference.shape),
            atol=1e-8,
            rtol=0.0,
        )
        self.assertEqual(
            shifted_primitives["primitives"][0]["points"],
            [[4.0, 2.0], [10.0, 8.0]],
        )
        self.assertEqual(shifted_layers["atoms"][0]["center"], [4.0, 2.0])
        self.assertEqual(shifted_layers["atoms"][0]["bbox"], [3.0, 1.0, 5.0, 3.0])
        self.assertEqual(shifted_layers["bonds"][0]["end"], [10.0, 8.0])
        self.assertEqual(primitives, original_primitives)
        self.assertEqual(layers, original_layers)

    def test_translate_sample_moves_raster_bond_layer_and_targets_together(self) -> None:
        source = sample(foreground_x=2)
        shifted = translate_sample(
            source,
            dx=2,
            dy=1,
            target_config=EvidenceCarrierExtentTargetConfig(),
        )
        self.assertEqual(shifted["image"].getpixel((4, 4)), (0, 0, 0))
        self.assertEqual(shifted["bond_layer"].getpixel((4, 4)), 255)
        self.assertEqual(shifted["layers"]["bonds"][0]["begin"], [4.0, 4.0])
        self.assertEqual(source["layers"]["bonds"][0]["begin"], [2.0, 3.0])

    def test_paired_selection_rejects_crop_without_reading_target_geometry_for_rank(self) -> None:
        cropped = sample(foreground_x=WIDTH - 1)
        safe = sample(foreground_x=2)
        with patch(
            "chemvector.audit_evidence_carrier_extent_translation."
            "EvidenceCarrierExtentWebDataset",
            return_value=[cropped, safe],
        ):
            selected, selection = select_paired_samples(
                "validation.tar",
                ((1, 0),),
                limit=1,
                target_config=EvidenceCarrierExtentTargetConfig(),
            )
        self.assertEqual([item["key"] for item in selected], [safe["key"]])
        self.assertEqual(selection["scanned"], 2)
        self.assertEqual(selection["rejected_for_crop_by_translation"]["x+1_y+0"], 1)
        self.assertIs(selection["selection_uses_target_geometry"], False)

    def test_structural_exact_excludes_empty_target_samples(self) -> None:
        summary = summarize_records(
            [
                record(target_lines=0, matched=0, exact=True),
                record(
                    target_lines=2,
                    matched=2,
                    exact=True,
                    single_matched=1,
                    single_expected=1,
                ),
            ]
        )
        self.assertEqual(summary["q8"]["structural_samples"]["exact"], 1)
        self.assertEqual(summary["q8"]["structural_samples"]["expected"], 1)
        self.assertEqual(
            summary["q8"]["structural_samples"]["empty_target_samples_excluded"],
            1,
        )

    def test_comparison_reports_signed_delta_and_regression_pp(self) -> None:
        base = {"geometry": summarize_records([record(target_lines=10, matched=9, exact=False)])}
        translated = {
            "geometry": summarize_records([record(target_lines=10, matched=8, exact=False)])
        }
        comparison = compare_scenarios(base, translated)
        recall = comparison["q8"]["ordinary_member_recall"]
        self.assertAlmostEqual(recall["base"], 0.9)
        self.assertAlmostEqual(recall["translated"], 0.8)
        self.assertAlmostEqual(recall["delta_pp"], -10.0)
        self.assertAlmostEqual(recall["regression_pp"], 10.0)

    def test_scenario_evaluation_forces_untrained_keep_head_active(self) -> None:
        mask = torch.ones((1, 1), dtype=torch.bool)
        batch = {
            "pixel_values": torch.zeros((1, 3, 10, 10)),
            "carrier_segments": torch.tensor(
                [[[[0.1, 0.5], [0.9, 0.5]]]], dtype=torch.float32
            ),
            "carrier_mask": mask,
            "keep_labels": torch.ones((1, 1), dtype=torch.long),
            "keep_loss_mask": mask,
            "equivalence_weight": torch.ones((1, 1)),
            "normal_offset_bin": torch.zeros((1, 1), dtype=torch.long),
            "extent_left_bin": torch.zeros((1, 1), dtype=torch.long),
            "extent_right_bin": torch.ones((1, 1), dtype=torch.long),
            "geometry_loss_mask": mask,
            "target_line_id": torch.zeros((1, 1), dtype=torch.long),
            "group_id": torch.zeros((1, 1), dtype=torch.long),
            "group_type": torch.ones((1, 1), dtype=torch.long),
            "keys": ["validation-sample"],
            "meta": [{"split": "validation"}],
            "primitives": [{}],
            "layers": [{"width": 10, "height": 10}],
        }

        class Model:
            def eval(self):
                return self

            def __call__(self, pixel_values, carrier_segments, carrier_mask, **kwargs):
                self.inputs = (pixel_values, carrier_segments, carrier_mask, kwargs)
                return {"unused": torch.tensor(0.0)}

        model = Model()
        decoded = {
            "refined_segments_pixels": torch.tensor(
                [[[[1.0, 5.0], [8.0, 5.0]]]], dtype=torch.float32
            )
        }
        evaluated = record(target_lines=1, matched=1, exact=True)
        with patch(
            "chemvector.audit_evidence_carrier_extent_translation."
            "decode_evidence_carrier_extent",
            return_value=decoded,
        ) as decoder, patch(
            "chemvector.audit_evidence_carrier_extent_translation."
            "evaluate_sample_geometry",
            return_value=evaluated,
        ):
            metrics = evaluate_scenario(
                model,
                [batch],
                torch.device("cpu"),
                target_config=EvidenceCarrierExtentTargetConfig(),
            )

        decoder.assert_called_once()
        self.assertEqual(decoder.call_args.kwargs["keep_threshold"], 0.0)
        self.assertEqual(model.inputs[3]["coordinate_mode"], "normalized")
        self.assertEqual(metrics["ordinary_q8_member_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
