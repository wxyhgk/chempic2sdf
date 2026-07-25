from __future__ import annotations

from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image
import torch

from chemvector.bond_group_set import PARALLEL_2, SINGLE_LINE
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.evidence_carrier_pair_evaluation import (
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evaluate_evidence_carrier_pair_extent import (
    P0_REFERENCE_Q8_ORDINARY_RECALL,
    build_parser,
    evaluate_pair_extent_sample,
    pack_selected_pair_carriers,
    summarize_pair_extent_validation,
)


WIDTH = HEIGHT = 100


def qpoint(x: float, y: float) -> list[float]:
    return [x / (WIDTH - 1) * 2047.0, y / (HEIGHT - 1) * 2047.0]


def sidecars() -> tuple[dict, dict]:
    primitives = {
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


def pair_target(
    target_ids: list[int],
    pairs: list[tuple[int, int]],
    *,
    group_ids: list[int] | None = None,
    group_types: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    count = len(target_ids)
    anchors = torch.tensor(
        [
            ((10.0, 50.0 + 2.0 * index), (90.0, 50.0 + 2.0 * index))
            for index in range(count)
        ],
        dtype=torch.float32,
    )
    if not count:
        anchors = torch.empty((0, 2, 2), dtype=torch.float32)
    pair_indices = torch.tensor(pairs, dtype=torch.long)
    if not pairs:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    centerlines = []
    for first, second in pairs:
        centerlines.append(0.5 * (anchors[first] + anchors[second]))
    centerlines_tensor = (
        torch.stack(centerlines)
        if centerlines
        else torch.empty((0, 2, 2), dtype=torch.float32)
    )
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    group_ids = group_ids or list(target_ids)
    group_types = group_types or [SINGLE_LINE] * count
    pair_labels = (
        target_tensor[pair_indices[:, 0]] == target_tensor[pair_indices[:, 1]]
    ).long() if len(pair_indices) else torch.empty((0,), dtype=torch.long)
    return {
        "anchor_segments_pixels": anchors,
        "anchor_target_line_ids": target_tensor,
        "anchor_source_indices": torch.arange(count),
        "anchor_mask": torch.ones(count, dtype=torch.bool),
        "pair_indices": pair_indices,
        "pair_features": torch.tensor(
            [[1.0, 3.0, 40.0, 1.0, 0.95, 1.0, 0.7, 0.95]] * len(pairs),
            dtype=torch.float32,
        ).reshape(len(pairs), 8),
        "pair_labels": pair_labels,
        "pair_mask": torch.ones(len(pairs), dtype=torch.bool),
        "pair_overlap_centerline_pixels": centerlines_tensor,
        "raw_keep_labels": torch.ones(count, dtype=torch.long),
        "raw_target_line_ids": target_tensor.clone(),
        "raw_group_ids": torch.tensor(group_ids, dtype=torch.long),
        "raw_group_types": torch.tensor(group_types, dtype=torch.long),
    }


def geometry_record(*, matched: int, expected: int, exact: bool) -> dict:
    tolerances = {}
    for name in ("q8", "q16", "q32"):
        tolerances[name] = {
            "ordinary_members": {"matched": matched, "expected": expected},
            "groups": {
                "single": {"matched": matched, "expected": expected},
                "p2": {"matched": matched, "expected": expected},
                "p3": {"matched": 0, "expected": 0},
            },
            "long_groups": {"matched": matched, "expected": expected},
            "structural_sample_eligible": expected > 0,
            "structural_sample_exact": exact,
        }
    return {"tolerances": tolerances}


class FakePairModel:
    def eval(self):
        return self

    def __call__(self, pair_features, pair_mask):
        self.inputs = (pair_features, pair_mask)
        return {"pair_gains": torch.full(pair_mask.shape, 2.0, device=pair_mask.device)}


class FakeP0Model:
    def eval(self):
        return self

    def __call__(self, pixel_values, carrier_segments, carrier_mask, **kwargs):
        self.inputs = (pixel_values, carrier_segments, carrier_mask, kwargs)
        return {
            "carrier_segments": carrier_segments,
            "carrier_mask": carrier_mask,
        }


class EvidenceCarrierPairExtentEvaluatorTest(unittest.TestCase):
    def test_packing_uses_pair_centerline_and_retains_unmatched_anchor(self) -> None:
        target = pair_target([0, 0, 1], [(0, 1)])
        evaluated = evaluate_evidence_carrier_pair_sample(
            target,
            [2.0],
            threshold=1.0,
        )

        packed = pack_selected_pair_carriers(target, evaluated)

        self.assertEqual(packed["source_anchor_groups"], [[0, 1], [2]])
        torch.testing.assert_close(
            packed["carrier_segments_pixels"][0],
            target["pair_overlap_centerline_pixels"][0],
        )
        torch.testing.assert_close(
            packed["carrier_segments_pixels"][1],
            target["anchor_segments_pixels"][2],
        )
        self.assertEqual(packed["target_line_id"].tolist(), [0, 1])
        self.assertEqual(packed["geometry_mask"].tolist(), [True, True])

    def test_cross_target_centerline_receives_no_geometry_credit(self) -> None:
        target = pair_target(
            [0, 1],
            [(0, 1)],
            group_ids=[5, 5],
            group_types=[PARALLEL_2, PARALLEL_2],
        )
        evaluated = evaluate_evidence_carrier_pair_sample(target, [2.0])

        packed = pack_selected_pair_carriers(target, evaluated)

        self.assertEqual(packed["target_line_id"].tolist(), [-1])
        self.assertEqual(packed["geometry_mask"].tolist(), [False])

    def test_sample_runs_one_target_build_and_one_batched_p0_forward(self) -> None:
        primitives, layers = sidecars()
        target = pair_target([0, 0], [(0, 1)])
        target["anchor_segments_pixels"][:] = torch.tensor(
            ((10.0, 50.0), (90.0, 50.0))
        )
        target["pair_overlap_centerline_pixels"][:] = torch.tensor(
            ((10.0, 50.0), (90.0, 50.0))
        )
        sample = {
            "key": "validation-1",
            "meta": {
                "sample_key": "validation-1",
                "molecule_id": "mol-1",
                "source_pilot": "pilot",
                "source_split": "validation",
                "source_sample_key": "validation_1_original",
                "source_variant": "original",
                "split": "validation",
                "variant": "original",
            },
            "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
            "primitives": primitives,
            "layers": layers,
        }
        pair_model = FakePairModel()
        p0_model = FakeP0Model()

        def decode(output, *, keep_threshold):
            self.assertEqual(keep_threshold, 0.0)
            scale = output["carrier_segments"].new_tensor((WIDTH - 1, HEIGHT - 1))
            return {
                "refined_segments_pixels": output["carrier_segments"] * scale,
            }

        with patch(
            "chemvector.evaluate_evidence_carrier_pair_extent."
            "build_conditional_ordinary_pair_target",
            return_value=target,
        ) as builder, patch(
            "chemvector.evaluate_evidence_carrier_pair_extent."
            "decode_evidence_carrier_extent",
            side_effect=decode,
        ):
            result = evaluate_pair_extent_sample(
                pair_model,
                p0_model,
                sample,
                pair_threshold=1.0,
                pairing_config=object(),
                target_config=EvidenceCarrierExtentTargetConfig(),
                device=torch.device("cpu"),
                expected_image_size=None,
            )

        builder.assert_called_once()
        self.assertEqual(result["lsd_association_passes"], 1)
        self.assertEqual(p0_model.inputs[0].shape[0], 2)
        self.assertEqual(p0_model.inputs[3]["coordinate_mode"], "normalized")
        self.assertEqual(
            result["packed_geometry"]["tolerances"]["q8"]["ordinary_members"],
            {"matched": 1, "expected": 1},
        )

    def test_summary_reports_pair_metrics_and_q8_reference_gate(self) -> None:
        target = pair_target(
            [0, 1],
            [(0, 1)],
            group_ids=[5, 5],
            group_types=[PARALLEL_2, PARALLEL_2],
        )
        pair_result = evaluate_evidence_carrier_pair_sample(target, [2.0])
        passed = summarize_pair_extent_validation(
            [
                {
                    "pair_matching": pair_result,
                    "baseline_geometry": geometry_record(
                        matched=1000, expected=1000, exact=True
                    ),
                    "packed_geometry": geometry_record(
                        matched=1000, expected=1000, exact=True
                    ),
                    "lsd_association_passes": 1,
                }
            ]
        )

        self.assertEqual(passed["gate"]["status"], "passed")
        self.assertEqual(passed["pair_matching"]["selected_cross_target_pairs"], 1)
        self.assertEqual(
            passed["pair_matching"]["parallel_2_distinct_preservation"], 0.0
        )
        self.assertEqual(passed["pair_matching"]["ordinary_support_loss_pp"], 100.0)
        self.assertEqual(
            passed["packed_metrics"]["single_q8_group_coverage"], 1.0
        )
        self.assertEqual(
            passed["q8_reference_comparison"]["reference_recall"],
            P0_REFERENCE_Q8_ORDINARY_RECALL,
        )

        failed = summarize_pair_extent_validation(
            [
                {
                    "pair_matching": pair_result,
                    "baseline_geometry": geometry_record(
                        matched=981, expected=1000, exact=False
                    ),
                    "packed_geometry": geometry_record(
                        matched=981, expected=1000, exact=False
                    ),
                    "lsd_association_passes": 1,
                }
            ]
        )
        self.assertEqual(failed["gate"]["status"], "failed")
        self.assertGreater(
            failed["q8_reference_comparison"]["regression_percentage_points"],
            0.2,
        )

    def test_cli_has_no_validation_threshold_override(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("gain_threshold", destinations)
        self.assertNotIn("threshold", destinations)
        self.assertNotIn("calibration", destinations)


if __name__ == "__main__":
    unittest.main()
