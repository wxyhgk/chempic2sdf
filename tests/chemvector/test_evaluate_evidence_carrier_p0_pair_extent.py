from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image
import torch

from chemvector.bond_group_set import SINGLE_LINE
from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalPairSampleIdentity,
)
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.evidence_carrier_pair_evaluation import (
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evaluate_evidence_carrier_p0_pair_extent import (
    MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS,
    build_parser,
    evaluate_p0_pair_extent_sample,
    summarize_p0_pair_extent_validation,
)


WIDTH = HEIGHT = 100


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


def target() -> dict[str, torch.Tensor]:
    anchors = torch.tensor(
        [
            ((10.0, 50.0), (90.0, 50.0)),
            ((10.0, 54.0), (90.0, 54.0)),
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([4, 4], dtype=torch.long)
    return {
        "anchor_segments": anchors / 99.0,
        "anchor_segments_pixels": anchors,
        "anchor_source_indices": torch.tensor([0, 1], dtype=torch.long),
        "anchor_target_line_ids": target_ids,
        "anchor_mask": torch.ones(2, dtype=torch.bool),
        "pair_indices": torch.tensor([[0, 1]], dtype=torch.long),
        "pair_features": torch.tensor(
            [[1.0, 4.0, 80.0, 1.0, 0.95, 1.0, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "pair_labels": torch.ones(1, dtype=torch.long),
        "pair_mask": torch.ones(1, dtype=torch.bool),
        "pair_overlap_centerline_pixels": torch.tensor(
            [[[10.0, 52.0], [90.0, 52.0]]], dtype=torch.float32
        ),
        "raw_keep_labels": torch.ones(2, dtype=torch.long),
        "raw_target_line_ids": target_ids.clone(),
        "raw_group_ids": torch.tensor([7, 7], dtype=torch.long),
        "raw_group_types": torch.tensor(
            [SINGLE_LINE, SINGLE_LINE], dtype=torch.long
        ),
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


def sample_record(*, baseline_matched: int, packed_matched: int) -> dict:
    pair_matching = evaluate_evidence_carrier_pair_sample(
        target(), torch.tensor([2.0]), threshold=1.0
    )
    return {
        "pair_matching": pair_matching,
        "baseline_geometry": geometry_record(
            matched=baseline_matched, expected=1000, exact=baseline_matched == 1000
        ),
        "packed_geometry": geometry_record(
            matched=packed_matched, expected=1000, exact=packed_matched == 1000
        ),
        "target_builds": 1,
        "p0_pair_evidence_forward_passes": 1,
        "p0_baseline_packed_forward_passes": 1,
        "baseline_and_packed_share_forward": True,
    }


class EvaluateFrozenP0PairExtentTest(unittest.TestCase):
    def test_summary_compares_packed_q8_to_live_same_batch_baseline(self) -> None:
        summary = summarize_p0_pair_extent_validation(
            [sample_record(baseline_matched=1000, packed_matched=998)]
        )

        comparison = summary["q8_live_baseline_comparison"]
        self.assertEqual(comparison["baseline_recall"], 1.0)
        self.assertEqual(comparison["packed_recall"], 0.998)
        self.assertAlmostEqual(comparison["regression_percentage_points"], 0.2)
        self.assertFalse(comparison["hardcoded_reference_used"])
        self.assertEqual(
            comparison["maximum_allowed_regression_percentage_points"],
            MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS,
        )
        self.assertEqual(summary["extent_gate"]["status"], "GO")

    def test_summary_fails_when_live_baseline_regression_exceeds_point_two_pp(self) -> None:
        summary = summarize_p0_pair_extent_validation(
            [sample_record(baseline_matched=1000, packed_matched=997)]
        )

        self.assertEqual(summary["extent_gate"]["status"], "NO-GO")
        self.assertGreater(
            summary["q8_live_baseline_comparison"]["regression_percentage_points"],
            0.2,
        )

    def test_sample_uses_one_target_and_one_shared_baseline_packed_forward(self) -> None:
        current_target = target()
        pair_evaluation = evaluate_evidence_carrier_pair_sample(
            current_target, torch.tensor([2.0]), threshold=1.0
        )
        current_identity = identity()
        sample = {
            "identity": current_identity,
            "image": Image.new("RGB", (WIDTH, HEIGHT), "white"),
            "primitives": {},
            "layers": {"width": WIDTH, "height": HEIGHT},
        }
        baseline = current_target["anchor_segments_pixels"].clone()
        packed = torch.tensor([[[10.0, 52.0], [90.0, 52.0]]])
        with patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_extent."
            "evaluate_frozen_p0_pair_sample",
            return_value={
                "identity": current_identity,
                "target": current_target,
                "pair_evaluation": pair_evaluation,
                "target_builds": 1,
                "anchor_count": 2,
                "pair_count": 1,
                "p0_forward_passes": 1,
            },
        ) as pair_stage, patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_extent._native_image_tensor",
            return_value=torch.zeros((3, HEIGHT, WIDTH)),
        ), patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_extent."
            "_decode_baseline_and_packed",
            return_value=(baseline, packed),
        ) as decoder, patch(
            "chemvector.evaluate_evidence_carrier_p0_pair_extent."
            "evaluate_sample_geometry",
            side_effect=[
                geometry_record(matched=2, expected=2, exact=True),
                geometry_record(matched=1, expected=1, exact=True),
            ],
        ):
            result = evaluate_p0_pair_extent_sample(
                object(),
                object(),
                sample,
                pair_threshold=1.0,
                pairing_config=object(),
                target_config=EvidenceCarrierExtentTargetConfig(),
                device=torch.device("cpu"),
                expected_image_size=100,
            )

        pair_stage.assert_called_once()
        decoder.assert_called_once()
        self.assertEqual(result["target_builds"], 1)
        self.assertEqual(result["p0_pair_evidence_forward_passes"], 1)
        self.assertEqual(result["p0_baseline_packed_forward_passes"], 1)
        self.assertTrue(result["baseline_and_packed_share_forward"])
        self.assertEqual(result["packed_carrier_count"], 1)

    def test_cli_has_no_threshold_or_p0_checkpoint_override(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("threshold", destinations)
        self.assertNotIn("gain_threshold", destinations)
        self.assertNotIn("p0_checkpoint", destinations)
        self.assertNotIn("reference_recall", destinations)


if __name__ == "__main__":
    unittest.main()
