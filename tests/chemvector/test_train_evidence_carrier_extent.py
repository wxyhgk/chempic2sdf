from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from chemvector.audit_evidence_carrier_proposals import TargetLine
from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentConfig,
    EvidenceCarrierExtentPredictor,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
)
from chemvector.train_evidence_carrier_extent import (
    BEST_CHECKPOINT_METRIC_ORDER,
    CHECKPOINT_SCHEMA_VERSION,
    P0_EXTENT_BINS,
    P0_NORMAL_OFFSET_WEIGHT,
    assert_unscaled_full_raster_batch,
    assert_extent_config_contract,
    best_equivalent_line_predictions,
    compute_p0_loss,
    evidence_carrier_extent_config,
    evidence_carrier_target_config,
    evaluate_sample_geometry,
    expand_shards,
    freeze_p0_keep_head,
    load_checkpoint,
    reject_internal_test_paths,
    reject_train_validation_overlap,
    save_checkpoint,
    summarize_geometry_records,
    validation_checkpoint_key,
    validate_checkpoint_payload,
)


class EvidenceCarrierExtentTrainerTest(unittest.TestCase):
    def test_fixed_model_and_target_bin_contract(self) -> None:
        full = evidence_carrier_extent_config()
        tiny = evidence_carrier_extent_config(tiny=True)
        target = evidence_carrier_target_config()

        for model_config in (full, tiny):
            assert_extent_config_contract(model_config, target)
            self.assertEqual(model_config.extent_bins, P0_EXTENT_BINS)
            self.assertEqual(model_config.normal_offset_bins, 17)
            self.assertEqual(model_config.normal_offset_radius_pixels, 4.0)
            self.assertEqual(model_config.normal_offset_step_pixels, 0.5)
        self.assertEqual(target.extent_bins, P0_EXTENT_BINS)
        self.assertEqual(target.normal_offset_bins, 17)

        with self.assertRaisesRegex(ValueError, "extent_bins"):
            assert_extent_config_contract(
                replace(tiny, extent_bins=P0_EXTENT_BINS - 1), target
            )
        with self.assertRaisesRegex(ValueError, "normal_offset_bins"):
            assert_extent_config_contract(
                tiny, replace(target, normal_offset_bins=19)
            )
        with self.assertRaisesRegex(ValueError, "normal_offset_range"):
            assert_extent_config_contract(
                tiny,
                replace(
                    target,
                    normal_offset_limit_pixels=5.0,
                    positive_perpendicular_pixels=3.0,
                ),
            )

    def test_internal_test_and_resolved_overlap_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.tar"
            train.write_bytes(b"train")
            validation_alias = root / "validation.tar"
            validation_alias.symlink_to(train)
            with self.assertRaisesRegex(ValueError, "overlap"):
                reject_train_validation_overlap([train], [validation_alias])

            sealed_dir = root / "internal-test"
            sealed_dir.mkdir()
            sealed = sealed_dir / "sealed.tar"
            sealed.write_bytes(b"sealed")
            innocent_alias = root / "public-validation.tar"
            innocent_alias.symlink_to(sealed)
            with self.assertRaisesRegex(ValueError, "internal_test"):
                reject_internal_test_paths([innocent_alias])
            with self.assertRaisesRegex(ValueError, "internal_test"):
                expand_shards([str(innocent_alias)])

    def test_full_raster_resize_is_rejected_before_forward(self) -> None:
        batch = {
            "pixel_values": torch.ones((2, 3, 32, 32)),
            "layers": [
                {"width": 32, "height": 32},
                {"width": 32, "height": 32},
            ],
        }
        assert_unscaled_full_raster_batch(batch)
        batch["layers"][1]["width"] = 768
        batch["layers"][1]["height"] = 768
        with self.assertRaisesRegex(ValueError, "without resize"):
            assert_unscaled_full_raster_batch(batch)

    def test_equivalent_carriers_take_one_best_prediction_per_target_line(self) -> None:
        q_mid_x = 50.0 / 99.0 * 2047.0
        target_lines = [
            TargetLine(
                group_index=0,
                group_kind="single_line",
                points_q=((0.0, 0.0), (2047.0, 0.0)),
                long_group=False,
                text_adjacent=False,
            ),
            TargetLine(
                group_index=1,
                group_kind="single_line",
                points_q=((q_mid_x, 0.0), (q_mid_x, 2047.0)),
                long_group=False,
                text_adjacent=False,
            ),
        ]
        predicted = np.asarray(
            [
                [[0.0, 10.0], [99.0, 10.0]],
                [[0.0, 0.0], [99.0, 0.0]],
                [[50.0, 0.0], [50.0, 99.0]],
                [[0.0, 30.0], [99.0, 30.0]],
            ],
            dtype=np.float32,
        )

        best = best_equivalent_line_predictions(
            predicted,
            torch.tensor([0, 0, 1, -1]),
            torch.tensor([True, True, True, False]),
            target_lines,
            width=100,
            height=100,
        )

        self.assertEqual(set(best), {0, 1})
        self.assertEqual(best[0]["carrier_index"], 1)
        self.assertEqual(best[1]["carrier_index"], 2)
        self.assertAlmostEqual(float(best[0]["error_q"]), 0.0, places=5)
        self.assertAlmostEqual(float(best[1]["error_q"]), 0.0, places=5)

    def test_empty_target_sample_is_excluded_from_structural_exact(self) -> None:
        record = evaluate_sample_geometry(
            np.empty((0, 2, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=bool),
            {},
            {"width": 32, "height": 32, "atoms": [], "bonds": []},
            target_config=evidence_carrier_target_config(),
            tolerances_q=(8.0,),
        )

        self.assertFalse(record["tolerances"]["q8"]["structural_sample_eligible"])
        self.assertFalse(record["tolerances"]["q8"]["structural_sample_exact"])
        summary = summarize_geometry_records([record], tolerances_q=(8.0,))
        structural = summary["q8"]["structural_samples"]
        self.assertEqual(structural["exact"], 0)
        self.assertEqual(structural["expected"], 0)
        self.assertEqual(structural["excluded_no_ordinary_target"], 1)
        self.assertEqual(structural["total"], 1)
        self.assertIsNone(structural["exact_rate"])

    def test_structural_exact_denominator_uses_only_eligible_samples(self) -> None:
        empty = {
            "tolerances": {
                "q8": {
                    "ordinary_members": {"matched": 0, "expected": 0},
                    "groups": {
                        "single": {"matched": 0, "expected": 0},
                        "p2": {"matched": 0, "expected": 0},
                        "p3": {"matched": 0, "expected": 0},
                    },
                    "long_groups": {"matched": 0, "expected": 0},
                    "structural_sample_eligible": False,
                    "structural_sample_exact": False,
                }
            }
        }
        exact = {
            "tolerances": {
                "q8": {
                    "ordinary_members": {"matched": 1, "expected": 1},
                    "groups": {
                        "single": {"matched": 1, "expected": 1},
                        "p2": {"matched": 0, "expected": 0},
                        "p3": {"matched": 0, "expected": 0},
                    },
                    "long_groups": {"matched": 0, "expected": 0},
                    "structural_sample_eligible": True,
                    "structural_sample_exact": True,
                }
            }
        }

        structural = summarize_geometry_records(
            [empty, exact], tolerances_q=(8.0,)
        )["q8"]["structural_samples"]
        self.assertEqual(structural["exact"], 1)
        self.assertEqual(structural["expected"], 1)
        self.assertEqual(structural["excluded_no_ordinary_target"], 1)
        self.assertEqual(structural["exact_rate"], 1.0)

    def test_best_checkpoint_uses_route_metric_lexicographic_order(self) -> None:
        self.assertEqual(
            BEST_CHECKPOINT_METRIC_ORDER,
            (
                ("ordinary_q8_member_recall", "max"),
                ("structural_q8_sample_exact_rate", "max"),
                ("long_q8_group_coverage", "max"),
                ("loss", "min"),
            ),
        )

        def metrics(
            recall: float,
            structural: float,
            long_coverage: float,
            loss: float,
        ) -> dict[str, float]:
            return {
                "ordinary_q8_member_recall": recall,
                "structural_q8_sample_exact_rate": structural,
                "long_q8_group_coverage": long_coverage,
                "loss": loss,
            }

        baseline = metrics(0.98, 0.80, 0.95, 1.0)
        self.assertGreater(
            validation_checkpoint_key(metrics(0.981, 0.10, 0.10, 10.0)),
            validation_checkpoint_key(baseline),
        )
        self.assertGreater(
            validation_checkpoint_key(metrics(0.98, 0.81, 0.10, 10.0)),
            validation_checkpoint_key(baseline),
        )
        self.assertGreater(
            validation_checkpoint_key(metrics(0.98, 0.80, 0.96, 10.0)),
            validation_checkpoint_key(baseline),
        )
        self.assertGreater(
            validation_checkpoint_key(metrics(0.98, 0.80, 0.95, 0.9)),
            validation_checkpoint_key(baseline),
        )

    def test_checkpoint_roundtrip_preserves_protocol_and_weights(self) -> None:
        torch.manual_seed(7)
        model = EvidenceCarrierExtentPredictor(
            evidence_carrier_extent_config(tiny=True)
        )
        freeze_p0_keep_head(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        target_config = evidence_carrier_target_config()
        args = argparse.Namespace(
            train_shards=["train.tar"],
            validation_shards=["validation.tar"],
            tiny=True,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-000001.pt"
            digest = save_checkpoint(
                path,
                model,
                optimizer,
                target_config=target_config,
                global_step=1,
                samples_seen=2,
                args=args,
                validation={"loss": 1.25},
            )
            loaded, loaded_target, checkpoint = load_checkpoint(path)

            self.assertEqual(len(digest), 64)
            self.assertEqual(
                checkpoint["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION
            )
            self.assertIs(checkpoint["uses_internal_test"], False)
            self.assertEqual(checkpoint["loss_protocol"]["keep_weight"], 0.0)
            self.assertEqual(
                checkpoint["loss_protocol"]["normal_offset_weight"], 0.25
            )
            self.assertEqual(
                checkpoint["loss_protocol"]["equivalence_weight"],
                "target_line_id_balanced",
            )
            self.assertEqual(loaded_target, target_config)
            self.assertFalse(loaded.keep_head.weight.requires_grad)
            for name, expected in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], expected)

            checkpoint["training_args"]["validation_shards"] = [
                "shards/internal-test-000000.tar"
            ]
            with self.assertRaisesRegex(ValueError, "internal_test"):
                validate_checkpoint_payload(checkpoint)

    def test_cpu_tiny_forward_backward_updates_geometry_not_keep_head(self) -> None:
        torch.manual_seed(19)
        config = evidence_carrier_extent_config(tiny=True)
        target_config = evidence_carrier_target_config()
        assert_extent_config_contract(config, target_config)
        model = EvidenceCarrierExtentPredictor(config)
        freeze_p0_keep_head(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-2,
            weight_decay=0.0,
        )
        pixels = torch.ones((1, 3, 32, 32), dtype=torch.float32)
        pixels[:, :, 15:17, 3:29] = 0.0
        segments = torch.tensor(
            [[[[3.0 / 31.0, 15.0 / 31.0], [28.0 / 31.0, 15.0 / 31.0]]]],
            dtype=torch.float32,
        )
        mask = torch.ones((1, 1), dtype=torch.bool)
        tensors = {
            "keep_labels": torch.ones((1, 1), dtype=torch.long),
            "keep_loss_mask": mask.clone(),
            "equivalence_weight": torch.ones((1, 1)),
            "normal_offset_bin": torch.tensor([[8]]),
            "extent_left_bin": torch.tensor([[32]]),
            "extent_right_bin": torch.tensor([[352]]),
            "geometry_loss_mask": mask.clone(),
        }
        keep_before = model.keep_head.weight.detach().clone()
        normal_before = model.normal_offset_head.weight.detach().clone()

        output = model(pixels, segments, mask)
        losses = compute_p0_loss(output, tensors)
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertEqual(float(losses["geometry_supervision_weight"]), 1.0)
        self.assertEqual(P0_NORMAL_OFFSET_WEIGHT, 0.25)
        torch.testing.assert_close(
            losses["loss"],
            P0_NORMAL_OFFSET_WEIGHT * losses["normal_offset_loss"]
            + losses["extent_loss"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        torch.testing.assert_close(model.keep_head.weight, keep_before)
        self.assertGreater(
            float(
                (model.normal_offset_head.weight - normal_before)
                .detach()
                .abs()
                .sum()
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
