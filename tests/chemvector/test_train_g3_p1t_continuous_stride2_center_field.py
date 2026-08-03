"""Tests for the P1-T continuous stride-2 trainer protocol."""

from __future__ import annotations

import unittest

from chemvector.core.model import ChemVectorConfig, ChemVisionEncoder
from chemvector.training.atoms.train_g3_p1t_continuous_stride2_center_field import (
    TRAINABLE_PARAMETER_COUNT,
    VISION_BLOCK_PARAMETER_COUNT,
    capacity_gate,
    trainable_vision_block,
)


class P1TContinuousStride2TrainerTests(unittest.TestCase):
    def test_last_stride4_block_has_fixed_capacity(self) -> None:
        vision = ChemVisionEncoder(ChemVectorConfig())
        block = trainable_vision_block(vision)
        self.assertEqual(
            sum(value.numel() for value in block.parameters()),
            VISION_BLOCK_PARAMETER_COUNT,
        )
        self.assertEqual(TRAINABLE_PARAMETER_COUNT, 92_035)

    def test_capacity_gate_is_strict(self) -> None:
        metrics = {
            "known_full_count": {
                "matched_atoms": 1_424,
                "target_atoms": 1_424,
                "whole_set_exact_images": 32,
            },
            "top128_reachability": {
                "matched_targets": 1_424,
                "targets": 1_424,
                "exact_images": 32,
            },
            "regional_maxima": {"peak_shortfall_images": 0},
        }
        self.assertEqual(capacity_gate(metrics)[0], "overfit_capacity_go")
        metrics["known_full_count"]["whole_set_exact_images"] = 31
        self.assertEqual(capacity_gate(metrics)[0], "continuous_stride2_capacity_no_go")


if __name__ == "__main__":
    unittest.main()
