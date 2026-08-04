"""Protocol tests for train5k continuous center-field fine-tuning."""

from __future__ import annotations

import unittest

from chemvector.training.atoms.train_g3_p1z_continuous_center_field_train5k import (
    SCALE_PROTOCOLS,
    build_parser,
    gate,
    validate_args,
)


def metrics(matched: int, exact: int) -> dict:
    return {
        "top128_reachability": {
            "matched_targets": matched,
            "targets": 3_080,
            "atom_recall": matched / 3_080,
            "exact_images": exact,
        }
    }


class G3P1ZContinuousCenterFieldTrain5kTests(unittest.TestCase):
    def test_scale_protocols_use_distinct_model_outputs(self) -> None:
        self.assertNotEqual(
            SCALE_PROTOCOLS["stride2"]["model_type"],
            SCALE_PROTOCOLS["stride1"]["model_type"],
        )
        self.assertNotEqual(
            SCALE_PROTOCOLS["stride2"]["output"],
            SCALE_PROTOCOLS["stride1"]["output"],
        )

    def test_cli_is_fixed_except_scale(self) -> None:
        args = build_parser().parse_args(["--scale", "stride2"])
        protocol = validate_args(args)
        self.assertEqual(protocol["model_type"], SCALE_PROTOCOLS["stride2"]["model_type"])
        with self.assertRaisesRegex(ValueError, "fixed batch_size"):
            validate_args(
                build_parser().parse_args(["--scale", "stride1", "--batch-size", "2"])
            )

    def test_gate_requires_independent_step0_and_two_route_gains(self) -> None:
        protocol = SCALE_PROTOCOLS["stride2"]
        decision, checks = gate(protocol, metrics(2_989, 49), metrics(3_030, 60))
        self.assertEqual(decision, "route_causal_go")
        self.assertTrue(all(checks.values()))
        decision, _ = gate(protocol, metrics(2_989, 49), metrics(3_000, 60))
        self.assertEqual(decision, "route_causal_no_go")


if __name__ == "__main__":
    unittest.main()
