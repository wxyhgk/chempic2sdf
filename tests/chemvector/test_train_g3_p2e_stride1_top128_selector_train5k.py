"""Tests for the P2-E selector train5k protocol."""

import unittest

from chemvector.training.atoms.train_g3_p2e_stride1_top128_selector_train5k import (
    STEP0_ROUTE_EXACT,
    STEP0_ROUTE_MATCHED,
    build_parser,
    gate,
    validate_args,
)


def result(matched: int, exact: int) -> dict:
    return {
        "full_known_count": {
            "matched_atoms": matched,
            "target_atoms": 3_080,
            "recall_6px": matched / 3_080,
            "whole_set_exact_images": exact,
        }
    }


class P2EStride1Top128SelectorTrain5kTests(unittest.TestCase):
    def test_cli_is_fixed(self) -> None:
        validate_args(build_parser().parse_args([]))
        with self.assertRaisesRegex(ValueError, "fixed batch_size"):
            validate_args(build_parser().parse_args(["--batch-size", "2"]))

    def test_gate_requires_causal_and_capability_thresholds(self) -> None:
        step0 = result(STEP0_ROUTE_MATCHED, STEP0_ROUTE_EXACT)
        decision, checks = gate(step0, result(3_060, 90))
        self.assertEqual(decision, "route_causal_go_capability_go")
        self.assertTrue(all(checks.values()))
        decision, _ = gate(step0, result(3_045, 90))
        self.assertEqual(decision, "route_causal_go_capability_no_go")


if __name__ == "__main__":
    unittest.main()
