"""Tests for vertex threshold evaluation helpers."""

from __future__ import annotations

import argparse
import unittest

from chemvector.evaluate_vertex import (
    parse_positive_floats,
    parse_positive_ints,
    parse_thresholds,
)


class EvaluateVertexTests(unittest.TestCase):
    def test_threshold_parser_preserves_requested_order(self) -> None:
        self.assertEqual(parse_thresholds("0.2, 0.1,0.3"), (0.2, 0.1, 0.3))

    def test_threshold_parser_rejects_invalid_or_duplicate_values(self) -> None:
        for value in ("", "0", "1", "0.2,0.2"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_thresholds(value)

    def test_positive_grid_parsers(self) -> None:
        self.assertEqual(parse_positive_floats("4, 6.5,16"), (4.0, 6.5, 16.0))
        self.assertEqual(parse_positive_ints("64,128,192"), (64, 128, 192))
        for parser, value in (
            (parse_positive_floats, "0,4"),
            (parse_positive_floats, "4,4"),
            (parse_positive_ints, "0,128"),
            (parse_positive_ints, "128,128"),
        ):
            with self.subTest(parser=parser.__name__, value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parser(value)


if __name__ == "__main__":
    unittest.main()
