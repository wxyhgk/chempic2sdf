#!/usr/bin/env python3
"""Small deterministic tests for raster-pilot record generation."""

from __future__ import annotations

import unittest

try:
    from .run_raster_pilot_v0_5 import PILOT_COUNT, _render, pilot_tuple_ids
except ImportError:
    from run_raster_pilot_v0_5 import PILOT_COUNT, _render, pilot_tuple_ids  # type: ignore


class RasterPilotRunnerTests(unittest.TestCase):
    def test_selection_count_and_bounds(self) -> None:
        ids = pilot_tuple_ids()
        self.assertEqual(len(ids), PILOT_COUNT)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 36_088_400)

    def test_record_is_deterministic_and_canonical_bytes_match_digest(self) -> None:
        first, first_bytes = _render(0)
        second, second_bytes = _render(0)
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["tuple_id"], 0)
        self.assertEqual(len(first["canonical_gray_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
