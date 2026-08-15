from __future__ import annotations

import json
import unittest
from pathlib import Path

from .ownership_realization import (
    OwnershipRealizationError,
    _partition_is_locally_admissible,
    analyze_configuration,
    build_matrix,
    canonical_bytes,
    representative_witness,
)


ROOT = Path(__file__).resolve().parent


def _decode_runs(runs: list[list[int]]) -> set[int]:
    return {
        row * 768 + column
        for row, start, length in runs
        for column in range(start, start + length)
    }


class OwnershipRealizationTests(unittest.TestCase):
    def test_edge_width_five_realizes_873_partitions(self) -> None:
        result = analyze_configuration("edge", 5, "round", "box_4x4")
        self.assertTrue(result["support_union_exact"])
        self.assertEqual(result["triple_overlap_pixel_count"], 10)
        self.assertEqual(result["locally_admissible_partition_count"], 873)
        self.assertEqual(result["unowned_overlap_pixel_count"], 0)

    def test_center_width_five_realizes_32_partitions(self) -> None:
        result = analyze_configuration("center", 5, "round", "box_4x4")
        self.assertTrue(result["support_union_exact"])
        self.assertEqual(result["triple_overlap_pixel_count"], 5)
        self.assertEqual(result["locally_admissible_partition_count"], 32)
        self.assertEqual(result["unowned_overlap_pixel_count"], 0)

    def test_additional_frozen_partition_counts(self) -> None:
        square = analyze_configuration("edge", 5, "square", "box_4x4")
        minimal = analyze_configuration("center", 4, "butt", "box_4x4")
        self.assertEqual(square["locally_admissible_partition_count"], 2415)
        self.assertEqual(minimal["locally_admissible_partition_count"], 2)

    def test_configuration_outside_preregistration_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            OwnershipRealizationError, "CONFIGURATION_OUTSIDE_PREREGISTRATION"
        ):
            analyze_configuration("edge", 7, "round", "box_4x4")

    def test_frozen_matrix_replays_byte_exact(self) -> None:
        matrix = build_matrix()
        self.assertEqual(
            ROOT.joinpath("results.json").read_bytes(),
            canonical_bytes(matrix),
        )
        over_capacity = [
            item for item in matrix["configurations"]
            if item["enumeration_status"] == "OVER_CAPACITY"
        ]
        self.assertEqual(len(over_capacity), 3)
        self.assertTrue(all(
            item["locally_admissible_partition_count"] is None
            for item in over_capacity
        ))

    def test_representative_witness_and_png_replay_byte_exact(self) -> None:
        witness, png = representative_witness()
        self.assertEqual(
            ROOT.joinpath("representative_witness.json").read_bytes(),
            canonical_bytes(witness),
        )
        self.assertEqual(
            ROOT.joinpath("representative_edge_w5_round_box4.png").read_bytes(), png
        )
        parsed = json.loads(ROOT.joinpath("representative_witness.json").read_bytes())
        configuration = parsed["configuration"]
        annotations = configuration["partition_annotations"]
        self.assertEqual(len(annotations), 2)
        self.assertNotEqual(
            annotations[0], annotations[1],
        )
        carrier_supports = [
            _decode_runs(item["support_runs"])
            for item in configuration["carrier_supports"]
        ]
        universe = _decode_runs(
            configuration["ownership_sets"]["ownership_universe_runs"]
        )
        fibers = {
            pixel: tuple(
                index for index, support in enumerate(carrier_supports)
                if pixel in support
            )
            for pixel in set().union(*carrier_supports)
        }
        seeds = 20 * 768 + 18, 20 * 768 + 22
        for annotation in annotations:
            self.assertTrue(_partition_is_locally_admissible(
                universe,
                _decode_runs(annotation["left_region_runs"]),
                _decode_runs(annotation["right_region_runs"]),
                seeds,
                fibers,
            ))


if __name__ == "__main__":
    unittest.main()
