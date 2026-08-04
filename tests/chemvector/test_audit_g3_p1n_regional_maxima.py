"""Tests for the G3 P1-N regional-maxima audit operator."""

from __future__ import annotations

import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    _decision,
    collapse_slots_to_cell_field,
    extract_regional_maximum_objects,
)


class G3P1NRegionalMaximaTests(unittest.TestCase):
    def test_cell_collapse_uses_best_slot_and_preserves_its_anchor(self) -> None:
        logits = torch.tensor(
            [[[[0.0, 3.0]], [[2.0, 1.0]], [[1.0, 0.0]]]],
            dtype=torch.float32,
        )
        anchors = torch.zeros(1, 3, 2, 1, 2)
        anchors[0, 1, :, 0, 0] = torch.tensor([0.2, 0.3])
        anchors[0, 0, :, 0, 1] = torch.tensor([0.7, 0.8])

        field = collapse_slots_to_cell_field(
            {"presence_logits": logits, "anchors": anchors}
        )

        torch.testing.assert_close(field["scores"], torch.tensor([[[2.0, 3.0]]]))
        torch.testing.assert_close(
            field["anchors"].permute(0, 2, 3, 1),
            torch.tensor([[[[0.2, 0.3], [0.7, 0.8]]]]),
        )
        torch.testing.assert_close(field["slot_ids"], torch.tensor([[[1, 0]]]))

    def test_equal_slot_tie_uses_lowest_slot(self) -> None:
        logits = torch.ones(1, 3, 1, 1)
        anchors = torch.zeros(1, 3, 2, 1, 1)
        anchors[0, 0, :, 0, 0] = torch.tensor([0.1, 0.2])
        anchors[0, 1, :, 0, 0] = torch.tensor([0.8, 0.9])

        field = collapse_slots_to_cell_field(
            {"presence_logits": logits, "anchors": anchors}
        )

        self.assertEqual(int(field["slot_ids"][0, 0, 0]), 0)
        torch.testing.assert_close(
            field["anchors"][0, :, 0, 0], torch.tensor([0.1, 0.2])
        )

    def test_plateau_becomes_one_object_with_deterministic_identity(self) -> None:
        scores = torch.tensor(
            [[[0.0, 2.0, 2.0], [0.0, 2.0, 2.0], [1.0, 0.0, 3.0]]],
            dtype=torch.float32,
        )
        anchors = torch.zeros(1, 2, 3, 3)
        for y in range(3):
            for x in range(3):
                anchors[0, :, y, x] = torch.tensor([x / 3.0, y / 3.0])
        source_ids = torch.arange(9).reshape(1, 3, 3)

        objects = extract_regional_maximum_objects(
            {"scores": scores, "anchors": anchors, "source_ids": source_ids}
        )[0]

        # The value-2 plateau touches the higher value 3 diagonally and is not a
        # regional maximum. Only the value-3 object remains.
        self.assertEqual(objects["regional_maxima_count"], 1)
        torch.testing.assert_close(objects["source_ids"], torch.tensor([8]))
        torch.testing.assert_close(
            objects["anchors"], torch.tensor([[2.0 / 3.0, 2.0 / 3.0]])
        )

    def test_never_backfills_when_requested_capacity_exceeds_peaks(self) -> None:
        scores = torch.ones(1, 2, 2, dtype=torch.float32)
        anchors = torch.zeros(1, 2, 2, 2)
        source_ids = torch.arange(4).reshape(1, 2, 2)

        objects = extract_regional_maximum_objects(
            {"scores": scores, "anchors": anchors, "source_ids": source_ids},
            maximum_objects=128,
        )[0]

        self.assertEqual(objects["regional_maxima_count"], 1)
        self.assertEqual(len(objects["anchors"]), 1)
        self.assertEqual(objects["plateau_sizes"], [4])

    def test_multiple_plateaus_use_row_major_representatives(self) -> None:
        scores = torch.tensor(
            [[[3.0, 3.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 4.0]]],
            dtype=torch.float32,
        )
        anchors = torch.zeros(1, 2, 3, 4)
        source_ids = torch.arange(12).reshape(1, 3, 4)

        objects = extract_regional_maximum_objects(
            {"scores": scores, "anchors": anchors, "source_ids": source_ids}
        )[0]

        self.assertEqual(objects["source_ids"].tolist(), [10, 0, 3])
        self.assertEqual(objects["plateau_sizes"], [2, 2, 1])
        self.assertEqual(sorted(objects["all_plateau_sizes"]), [1, 2, 2])

    def test_decision_uses_reachability_not_top128_count_exact(self) -> None:
        train = {
            "top128_reachability": {"atom_recall": 0.999, "exact_rate": 0.96},
            "modes": {
                "regional_top128_full": {"exact_rate_6px": 0.0},
                "regional_known_representable": {
                    "recall_6px": 0.99,
                    "exact_rate_6px": 0.75,
                },
                "component_known_representable": {
                    "recall_6px": 0.99,
                    "exact_rate_6px": 0.68,
                },
            },
            "regional_maxima": {"peak_shortfall_full_count_images": 0},
        }

        decision, checks = _decision(train)

        self.assertEqual(decision, "mechanism_go_requires_fresh_confirmation")
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
