import unittest

import torch

from chemvector.audit_evidence_carrier_pair_local_raster import (
    collect_changed_cross_pairs,
)


class PairLocalRasterAuditTest(unittest.TestCase):
    def test_collects_only_candidate_selected_negative_pairs(self) -> None:
        records = [
            {
                "identity": {"sample_key": "sample-a"},
                "target_free": {
                    "pair_indices": torch.tensor([[0, 1], [2, 3], [4, 5]]),
                    "pair_tokens": torch.tensor(
                        [[0.0] * 20 + [0.5], [0.0] * 20 + [0.6], [0.0] * 20 + [0.7]]
                    ),
                },
                "target_aware": {"pair_labels": torch.tensor([1, 0, 0])},
            }
        ]
        report = {
            "changed_samples": [
                {
                    "identity": {
                        "sample_key": "sample-a",
                        "render_variant": "stress",
                    },
                    "baseline": {"selected_candidate_indices": [0, 1]},
                    "candidate": {"selected_candidate_indices": [0, 2]},
                }
            ]
        }
        rows = collect_changed_cross_pairs(report, records, fold=2)
        self.assertEqual(
            rows,
            [
                {
                    "fold": 2,
                    "sample_key": "sample-a",
                    "render_variant": "stress",
                    "candidate_index": 2,
                    "anchor_pair": [4, 5],
                    "raw_gain": rows[0]["raw_gain"],
                    "introduced_by_residual": True,
                }
            ],
        )
        self.assertAlmostEqual(rows[0]["raw_gain"], 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
