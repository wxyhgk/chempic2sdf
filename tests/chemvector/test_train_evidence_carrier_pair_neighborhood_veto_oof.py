import unittest

import torch

from chemvector.train_evidence_carrier_pair_neighborhood_veto_oof import (
    PairNeighborhoodVetoOOFTrainingError,
    collate_neighborhood_records,
    merge_neighborhood_records,
)


class PairNeighborhoodVetoOOFTrainingTest(unittest.TestCase):
    def _record(self, key: str, count: int):
        return {
            "identity": {"sample_key": key},
            "target_free": {
                "pair_tokens": torch.zeros(count, 21),
                "pair_indices": torch.tensor(
                    [[index, index + 1] for index in range(count)], dtype=torch.long
                ).reshape(count, 2),
                "global_count_features": torch.zeros(4),
            },
            "target_aware": {"pair_labels": torch.arange(count).remainder(2)},
            "pair_local_rasters": torch.zeros(
                count, 2, 24, 48, dtype=torch.uint8
            ),
        }

    def test_merge_and_collate_preserve_neighbor_order(self) -> None:
        records = [self._record("a", 2), self._record("b", 1)]
        a_features = torch.zeros(2, 12, 8, dtype=torch.float16)
        a_features[0, 0, 0] = -0.5
        a_features[0, 1, 0] = 0.5
        neighborhoods = {
            "a": (
                a_features,
                torch.tensor(
                    [[True, True] + [False] * 10, [False] * 12],
                    dtype=torch.bool,
                ),
            ),
            "b": (
                torch.ones(1, 12, 8, dtype=torch.float16),
                torch.ones(1, 12, dtype=torch.bool),
            ),
        }
        merged = merge_neighborhood_records(records, neighborhoods)
        batch = collate_neighborhood_records(merged)
        self.assertEqual(batch["neighbor_features"].dtype, torch.float16)
        self.assertEqual(batch["neighbor_features"].shape, (2, 2, 12, 8))
        self.assertEqual(batch["neighbor_mask"][0, 0, :3].tolist(), [True, True, False])
        self.assertEqual(float(batch["neighbor_features"][0, 0, 0, 0]), -0.5)
        self.assertEqual(float(batch["neighbor_features"][0, 0, 1, 0]), 0.5)

    def test_merge_rejects_missing_samples(self) -> None:
        with self.assertRaisesRegex(PairNeighborhoodVetoOOFTrainingError, "lacks"):
            merge_neighborhood_records([self._record("a", 1)], {})


if __name__ == "__main__":
    unittest.main()
