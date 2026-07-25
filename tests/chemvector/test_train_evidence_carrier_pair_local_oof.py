import unittest

import torch

from chemvector.train_evidence_carrier_pair_local_oof import (
    PairLocalOOFTrainingError,
    collate_pair_local_records,
    merge_pair_local_records,
)


class PairLocalOOFTrainingTest(unittest.TestCase):
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
        }

    def test_merge_and_collate_preserve_uint8_crops(self) -> None:
        records = [self._record("a", 2), self._record("b", 1)]
        crops = {
            "a": torch.ones(2, 2, 24, 48, dtype=torch.uint8),
            "b": torch.full((1, 2, 24, 48), 7, dtype=torch.uint8),
        }
        merged = merge_pair_local_records(records, crops)
        batch = collate_pair_local_records(merged)
        self.assertEqual(batch["pair_local_rasters"].dtype, torch.uint8)
        self.assertEqual(batch["pair_local_rasters"].shape, (2, 2, 2, 24, 48))
        self.assertEqual(batch["pair_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(int(batch["pair_local_rasters"][1, 0, 0, 0, 0]), 7)

    def test_merge_rejects_missing_and_extra_samples(self) -> None:
        records = [self._record("a", 1)]
        with self.assertRaisesRegex(PairLocalOOFTrainingError, "lacks"):
            merge_pair_local_records(records, {})
        crops = {
            "a": torch.zeros(1, 2, 24, 48, dtype=torch.uint8),
            "b": torch.zeros(1, 2, 24, 48, dtype=torch.uint8),
        }
        with self.assertRaisesRegex(PairLocalOOFTrainingError, "unmatched"):
            merge_pair_local_records(records, crops)


if __name__ == "__main__":
    unittest.main()
