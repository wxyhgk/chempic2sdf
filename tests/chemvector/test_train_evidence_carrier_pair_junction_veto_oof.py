import unittest

import torch

from chemvector.train_evidence_carrier_pair_junction_veto_oof import (
    collate_junction_records,
)


class PairJunctionVetoOOFTrainingTest(unittest.TestCase):
    def test_collates_pair_group_types_without_model_input_changes(self) -> None:
        record = {
            "identity": {"sample_key": "a"},
            "target_free": {
                "pair_tokens": torch.zeros(2, 21),
                "pair_indices": torch.tensor([[0, 2], [1, 2]]),
                "global_count_features": torch.zeros(4),
            },
            "target_aware": {
                "pair_labels": torch.tensor([0, 1]),
                "evaluation_target": {
                    "anchor_group_types": torch.tensor([1, 2, 3])
                },
            },
            "pair_local_rasters": torch.zeros(2, 2, 24, 48, dtype=torch.uint8),
        }
        batch = collate_junction_records([record])
        self.assertEqual(batch["pair_group_types"].tolist(), [[[1, 3], [2, 3]]])
        self.assertEqual(batch["pair_local_rasters"].dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
