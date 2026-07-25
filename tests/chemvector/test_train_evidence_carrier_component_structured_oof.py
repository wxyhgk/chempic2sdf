import unittest

import torch

from chemvector.evidence_carrier_component_structured import (
    EvidenceCarrierComponentStructured,
)
from chemvector.train_evidence_carrier_component_structured_oof import (
    collate_component_records,
    component_training_loss,
    merge_component_records,
)


class ComponentStructuredOOFTrainingTest(unittest.TestCase):
    def _record(self):
        return {
            "identity": {"sample_key": "a"},
            "target_free": {
                "pair_tokens": torch.zeros(2, 21),
                "pair_indices": torch.tensor([[0, 1], [1, 2]]),
                "global_count_features": torch.zeros(4),
            },
            "target_aware": {"pair_labels": torch.tensor([1, 0])},
            "pair_local_rasters": torch.zeros(2, 2, 24, 48, dtype=torch.uint8),
        }

    def test_merge_collate_and_structured_loss(self) -> None:
        sidecar = {
            "a": {
                "baseline_scores": torch.tensor([1.0, 0.9]),
                "component_anchor_counts": torch.tensor([3, 3], dtype=torch.uint8),
                "component_edge_counts": torch.tensor([2, 2], dtype=torch.uint8),
                "components": [
                    {
                        "active_pair_indices": torch.tensor([0, 1], dtype=torch.int32),
                        "matching_masks": torch.tensor(
                            [[False, False], [True, False], [False, True]],
                            dtype=torch.bool,
                        ),
                    }
                ],
            }
        }
        records = merge_component_records([self._record()], sidecar)
        batch = collate_component_records(records)
        self.assertEqual(batch["baseline_scores"].shape, (1, 2))
        model = EvidenceCarrierComponentStructured().eval()
        float_batch = {
            **batch,
            "pair_local_rasters": batch["pair_local_rasters"].float().div(255.0),
        }
        output = model(
            float_batch["pair_tokens"],
            float_batch["pair_local_rasters"],
            float_batch["pair_indices"],
            float_batch["pair_mask"],
            float_batch["global_count_features"],
            float_batch["baseline_scores"],
            float_batch["component_anchor_counts"],
            float_batch["component_edge_counts"],
        )
        losses = component_training_loss(output, float_batch)
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertEqual(losses["component_count"], 1)
        self.assertEqual(losses["active_negative_count"], 1)


if __name__ == "__main__":
    unittest.main()
