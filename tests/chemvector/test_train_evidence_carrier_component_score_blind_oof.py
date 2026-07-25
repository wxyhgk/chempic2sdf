import unittest

import torch

from chemvector.evidence_carrier_component_score_blind import (
    EvidenceCarrierComponentScoreBlind,
)
from chemvector.train_evidence_carrier_component_score_blind_oof import (
    branch_gate,
    collate_score_blind_records,
    prepare_counterfactual_records,
    score_blind_training_loss,
)


class ComponentScoreBlindOOFTrainingTest(unittest.TestCase):
    def _record(self):
        return {
            "identity": {"sample_key": "a"},
            "target_free": {
                "pair_tokens": torch.zeros(3, 21),
                "pair_indices": torch.tensor([[0, 1], [1, 2], [0, 2]]),
                "global_count_features": torch.zeros(4),
            },
            "target_aware": {"pair_labels": torch.tensor([1, 0, 1])},
            "pair_local_rasters": torch.zeros(3, 2, 24, 48, dtype=torch.uint8),
            "baseline_scores": torch.tensor([1.0, 0.2, 0.9]),
            "component_anchor_counts": torch.tensor([3, 3, 3], dtype=torch.uint8),
            "component_edge_counts": torch.tensor([3, 3, 3], dtype=torch.uint8),
            "components": [
                {
                    "active_pair_indices": torch.tensor([0, 2], dtype=torch.int32),
                    "matching_masks": torch.tensor(
                        [[False, False], [True, False], [False, True]],
                        dtype=torch.bool,
                    ),
                }
            ],
        }

    def test_counterfactual_loss_uses_inactive_negative(self) -> None:
        records = prepare_counterfactual_records([self._record()])
        batch = collate_score_blind_records(records)
        batch["pair_local_rasters"] = batch["pair_local_rasters"].float().div(255.0)
        model = EvidenceCarrierComponentScoreBlind().eval()
        output = model(
            batch["pair_tokens"],
            batch["pair_local_rasters"],
            batch["pair_indices"],
            batch["pair_mask"],
            batch["global_count_features"],
            batch["baseline_scores"],
            batch["component_anchor_counts"],
            batch["component_edge_counts"],
        )
        losses = score_blind_training_loss(output, batch)
        self.assertEqual(losses["counterfactual_negative_count"], 1)
        self.assertEqual(losses["counterfactual_positive_count"], 2)
        self.assertTrue(bool(torch.isfinite(losses["loss"])))

    def test_branch_gate_requires_cross_and_no_exact_regression(self) -> None:
        baseline = {
            "selected_cross_target_pairs": 16,
            "multiplicity_2_recall": 0.996,
            "pairable_sample_exact": 0.96,
            "selected_pair_precision": 0.9999,
            "parallel_2_violated_group_count": 1,
        }
        candidate = {**baseline, "selected_cross_target_pairs": 8}
        self.assertEqual(branch_gate(baseline, candidate)["status"], "GO")
        candidate["pairable_sample_exact"] = 0.95
        self.assertEqual(branch_gate(baseline, candidate)["status"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
