from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

import chemvector.train_evidence_carrier_endpoint_oof as trainer
from chemvector.evidence_carrier_endpoint_incidence_residual import (
    EvidenceCarrierEndpointIncidenceResidual,
)


def _record(key: str) -> dict:
    pair_tokens = torch.zeros((3, 21), dtype=torch.float32)
    pair_tokens[:, -1] = torch.tensor((1.2, 0.6, -0.5))
    return {
        "identity": {"sample_key": key},
        "partition": "fit",
        "target_free": {
            "pair_tokens": pair_tokens,
            "pair_indices": torch.tensor(((0, 1), (0, 2), (3, 4))),
            "global_count_features": torch.zeros(4),
        },
        "target_aware": {
            "pair_labels": torch.tensor((1, 0, 0)),
            "evaluation_target": {
                "anchor_segments_pixels": torch.zeros((5, 2, 2)),
                "anchor_target_line_ids": torch.tensor((0, 0, 1, 2, 3)),
                "anchor_source_indices": torch.arange(5),
                "anchor_group_ids": torch.arange(5),
                "anchor_group_types": torch.zeros(5, dtype=torch.long),
            },
        },
    }


def _metrics(**overrides: object) -> dict:
    result = {
        "sample_count": 2_400,
        "selected_cross_target_pairs": 0,
        "selected_pair_precision": 1.0,
        "multiplicity_2_recall": 1.0,
        "multiplicity_2_matched_count": 100,
        "pairable_sample_exact": 1.0,
        "pairable_sample_exact_count": 90,
        "parallel_2_distinct_preservation": 1.0,
        "parallel_2_violated_group_count": 0,
        "ordinary_support_loss_pp": 0.0,
        "ordinary_support_lost_target_count": 0,
        "matching_optimal_sample_count": 2_400,
        "permutation_deterministic": True,
    }
    result.update(overrides)
    return result


class EndpointOOFTrainerTest(unittest.TestCase):
    def test_collate_pads_pair_indices_with_minus_one(self) -> None:
        batch = trainer.collate_endpoint_records([_record("a"), _record("b")])
        self.assertEqual(batch["pair_tokens"].shape, (2, 3, 21))
        self.assertEqual(batch["pair_indices"].shape, (2, 3, 2))
        self.assertTrue(batch["pair_mask"].all())

    def test_deterministic_batches_repeat(self) -> None:
        first = list(
            trainer.deterministic_batch_indices(3, steps=4, batch_size=5, seed=9)
        )
        second = list(
            trainer.deterministic_batch_indices(3, steps=4, batch_size=5, seed=9)
        )
        self.assertEqual(first, second)

    def test_short_training_is_reproducible(self) -> None:
        records = [_record(str(index)) for index in range(4)]

        def run():
            torch.manual_seed(5)
            model = EvidenceCarrierEndpointIncidenceResidual()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            with patch.object(trainer, "EXPECTED_TRAIN_SAMPLE_COUNT", 4):
                metrics = trainer.train_endpoint_steps(
                    model,
                    optimizer,
                    records,
                    steps=4,
                    batch_size=3,
                    seed=12,
                )
            return deepcopy(model.state_dict()), metrics

        first_state, first_metrics = run()
        second_state, second_metrics = run()
        for name in first_state:
            torch.testing.assert_close(first_state[name], second_state[name])
        self.assertEqual(first_metrics, second_metrics)

    def test_gate_requires_zero_cross_and_strict_improvement(self) -> None:
        baseline = _metrics(selected_cross_target_pairs=2)
        candidate = _metrics(selected_cross_target_pairs=0)
        passed = trainer.oof_gate(
            baseline, candidate, expected_sample_count=2_400
        )
        self.assertEqual(passed["status"], "GO")

        unsafe = trainer.oof_gate(
            baseline,
            _metrics(selected_cross_target_pairs=1),
            expected_sample_count=2_400,
        )
        self.assertEqual(unsafe["status"], "NO-GO")

        no_op = trainer.oof_gate(
            candidate, candidate, expected_sample_count=2_400
        )
        self.assertEqual(no_op["status"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
