"""Tests for the P3-E21 cycle protocol and split allocator."""

import unittest

import torch

from chemvector.evaluation.g3_p3e21_cycle import AtomSelectionResult, ROUTES
from chemvector.evaluation.audits.atoms.build_g3_p3e21_cycle_manifests import (
    allocate_scaffold_disjoint,
)


class G3P3E21CycleTests(unittest.TestCase):
    def test_atom_selection_result_enforces_common_contract(self) -> None:
        result = AtomSelectionResult(
            sample_key="sample",
            predicted_count=2,
            anchors=torch.tensor([[0.1, 0.2], [0.8, 0.9]]),
            scores=torch.tensor([0.9, 0.8]),
            source_candidate_ids=torch.tensor([4, 7]),
            presence_logits=torch.zeros(8),
            count_log_probs=torch.zeros(9),
            route_name=ROUTES[0],
            checkpoint_sha256="0" * 64,
            diagnostics={},
        )
        self.assertEqual(result.predicted_count, 2)
        with self.assertRaisesRegex(ValueError, "anchor Count differs"):
            AtomSelectionResult(
                sample_key="sample",
                predicted_count=1,
                anchors=result.anchors,
                scores=result.scores,
                source_candidate_ids=result.source_candidate_ids,
                presence_logits=result.presence_logits,
                count_log_probs=result.count_log_probs,
                route_name=ROUTES[0],
                checkpoint_sha256="0" * 64,
                diagnostics={},
            )

    def test_allocator_is_scaffold_disjoint_and_count_balanced(self) -> None:
        groups = []
        buckets = ("1-20", "21-40", "41-60", "61-80", "81-100")
        for bucket in buckets:
            for index in range(9):
                groups.append({
                    "bucket": bucket,
                    "molecule_id": f"{bucket}-{index}",
                    "scaffold": f"{bucket}-scaffold-{index}",
                })
        selected, audit = allocate_scaffold_disjoint(
            groups, split_counts={"train": 3, "dev": 2, "confirm": 2}
        )
        self.assertEqual({name: len(values) for name, values in selected.items()}, {
            "train": 15,
            "dev": 10,
            "confirm": 10,
        })
        self.assertTrue(
            all(value == 0 for value in audit["cross_split_scaffold_overlap"].values())
        )


if __name__ == "__main__":
    unittest.main()
