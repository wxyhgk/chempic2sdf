import unittest

import torch

from chemvector.audit_evidence_carrier_pair_local_oof_failures import (
    classify_cross_pair,
)


class PairLocalOOFFailureAuditTest(unittest.TestCase):
    def test_classifies_incidence_and_introduction(self) -> None:
        pairs = torch.tensor([[0, 1], [1, 2], [3, 4]])
        labels = torch.tensor([0, 1, 0])
        row = classify_cross_pair(0, pairs, labels, baseline_selected=[2])
        self.assertEqual(row["anchor_pair"], [0, 1])
        self.assertEqual(row["first_endpoint_degree"], 1)
        self.assertEqual(row["second_endpoint_degree"], 2)
        self.assertTrue(row["has_incident_positive_candidate"])
        self.assertFalse(row["isolated_endpoint_pair"])
        self.assertTrue(row["introduced_by_pair_local_model"])


if __name__ == "__main__":
    unittest.main()
