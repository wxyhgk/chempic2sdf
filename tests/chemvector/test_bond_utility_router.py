import unittest

import torch

from chemvector.bond_utility_router import (
    BondUtilityRouter,
    conservative_frontier,
    route_summary,
)


class BondUtilityRouterTest(unittest.TestCase):
    def test_router_emits_one_logit_per_sample(self) -> None:
        model = BondUtilityRouter(5, hidden_dim=8)
        self.assertEqual(model(torch.randn(3, 5)).shape, (3,))

    def test_frontier_prefers_rescue_without_break(self) -> None:
        probabilities = torch.tensor((0.9, 0.8, 0.7, 0.1))
        labels = ["expert_rescue", "both_wrong", "expert_break", "both_exact"]
        domains = ["a", "a", "b", "b"]

        frontier = conservative_frontier(probabilities, labels, domains)

        self.assertEqual(frontier["0"]["overall"]["expert_rescue"], 1)
        self.assertEqual(frontier["0"]["overall"]["expert_break"], 0)
        self.assertEqual(frontier["0"]["overall"]["net_topology_exact_gain"], 1)

    def test_summary_reports_domain_net_gain(self) -> None:
        result = route_summary(
            torch.tensor((0.9, 0.8)),
            ["expert_rescue", "expert_break"],
            ["a", "b"],
            0.85,
        )
        self.assertEqual(result["overall"]["net_topology_exact_gain"], 1)
        self.assertEqual(result["domains"]["b"]["selected"], 0)


if __name__ == "__main__":
    unittest.main()
