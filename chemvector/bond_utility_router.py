"""Small router that predicts whether the style BondObject expert is useful."""

from __future__ import annotations

import torch
from torch import nn


class BondUtilityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("Bond utility router dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        bottleneck = max(hidden_dim // 2, 4)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError("Bond utility router input must be a matching matrix")
        return self.network(features).squeeze(-1)


def route_summary(
    probabilities: torch.Tensor,
    labels: list[str],
    domains: list[str],
    threshold: float,
) -> dict:
    if probabilities.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError("Router probabilities and labels differ")
    selected = probabilities >= threshold

    def summarize(indices: list[int]) -> dict:
        selected_indices = [index for index in indices if bool(selected[index])]
        counts = {
            name: sum(labels[index] == name for index in selected_indices)
            for name in ("expert_rescue", "expert_break", "both_exact", "both_wrong")
        }
        return {
            "samples": len(indices),
            "selected": len(selected_indices),
            "route_rate": len(selected_indices) / max(len(indices), 1),
            **counts,
            "net_topology_exact_gain": counts["expert_rescue"]
            - counts["expert_break"],
        }

    result = {"threshold": threshold, "overall": summarize(list(range(len(labels))))}
    result["domains"] = {
        domain: summarize([index for index, value in enumerate(domains) if value == domain])
        for domain in sorted(set(domains))
    }
    return result


def conservative_frontier(
    probabilities: torch.Tensor,
    labels: list[str],
    domains: list[str],
    break_budgets: tuple[int, ...] = (0, 1, 2, 5, 10),
) -> dict[str, dict | None]:
    thresholds = sorted(
        {1.0, 0.0, *(float(value) for value in probabilities.tolist())},
        reverse=True,
    )
    summaries = [route_summary(probabilities, labels, domains, value) for value in thresholds]
    frontier = {}
    for budget in break_budgets:
        candidates = [
            item
            for item in summaries
            if item["overall"]["expert_break"] <= budget
        ]
        frontier[str(budget)] = max(
            candidates,
            key=lambda item: (
                item["overall"]["net_topology_exact_gain"],
                item["overall"]["expert_rescue"],
                -item["overall"]["both_wrong"],
                item["threshold"],
            ),
            default=None,
        )
    return frontier
