"""Finite certification utilities for constrained graph recovery v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from formal.chemgraph.evidence_core_v0_4.candidate.candidate import graphs_isomorphic


@dataclass(frozen=True, slots=True)
class ScoreInterval:
    """Closed score interval; lower scores are better."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise ValueError("score bounds must be integers")
        if not isinstance(self.lower, int) or not isinstance(self.upper, int):
            raise ValueError("score bounds must be integers")
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("score interval must satisfy 0 <= lower <= upper")


def graph_equivalence_classes(candidates: list[dict[str, Any]]) -> list[list[int]]:
    """Partition candidates by the frozen anonymous graph isomorphism relation."""

    classes: list[list[int]] = []
    for candidate_index, candidate in enumerate(candidates):
        for graph_class in classes:
            if graphs_isomorphic(candidate, candidates[graph_class[0]]):
                graph_class.append(candidate_index)
                break
        else:
            classes.append([candidate_index])
    return classes


def certify_class_intervals(
    candidates: list[dict[str, Any]],
    intervals: list[ScoreInterval] | None = None,
) -> dict[str, Any]:
    """Return Graph only for one class or one interval-separated winning class."""

    classes = graph_equivalence_classes(candidates)
    if not classes:
        return {"decision": "Reject", "class_count": 0, "winner_class": None}
    if len(classes) == 1:
        return {"decision": "Graph", "class_count": 1, "winner_class": 0}
    if intervals is None:
        return {
            "decision": "Ambiguous",
            "class_count": len(classes),
            "winner_class": None,
        }
    if len(intervals) != len(classes):
        raise ValueError("one score interval is required per graph equivalence class")

    certified = []
    for index, interval in enumerate(intervals):
        if all(
            interval.upper < other.lower
            for other_index, other in enumerate(intervals)
            if other_index != index
        ):
            certified.append(index)
    if len(certified) == 1:
        return {
            "decision": "Graph",
            "class_count": len(classes),
            "winner_class": certified[0],
        }
    return {
        "decision": "Ambiguous",
        "class_count": len(classes),
        "winner_class": None,
    }


def bounded_score_values(clean: int, epsilon: int) -> range:
    if clean < 0 or epsilon < 0:
        raise ValueError("clean score and epsilon must be nonnegative")
    return range(max(0, clean - epsilon), clean + epsilon + 1)


__all__ = [
    "ScoreInterval",
    "bounded_score_values",
    "certify_class_intervals",
    "graph_equivalence_classes",
]
