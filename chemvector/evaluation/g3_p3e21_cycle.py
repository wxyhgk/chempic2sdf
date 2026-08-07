"""Frozen protocol and common interfaces for the G3 P3-E21 route cycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


CYCLE_NAME = "chemvector-g3-p3e21-five-route-sdf-cycle"
OUTPUT = Path(f"outputs/audits/{CYCLE_NAME}")
SEED = 20260821
COUNT_BUCKETS = (
    ("1-20", 1, 20),
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
)
SPLIT_MOLECULES_PER_BUCKET = {
    "train": 160,
    "dev": 32,
    "confirm": 32,
}
VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
ROUTES = (
    "hard_latent_occupancy",
    "soft_coverage_occupancy",
    "query_slot_occupancy",
    "bond_conditioned_occupancy",
    "hybrid_structured_count",
)
DOWNSTREAM_PROFILES = ("v0.10.10", "v0.10.21")


def count_bucket(value: int) -> str:
    for name, lower, upper in COUNT_BUCKETS:
        if lower <= value <= upper:
            return name
    raise ValueError(f"P3-E21 Count is outside the cycle buckets: {value}")


@dataclass(slots=True)
class AtomSelectionResult:
    sample_key: str
    predicted_count: int
    anchors: torch.Tensor
    scores: torch.Tensor
    source_candidate_ids: torch.Tensor
    presence_logits: torch.Tensor
    count_log_probs: torch.Tensor
    route_name: str
    checkpoint_sha256: str
    diagnostics: dict

    def __post_init__(self) -> None:
        if not self.sample_key or self.route_name not in ROUTES:
            raise ValueError("P3-E21 Atom selection identity is invalid")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("P3-E21 checkpoint SHA256 is invalid")
        if self.anchors.ndim != 2 or self.anchors.size(1) != 2:
            raise ValueError("P3-E21 selected anchors must have shape [N,2]")
        if not 0 <= self.predicted_count <= len(self.presence_logits):
            raise ValueError("P3-E21 predicted Count is outside candidate capacity")
        if len(self.anchors) != self.predicted_count:
            raise ValueError("P3-E21 selected anchor Count differs")
        if self.scores.shape != (self.predicted_count,):
            raise ValueError("P3-E21 selected scores differ")
        if self.source_candidate_ids.shape != (self.predicted_count,):
            raise ValueError("P3-E21 selected candidate IDs differ")
        if self.presence_logits.ndim != 1:
            raise ValueError("P3-E21 presence logits must be one vector")
        if self.count_log_probs.shape != (len(self.presence_logits) + 1,):
            raise ValueError("P3-E21 Count distribution width differs")
        if not bool(torch.isfinite(self.anchors).all()) or not bool(
            ((self.anchors >= 0.0) & (self.anchors <= 1.0)).all()
        ):
            raise ValueError("P3-E21 selected anchor is outside the image")
        if len(self.source_candidate_ids.unique()) != self.predicted_count:
            raise ValueError("P3-E21 selected candidate IDs must be unique")


__all__ = [
    "AtomSelectionResult",
    "COUNT_BUCKETS",
    "CYCLE_NAME",
    "DOWNSTREAM_PROFILES",
    "OUTPUT",
    "ROUTES",
    "SEED",
    "SPLIT_MOLECULES_PER_BUCKET",
    "VARIANTS",
    "count_bucket",
]
