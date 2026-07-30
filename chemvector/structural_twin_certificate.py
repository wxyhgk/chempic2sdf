"""Target-free structural certificate for deleting duplicate AtomObjects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from chemvector.safe_atom_suppression import predicted_adjacency


@dataclass(frozen=True, slots=True)
class StructuralTwinCertificateConfig:
    maximum_distance_px: float = 2.0
    image_size: float = 768.0
    require_nonempty_neighborhood: bool = True

    def __post_init__(self) -> None:
        if self.maximum_distance_px <= 0 or self.image_size <= 0:
            raise ValueError("Structural-twin distances must be positive")


def structural_twin_witnesses(
    sample: Mapping,
    removed: int,
    *,
    presence_threshold: float,
    config: StructuralTwinCertificateConfig | None = None,
) -> tuple[int, ...]:
    """Return witnesses that preserve every selected typed neighbor relation."""

    config = config or StructuralTwinCertificateConfig()
    anchors = sample["anchors"]
    if not 0 <= removed < len(anchors):
        raise IndexError("Removed AtomObject lies outside the sample")
    adjacency = predicted_adjacency(sample, presence_threshold)
    distances = (
        torch.linalg.vector_norm(anchors - anchors[removed], dim=-1)
        * config.image_size
    )
    coordinate_tolerance = (
        torch.finfo(anchors.dtype).eps * config.image_size * 2.0
        if anchors.dtype.is_floating_point
        else 0.0
    )
    pair_rows = {
        tuple(sorted((int(begin), int(end)))): row
        for row, (begin, end) in enumerate(sample["pairs"].tolist())
    }
    witnesses = []
    for witness in torch.where(
        (distances <= config.maximum_distance_px + coordinate_tolerance)
        & (torch.arange(len(anchors)) != removed)
    )[0].tolist():
        # A selected relation between the two nodes is evidence for two atoms,
        # not a duplicated proposal of one atom.
        if witness in adjacency[removed]:
            continue
        removed_neighbors = adjacency[removed]
        witness_neighbors = adjacency[witness]
        if removed_neighbors != witness_neighbors:
            continue
        if config.require_nonempty_neighborhood and not removed_neighbors:
            continue
        typed_relations_match = True
        for neighbor in removed_neighbors:
            removed_row = pair_rows.get(tuple(sorted((removed, neighbor))))
            witness_row = pair_rows.get(tuple(sorted((witness, neighbor))))
            if removed_row is None or witness_row is None:
                typed_relations_match = False
                break
            if int(sample["chemical"][removed_row]) != int(
                sample["chemical"][witness_row]
            ):
                typed_relations_match = False
                break
        if typed_relations_match:
            witnesses.append(witness)
    return tuple(witnesses)


def structural_twin_delete_is_certified(
    sample: Mapping,
    removed: int,
    *,
    presence_threshold: float,
    config: StructuralTwinCertificateConfig | None = None,
) -> bool:
    return bool(
        structural_twin_witnesses(
            sample,
            removed,
            presence_threshold=presence_threshold,
            config=config,
        )
    )


__all__ = [
    "StructuralTwinCertificateConfig",
    "structural_twin_delete_is_certified",
    "structural_twin_witnesses",
]
