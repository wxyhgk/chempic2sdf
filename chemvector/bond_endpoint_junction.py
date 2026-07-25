"""Derive semantic Junctions from complete Bond Object endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chemvector.geometry import stable_pairwise_distances


@dataclass(frozen=True, slots=True)
class BondEndpointJunctionResult:
    points: torch.Tensor
    bond_objects: tuple[dict, ...]
    raw_clusters: int
    self_loops: int
    duplicate_pairs: int


def project_unique_junction_pairs(
    bond_objects: list[dict],
    endpoint_groups: torch.Tensor,
    *,
    ranking_scores: torch.Tensor,
) -> BondEndpointJunctionResult:
    """Project thresholded Bond candidates onto unique predicted Junction pairs.

    ``ranking_scores`` decides which candidate represents a duplicated undirected
    Junction pair. Detector confidence remains stored in ``item["score"]`` and is
    still used to weight the retained endpoints when computing Junction points.
    """

    expected_shape = (len(bond_objects) * 2,)
    if endpoint_groups.shape != expected_shape:
        raise ValueError("Endpoint group IDs differ from Bond endpoints")
    if ranking_scores.shape != (len(bond_objects),):
        raise ValueError("Projection ranking scores differ from Bond objects")
    scores = ranking_scores.detach().float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("Projection ranking scores must be finite")
    groups = endpoint_groups.detach().long().cpu()
    raw_clusters = len(set(groups.tolist()))
    if not bond_objects:
        return BondEndpointJunctionResult(
            points=torch.empty((0, 2), dtype=torch.float32),
            bond_objects=(),
            raw_clusters=raw_clusters,
            self_loops=0,
            duplicate_pairs=0,
        )

    selected_indices = []
    used_pairs: set[tuple[int, int]] = set()
    self_loops = duplicate_pairs = 0
    order = sorted(
        range(len(bond_objects)),
        key=lambda index: (-float(scores[index]), index),
    )
    for object_index in order:
        begin = int(groups[object_index * 2])
        end = int(groups[object_index * 2 + 1])
        if begin == end:
            self_loops += 1
            continue
        pair = tuple(sorted((begin, end)))
        if pair in used_pairs:
            duplicate_pairs += 1
            continue
        selected_indices.append(object_index)
        used_pairs.add(pair)

    if not selected_indices:
        return BondEndpointJunctionResult(
            points=torch.empty((0, 2), dtype=torch.float32),
            bond_objects=(),
            raw_clusters=raw_clusters,
            self_loops=self_loops,
            duplicate_pairs=duplicate_pairs,
        )

    selected_objects = [bond_objects[index] for index in selected_indices]
    selected_groups = torch.tensor(
        [
            int(groups[object_index * 2 + side])
            for object_index in selected_indices
            for side in (0, 1)
        ],
        dtype=torch.long,
    )
    projected = junctions_from_endpoint_groups(selected_objects, selected_groups)
    return BondEndpointJunctionResult(
        points=projected.points,
        bond_objects=projected.bond_objects,
        raw_clusters=raw_clusters,
        self_loops=self_loops,
        duplicate_pairs=duplicate_pairs,
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


def junctions_from_endpoint_groups(
    bond_objects: list[dict],
    endpoint_groups: torch.Tensor,
) -> BondEndpointJunctionResult:
    """Build Junctions from an explicit group ID for every Bond endpoint."""

    if endpoint_groups.shape != (len(bond_objects) * 2,):
        raise ValueError("Endpoint group IDs differ from Bond endpoints")
    if not bond_objects:
        return BondEndpointJunctionResult(
            points=torch.empty((0, 2), dtype=torch.float32),
            bond_objects=(),
            raw_clusters=0,
            self_loops=0,
            duplicate_pairs=0,
        )
    endpoints = torch.stack(
        [item["endpoints"].detach().float().cpu() for item in bond_objects]
    ).reshape(-1, 2)
    group_values = sorted({int(value) for value in endpoint_groups.tolist()})
    group_index = {value: index for index, value in enumerate(group_values)}
    cluster_by_endpoint = {
        index: group_index[int(value)]
        for index, value in enumerate(endpoint_groups.tolist())
    }
    cluster_points = []
    for value in group_values:
        indices = torch.where(endpoint_groups.cpu() == value)[0].tolist()
        weights = torch.tensor(
            [max(float(bond_objects[index // 2]["score"]), 1e-6) for index in indices]
        )
        cluster_points.append(
            (endpoints[indices] * weights[:, None]).sum(0) / weights.sum()
        )
    return _finalize_junction_result(
        bond_objects,
        cluster_by_endpoint,
        cluster_points,
    )


def _finalize_junction_result(
    bond_objects: list[dict],
    cluster_by_endpoint: dict[int, int],
    cluster_points: list[torch.Tensor],
) -> BondEndpointJunctionResult:
    pair_objects: dict[tuple[int, int], dict] = {}
    self_loops = duplicate_pairs = 0
    for object_index, item in sorted(
        enumerate(bond_objects),
        key=lambda pair: float(pair[1]["score"]),
        reverse=True,
    ):
        begin = cluster_by_endpoint[object_index * 2]
        end = cluster_by_endpoint[object_index * 2 + 1]
        if begin == end:
            self_loops += 1
            continue
        pair = tuple(sorted((begin, end)))
        if pair in pair_objects:
            duplicate_pairs += 1
            continue
        enriched = dict(item)
        enriched["junction_pair"] = pair
        pair_objects[pair] = enriched

    used_clusters = sorted({index for pair in pair_objects for index in pair})
    remap = {old: new for new, old in enumerate(used_clusters)}
    points = (
        torch.stack([cluster_points[index] for index in used_clusters])
        if used_clusters
        else torch.empty((0, 2), dtype=torch.float32)
    )
    selected = []
    for item in pair_objects.values():
        enriched = dict(item)
        enriched["junction_pair"] = tuple(
            remap[index] for index in item["junction_pair"]
        )
        selected.append(enriched)
    selected.sort(key=lambda item: item["junction_pair"])
    return BondEndpointJunctionResult(
        points=points,
        bond_objects=tuple(selected),
        raw_clusters=len(cluster_points),
        self_loops=self_loops,
        duplicate_pairs=duplicate_pairs,
    )


def junctions_from_bond_endpoints(
    bond_objects: list[dict],
    *,
    width: int,
    height: int,
    radius_pixels: float,
) -> BondEndpointJunctionResult:
    """Cluster two semantic endpoints per Bond Object into graph Junctions."""

    if min(width, height) < 1:
        raise ValueError("Bond endpoint canvas dimensions must be positive")
    if radius_pixels <= 0.0:
        raise ValueError("Bond endpoint cluster radius must be positive")
    if not bond_objects:
        return BondEndpointJunctionResult(
            points=torch.empty((0, 2), dtype=torch.float32),
            bond_objects=(),
            raw_clusters=0,
            self_loops=0,
            duplicate_pairs=0,
        )

    endpoints = torch.stack(
        [item["endpoints"].detach().float().cpu() for item in bond_objects]
    )
    if endpoints.shape != (len(bond_objects), 2, 2):
        raise ValueError("Every Bond Object must contain exactly two endpoints")
    flat = endpoints.reshape(-1, 2)
    scale = torch.tensor((float(width), float(height)))
    distances = stable_pairwise_distances(flat * scale, flat * scale)
    groups = _DisjointSet(len(flat))
    for first in range(len(flat)):
        for second in range(first + 1, len(flat)):
            if first // 2 == second // 2:
                continue
            if float(distances[first, second]) <= radius_pixels:
                groups.union(first, second)

    members: dict[int, list[int]] = {}
    for endpoint_index in range(len(flat)):
        members.setdefault(groups.find(endpoint_index), []).append(endpoint_index)
    roots = sorted(members, key=lambda root: min(members[root]))
    cluster_by_endpoint = {}
    cluster_points = []
    for cluster_index, root in enumerate(roots):
        indices = members[root]
        weights = torch.tensor(
            [max(float(bond_objects[index // 2]["score"]), 1e-6) for index in indices]
        )
        point = (flat[indices] * weights[:, None]).sum(0) / weights.sum()
        cluster_points.append(point)
        cluster_by_endpoint.update({index: cluster_index for index in indices})

    return _finalize_junction_result(
        bond_objects,
        cluster_by_endpoint,
        cluster_points,
    )
