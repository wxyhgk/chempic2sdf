#!/usr/bin/env python3
"""EvidenceCore v0.4 to anonymous chemical graph candidates (stdlib only)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import sys
from typing import Any, Hashable, Iterable


MAX_INPUT_BYTES = 65_536
MAX_PRIMITIVES = 64
MAX_EVENTS = 64
MAX_CANDIDATES = 64
MAX_NODES = 128
MAX_EDGES = 128
ENDPOINT_CLUSTER_RADIUS_Q4 = 16
END_PROJECTION_TOLERANCE_Q4 = 24
ORIENTATION_BUCKET_TOLERANCE = 3
PARALLEL_SINE_NUMERATOR = 1
PARALLEL_SINE_DENOMINATOR = 64
GLYPH_LIKE_MIN_SCORE = 9_000
ALLOWED_EVENT_KINDS = {"terminal", "junction", "intersection", "glyph_like", "unknown"}

NodeKey = Hashable
Point = tuple[int, int]


class CandidateError(ValueError):
    """Evidence is invalid or outside the frozen finite interpretation grammar."""


@dataclass(frozen=True)
class Primitive:
    primitive_id: int
    begin: Point
    end: Point
    width_q4: int
    support_score: int
    local_residual_count: int
    orientation_bucket: int
    closed: bool
    straight: bool


@dataclass(frozen=True)
class Event:
    event_id: int
    centroid: Point
    bbox: tuple[int, int, int, int]
    incident_primitive_ids: tuple[int, ...]
    kind_hypothesis: str
    kind_score: int


@dataclass(frozen=True)
class Edge:
    begin: NodeKey
    end: NodeKey
    order: int
    primitive_ids: tuple[int, ...]


@dataclass
class WorkingGraph:
    node_roles: dict[NodeKey, str]
    edges: list[Edge]
    role_hypotheses: dict[int, str]

    def clone(self) -> "WorkingGraph":
        return WorkingGraph(dict(self.node_roles), list(self.edges), dict(self.role_hypotheses))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateError(f"{path}: expected object")
    actual = set(value)
    if actual != expected:
        raise CandidateError(
            f"{path}: closed fields mismatch missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _integer(value: Any, lower: int, upper: int, path: str) -> int:
    if not _is_integer(value) or not lower <= value <= upper:
        raise CandidateError(f"{path}: expected integer in [{lower},{upper}]")
    return value


def _point(value: Any, path: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise CandidateError(f"{path}: expected two-element point")
    return (
        _integer(value[0], 0, 768 * 16, f"{path}/0"),
        _integer(value[1], 0, 768 * 16, f"{path}/1"),
    )


def _bbox(value: Any, path: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise CandidateError(f"{path}: expected four-element bbox")
    result = tuple(_integer(item, 0, 768 * 16, f"{path}/{index}")
                   for index, item in enumerate(value))
    if result[0] > result[2] or result[1] > result[3]:
        raise CandidateError(f"{path}: inverted bbox")
    return result  # type: ignore[return-value]


def parse_evidence(value: Any) -> tuple[list[Primitive], list[Event]] | None:
    if not isinstance(value, dict):
        raise CandidateError("$: expected object")
    if value.get("status") == "reject":
        rejected = _require_keys(
            value, {"schema_version", "status", "reject_reason"}, "$",
        )
        if rejected["schema_version"] != "chemgraph-evidence-core-v0.4":
            raise CandidateError("$/schema_version: unsupported version")
        reason = rejected["reject_reason"]
        if not isinstance(reason, str) or not reason or len(reason) > 256:
            raise CandidateError("$/reject_reason: expected non-empty string up to 256 chars")
        return None
    root = _require_keys(
        value,
        {"schema_version", "status", "image", "stroke_primitives", "event_regions", "residual"},
        "$",
    )
    if root["schema_version"] != "chemgraph-evidence-core-v0.4":
        raise CandidateError("$/schema_version: unsupported version")
    if root["status"] != "ok":
        raise CandidateError("$/status: expected ok")
    image = _require_keys(root["image"], {"width", "height", "coordinate_format"}, "$/image")
    if image != {"width": 768, "height": 768, "coordinate_format": "Q4-pixel"}:
        raise CandidateError("$/image: unsupported image profile")
    primitive_values = root["stroke_primitives"]
    if not isinstance(primitive_values, list) or len(primitive_values) > MAX_PRIMITIVES:
        raise CandidateError("$/stroke_primitives: capacity exceeded")
    primitives = []
    for index, raw in enumerate(primitive_values):
        item = _require_keys(
            raw,
            {
                "id", "begin_q4", "end_q4", "width_q4", "support_score",
                "local_residual_count", "orientation_bucket", "closed", "straight",
            },
            f"$/stroke_primitives/{index}",
        )
        primitive_id = _integer(item["id"], 0, 63, f"$/stroke_primitives/{index}/id")
        raw_begin = _point(item["begin_q4"], f"$/stroke_primitives/{index}/begin_q4")
        raw_end = _point(item["end_q4"], f"$/stroke_primitives/{index}/end_q4")
        begin, end = sorted((raw_begin, raw_end))
        if not isinstance(item["closed"], bool) or not isinstance(item["straight"], bool):
            raise CandidateError(f"$/stroke_primitives/{index}: flags must be boolean")
        primitives.append(Primitive(
            primitive_id=primitive_id,
            begin=begin,
            end=end,
            width_q4=_integer(item["width_q4"], 1, 64 * 16,
                              f"$/stroke_primitives/{index}/width_q4"),
            support_score=_integer(item["support_score"], 0, 10_000,
                                   f"$/stroke_primitives/{index}/support_score"),
            local_residual_count=_integer(item["local_residual_count"], 0, 768 * 768,
                                          f"$/stroke_primitives/{index}/local_residual_count"),
            orientation_bucket=_integer(item["orientation_bucket"], 0, 719,
                                        f"$/stroke_primitives/{index}/orientation_bucket"),
            closed=item["closed"],
            straight=item["straight"],
        ))
    primitives.sort(key=lambda item: item.primitive_id)
    if [item.primitive_id for item in primitives] != list(range(len(primitives))):
        raise CandidateError("$/stroke_primitives: IDs must be unique and contiguous")

    event_values = root["event_regions"]
    if not isinstance(event_values, list) or len(event_values) > MAX_EVENTS:
        raise CandidateError("$/event_regions: capacity exceeded")
    events = []
    primitive_ids = set(range(len(primitives)))
    for index, raw in enumerate(event_values):
        item = _require_keys(
            raw,
            {"id", "centroid_q4", "bbox_q4", "incident_primitive_ids",
             "kind_hypothesis", "score"},
            f"$/event_regions/{index}",
        )
        event_id = _integer(item["id"], 0, 63, f"$/event_regions/{index}/id")
        incident = item["incident_primitive_ids"]
        if (not isinstance(incident, list) or len(incident) > 16
                or any(not _is_integer(value) for value in incident)
                or len(incident) != len(set(incident))
                or not set(incident) <= primitive_ids):
            raise CandidateError(f"$/event_regions/{index}/incident_primitive_ids: invalid")
        kind = item["kind_hypothesis"]
        if kind not in ALLOWED_EVENT_KINDS:
            raise CandidateError(f"$/event_regions/{index}/kind_hypothesis: invalid")
        events.append(Event(
            event_id=event_id,
            centroid=_point(item["centroid_q4"], f"$/event_regions/{index}/centroid_q4"),
            bbox=_bbox(item["bbox_q4"], f"$/event_regions/{index}/bbox_q4"),
            incident_primitive_ids=tuple(sorted(incident)),
            kind_hypothesis=kind,
            kind_score=_integer(item["score"], 0, 10_000,
                                f"$/event_regions/{index}/score"),
        ))
    events.sort(key=lambda item: item.event_id)
    if [item.event_id for item in events] != list(range(len(events))):
        raise CandidateError("$/event_regions: IDs must be unique and contiguous")

    residual = _require_keys(
        root["residual"], {"foreground_pixel_count", "component_count", "bbox_q4"},
        "$/residual",
    )
    _integer(residual["foreground_pixel_count"], 0, 768 * 768,
             "$/residual/foreground_pixel_count")
    _integer(residual["component_count"], 0, 4096, "$/residual/component_count")
    _bbox(residual["bbox_q4"], "$/residual/bbox_q4")
    return primitives, events


class _DisjointSet:
    def __init__(self, values: Iterable[Hashable]):
        self.parent = {value: value for value in values}

    def find(self, value: Hashable) -> Hashable:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: Hashable, right: Hashable) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if repr(a) > repr(b):
                a, b = b, a
            self.parent[b] = a


def _cluster_endpoints(
    primitives: list[Primitive], events: list[Event]
) -> tuple[dict[tuple[int, int], int], dict[int, str]]:
    references = [(primitive.primitive_id, endpoint_index)
                  for primitive in primitives for endpoint_index in (0, 1)]
    points = {
        (primitive.primitive_id, 0): primitive.begin for primitive in primitives
    } | {
        (primitive.primitive_id, 1): primitive.end for primitive in primitives
    }
    dsu = _DisjointSet(references)
    for index, left in enumerate(references):
        for right in references[index + 1:]:
            a, b = points[left], points[right]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= ENDPOINT_CLUSTER_RADIUS_Q4:
                dsu.union(left, right)
    for event in events:
        if event.kind_hypothesis != "junction":
            continue
        incident_endpoints = []
        for primitive_id in event.incident_primitive_ids:
            primitive = primitives[primitive_id]
            choices = ((primitive_id, 0), (primitive_id, 1))
            nearest = min(choices, key=lambda ref: (
                (points[ref][0] - event.centroid[0]) ** 2
                + (points[ref][1] - event.centroid[1]) ** 2,
                ref,
            ))
            point = points[nearest]
            tolerance = max(ENDPOINT_CLUSTER_RADIUS_Q4, primitive.width_q4)
            if max(abs(point[0] - event.centroid[0]),
                   abs(point[1] - event.centroid[1])) <= tolerance:
                incident_endpoints.append(nearest)
        if len(incident_endpoints) >= 2:
            for reference in incident_endpoints[1:]:
                dsu.union(incident_endpoints[0], reference)
    groups: dict[Hashable, list[tuple[int, int]]] = defaultdict(list)
    for reference in references:
        groups[dsu.find(reference)].append(reference)
    ordered = sorted(groups.values(), key=lambda group: (
        min(points[reference] for reference in group), tuple(sorted(group))
    ))
    endpoint_nodes = {
        reference: node_id for node_id, group in enumerate(ordered) for reference in group
    }
    return endpoint_nodes, {node_id: "implicit" for node_id in range(len(ordered))}


def _orientation(a: Point, b: Point, c: Point) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_crossing(left: Primitive, right: Primitive) -> bool:
    a, b, c, d = left.begin, left.end, right.begin, right.end
    first = _orientation(a, b, c), _orientation(a, b, d)
    second = _orientation(c, d, a), _orientation(c, d, b)
    return ((first[0] < 0 < first[1] or first[1] < 0 < first[0])
            and (second[0] < 0 < second[1] or second[1] < 0 < second[0]))


def _aligned_parallel(left: Primitive, right: Primitive) -> bool:
    if not (left.straight and right.straight and not left.closed and not right.closed):
        return False
    bucket_difference = abs(left.orientation_bucket - right.orientation_bucket)
    if min(bucket_difference, 720 - bucket_difference) > ORIENTATION_BUCKET_TOLERANCE:
        return False
    lv = left.end[0] - left.begin[0], left.end[1] - left.begin[1]
    rv = right.end[0] - right.begin[0], right.end[1] - right.begin[1]
    cross = lv[0] * rv[1] - lv[1] * rv[0]
    left_length_squared = lv[0] * lv[0] + lv[1] * lv[1]
    right_length_squared = rv[0] * rv[0] + rv[1] * rv[1]
    if (lv[0] * rv[0] + lv[1] * rv[1] <= 0
            or cross * cross * PARALLEL_SINE_DENOMINATOR ** 2
            > PARALLEL_SINE_NUMERATOR ** 2 * left_length_squared * right_length_squared):
        return False
    for a, b in ((left.begin, right.begin), (left.end, right.end)):
        projection = (b[0] - a[0]) * lv[0] + (b[1] - a[1]) * lv[1]
        if projection * projection > END_PROJECTION_TOLERANCE_Q4 ** 2 * left_length_squared:
            return False
    return True


def _bundle_groups(
    primitives: list[Primitive],
    endpoint_nodes: dict[tuple[int, int], int],
    excluded: set[int],
) -> list[tuple[int, ...]]:
    eligible = [item.primitive_id for item in primitives if item.primitive_id not in excluded]
    related: dict[int, set[int]] = {item: set() for item in eligible}
    for left_id, right_id in _pairs(eligible):
        left, right = primitives[left_id], primitives[right_id]
        distinct_ends = (
            endpoint_nodes[(left_id, 0)] != endpoint_nodes[(right_id, 0)]
            and endpoint_nodes[(left_id, 1)] != endpoint_nodes[(right_id, 1)]
        )
        if distinct_ends and _aligned_parallel(left, right):
            related[left_id].add(right_id)
            related[right_id].add(left_id)
    groups = []
    unseen = set(eligible)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            item = stack.pop()
            if item not in unseen:
                continue
            unseen.remove(item)
            component.add(item)
            stack.extend(related[item])
        if len(component) in (2, 3) and all(
            right in related[left] for left, right in _pairs(sorted(component))
        ):
            groups.append(tuple(sorted(component)))
    return groups


def _pairs(values: Iterable[int]) -> Iterable[tuple[int, int]]:
    items = list(values)
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            yield left, right


def _base_graph(
    primitives: list[Primitive], events: list[Event]
) -> tuple[WorkingGraph, dict[tuple[int, int], int]]:
    endpoint_nodes, roles = _cluster_endpoints(primitives, events)
    edges = [Edge(
        endpoint_nodes[(primitive.primitive_id, 0)],
        endpoint_nodes[(primitive.primitive_id, 1)],
        1,
        (primitive.primitive_id,),
    ) for primitive in primitives]
    return WorkingGraph(roles, edges, {}), endpoint_nodes


def _junction_variant(graph: WorkingGraph, primitive_ids: tuple[int, int], ordinal: int) -> WorkingGraph:
    result = graph.clone()
    junction: NodeKey = ("junction", ordinal, primitive_ids)
    result.node_roles[junction] = "implicit"
    new_edges = []
    found = Counter()
    for edge in result.edges:
        matched = [item for item in primitive_ids if item in edge.primitive_ids]
        if matched and len(edge.primitive_ids) == 1:
            primitive_id = matched[0]
            found[primitive_id] += 1
            new_edges.extend((
                Edge(edge.begin, junction, 1, (primitive_id,)),
                Edge(junction, edge.end, 1, (primitive_id,)),
            ))
        else:
            new_edges.append(edge)
    if any(found[item] != 1 for item in primitive_ids):
        raise CandidateError("interior crossing conflicts with another transformation")
    result.edges = new_edges
    return result


def _merge_nodes(graph: WorkingGraph, groups: list[list[NodeKey]]) -> WorkingGraph:
    result = graph.clone()
    dsu = _DisjointSet(result.node_roles)
    for group in groups:
        for item in group[1:]:
            dsu.union(group[0], item)
    roles: dict[NodeKey, str] = {}
    for node, role in result.node_roles.items():
        representative = dsu.find(node)
        previous = roles.get(representative)
        if previous is not None and previous != role:
            raise CandidateError("cannot merge nodes with different roles")
        roles[representative] = role
    result.node_roles = roles
    result.edges = [Edge(dsu.find(edge.begin), dsu.find(edge.end), edge.order, edge.primitive_ids)
                    for edge in result.edges]
    if any(edge.begin == edge.end for edge in result.edges):
        raise CandidateError("node merge produced self loop")
    return result


def _bundle_variant(
    graph: WorkingGraph,
    group: tuple[int, ...],
    endpoint_nodes: dict[tuple[int, int], int],
) -> WorkingGraph:
    result = graph.clone()
    result.edges = [edge for edge in result.edges
                    if not (len(edge.primitive_ids) == 1 and edge.primitive_ids[0] in group)]
    result = _merge_nodes(result, [
        [endpoint_nodes[(item, 0)] for item in group],
        [endpoint_nodes[(item, 1)] for item in group],
    ])
    probe = _DisjointSet(graph.node_roles)
    for endpoint_index in (0, 1):
        values = [endpoint_nodes[(item, endpoint_index)] for item in group]
        for value in values[1:]:
            probe.union(values[0], value)
    begin = probe.find(endpoint_nodes[(group[0], 0)])
    end = probe.find(endpoint_nodes[(group[0], 1)])
    result.edges.append(Edge(begin, end, len(group), group))
    return result


def _glyph_text_variant(graph: WorkingGraph, primitive_id: int) -> WorkingGraph:
    result = graph.clone()
    removed = [edge for edge in result.edges if primitive_id in edge.primitive_ids]
    if len(removed) != 1 or len(removed[0].primitive_ids) != 1:
        raise CandidateError("glyph_like primitive is not one independent lane")
    result.edges = [edge for edge in result.edges if primitive_id not in edge.primitive_ids]
    used = {edge.begin for edge in result.edges} | {edge.end for edge in result.edges}
    for endpoint in (removed[0].begin, removed[0].end):
        if endpoint not in used:
            result.node_roles.pop(endpoint, None)
        else:
            raise CandidateError("glyph_like primitive shares an endpoint cluster")
    text_node: NodeKey = ("text_stroke", primitive_id)
    result.node_roles[text_node] = "text_stroke"
    result.role_hypotheses[primitive_id] = "text_stroke"
    return result


def _glyph_bond_variant(graph: WorkingGraph, primitive_id: int) -> WorkingGraph:
    result = graph.clone()
    result.role_hypotheses[primitive_id] = "bond_lane"
    return result


def _graph_payload(graph: WorkingGraph) -> dict[str, Any]:
    used = {edge.begin for edge in graph.edges} | {edge.end for edge in graph.edges}
    nodes = [node for node in graph.node_roles
             if node in used or graph.node_roles[node] == "text_stroke"]
    nodes.sort(key=lambda node: (0 if graph.node_roles[node] == "implicit" else 1, repr(node)))
    node_ids = {node: index for index, node in enumerate(nodes)}
    edges = []
    seen_pairs = set()
    for edge in graph.edges:
        begin = min(node_ids[edge.begin], node_ids[edge.end])
        end = max(node_ids[edge.begin], node_ids[edge.end])
        if begin == end or (begin, end) in seen_pairs:
            raise CandidateError("self-loop or duplicate anonymous edge is outside grammar")
        seen_pairs.add((begin, end))
        edges.append({"begin": begin, "end": end, "order": edge.order})
    edges.sort(key=lambda item: (item["begin"], item["end"], item["order"]))
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise CandidateError("candidate graph capacity exceeded")
    return {
        "nodes": [{"id": index, "role": graph.node_roles[node]}
                  for index, node in enumerate(nodes)],
        "edges": edges,
        "role_hypotheses": [
            {"primitive_id": primitive_id, "role": role}
            for primitive_id, role in sorted(graph.role_hypotheses.items())
        ],
    }


def _adjacency(graph: dict[str, Any]) -> list[dict[int, int]]:
    result = [dict() for _ in graph["nodes"]]
    for edge in graph["edges"]:
        left, right, order = edge["begin"], edge["end"], edge["order"]
        if right in result[left]:
            raise CandidateError("parallel anonymous edges are not supported")
        result[left][right] = order
        result[right][left] = order
    return result


def graphs_isomorphic(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Independent color-refinement plus backtracking graph isomorphism."""
    if len(left["nodes"]) != len(right["nodes"]) or len(left["edges"]) != len(right["edges"]):
        return False
    left_adj, right_adj = _adjacency(left), _adjacency(right)
    left_roles = [item["role"] for item in left["nodes"]]
    right_roles = [item["role"] for item in right["nodes"]]
    if Counter(left_roles) != Counter(right_roles):
        return False
    signatures = []
    for roles, adjacency in ((left_roles, left_adj), (right_roles, right_adj)):
        signatures.append([
            (roles[index], tuple(sorted(adjacency[index].values())))
            for index in range(len(roles))
        ])
    universe = sorted(set(signatures[0] + signatures[1]), key=repr)
    rank = {signature: index for index, signature in enumerate(universe)}
    colors = [[rank[item] for item in side] for side in signatures]
    for _ in range(len(left_roles)):
        refined = []
        for side, adjacency in enumerate((left_adj, right_adj)):
            refined.append([
                (colors[side][node], tuple(sorted(
                    (order, colors[side][neighbor])
                    for neighbor, order in adjacency[node].items()
                ))) for node in range(len(adjacency))
            ])
        universe = sorted(set(refined[0] + refined[1]), key=repr)
        rank = {signature: index for index, signature in enumerate(universe)}
        updated = [[rank[item] for item in side] for side in refined]
        if updated == colors:
            break
        colors = updated
    if Counter(colors[0]) != Counter(colors[1]):
        return False
    right_by_color: dict[int, list[int]] = defaultdict(list)
    for node, color in enumerate(colors[1]):
        right_by_color[color].append(node)
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def compatible(source: int, target: int) -> bool:
        for mapped_source, mapped_target in mapping.items():
            if left_adj[source].get(mapped_source) != right_adj[target].get(mapped_target):
                return False
        return True

    def search() -> bool:
        if len(mapping) == len(left_roles):
            return True
        remaining = [node for node in range(len(left_roles)) if node not in mapping]
        source = min(remaining, key=lambda node: (
            sum(candidate not in used for candidate in right_by_color[colors[0][node]]),
            -len(left_adj[node]), node,
        ))
        for target in right_by_color[colors[0][source]]:
            if target not in used and compatible(source, target):
                mapping[source] = target
                used.add(target)
                if search():
                    return True
                used.remove(target)
                del mapping[source]
        return False

    return search()


def _deduplicate(graphs: list[WorkingGraph]) -> list[dict[str, Any]]:
    result = []
    for working in graphs:
        payload = _graph_payload(working)
        if not any(graphs_isomorphic(payload, existing) for existing in result):
            result.append(payload)
            if len(result) > MAX_CANDIDATES:
                raise CandidateError("candidate set capacity exceeded")
    return result


def _interpret_candidates(primitives: list[Primitive], events: list[Event]) -> list[dict[str, Any]]:
    if any(item.begin == item.end or item.closed or not item.straight for item in primitives):
        raise CandidateError("closed or non-straight primitive is outside grammar")
    base, endpoint_nodes = _base_graph(primitives, events)
    intersection_pairs = {
        tuple(sorted(pair))
        for event in events if event.kind_hypothesis == "intersection"
        for pair in _pairs(event.incident_primitive_ids)
    }
    crossing_pairs = tuple(
        (left.primitive_id, right.primitive_id)
        for index, left in enumerate(primitives)
        for right in primitives[index + 1:]
        if ((left.primitive_id, right.primitive_id) in intersection_pairs
            and not ({endpoint_nodes[(left.primitive_id, 0)],
                      endpoint_nodes[(left.primitive_id, 1)]}
                     & {endpoint_nodes[(right.primitive_id, 0)],
                        endpoint_nodes[(right.primitive_id, 1)]})
            and _proper_crossing(left, right))
    )
    crossing_incidence = Counter(item for pair in crossing_pairs for item in pair)
    if any(count > 1 for count in crossing_incidence.values()):
        raise CandidateError("multiple proper crossings on one primitive are outside grammar")
    glyph_ids = []
    for event in events:
        if event.kind_hypothesis == "glyph_like" and event.kind_score >= GLYPH_LIKE_MIN_SCORE:
            if len(event.incident_primitive_ids) != 1:
                raise CandidateError("glyph_like event must own exactly one primitive")
            primitive_id = event.incident_primitive_ids[0]
            if primitive_id not in glyph_ids:
                glyph_ids.append(primitive_id)
    excluded = set(crossing_incidence) | set(glyph_ids)
    bundles = _bundle_groups(primitives, endpoint_nodes, excluded)
    bundle_node_sets = [
        {endpoint_nodes[(primitive_id, endpoint_index)]
         for primitive_id in group for endpoint_index in (0, 1)}
        for group in bundles
    ]
    if any(left & right for index, left in enumerate(bundle_node_sets)
           for right in bundle_node_sets[index + 1:]):
        raise CandidateError("connected bundle groups are outside grammar")
    if len(crossing_pairs) + len(bundles) + len(glyph_ids) > 6:
        raise CandidateError("more than 64 ambiguity combinations are outside grammar")

    working = [base]
    for ordinal, pair in enumerate(crossing_pairs):
        working = working + [_junction_variant(item, pair, ordinal) for item in working]
    for group in bundles:
        working = working + [_bundle_variant(item, group, endpoint_nodes) for item in working]
    for primitive_id in glyph_ids:
        expanded = []
        for item in working:
            expanded.extend((
                _glyph_bond_variant(item, primitive_id),
                _glyph_text_variant(item, primitive_id),
            ))
        working = expanded
    return _deduplicate(working)


def build_candidate_set(evidence: Any) -> dict[str, Any]:
    parsed = parse_evidence(evidence)
    if parsed is None:
        primitives, events = [], []
    else:
        primitives, events = parsed
    try:
        candidates = _interpret_candidates(primitives, events) if primitives else []
    except CandidateError:
        candidates = []
    decode_kind = "Reject" if not candidates else "Graph" if len(candidates) == 1 else "Ambiguous"
    return {
        "candidates": candidates,
        "decode_kind": decode_kind,
        "schema_version": "anonymous-candidate-set-v0.4",
    }


def main() -> None:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise SystemExit("EvidenceCore exceeds 65536 bytes")
    try:
        evidence = json.loads(raw)
        if raw != canonical_json_bytes(evidence):
            raise CandidateError("stdin must be canonical JSON with one final newline")
        result = build_candidate_set(evidence)
        output = canonical_json_bytes(result)
        if len(output) > MAX_INPUT_BYTES:
            raise CandidateError("CandidateSet exceeds 65536 bytes")
        sys.stdout.buffer.write(output)
    except (CandidateError, json.JSONDecodeError) as error:
        raise SystemExit(f"candidate rejected evidence: {error}") from error


if __name__ == "__main__":
    main()
