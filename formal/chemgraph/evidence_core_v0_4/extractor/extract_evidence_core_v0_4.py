#!/usr/bin/env python3
"""Extract bounded neutral line/event evidence from one canonical PNG on stdin."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np


SCHEMA_VERSION = "chemgraph-evidence-core-v0.4"
IMAGE_SIZE = 768
MAX_OUTPUT_BYTES = 65_536
MAX_PRIMITIVES = 64
MAX_EVENTS = 64
ANGLE_BUCKETS = 720


class EvidenceError(ValueError):
    """Input or bounded-output contract failure."""


@dataclass(frozen=True)
class RawLine:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class LineCluster:
    angle: float
    rho: float
    intervals: list[tuple[float, float]]
    lines: list[RawLine]


@dataclass(frozen=True)
class Primitive:
    begin: tuple[float, float]
    end: tuple[float, float]
    width: float
    support_score: int
    local_residual: int

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.begin[0], self.end[1] - self.begin[1])


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def decode_png(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE) or image.dtype != np.uint8:
        raise EvidenceError("unsupported_png")
    return image


def morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Return a one-pixel Zhang-Suen centerline without diagonal double ridges."""
    current = np.pad(binary.astype(bool), 1, mode="constant")
    for _ in range(96):
        changed = False
        for first_subiteration in (True, False):
            center = current[1:-1, 1:-1]
            north = current[:-2, 1:-1]
            north_east = current[:-2, 2:]
            east = current[1:-1, 2:]
            south_east = current[2:, 2:]
            south = current[2:, 1:-1]
            south_west = current[2:, :-2]
            west = current[1:-1, :-2]
            north_west = current[:-2, :-2]
            neighbours = (
                north.astype(np.uint8)
                + north_east
                + east
                + south_east
                + south
                + south_west
                + west
                + north_west
            )
            ordered = (
                north,
                north_east,
                east,
                south_east,
                south,
                south_west,
                west,
                north_west,
                north,
            )
            transitions = np.zeros_like(neighbours)
            for left, right in zip(ordered, ordered[1:]):
                transitions += ((~left) & right).astype(np.uint8)
            remove = center & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if first_subiteration:
                remove &= ~(north & east & south)
                remove &= ~(east & south & west)
            else:
                remove &= ~(north & east & west)
                remove &= ~(north & south & west)
            if np.any(remove):
                center[remove] = False
                changed = True
        if not changed:
            break
    return np.where(current[1:-1, 1:-1], 255, 0).astype(np.uint8)


def canonical_direction(line: RawLine) -> tuple[float, float]:
    dx, dy = line.x1 - line.x0, line.y1 - line.y0
    length = math.hypot(dx, dy)
    if length == 0:
        raise EvidenceError("zero_length_hough_line")
    return dx / length, dy / length


def canonical_angle(line: RawLine) -> float:
    direction = canonical_direction(line)
    return math.atan2(direction[1], direction[0])


def line_coordinates(line: RawLine) -> tuple[float, float, float, float]:
    direction = canonical_direction(line)
    angle = math.atan2(direction[1], direction[0])
    normal = (-direction[1], direction[0])
    midpoint = ((line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2)
    rho = normal[0] * midpoint[0] + normal[1] * midpoint[1]
    t0 = direction[0] * line.x0 + direction[1] * line.y0
    t1 = direction[0] * line.x1 + direction[1] * line.y1
    return angle, rho, min(t0, t1), max(t0, t1)


def align_line_coordinates(
    coordinates: tuple[float, float, float, float], reference_angle: float
) -> tuple[float, float, float, float]:
    """Choose the unoriented representative nearest a cluster direction."""
    angle, rho, begin, end = coordinates
    if math.cos(angle - reference_angle) < 0:
        angle = angle + math.pi if angle < reference_angle else angle - math.pi
        rho = -rho
        begin, end = -end, -begin
    while angle - reference_angle > math.pi / 2:
        angle -= math.pi
        rho = -rho
        begin, end = -end, -begin
    while angle - reference_angle < -math.pi / 2:
        angle += math.pi
        rho = -rho
        begin, end = -end, -begin
    return angle, rho, begin, end


def angle_distance(left: float, right: float) -> float:
    value = abs(left - right) % math.pi
    return min(value, math.pi - value)


def intervals_compatible(intervals: list[tuple[float, float]], candidate: tuple[float, float]) -> bool:
    return any(not (candidate[0] > end + 14 or candidate[1] < begin - 14) for begin, end in intervals)


def cluster_hough_lines(lines: list[RawLine]) -> list[LineCluster]:
    clusters: list[LineCluster] = []
    for line in sorted(lines, key=lambda item: (-item.length, item.y0, item.x0, item.y1, item.x1)):
        angle, rho, begin, end = line_coordinates(line)
        best: tuple[float, LineCluster] | None = None
        for cluster in clusters:
            aligned = align_line_coordinates((angle, rho, begin, end), cluster.angle)
            aligned_angle, aligned_rho, aligned_begin, aligned_end = aligned
            angular = abs(aligned_angle - cluster.angle)
            rho_distance = abs(aligned_rho - cluster.rho)
            if angular <= math.radians(2.0) and rho_distance <= 2.25 and intervals_compatible(
                cluster.intervals, (aligned_begin, aligned_end)
            ):
                cost = angular * 100 + rho_distance
                if best is None or cost < best[0]:
                    best = (cost, cluster)
        if best is None:
            clusters.append(LineCluster(angle, rho, [(begin, end)], [line]))
        else:
            cluster = best[1]
            _angle, _rho, aligned_begin, aligned_end = align_line_coordinates(
                (angle, rho, begin, end), cluster.angle
            )
            cluster.intervals.append((aligned_begin, aligned_end))
            cluster.lines.append(line)
            weights = [max(item.length, 1.0) for item in cluster.lines]
            aligned_members = [
                align_line_coordinates(line_coordinates(item), cluster.angle)
                for item in cluster.lines
            ]
            cluster.rho = sum(item[1] * weight for item, weight in zip(aligned_members, weights)) / sum(weights)
            x = sum(math.cos(item[0]) * weight for item, weight in zip(aligned_members, weights))
            y = sum(math.sin(item[0]) * weight for item, weight in zip(aligned_members, weights))
            cluster.angle = math.atan2(y, x)
    return clusters


def merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1] + 14:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return [(item[0], item[1]) for item in result]


def segment_mask(shape: tuple[int, int], begin: tuple[float, float], end: tuple[float, float], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(begin[0])), int(round(begin[1]))),
        (int(round(end[0])), int(round(end[1]))),
        255,
        max(1, width),
        cv2.LINE_8,
    )
    return mask


def clusters_to_primitives(
    clusters: list[LineCluster], binary: np.ndarray, distance: np.ndarray
) -> list[Primitive]:
    output: list[Primitive] = []
    for cluster in clusters:
        direction = (math.cos(cluster.angle), math.sin(cluster.angle))
        normal = (-direction[1], direction[0])
        for begin_t, end_t in merged_intervals(cluster.intervals):
            if end_t - begin_t < 18:
                continue
            begin = (
                direction[0] * begin_t + normal[0] * cluster.rho,
                direction[1] * begin_t + normal[1] * cluster.rho,
            )
            end = (
                direction[0] * end_t + normal[0] * cluster.rho,
                direction[1] * end_t + normal[1] * cluster.rho,
            )
            thin_mask = segment_mask(binary.shape, begin, end, 3)
            width_values = distance[thin_mask > 0]
            width = max(1.0, float(np.median(width_values) * 2.0)) if width_values.size else 1.0
            corridor_width = max(3, int(math.ceil(width + 3)))
            corridor = segment_mask(binary.shape, begin, end, corridor_width)
            supported = int(np.count_nonzero(binary & (corridor > 0)))
            corridor_pixels = int(np.count_nonzero(corridor))
            expected = max(1, int(round((end_t - begin_t) * max(width, 1.0))))
            support_score = min(10000, (supported * 10000) // max(1, min(corridor_pixels, expected + corridor_width * 2)))
            local_residual = max(0, supported - expected)
            output.append(Primitive(begin, end, width, support_score, local_residual))
    output.sort(
        key=lambda item: (
            round(min(item.begin[1], item.end[1]), 3),
            round(min(item.begin[0], item.end[0]), 3),
            round(max(item.begin[1], item.end[1]), 3),
            round(max(item.begin[0], item.end[0]), 3),
        )
    )
    if len(output) > MAX_PRIMITIVES:
        raise EvidenceError("primitive_capacity_exceeded")
    return output


def primitive_coordinates(primitive: Primitive) -> tuple[float, float, float, float]:
    return line_coordinates(RawLine(*primitive.begin, *primitive.end))


def midline_support_ratio(
    left: Primitive, right: Primitive, binary: np.ndarray
) -> tuple[float, float]:
    left_coordinates = primitive_coordinates(left)
    right_coordinates = align_line_coordinates(primitive_coordinates(right), left_coordinates[0])
    angle = (left_coordinates[0] + right_coordinates[0]) / 2
    rho = (left_coordinates[1] + right_coordinates[1]) / 2
    overlap_begin = max(left_coordinates[2], right_coordinates[2])
    overlap_end = min(left_coordinates[3], right_coordinates[3])
    interval_gap = max(0.0, overlap_begin - overlap_end)
    if overlap_end > overlap_begin:
        sample_begin, sample_end = overlap_begin, overlap_end
    else:
        sample_begin, sample_end = overlap_end, overlap_begin
    direction = (math.cos(angle), math.sin(angle))
    normal = (-direction[1], direction[0])
    samples = max(3, min(33, int(abs(sample_end - sample_begin)) + 1))
    supported = 0
    for index in range(samples):
        fraction = 0.5 if samples == 1 else index / (samples - 1)
        position = sample_begin + (sample_end - sample_begin) * fraction
        x = int(round(direction[0] * position + normal[0] * rho))
        y = int(round(direction[1] * position + normal[1] * rho))
        if 0 <= x < binary.shape[1] and 0 <= y < binary.shape[0] and binary[y, x]:
            supported += 1
    return supported / samples, interval_gap


def merge_duplicate_primitives(
    primitives: list[Primitive], binary: np.ndarray, distance: np.ndarray
) -> list[Primitive]:
    """Merge Hough duplicates only when the ink between centerlines supports it."""
    parent = list(range(len(primitives)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_id, left in enumerate(primitives):
        left_coordinates = primitive_coordinates(left)
        for right_id in range(left_id + 1, len(primitives)):
            right = primitives[right_id]
            right_coordinates = align_line_coordinates(primitive_coordinates(right), left_coordinates[0])
            angular = abs(left_coordinates[0] - right_coordinates[0])
            rho_distance = abs(left_coordinates[1] - right_coordinates[1])
            rho_limit = max(6.0, 0.6 * max(left.width, right.width))
            support_ratio, interval_gap = midline_support_ratio(left, right, binary)
            if (
                angular <= math.radians(3.0)
                and rho_distance <= rho_limit
                and interval_gap <= 14.0
                and support_ratio >= 0.75
            ):
                union(left_id, right_id)

    groups: dict[int, list[Primitive]] = {}
    for primitive_id, primitive in enumerate(primitives):
        groups.setdefault(find(primitive_id), []).append(primitive)
    output = []
    for group in groups.values():
        reference = max(group, key=lambda item: (item.length, -item.begin[1], -item.begin[0]))
        reference_coordinates = primitive_coordinates(reference)
        aligned = [
            align_line_coordinates(primitive_coordinates(item), reference_coordinates[0])
            for item in group
        ]
        weights = [max(item.length, 1.0) for item in group]
        rho = sum(item[1] * weight for item, weight in zip(aligned, weights)) / sum(weights)
        begin_t = min(item[2] for item in aligned)
        end_t = max(item[3] for item in aligned)
        direction = (math.cos(reference_coordinates[0]), math.sin(reference_coordinates[0]))
        normal = (-direction[1], direction[0])
        begin = (direction[0] * begin_t + normal[0] * rho, direction[1] * begin_t + normal[1] * rho)
        end = (direction[0] * end_t + normal[0] * rho, direction[1] * end_t + normal[1] * rho)
        thin_mask = segment_mask(binary.shape, begin, end, 3)
        width_values = distance[thin_mask > 0]
        width = max(1.0, float(np.median(width_values) * 2.0)) if width_values.size else 1.0
        corridor_width = max(3, int(math.ceil(width + 3)))
        corridor = segment_mask(binary.shape, begin, end, corridor_width)
        supported = int(np.count_nonzero(binary & (corridor > 0)))
        corridor_pixels = int(np.count_nonzero(corridor))
        expected = max(1, int(round(math.hypot(end[0] - begin[0], end[1] - begin[1]) * width)))
        score = min(10000, (supported * 10000) // max(1, min(corridor_pixels, expected + corridor_width * 2)))
        output.append(Primitive(begin, end, width, score, max(0, supported - expected)))
    output.sort(key=lambda item: (item.begin[1], item.begin[0], item.end[1], item.end[0]))
    return output


def infinite_intersection(left: Primitive, right: Primitive) -> tuple[float, float] | None:
    p = left.begin
    r = (left.end[0] - left.begin[0], left.end[1] - left.begin[1])
    q = right.begin
    s = (right.end[0] - right.begin[0], right.end[1] - right.begin[1])
    denominator = cross(r, s)
    if abs(denominator) < 1e-6:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    scale = cross(q_minus_p, s) / denominator
    return p[0] + scale * r[0], p[1] + scale * r[1]


def extension_support_ratio(
    begin: tuple[float, float], end: tuple[float, float], binary: np.ndarray
) -> float:
    distance = point_distance(begin, end)
    samples = max(3, min(41, int(math.ceil(distance)) + 1))
    supported = 0
    for index in range(samples):
        fraction = index / (samples - 1)
        x = int(round(begin[0] + (end[0] - begin[0]) * fraction))
        y = int(round(begin[1] + (end[1] - begin[1]) * fraction))
        x0, x1 = max(0, x - 1), min(binary.shape[1], x + 2)
        y0, y1 = max(0, y - 1), min(binary.shape[0], y + 2)
        if x0 < x1 and y0 < y1 and np.any(binary[y0:y1, x0:x1]):
            supported += 1
    return supported / samples


def snap_nearby_junctions(primitives: list[Primitive], binary: np.ndarray) -> list[Primitive]:
    proposals: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for left_id, left in enumerate(primitives):
        for right_id in range(left_id + 1, len(primitives)):
            right = primitives[right_id]
            point = infinite_intersection(left, right)
            if point is None:
                continue
            left_endpoint = min(range(2), key=lambda index: point_distance(point, (left.begin, left.end)[index]))
            right_endpoint = min(range(2), key=lambda index: point_distance(point, (right.begin, right.end)[index]))
            left_distance = point_distance(point, (left.begin, left.end)[left_endpoint])
            right_distance = point_distance(point, (right.begin, right.end)[right_endpoint])
            tolerance = max(36.0, (left.width + right.width) / 2 + 6.0)
            if (
                left_distance <= tolerance
                and right_distance <= tolerance
                and extension_support_ratio((left.begin, left.end)[left_endpoint], point, binary) >= 0.75
                and extension_support_ratio((right.begin, right.end)[right_endpoint], point, binary) >= 0.75
            ):
                proposals.setdefault((left_id, left_endpoint), []).append(point)
                proposals.setdefault((right_id, right_endpoint), []).append(point)
    output = []
    for primitive_id, primitive in enumerate(primitives):
        endpoints = [primitive.begin, primitive.end]
        for endpoint_id in range(2):
            values = proposals.get((primitive_id, endpoint_id), [])
            if values:
                endpoints[endpoint_id] = (
                    sum(item[0] for item in values) / len(values),
                    sum(item[1] for item in values) / len(values),
                )
        output.append(
            Primitive(endpoints[0], endpoints[1], primitive.width, primitive.support_score, primitive.local_residual)
        )
    output.sort(key=lambda item: (item.begin[1], item.begin[0], item.end[1], item.end[0]))
    return output


def extend_terminal_endpoints(primitives: list[Primitive], binary: np.ndarray) -> list[Primitive]:
    terminal_references: set[tuple[int, int]] = set()
    endpoints = [
        (point, primitive_id, endpoint_id)
        for primitive_id, primitive in enumerate(primitives)
        for endpoint_id, point in enumerate((primitive.begin, primitive.end))
    ]
    groups: list[list[tuple[tuple[float, float], int, int]]] = []
    for item in endpoints:
        target = next(
            (
                group
                for group in groups
                if point_distance(
                    item[0],
                    (
                        sum(member[0][0] for member in group) / len(group),
                        sum(member[0][1] for member in group) / len(group),
                    ),
                )
                <= 6.0
            ),
            None,
        )
        if target is None:
            groups.append([item])
        else:
            target.append(item)
    for group in groups:
        if len({item[1] for item in group}) == 1:
            terminal_references.add((group[0][1], group[0][2]))

    output = []
    for primitive_id, primitive in enumerate(primitives):
        points = [primitive.begin, primitive.end]
        for endpoint_id in range(2):
            if (primitive_id, endpoint_id) not in terminal_references:
                continue
            other = points[1 - endpoint_id]
            endpoint = points[endpoint_id]
            length = point_distance(other, endpoint)
            direction = ((endpoint[0] - other[0]) / length, (endpoint[1] - other[1]) / length)
            last_supported = endpoint
            missing_run = 0
            for step in range(1, 49):
                candidate = (endpoint[0] + direction[0] * step, endpoint[1] + direction[1] * step)
                x, y = int(round(candidate[0])), int(round(candidate[1]))
                x0, x1 = max(0, x - 1), min(binary.shape[1], x + 2)
                y0, y1 = max(0, y - 1), min(binary.shape[0], y + 2)
                if x0 < x1 and y0 < y1 and np.any(binary[y0:y1, x0:x1]):
                    last_supported = candidate
                    missing_run = 0
                else:
                    missing_run += 1
                    if missing_run >= 3:
                        break
            points[endpoint_id] = last_supported
        output.append(
            Primitive(points[0], points[1], primitive.width, primitive.support_score, primitive.local_residual)
        )
    output.sort(key=lambda item: (item.begin[1], item.begin[0], item.end[1], item.end[0]))
    return output


def canonicalize_primitives(primitives: list[Primitive]) -> list[Primitive]:
    output = []
    for primitive in primitives:
        begin, end = primitive.begin, primitive.end
        if (round(end[1], 4), round(end[0], 4)) < (round(begin[1], 4), round(begin[0], 4)):
            begin, end = end, begin
        output.append(
            Primitive(begin, end, primitive.width, primitive.support_score, primitive.local_residual)
        )
    output.sort(
        key=lambda item: (
            round(item.begin[1], 4),
            round(item.begin[0], 4),
            round(item.end[1], 4),
            round(item.end[0], 4),
        )
    )
    return output


def line_source_without_tiny_components(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.where(binary, 255, 0).astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(binary)
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area >= 18 and max(width, height) >= 18:
            retained[labels == component_id] = True
    return retained


def cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def proper_intersection(left: Primitive, right: Primitive) -> tuple[float, float] | None:
    p = left.begin
    r = (left.end[0] - left.begin[0], left.end[1] - left.begin[1])
    q = right.begin
    s = (right.end[0] - right.begin[0], right.end[1] - right.begin[1])
    denominator = cross(r, s)
    if abs(denominator) < 1e-6:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    t = cross(q_minus_p, s) / denominator
    u = cross(q_minus_p, r) / denominator
    margin_left = min(0.12, 5.0 / max(left.length, 1.0))
    margin_right = min(0.12, 5.0 / max(right.length, 1.0))
    if margin_left < t < 1 - margin_left and margin_right < u < 1 - margin_right:
        return p[0] + t * r[0], p[1] + t * r[1]
    return None


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def endpoint_groups(primitives: list[Primitive]) -> list[tuple[tuple[float, float], list[int]]]:
    endpoints = [
        (point, primitive_id)
        for primitive_id, primitive in enumerate(primitives)
        for point in (primitive.begin, primitive.end)
    ]
    groups: list[list[tuple[tuple[float, float], int]]] = []
    for point, primitive_id in endpoints:
        target = None
        for group in groups:
            center = (
                sum(item[0][0] for item in group) / len(group),
                sum(item[0][1] for item in group) / len(group),
            )
            widths = [primitives[item[1]].width for item in group]
            tolerance = max(5.0, (max(widths) + primitives[primitive_id].width) / 2 + 2)
            if point_distance(point, center) <= tolerance:
                target = group
                break
        if target is None:
            groups.append([(point, primitive_id)])
        else:
            target.append((point, primitive_id))
    result = []
    for group in groups:
        center = (
            sum(item[0][0] for item in group) / len(group),
            sum(item[0][1] for item in group) / len(group),
        )
        result.append((center, sorted(set(item[1] for item in group))))
    return result


def bbox_q4(center: tuple[float, float], radius: float) -> list[int]:
    return [
        max(0, int(round((center[0] - radius) * 16))),
        max(0, int(round((center[1] - radius) * 16))),
        min(IMAGE_SIZE * 16, int(round((center[0] + radius) * 16))),
        min(IMAGE_SIZE * 16, int(round((center[1] + radius) * 16))),
    ]


def event_records(primitives: list[Primitive]) -> list[dict[str, object]]:
    raw_events: list[tuple[tuple[float, float], list[int], str, int]] = []
    for center, incident in endpoint_groups(primitives):
        if len(incident) == 1:
            raw_events.append((center, incident, "terminal", 9000))
        else:
            raw_events.append((center, incident, "junction", 9000))
    for left_index in range(len(primitives)):
        for right_index in range(left_index + 1, len(primitives)):
            point = proper_intersection(primitives[left_index], primitives[right_index])
            if point is not None:
                raw_events.append((point, [left_index, right_index], "intersection", 9000))

    # A bounded neutral glyph hypothesis: a short isolated stroke placed beside
    # a substantially longer carrier. It does not predict the character label.
    if len(primitives) == 2:
        ordered = sorted(enumerate(primitives), key=lambda item: item[1].length)
        short_id, short = ordered[0]
        long_id, long = ordered[1]
        separation = min(
            point_distance(point, endpoint)
            for point in (short.begin, short.end)
            for endpoint in (long.begin, long.end)
        )
        if short.length / max(long.length, 1.0) < 0.65 and separation / max(long.length, 1.0) > 0.25:
            center = ((short.begin[0] + short.end[0]) / 2, (short.begin[1] + short.end[1]) / 2)
            raw_events.append((center, [short_id], "glyph_like", 7500))

    raw_events.sort(key=lambda item: (item[0][1], item[0][0], item[2], item[1]))
    if len(raw_events) > MAX_EVENTS:
        raise EvidenceError("event_capacity_exceeded")
    records = []
    for event_id, (center, incident, kind, score) in enumerate(raw_events):
        radius = max([primitives[index].width for index in incident] or [2.0]) + 3
        records.append(
            {
                "bbox_q4": bbox_q4(center, radius),
                "centroid_q4": [int(round(center[0] * 16)), int(round(center[1] * 16))],
                "id": event_id,
                "incident_primitive_ids": incident,
                "kind_hypothesis": kind,
                "score": score,
            }
        )
    return records


def primitive_record(primitive_id: int, primitive: Primitive) -> dict[str, object]:
    begin, end = primitive.begin, primitive.end
    if (round(end[1], 4), round(end[0], 4)) < (round(begin[1], 4), round(begin[0], 4)):
        begin, end = end, begin
    angle = math.atan2(end[1] - begin[1], end[0] - begin[0]) % math.pi
    return {
        "begin_q4": [int(round(begin[0] * 16)), int(round(begin[1] * 16))],
        "closed": False,
        "end_q4": [int(round(end[0] * 16)), int(round(end[1] * 16))],
        "id": primitive_id,
        "local_residual_count": primitive.local_residual,
        "orientation_bucket": int(round(angle * ANGLE_BUCKETS / math.pi)) % ANGLE_BUCKETS,
        "straight": True,
        "support_score": primitive.support_score,
        "width_q4": max(1, int(round(primitive.width * 16))),
    }


def extract(data: bytes) -> dict[str, object]:
    image = decode_png(data)
    binary = image < 250
    line_source = line_source_without_tiny_components(binary)
    skeleton = morphological_skeleton(line_source)
    raw = cv2.HoughLinesP(
        skeleton,
        1,
        math.pi / ANGLE_BUCKETS,
        threshold=22,
        minLineLength=18,
        maxLineGap=10,
    )
    if raw is None:
        raise EvidenceError("no_line_evidence")
    lines = [RawLine(*map(float, np.asarray(item).reshape(-1)[:4])) for item in raw]
    clusters = cluster_hough_lines(lines)
    distance = cv2.distanceTransform(np.where(binary, 255, 0).astype(np.uint8), cv2.DIST_L2, 5)
    primitives = clusters_to_primitives(clusters, binary, distance)
    primitives = merge_duplicate_primitives(primitives, binary, distance)
    primitives = snap_nearby_junctions(primitives, binary)
    primitives = extend_terminal_endpoints(primitives, binary)
    primitives = canonicalize_primitives(primitives)
    if not primitives:
        raise EvidenceError("no_primitive_evidence")
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.where(binary, 255, 0).astype(np.uint8), connectivity=8
    )
    if np.any(binary):
        ys, xs = np.nonzero(binary)
        residual_bbox = [
            int(xs.min()) * 16,
            int(ys.min()) * 16,
            int(xs.max()) * 16,
            int(ys.max()) * 16,
        ]
    else:
        residual_bbox = [0, 0, 0, 0]
    evidence = {
        "event_regions": event_records(primitives),
        "image": {"coordinate_format": "Q4-pixel", "height": IMAGE_SIZE, "width": IMAGE_SIZE},
        "residual": {
            "bbox_q4": residual_bbox,
            "component_count": max(0, int(count) - 1),
            "foreground_pixel_count": int(np.count_nonzero(binary)),
        },
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "stroke_primitives": [
            primitive_record(index, primitive) for index, primitive in enumerate(primitives)
        ],
    }
    encoded = canonical_bytes(evidence)
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise EvidenceError("output_capacity_exceeded")
    return evidence


def main() -> int:
    try:
        result = extract(sys.stdin.buffer.read())
    except (EvidenceError, cv2.error, ValueError) as error:
        result = {
            "reject_reason": str(error),
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
        }
    encoded = canonical_bytes(result)
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise SystemExit("bounded output contract violated")
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
