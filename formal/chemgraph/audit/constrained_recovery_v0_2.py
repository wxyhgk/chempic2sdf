"""Executable finite specification for constrained recovery v0.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from formal.chemgraph.evidence_core_v0_4.candidate.candidate import graphs_isomorphic


Point = tuple[int, int]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class SegmentGeometry:
    segment_id: str
    begin: Point
    end: Point

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be nonempty")
        if self.begin == self.end:
            raise ValueError("segment geometry must be nonzero")

    def record(self) -> dict[str, Any]:
        return {
            "begin": list(self.begin),
            "end": list(self.end),
            "segment_id": self.segment_id,
        }


@dataclass(frozen=True, slots=True)
class CarrierEvidence:
    carrier_id: str
    primitive_id: str
    support_pixels: tuple[str, ...]
    support_segments: tuple[SegmentGeometry, ...]
    support_score: int
    minimum_support_score: int
    owner_kind: str
    owner_id: str

    def admissibility_errors(self) -> tuple[str, ...]:
        errors = []
        if not self.carrier_id:
            errors.append("empty_carrier_id")
        if not self.primitive_id:
            errors.append("empty_primitive_id")
        if not self.support_pixels:
            errors.append("empty_pixel_support")
        if any(not token for token in self.support_pixels):
            errors.append("unnamed_pixel_support")
        if len(self.support_pixels) != len(set(self.support_pixels)):
            errors.append("duplicate_pixel_support")
        if not self.support_segments:
            errors.append("empty_segment_support")
        segment_ids = [segment.segment_id for segment in self.support_segments]
        if len(segment_ids) != len(set(segment_ids)):
            errors.append("duplicate_segment_support")
        if isinstance(self.support_score, bool) or not isinstance(self.support_score, int):
            errors.append("invalid_support_score")
        elif self.support_score < self.minimum_support_score:
            errors.append("support_below_threshold")
        if self.owner_kind not in {"bond_lane", "glyph_stroke", "marker"}:
            errors.append("invalid_owner_kind")
        if not self.owner_id:
            errors.append("empty_owner_id")
        return tuple(errors)

    @property
    def admissible(self) -> bool:
        return not self.admissibility_errors()

    def support_tokens(self) -> set[tuple[str, str]]:
        return ({("pixel", token) for token in self.support_pixels}
                | {("segment", item.segment_id) for item in self.support_segments})

    def record(self) -> dict[str, Any]:
        return {
            "carrier_id": self.carrier_id,
            "minimum_support_score": self.minimum_support_score,
            "owner": {"id": self.owner_id, "kind": self.owner_kind},
            "primitive_id": self.primitive_id,
            "support_pixels": list(self.support_pixels),
            "support_score": self.support_score,
            "support_segments": [segment.record() for segment in self.support_segments],
        }


@dataclass(frozen=True, slots=True)
class BondLane:
    primitive_id: str
    geometry: SegmentGeometry
    support: CarrierEvidence
    bond_id: str
    begin_node: int
    end_node: int

    def __post_init__(self) -> None:
        if self.begin_node == self.end_node:
            raise ValueError("bond lane must have two distinct endpoints")
        if not self.support.admissible:
            raise ValueError("bond lane requires admissible carrier support")
        if self.support.primitive_id != self.primitive_id:
            raise ValueError("carrier primitive does not match lane primitive")
        if self.support.owner_kind != "bond_lane" or self.support.owner_id != self.bond_id:
            raise ValueError("carrier owner does not match bond lane owner")
        if self.geometry not in self.support.support_segments:
            raise ValueError("lane geometry must be named by carrier segment support")

    def endpoint(self, at_end: bool) -> int:
        return self.end_node if at_end else self.begin_node

    def record(self) -> dict[str, Any]:
        return {
            "bond_id": self.bond_id,
            "carrier": self.support.record(),
            "geometry": self.geometry.record(),
            "primitive_id": self.primitive_id,
        }


@dataclass(frozen=True, slots=True)
class BondBundle:
    bond_id: str
    order: int
    lanes: tuple[BondLane, ...]

    def __post_init__(self) -> None:
        if self.order not in {1, 2, 3}:
            raise ValueError("bond order must be 1, 2, or 3")
        if len(self.lanes) != self.order:
            raise ValueError("bond order requires the exact physical lane count")
        if any(lane.bond_id != self.bond_id for lane in self.lanes):
            raise ValueError("all lanes must have the bundle owner")
        endpoint_pairs = {(lane.begin_node, lane.end_node) for lane in self.lanes}
        if len(endpoint_pairs) != 1:
            raise ValueError("all lanes must share the same semantic endpoints")
        primitive_ids = [lane.primitive_id for lane in self.lanes]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("each lane must own a distinct primitive")
        seen_support: set[tuple[str, str]] = set()
        for lane in self.lanes:
            tokens = lane.support.support_tokens()
            if seen_support & tokens:
                raise ValueError("lane support ownership must be mutually exclusive")
            seen_support |= tokens

    @property
    def begin_node(self) -> int:
        return self.lanes[0].endpoint(False)

    @property
    def end_node(self) -> int:
        return self.lanes[0].endpoint(True)

    def edge_record(self) -> dict[str, Any]:
        return {
            "begin": self.begin_node,
            "bond_id": self.bond_id,
            "carrier_ids": [lane.support.carrier_id for lane in self.lanes],
            "end": self.end_node,
            "lane_provenance": [lane.record() for lane in self.lanes],
            "order": self.order,
            "primitive_ids": [lane.primitive_id for lane in self.lanes],
        }


def graph_record(node_ids: Iterable[int], bundles: Iterable[BondBundle]) -> dict[str, Any]:
    bundle_values = list(bundles)
    lanes = [lane for bundle in bundle_values for lane in bundle.lanes]
    primitive_ids = [lane.primitive_id for lane in lanes]
    if len(primitive_ids) != len(set(primitive_ids)):
        raise ValueError("primitive ownership must be exact across the graph")
    seen_support: set[tuple[str, str]] = set()
    for lane in lanes:
        tokens = lane.support.support_tokens()
        if seen_support & tokens:
            raise ValueError("support ownership must be mutually exclusive across the graph")
        seen_support |= tokens
    return {
        "edges": [bundle.edge_record() for bundle in bundle_values],
        "nodes": [{"id": node_id, "role": "implicit"} for node_id in sorted(node_ids)],
        "role_hypotheses": [],
    }


def graph_has_complete_provenance(graph: dict[str, Any]) -> bool:
    all_primitive_ids: list[str] = []
    all_carrier_ids: list[str] = []
    all_support_tokens: set[tuple[str, str]] = set()
    for edge in graph.get("edges", []):
        order = edge.get("order")
        if order not in {1, 2, 3}:
            return False
        primitive_ids = edge.get("primitive_ids")
        carrier_ids = edge.get("carrier_ids")
        lanes = edge.get("lane_provenance")
        if not all(isinstance(items, list) and len(items) == order
                   for items in (primitive_ids, carrier_ids, lanes)):
            return False
        if len(set(primitive_ids)) != order or len(set(carrier_ids)) != order:
            return False
        all_primitive_ids.extend(primitive_ids)
        all_carrier_ids.extend(carrier_ids)
        for primitive_id, carrier_id, lane in zip(
            primitive_ids, carrier_ids, lanes, strict=True,
        ):
            if lane.get("primitive_id") != primitive_id:
                return False
            carrier = lane.get("carrier", {})
            if carrier.get("carrier_id") != carrier_id:
                return False
            if lane.get("bond_id") != edge.get("bond_id"):
                return False
            if carrier.get("primitive_id") != primitive_id:
                return False
            if carrier.get("owner") != {"id": edge.get("bond_id"), "kind": "bond_lane"}:
                return False
            if not carrier.get("support_pixels") or not carrier.get("support_segments"):
                return False
            if carrier.get("support_score", -1) < carrier.get("minimum_support_score", 0):
                return False
            geometry = lane.get("geometry")
            if geometry not in carrier.get("support_segments", []):
                return False
            tokens = ({("pixel", str(token)) for token in carrier["support_pixels"]}
                      | {("segment", str(item.get("segment_id")))
                         for item in carrier["support_segments"]})
            if all_support_tokens & tokens:
                return False
            all_support_tokens |= tokens
    return (len(all_primitive_ids) == len(set(all_primitive_ids))
            and len(all_carrier_ids) == len(set(all_carrier_ids)))


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    candidate_id: str
    graph: dict[str, Any]


def candidate_set_sha256(candidates: list[CandidateHypothesis]) -> str:
    payload = [
        {"candidate_id": candidate.candidate_id, "graph": candidate.graph}
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    ]
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class CoverageCertificate:
    evidence_id: str
    evidence_sha256: str
    enumerator_id: str
    coverage_audit_sha256: str
    candidate_ids: tuple[str, ...]
    supported_hypothesis_ids: tuple[str, ...]
    candidate_set_sha256: str

    def valid_for(self, candidates: list[CandidateHypothesis]) -> bool:
        actual_ids = tuple(sorted(candidate.candidate_id for candidate in candidates))
        return (
            bool(self.evidence_id)
            and _is_sha256(self.evidence_sha256)
            and bool(self.enumerator_id)
            and _is_sha256(self.coverage_audit_sha256)
            and bool(self.supported_hypothesis_ids)
            and len(actual_ids) == len(set(actual_ids))
            and tuple(sorted(self.candidate_ids)) == actual_ids
            and set(self.supported_hypothesis_ids) <= set(actual_ids)
            and self.candidate_set_sha256 == candidate_set_sha256(candidates)
        )


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    calibration_id: str
    calibration_sha256: str
    sample_count: int
    epsilon: int
    maximum_observed_error: int

    @property
    def valid(self) -> bool:
        return (
            bool(self.calibration_id)
            and _is_sha256(self.calibration_sha256)
            and isinstance(self.sample_count, int)
            and not isinstance(self.sample_count, bool)
            and self.sample_count > 0
            and isinstance(self.epsilon, int)
            and self.epsilon >= 0
            and 0 <= self.maximum_observed_error <= self.epsilon
        )


@dataclass(frozen=True, slots=True)
class CalibratedInterval:
    candidate_id: str
    observed_score: int
    lower: int
    upper: int
    calibration_id: str
    calibration_sha256: str

    @classmethod
    def from_score(
        cls, candidate_id: str, observed_score: int, calibration: CalibrationEvidence,
    ) -> "CalibratedInterval":
        if not calibration.valid or observed_score < 0:
            raise ValueError("valid calibration and nonnegative score are required")
        return cls(
            candidate_id=candidate_id,
            observed_score=observed_score,
            lower=max(0, observed_score - calibration.epsilon),
            upper=observed_score + calibration.epsilon,
            calibration_id=calibration.calibration_id,
            calibration_sha256=calibration.calibration_sha256,
        )

    def valid_for(self, calibration: CalibrationEvidence) -> bool:
        return (
            calibration.valid
            and self.observed_score >= 0
            and self.calibration_id == calibration.calibration_id
            and self.calibration_sha256 == calibration.calibration_sha256
            and self.lower == max(0, self.observed_score - calibration.epsilon)
            and self.upper == self.observed_score + calibration.epsilon
        )


def _equivalence_classes(candidates: list[CandidateHypothesis]) -> list[list[int]]:
    classes: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        for graph_class in classes:
            if graphs_isomorphic(candidate.graph, candidates[graph_class[0]].graph):
                graph_class.append(index)
                break
        else:
            classes.append([index])
    return classes


def certify_graph(
    candidates: list[CandidateHypothesis],
    *,
    coverage: CoverageCertificate | None,
    calibration: CalibrationEvidence | None = None,
    intervals: list[CalibratedInterval] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {"decision": "Reject", "reason": "empty_candidate_set"}
    if coverage is None or not coverage.valid_for(candidates):
        return {"decision": "Ambiguous", "reason": "coverage_not_certified"}
    if any(not graph_has_complete_provenance(candidate.graph) for candidate in candidates):
        return {"decision": "Ambiguous", "reason": "incomplete_edge_provenance"}

    classes = _equivalence_classes(candidates)
    if len(classes) == 1:
        return {
            "authority": "coverage_singleton",
            "decision": "Graph",
            "graph": candidates[classes[0][0]].graph,
        }
    if calibration is None or intervals is None or not calibration.valid:
        return {"decision": "Ambiguous", "reason": "calibration_not_certified"}
    by_id = {interval.candidate_id: interval for interval in intervals}
    expected_ids = {candidate.candidate_id for candidate in candidates}
    if set(by_id) != expected_ids or len(by_id) != len(intervals):
        return {"decision": "Ambiguous", "reason": "interval_coverage_invalid"}
    if any(not interval.valid_for(calibration) for interval in intervals):
        return {"decision": "Ambiguous", "reason": "interval_binding_invalid"}

    class_intervals = []
    for graph_class in classes:
        members = [by_id[candidates[index].candidate_id] for index in graph_class]
        class_intervals.append((min(item.lower for item in members),
                                min(item.upper for item in members)))
    winners = [
        index for index, (_, upper) in enumerate(class_intervals)
        if all(upper < other_lower for other_index, (other_lower, _) in enumerate(class_intervals)
               if other_index != index)
    ]
    if len(winners) != 1:
        return {"decision": "Ambiguous", "reason": "score_intervals_overlap"}
    winner = winners[0]
    return {
        "authority": "coverage_and_calibrated_interval",
        "calibration_id": calibration.calibration_id,
        "decision": "Graph",
        "graph": candidates[classes[winner][0]].graph,
    }


__all__ = [
    "BondBundle",
    "BondLane",
    "CalibratedInterval",
    "CalibrationEvidence",
    "CandidateHypothesis",
    "CarrierEvidence",
    "CoverageCertificate",
    "SegmentGeometry",
    "candidate_set_sha256",
    "certify_graph",
    "graph_has_complete_provenance",
    "graph_record",
]
