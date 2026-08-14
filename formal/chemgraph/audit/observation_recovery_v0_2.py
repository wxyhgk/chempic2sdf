"""Finite observation-class recovery audit for Grammar v0.2.

This module is deliberately an *audit oracle*, not an image decoder.  It uses
the frozen v0.1 scene manifest to enumerate every interpretation that produced
one opaque observation.  Semantic ownership in an ``OracleField`` therefore
comes from a finite witness (or a declared global prior); it is never presented
as something computed from the observation hash.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import permutations
import json
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .finite_micro_scene_v0_1 import (
    Bond,
    Node,
    PhysicalMember,
    Scene,
    canonical_graph_signature,
    fixture_scenes,
    iter_core_scenes,
)


SCHEMA_VERSION = "observation-recovery-v0.2"
ORACLE_FIELD_SCHEMA_VERSION = "chemgraph-oracle-field-v0.2"

FROZEN_MANIFEST_SHA256 = (
    "f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7"
)
FROZEN_CERTIFICATES_SHA256 = (
    "f97262008a50edf26b263745b9058533820be8bb72fdffcc3f9c38e6f9839485"
)
EXPECTED_SCENE_COUNT = 66_716
EXPECTED_CERTIFICATE_COUNT = 3_518

PARALLEL_PRIOR_ID = (
    "terminal_aligned_parallel_strokes_are_one_multiple_bond_v1"
)
CROSSING_PRIOR_ID = "unmarked_transversal_crossing_is_disconnected_v1"
STRATEGY_B_PRIOR_IDS = (PARALLEL_PRIOR_ID, CROSSING_PRIOR_ID)

SIMPLE_LAYER = "simple_line_angle_valid"
CROSSING_LAYER = "crossing_allowed"
SUPPORTED_DOMAINS = {
    "simple_domain": frozenset({SIMPLE_LAYER}),
    "crossing_domain": frozenset({SIMPLE_LAYER, CROSSING_LAYER}),
}

AMBIGUITY_NONE = "none"
AMBIGUITY_PARALLEL = "parallel_multiplicity"
AMBIGUITY_CROSSING = "crossing_junction"
AMBIGUITY_COMBINED = "parallel_multiplicity+crossing_junction"
AMBIGUITY_UNCLASSIFIED = "unclassified_graph_collision"


class RecoveryAuditError(RuntimeError):
    """Raised when a frozen-input or recovery invariant is violated."""


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    manifest_path: Path
    certificates_path: Path
    manifest_sha256: str
    certificates_sha256: str
    manifest_rows: tuple[dict[str, Any], ...]
    certificates: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One finite-domain semantic witness for an observation.

    ``field`` includes semantic bundle ownership from the Scene witness.  That
    ownership is audit-only and must not be treated as image-computable.
    """

    scene_id: str
    graph_signature: str
    grammar_layer: str
    field: dict[str, Any]

    @property
    def field_key(self) -> str:
        return canonical_json(self.field)

    @property
    def crossing_score(self) -> int:
        return len(self.field["protocol_interpretation"]["crossings"])

    @property
    def parallel_score(self) -> tuple[int, int, tuple[int, ...]]:
        orders = tuple(
            sorted(
                (
                    bundle["order_candidate"]
                    for bundle in self.field["protocol_interpretation"][
                        "bond_bundles"
                    ]
                    if bundle["order_candidate"] > 1
                ),
                reverse=True,
            )
        )
        return (sum(order - 1 for order in orders), len(orders), orders)


@dataclass(frozen=True, slots=True)
class ObservationClass:
    observation_sha256: str
    visible_lanes: tuple[dict[str, Any], ...]
    candidates: tuple[Candidate, ...]

    @property
    def graph_signatures(self) -> frozenset[str]:
        return frozenset(candidate.graph_signature for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    result_type: str
    graph_signature: str | None
    protocol_choice_graph_signature: str | None
    ambiguity_kind: str
    oracle_field: dict[str, Any] | None
    candidate_count: int
    survivor_count: int
    survivor_graph_count: int
    field_strict: bool
    decode_left_inverse: bool | None
    active_prior_ids: tuple[str, ...]
    evidence_factorizes_on_finite_domain: bool = True
    observation_extractor_implemented: bool = False
    semantic_field_factorizes_through_evidence: bool = False
    audit_oracle_only: bool = True
    deployable_from_opaque_observation: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "active_prior_ids": list(self.active_prior_ids),
            "ambiguity_kind": self.ambiguity_kind,
            "audit_oracle_only": self.audit_oracle_only,
            "candidate_count": self.candidate_count,
            "decode_left_inverse": self.decode_left_inverse,
            "deployable_from_opaque_observation": (
                self.deployable_from_opaque_observation
            ),
            "evidence_factorizes_on_finite_domain": (
                self.evidence_factorizes_on_finite_domain
            ),
            "field_strict": self.field_strict,
            "graph_signature": self.graph_signature,
            "observation_extractor_implemented": (
                self.observation_extractor_implemented
            ),
            "oracle_field": self.oracle_field,
            "protocol_choice_graph_signature": (
                self.protocol_choice_graph_signature
            ),
            "result_type": self.result_type,
            "survivor_count": self.survivor_count,
            "survivor_graph_count": self.survivor_graph_count,
            "semantic_field_factorizes_through_evidence": (
                self.semantic_field_factorizes_through_evidence
            ),
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecoveryAuditError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise RecoveryAuditError(
                    f"Manifest row {line_number} is not a JSON object"
                )
            rows.append(value)
    return tuple(rows)


def load_frozen_inputs(
    manifest_path: Path,
    certificates_path: Path,
) -> FrozenInputs:
    """Load and fully authenticate the frozen v0.1 inputs."""

    manifest_path = manifest_path.resolve()
    certificates_path = certificates_path.resolve()
    manifest_sha = sha256_file(manifest_path)
    certificate_sha = sha256_file(certificates_path)
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RecoveryAuditError(
            "Frozen manifest SHA256 mismatch: "
            f"expected {FROZEN_MANIFEST_SHA256}, got {manifest_sha}"
        )
    if certificate_sha != FROZEN_CERTIFICATES_SHA256:
        raise RecoveryAuditError(
            "Frozen collision certificate SHA256 mismatch: "
            f"expected {FROZEN_CERTIFICATES_SHA256}, got {certificate_sha}"
        )

    rows = _read_jsonl(manifest_path)
    if len(rows) != EXPECTED_SCENE_COUNT:
        raise RecoveryAuditError(
            f"Expected {EXPECTED_SCENE_COUNT} manifest rows, got {len(rows)}"
        )
    scene_ids = [row.get("scene_id") for row in rows]
    if len(set(scene_ids)) != len(scene_ids) or None in scene_ids:
        raise RecoveryAuditError("Manifest scene IDs are missing or non-unique")

    certificate_document = json.loads(certificates_path.read_text(encoding="utf-8"))
    certificates = tuple(certificate_document.get("certificates", ()))
    if (
        certificate_document.get("certificate_count")
        != EXPECTED_CERTIFICATE_COUNT
        or len(certificates) != EXPECTED_CERTIFICATE_COUNT
    ):
        raise RecoveryAuditError(
            "Frozen certificate count does not match the v0.1 commitment"
        )

    row_by_scene = {row["scene_id"]: row for row in rows}
    certificate_observations: set[str] = set()
    for certificate in certificates:
        observation = certificate.get("observation_sha256")
        if not isinstance(observation, str) or observation in certificate_observations:
            raise RecoveryAuditError("Certificate observations are missing or duplicated")
        certificate_observations.add(observation)
        for witness in certificate.get("witnesses", ()):
            scene_id = witness.get("scene_id")
            row = row_by_scene.get(scene_id)
            if row is None:
                raise RecoveryAuditError(
                    f"Certificate references unknown scene {scene_id!r}"
                )
            if row["production_observation_sha256"] != observation:
                raise RecoveryAuditError(
                    f"Certificate observation mismatch for scene {scene_id}"
                )
            if (
                row["canonical_graph_signature_structure"]
                != witness.get("graph_signature")
            ):
                raise RecoveryAuditError(
                    f"Certificate graph signature mismatch for scene {scene_id}"
                )

    manifest_collisions = {
        row["production_observation_sha256"]
        for row in rows
        if row["global_collision_class"] != "unique_graph"
    }
    if certificate_observations != manifest_collisions:
        raise RecoveryAuditError(
            "Certificate observations do not exactly cover manifest collisions"
        )

    return FrozenInputs(
        manifest_path=manifest_path,
        certificates_path=certificates_path,
        manifest_sha256=manifest_sha,
        certificates_sha256=certificate_sha,
        manifest_rows=rows,
        certificates=certificates,
    )


def _canonical_segment(
    begin: tuple[int, int], end: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    if begin == end:
        raise RecoveryAuditError("Oracle field cannot contain a zero-length lane")
    return tuple(sorted((tuple(begin), tuple(end))))  # type: ignore[return-value]


def _line_key(
    segment: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int]:
    begin, end = segment
    dx, dy = end[0] - begin[0], end[1] - begin[1]
    divisor = gcd(abs(dx), abs(dy))
    dx, dy = dx // divisor, dy // divisor
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    normal_x, normal_y = -dy, dx
    return dx, dy, normal_x * begin[0] + normal_y * begin[1]


def _parameter(point: tuple[int, int], line_key: tuple[int, int, int]) -> int:
    return line_key[0] * point[0] + line_key[1] * point[1]


def merge_visible_lanes(
    members: Iterable[PhysicalMember],
) -> tuple[dict[str, Any], ...]:
    """Canonicalize Scene-member lane evidence without semantic ownership.

    Adjacent collinear members are merged.  Consequently the crossing and
    junction witnesses of the same raster expose the same lanes.  This is an
    audit construction from Scene physical members, not a raster extractor.
    """

    by_line: dict[
        tuple[int, int, int], list[tuple[int, int, tuple[int, int], tuple[int, int]]]
    ] = defaultdict(list)
    for member in members:
        segment = _canonical_segment(member.begin, member.end)
        key = _line_key(segment)
        left, right = sorted(
            ((_parameter(point, key), point) for point in segment), key=lambda item: item[0]
        )
        by_line[key].append((left[0], right[0], left[1], right[1]))

    merged: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for key in sorted(by_line):
        intervals = sorted(by_line[key])
        if not intervals:
            continue
        start_t, end_t, start_point, end_point = intervals[0]
        for next_start, next_end, next_start_point, next_end_point in intervals[1:]:
            if next_start <= end_t:
                if next_end > end_t:
                    end_t, end_point = next_end, next_end_point
                continue
            merged.append(_canonical_segment(start_point, end_point))
            start_t, end_t = next_start, next_end
            start_point, end_point = next_start_point, next_end_point
        merged.append(_canonical_segment(start_point, end_point))

    merged.sort()
    return tuple(
        {
            "carrier_segment": [list(begin), list(end)],
            "lane_id": f"lane-{index:03d}",
            "terminal_ports": [list(begin), list(end)],
        }
        for index, (begin, end) in enumerate(merged)
    )


def _member_visible_lane_id(
    member: PhysicalMember, visible_lanes: Sequence[dict[str, Any]]
) -> str:
    segment = _canonical_segment(member.begin, member.end)
    member_key = _line_key(segment)
    member_parameters = sorted(_parameter(point, member_key) for point in segment)
    matches = []
    for lane in visible_lanes:
        lane_segment = tuple(tuple(point) for point in lane["carrier_segment"])
        if _line_key(lane_segment) != member_key:
            continue
        lane_parameters = sorted(_parameter(point, member_key) for point in lane_segment)
        if (
            lane_parameters[0] <= member_parameters[0]
            and member_parameters[1] <= lane_parameters[1]
        ):
            matches.append(lane["lane_id"])
    if len(matches) != 1:
        raise RecoveryAuditError(
            f"Physical member must belong to exactly one visible lane, got {matches}"
        )
    return matches[0]


def _orientation(
    begin: tuple[int, int], end: tuple[int, int], point: tuple[int, int]
) -> int:
    return (end[0] - begin[0]) * (point[1] - begin[1]) - (
        end[1] - begin[1]
    ) * (point[0] - begin[0])


def _proper_cross(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    return (
        _orientation(*left, right[0]) * _orientation(*left, right[1]) < 0
        and _orientation(*right, left[0]) * _orientation(*right, left[1]) < 0
    )


def _crossing_point(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> list[str]:
    # Exact rational strings avoid tolerance and platform-dependent floats.
    a, b = left
    c, d = right
    left_vector = (b[0] - a[0], b[1] - a[1])
    right_vector = (d[0] - c[0], d[1] - c[1])
    denominator = (
        left_vector[0] * right_vector[1]
        - left_vector[1] * right_vector[0]
    )
    displacement = (c[0] - a[0], c[1] - a[1])
    numerator = (
        displacement[0] * right_vector[1]
        - displacement[1] * right_vector[0]
    )
    x = Fraction(a[0] * denominator + numerator * left_vector[0], denominator)
    y = Fraction(a[1] * denominator + numerator * left_vector[1], denominator)
    return [str(x), str(y)]


def _scene_members(scene: Scene) -> Iterator[PhysicalMember]:
    for bond in scene.bonds:
        yield from bond.members


def oracle_field_for_scene(
    scene: Scene,
    visible_lanes: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Build one audit-only semantic witness field.

    The lane evidence is reconstructed from finite Scene physical members.
    Bundle ownership, terminal ownership and junction/crossing interpretation
    are copied from the finite Scene witness and explicitly labelled
    non-deployable.  Equality within each opaque-observation class establishes
    only a finite extensional factorization, not a computable raster extractor.
    """

    if visible_lanes is None:
        visible_lanes = merge_visible_lanes(_scene_members(scene))
    node_by_id = {node.atom_id: node for node in scene.nodes}

    raw_bundles = []
    for bond in scene.bonds:
        begin_node = node_by_id[bond.begin_atom]
        end_node = node_by_id[bond.end_atom]
        envelope = tuple(sorted(((begin_node.x, begin_node.y), (end_node.x, end_node.y))))
        members = tuple(
            sorted(_canonical_segment(member.begin, member.end) for member in bond.members)
        )
        lane_ids = tuple(
            sorted(_member_visible_lane_id(member, visible_lanes) for member in bond.members)
        )
        raw_bundles.append((envelope, bond.order, lane_ids, members))
    raw_bundles.sort()

    bundles = []
    ports = []
    ports_by_coordinate: dict[tuple[int, int], list[str]] = defaultdict(list)
    carrier_by_bundle: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for bundle_index, (envelope, order, lane_ids, members) in enumerate(raw_bundles):
        bundle_id = f"bundle-{bundle_index:03d}"
        carrier_by_bundle[bundle_id] = envelope
        port_ids = []
        for endpoint_index, coordinate in enumerate(envelope):
            port_id = f"port-{bundle_index:03d}-{endpoint_index}"
            port_ids.append(port_id)
            ports_by_coordinate[coordinate].append(port_id)
            ports.append(
                {
                    "bundle_id": bundle_id,
                    "coordinate": list(coordinate),
                    "port_id": port_id,
                }
            )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "lane_ids": list(lane_ids),
                "member_segments": [
                    [list(begin), list(end)] for begin, end in members
                ],
                "order_candidate": order,
                "shared_terminal_envelope": [
                    list(envelope[0]), list(envelope[1])
                ],
                "terminal_port_ids": port_ids,
            }
        )

    junctions = [
        {
            "coordinate": list(coordinate),
            "junction_id": f"junction-{index:03d}",
            "terminal_port_ids": sorted(port_ids),
        }
        for index, (coordinate, port_ids) in enumerate(sorted(ports_by_coordinate.items()))
    ]

    crossings = []
    bundle_ids = sorted(carrier_by_bundle)
    for left_index, left_id in enumerate(bundle_ids):
        left = carrier_by_bundle[left_id]
        for right_id in bundle_ids[left_index + 1 :]:
            right = carrier_by_bundle[right_id]
            if set(left).intersection(right) or not _proper_cross(left, right):
                continue
            crossings.append(
                {
                    "crossing_id": f"crossing-{len(crossings):03d}",
                    "marker_state": "unmarked",
                    "owner_bundle_ids": [left_id, right_id],
                    "point": _crossing_point(left, right),
                }
            )

    return {
        "ambiguity": AMBIGUITY_NONE,
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "evidence_source": "finite_scene_physical_members_not_opaque_tensor",
        "observation_extractor_implemented": False,
        "protocol_interpretation": {
            "bond_bundles": bundles,
            "crossings": crossings,
            "junctions": junctions,
            "terminal_ports": ports,
        },
        "provenance": "finite_scene_semantic_witness_or_declared_prior",
        "schema_version": ORACLE_FIELD_SCHEMA_VERSION,
        "visible_observation_primitives": {"lanes": list(visible_lanes)},
    }


def ambiguous_observation_field(
    visible_lanes: tuple[dict[str, Any], ...], ambiguity_kind: str
) -> dict[str, Any]:
    """Return a conservative field without inventing semantic ownership."""

    return {
        "ambiguity": ambiguity_kind,
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "evidence_source": "finite_scene_physical_members_not_opaque_tensor",
        "observation_extractor_implemented": False,
        "protocol_interpretation": None,
        "provenance": "visible_lanes_plus_observation_class_ambiguity_certificate",
        "schema_version": ORACLE_FIELD_SCHEMA_VERSION,
        "visible_observation_primitives": {"lanes": list(visible_lanes)},
    }


def decode_oracle_field(field: dict[str, Any]) -> str:
    """Decode a field-strict semantic interpretation into a graph signature."""

    if field.get("ambiguity") != AMBIGUITY_NONE:
        raise RecoveryAuditError("An ambiguous Oracle Field cannot decode to a Graph")
    interpretation = field.get("protocol_interpretation")
    if not isinstance(interpretation, dict):
        raise RecoveryAuditError("A strict Oracle Field needs one interpretation")
    bundles = interpretation.get("bond_bundles")
    if not isinstance(bundles, list):
        raise RecoveryAuditError("Oracle Field bond bundles are malformed")

    coordinates = sorted(
        {
            tuple(coordinate)
            for bundle in bundles
            for coordinate in bundle["shared_terminal_envelope"]
        }
    )
    node_id = {coordinate: index for index, coordinate in enumerate(coordinates)}
    nodes = tuple(
        Node(atom_id=index, x=coordinate[0], y=coordinate[1])
        for index, coordinate in enumerate(coordinates)
    )
    bonds = []
    for bundle in bundles:
        begin, end = (
            tuple(coordinate) for coordinate in bundle["shared_terminal_envelope"]
        )
        order = int(bundle["order_candidate"])
        members = tuple(
            PhysicalMember(tuple(member[0]), tuple(member[1]))
            for member in bundle["member_segments"]
        )
        bonds.append(
            Bond(
                node_id[begin],
                node_id[end],
                order=order,
                display_type={1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}[order],
                members=members,
            )
        )
    scene = Scene("decoded-oracle-field", "audit", nodes, tuple(bonds))
    return canonical_graph_signature(scene)


def _fraction_string(value: Fraction) -> str:
    return str(value)


def _fraction_point(point: Sequence[int | str]) -> tuple[Fraction, Fraction]:
    return Fraction(point[0]), Fraction(point[1])


def _fraction_orientation(
    begin: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
    point: tuple[Fraction, Fraction],
) -> Fraction:
    return (end[0] - begin[0]) * (point[1] - begin[1]) - (
        end[1] - begin[1]
    ) * (point[0] - begin[0])


def _fraction_proper_cross(
    left: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    right: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
) -> bool:
    return (
        _fraction_orientation(*left, right[0])
        * _fraction_orientation(*left, right[1])
        < 0
        and _fraction_orientation(*right, left[0])
        * _fraction_orientation(*right, left[1])
        < 0
    )


def _fraction_crossing_point(
    left: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    right: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
) -> list[str]:
    a, b = left
    c, d = right
    left_vector = (b[0] - a[0], b[1] - a[1])
    right_vector = (d[0] - c[0], d[1] - c[1])
    denominator = (
        left_vector[0] * right_vector[1]
        - left_vector[1] * right_vector[0]
    )
    displacement = (c[0] - a[0], c[1] - a[1])
    numerator = (
        displacement[0] * right_vector[1]
        - displacement[1] * right_vector[0]
    )
    point = (
        a[0] + numerator * left_vector[0] / denominator,
        a[1] + numerator * left_vector[1] / denominator,
    )
    return [_fraction_string(point[0]), _fraction_string(point[1])]


def _evidence_bundle_groups(
    visible_lanes: Sequence[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Group terminal-aligned parallel lanes using only vector evidence.

    The grouping is intentionally global.  A group of two or three lanes is
    always treated as one multiple bond.  More than three aligned lanes are
    left as singles because the frozen chemical protocol supports orders 1--3.
    This rule is not conditioned on a GT scene or graph signature.
    """

    aligned: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for lane in visible_lanes:
        segment = tuple(tuple(point) for point in lane["carrier_segment"])
        direction_x, direction_y, _ = _line_key(segment)
        parameters = sorted(_parameter(point, (direction_x, direction_y, 0)) for point in segment)
        aligned[(direction_x, direction_y, parameters[0], parameters[1])].append(lane)

    groups: list[tuple[dict[str, Any], ...]] = []
    for key in sorted(aligned):
        lanes = tuple(sorted(aligned[key], key=lambda lane: lane["lane_id"]))
        if len(lanes) in {2, 3}:
            groups.append(lanes)
        else:
            groups.extend((lane,) for lane in lanes)
    return tuple(groups)


def _average_terminal_envelope(
    lanes: Sequence[dict[str, Any]],
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    first_segment = tuple(tuple(point) for point in lanes[0]["carrier_segment"])
    direction_x, direction_y, _ = _line_key(first_segment)
    aligned_endpoints = []
    for lane in lanes:
        points = [tuple(point) for point in lane["carrier_segment"]]
        points.sort(key=lambda point: direction_x * point[0] + direction_y * point[1])
        aligned_endpoints.append(points)
    count = len(aligned_endpoints)
    return tuple(
        (
            sum(Fraction(points[index][0]) for points in aligned_endpoints) / count,
            sum(Fraction(points[index][1]) for points in aligned_endpoints) / count,
        )
        for index in (0, 1)
    )  # type: ignore[return-value]


def oracle_field_from_observable_evidence(
    visible_lanes: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Apply the two public priors without reading finite-scene meanings.

    This function is deterministic in ``visible_lanes``.  The current audit has
    only proved that this evidence is constant on each finite observation
    class; it has *not* implemented an extractor from an arbitrary opaque
    raster/tensor, so the returned field remains non-deployable.
    """

    groups = _evidence_bundle_groups(visible_lanes)
    bundles = []
    ports = []
    ports_by_coordinate: dict[tuple[str, str], list[str]] = defaultdict(list)
    carrier_by_bundle: dict[
        str, tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    ] = {}
    for bundle_index, lanes in enumerate(groups):
        bundle_id = f"bundle-{bundle_index:03d}"
        envelope = _average_terminal_envelope(lanes)
        carrier_by_bundle[bundle_id] = envelope
        envelope_strings = tuple(
            (_fraction_string(point[0]), _fraction_string(point[1]))
            for point in envelope
        )
        port_ids = []
        for endpoint_index, coordinate in enumerate(envelope_strings):
            port_id = f"port-{bundle_index:03d}-{endpoint_index}"
            port_ids.append(port_id)
            ports_by_coordinate[coordinate].append(port_id)
            ports.append(
                {
                    "bundle_id": bundle_id,
                    "coordinate": list(coordinate),
                    "port_id": port_id,
                }
            )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "lane_ids": [lane["lane_id"] for lane in lanes],
                "member_segments": [lane["carrier_segment"] for lane in lanes],
                "order_candidate": len(lanes),
                "shared_terminal_envelope": [list(point) for point in envelope_strings],
                "terminal_port_ids": port_ids,
            }
        )

    junctions = [
        {
            "coordinate": list(coordinate),
            "junction_id": f"junction-{index:03d}",
            "terminal_port_ids": sorted(port_ids),
        }
        for index, (coordinate, port_ids) in enumerate(sorted(ports_by_coordinate.items()))
    ]
    crossings = []
    bundle_ids = sorted(carrier_by_bundle)
    for left_index, left_id in enumerate(bundle_ids):
        left = carrier_by_bundle[left_id]
        for right_id in bundle_ids[left_index + 1 :]:
            right = carrier_by_bundle[right_id]
            if set(left).intersection(right) or not _fraction_proper_cross(left, right):
                continue
            crossings.append(
                {
                    "crossing_id": f"crossing-{len(crossings):03d}",
                    "marker_state": "unmarked",
                    "owner_bundle_ids": [left_id, right_id],
                    "point": _fraction_crossing_point(left, right),
                }
            )

    return {
        "active_prior_ids": list(STRATEGY_B_PRIOR_IDS),
        "ambiguity": AMBIGUITY_NONE,
        "audit_oracle_only": True,
        "deployable_from_opaque_observation": False,
        "evidence_factorizes_on_finite_domain": True,
        "observation_extractor_implemented": False,
        "protocol_interpretation": {
            "bond_bundles": bundles,
            "crossings": crossings,
            "junctions": junctions,
            "terminal_ports": ports,
        },
        "provenance": "deterministic_global_prior_over_canonical_visible_lane_evidence",
        "schema_version": ORACLE_FIELD_SCHEMA_VERSION,
        "visible_observation_primitives": {"lanes": list(visible_lanes)},
    }


def _generic_graph_signature(field: dict[str, Any]) -> str:
    """Decode prior fields whose terminal coordinates may be rational."""

    bundles = field["protocol_interpretation"]["bond_bundles"]
    coordinates = sorted(
        {
            tuple(coordinate)
            for bundle in bundles
            for coordinate in bundle["shared_terminal_envelope"]
        }
    )
    node_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    edges: dict[tuple[int, int], str] = {}
    for bundle in bundles:
        left, right = (
            node_index[tuple(coordinate)]
            for coordinate in bundle["shared_terminal_envelope"]
        )
        endpoints = tuple(sorted((left, right)))
        if endpoints in edges:
            raise RecoveryAuditError(
                "Observable prior produced repeated semantic endpoint ownership"
            )
        edges[endpoints] = str(bundle["order_candidate"])

    adjacency = {index: set() for index in range(len(coordinates))}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = []
    unseen = set(adjacency)
    while unseen:
        stack = [min(unseen)]
        current = set()
        while stack:
            node = stack.pop()
            if node in current:
                continue
            current.add(node)
            stack.extend(adjacency[node] - current)
        unseen -= current
        components.append(tuple(sorted(current)))

    component_signatures = []
    for component in components:
        candidates = []
        for order in permutations(component):
            edge_part = []
            for left_index, left in enumerate(order):
                for right in order[left_index + 1 :]:
                    edge_part.append(edges.get(tuple(sorted((left, right))), "."))
            candidates.append(
                f"N[{','.join('IMPLICIT' for _ in order)}]E[{','.join(edge_part)}]"
            )
        component_signatures.append(min(candidates))
    return "G{" + "|".join(sorted(component_signatures)) + "}"


def _ambiguity_kind(candidates: Sequence[Candidate]) -> str:
    if len({candidate.graph_signature for candidate in candidates}) <= 1:
        return AMBIGUITY_NONE
    parallel_varies = len({candidate.parallel_score for candidate in candidates}) > 1
    crossing_varies = len({candidate.crossing_score for candidate in candidates}) > 1
    if parallel_varies and crossing_varies:
        return AMBIGUITY_COMBINED
    if parallel_varies:
        return AMBIGUITY_PARALLEL
    if crossing_varies:
        return AMBIGUITY_CROSSING
    return AMBIGUITY_UNCLASSIFIED


def _strict_field(candidates: Sequence[Candidate]) -> tuple[dict[str, Any] | None, bool]:
    fields = {candidate.field_key: candidate.field for candidate in candidates}
    if len(fields) != 1:
        return None, False
    field = next(iter(fields.values()))
    decoded = decode_oracle_field(field)
    signatures = {candidate.graph_signature for candidate in candidates}
    return field, len(signatures) == 1 and decoded in signatures


def strategy_a_conservative(observation: ObservationClass) -> RecoveryDecision:
    """Conservative finite-class oracle: return Graph only for a strict class."""

    kind = _ambiguity_kind(observation.candidates)
    if kind != AMBIGUITY_NONE:
        return RecoveryDecision(
            result_type="Ambiguous",
            graph_signature=None,
            protocol_choice_graph_signature=None,
            ambiguity_kind=kind,
            oracle_field=ambiguous_observation_field(observation.visible_lanes, kind),
            candidate_count=len(observation.candidates),
            survivor_count=len(observation.candidates),
            survivor_graph_count=len(observation.graph_signatures),
            field_strict=False,
            decode_left_inverse=None,
            active_prior_ids=(),
            semantic_field_factorizes_through_evidence=False,
        )

    field, left_inverse = _strict_field(observation.candidates)
    graph = next(iter(observation.graph_signatures))
    return RecoveryDecision(
        result_type="Graph",
        graph_signature=graph,
        protocol_choice_graph_signature=graph,
        ambiguity_kind=AMBIGUITY_NONE,
        oracle_field=field,
        candidate_count=len(observation.candidates),
        survivor_count=len(observation.candidates),
        survivor_graph_count=1,
        field_strict=field is not None,
        decode_left_inverse=left_inverse if field is not None else None,
        active_prior_ids=(),
        semantic_field_factorizes_through_evidence=field is not None,
    )


def strategy_b_prior_survivors(
    observation: ObservationClass,
) -> tuple[Candidate, ...]:
    """Apply both global priors to finite semantic witnesses.

    The forward chooser reads only canonical lane evidence.  Scene meanings are
    consulted *after* decoding solely to measure whether the chosen graph has a
    survivor in the frozen finite domain.  No hash, scene ID, layer, or graph
    signature influences the forward choice.
    """

    chosen_field = oracle_field_from_observable_evidence(observation.visible_lanes)
    chosen_graph = _generic_graph_signature(chosen_field)
    return tuple(
        candidate
        for candidate in observation.candidates
        if candidate.graph_signature == chosen_graph
    )


def strategy_b_resolved(observation: ObservationClass) -> RecoveryDecision:
    chosen_field = oracle_field_from_observable_evidence(observation.visible_lanes)
    chosen_graph = _generic_graph_signature(chosen_field)
    survivors = strategy_b_prior_survivors(observation)
    if not survivors:
        return RecoveryDecision(
            result_type="Reject",
            graph_signature=None,
            protocol_choice_graph_signature=chosen_graph,
            ambiguity_kind=_ambiguity_kind(observation.candidates),
            oracle_field=None,
            candidate_count=len(observation.candidates),
            survivor_count=0,
            survivor_graph_count=0,
            field_strict=False,
            decode_left_inverse=None,
            active_prior_ids=STRATEGY_B_PRIOR_IDS,
            semantic_field_factorizes_through_evidence=True,
        )

    survivor_graphs = {candidate.graph_signature for candidate in survivors}
    kind = _ambiguity_kind(survivors)
    if len(survivor_graphs) != 1:
        return RecoveryDecision(
            result_type="Ambiguous",
            graph_signature=None,
            protocol_choice_graph_signature=chosen_graph,
            ambiguity_kind=kind,
            oracle_field=ambiguous_observation_field(observation.visible_lanes, kind),
            candidate_count=len(observation.candidates),
            survivor_count=len(survivors),
            survivor_graph_count=len(survivor_graphs),
            field_strict=False,
            decode_left_inverse=None,
            active_prior_ids=STRATEGY_B_PRIOR_IDS,
        )

    left_inverse = all(
        candidate.graph_signature == chosen_graph for candidate in survivors
    )
    return RecoveryDecision(
        result_type="Graph",
        graph_signature=chosen_graph,
        protocol_choice_graph_signature=chosen_graph,
        ambiguity_kind=AMBIGUITY_NONE,
        oracle_field=chosen_field,
        candidate_count=len(observation.candidates),
        survivor_count=len(survivors),
        survivor_graph_count=1,
        field_strict=True,
        decode_left_inverse=left_inverse,
        active_prior_ids=STRATEGY_B_PRIOR_IDS,
        semantic_field_factorizes_through_evidence=True,
    )


def _scene_catalog() -> dict[str, Scene]:
    scenes = (*iter_core_scenes(), *fixture_scenes())
    catalog = {scene.scene_id: scene for scene in scenes}
    if len(catalog) != EXPECTED_SCENE_COUNT:
        raise RecoveryAuditError(
            f"Reconstructed {len(catalog)} scenes, expected {EXPECTED_SCENE_COUNT}"
        )
    return catalog


def build_observation_classes(
    frozen: FrozenInputs,
    domain: str,
) -> tuple[ObservationClass, ...]:
    """Reconstruct and verify the declared finite domain."""

    allowed_layers = SUPPORTED_DOMAINS.get(domain)
    if allowed_layers is None:
        raise RecoveryAuditError(f"Unsupported recovery domain {domain!r}")
    scene_catalog = _scene_catalog()
    rows_by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen.manifest_rows:
        scene = scene_catalog.get(row["scene_id"])
        if scene is None:
            raise RecoveryAuditError(f"Manifest scene {row['scene_id']} was not reconstructed")
        reconstructed_graph = canonical_graph_signature(scene)
        if reconstructed_graph != row["canonical_graph_signature_structure"]:
            raise RecoveryAuditError(
                f"Reconstructed graph mismatch for {scene.scene_id}"
            )
        if row["exclusive_grammar_layer"] in allowed_layers:
            rows_by_observation[row["production_observation_sha256"]].append(row)

    classes = []
    for observation_sha, rows in sorted(rows_by_observation.items()):
        scene_visible_lanes = {
            canonical_json(merge_visible_lanes(_scene_members(scene_catalog[row["scene_id"]])))
            for row in rows
        }
        if len(scene_visible_lanes) != 1:
            raise RecoveryAuditError(
                "Same opaque observation produced inconsistent canonical visible lanes: "
                f"{observation_sha}"
            )
        visible_lanes = tuple(json.loads(next(iter(scene_visible_lanes))))
        candidates = tuple(
            Candidate(
                scene_id=row["scene_id"],
                graph_signature=row["canonical_graph_signature_structure"],
                grammar_layer=row["exclusive_grammar_layer"],
                field=oracle_field_for_scene(
                    scene_catalog[row["scene_id"]], visible_lanes
                ),
            )
            for row in sorted(rows, key=lambda item: item["scene_id"])
        )
        classes.append(ObservationClass(observation_sha, visible_lanes, candidates))
    return tuple(classes)


def fixture_observation_hashes(frozen: FrozenInputs) -> frozenset[str]:
    return frozenset(
        row["production_observation_sha256"]
        for row in frozen.manifest_rows
        if row["source"] == "fixture"
    )


__all__ = [
    "AMBIGUITY_CROSSING",
    "AMBIGUITY_NONE",
    "AMBIGUITY_PARALLEL",
    "CROSSING_PRIOR_ID",
    "EXPECTED_SCENE_COUNT",
    "FROZEN_CERTIFICATES_SHA256",
    "FROZEN_MANIFEST_SHA256",
    "FrozenInputs",
    "ObservationClass",
    "PARALLEL_PRIOR_ID",
    "RecoveryAuditError",
    "RecoveryDecision",
    "SCHEMA_VERSION",
    "STRATEGY_B_PRIOR_IDS",
    "SUPPORTED_DOMAINS",
    "ambiguous_observation_field",
    "build_observation_classes",
    "decode_oracle_field",
    "fixture_observation_hashes",
    "load_frozen_inputs",
    "merge_visible_lanes",
    "oracle_field_for_scene",
    "sha256_file",
    "strategy_a_conservative",
    "strategy_b_prior_survivors",
    "strategy_b_resolved",
]
