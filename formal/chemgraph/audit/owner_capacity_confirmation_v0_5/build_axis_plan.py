"""Freeze synthetic L/H sources and the complete capacity axis plan.

This module never imports or calls the R0 candidate constructor.
"""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    SCHEMA_VERSION as SUPPORT_SCHEMA,
    compile_support_partition,
)

from .certificate_checker import LIMITS, check_certificates
from .capacity_png import render_support_png, support_union
from .source_builder import AXIS_SIDES, build_source


ROOT = Path(__file__).resolve().parent
WITNESS_ROOT = ROOT / "witnesses"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONSTRUCTOR_PATH = (
    REPOSITORY_ROOT
    / "formal/chemgraph/audit/owner_candidate_domain_v0_5/candidate_domain.py"
)
SOURCE_BUILDER_PATH = ROOT / "source_builder.py"
CERTIFICATE_CHECKER_PATH = ROOT / "certificate_checker.py"
CAPACITY_RENDERER_PATH = ROOT / "capacity_png.py"
PLAN_PATH = ROOT / "axis_plan.json"
SOURCE_MANIFEST_PATH = ROOT / "source_manifest.json"
EXCLUSION_PATH = ROOT / "EXPLORATORY_EXCLUSION.md"
AXIS_PLAN_BUILDER_PATH = ROOT / "build_axis_plan.py"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _anchor_pixel(point_q4_exact: list[list[int]]) -> tuple[int, int]:
    x = Fraction(*point_q4_exact[0])
    y = Fraction(*point_q4_exact[1])
    return y.numerator // (16 * y.denominator), x.numerator // (16 * x.denominator)


def _measure_source(
    source: dict[str, Any], output_byte_upper_bound: int
) -> tuple[dict[str, int], dict[str, str], dict[str, int]]:
    support_input = {
        "geometry_root": source["geometry_root"],
        "parent_supports": source["parent_supports"],
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    geometry = compile_fragment_root(source["geometry_root"])
    support = compile_support_partition(support_input)
    count_vector = {
        "max_adjacency_edges": 0,
        "max_ambiguous_variables": 0,
        "max_atomic_fragments": geometry["atomic_fragment_count"],
        "max_carriers": geometry["base_stroke_count"],
        "max_constraints": 1 + 3 * geometry["intersection_count"],
        "max_domain_classes": 0,
        "max_intersections": geometry["intersection_count"],
        "max_output_bytes": output_byte_upper_bound,
        "max_owner_options": geometry["intersection_count"],
        "max_pixel_variables": support["cross_carrier_overlap_pixel_count"],
        "max_raw_assignments": 1,
        "max_total_rle_runs": support["child_support_run_count_total"],
    }
    count_kinds = {axis: "exact" for axis in LIMITS}
    count_kinds["max_output_bytes"] = "manual_schema_envelope_pending_binding"
    if geometry["intersection_count"] != 2:
        count_kinds.update({
            "max_adjacency_edges": "pre_gate_upper_bound",
            "max_ambiguous_variables": "pre_gate_upper_bound",
            "max_constraints": "pre_gate_upper_bound",
            "max_domain_classes": "pre_gate_upper_bound",
            "max_raw_assignments": "pre_gate_upper_bound",
        })
        return count_vector, count_kinds, _measurement_layers(source, geometry, support)
    options = geometry["intersections"]
    domains: list[tuple[int, ...]] = []
    pixels: list[tuple[int, int]] = []
    for overlap in support["cross_carrier_overlaps"]:
        carriers = set(overlap["base_stroke_ids"])
        domain = tuple(
            option["id"] for option in options
            if set(option["base_stroke_ids"]).issubset(carriers)
        )
        if not domain:
            raise RuntimeError("PLAN_SOURCE_HAS_UNOWNED_OVERLAP")
        domains.append(domain)
        pixels.append(tuple(overlap["pixel"]))
    pixel_to_id = {pixel: index for index, pixel in enumerate(pixels)}
    edges = 0
    for left_id, (row, column) in enumerate(pixels):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                right_id = pixel_to_id.get((row + dr, column + dc))
                if (dr or dc) and right_id is not None and left_id < right_id:
                    edges += 1
    for option in options:
        seed = _anchor_pixel(option["point_q4_exact"])
        seed_id = pixel_to_id.get(seed)
        if seed_id is None or option["id"] not in domains[seed_id]:
            raise RuntimeError("PLAN_SOURCE_SEED_UNAVAILABLE")
    ambiguous = sum(len(domain) > 1 for domain in domains)
    raw = 1
    for domain in domains:
        raw *= len(domain)
    count_vector.update({
        "max_adjacency_edges": edges,
        "max_ambiguous_variables": ambiguous,
        "max_constraints": 1 + 3 * len(options),
        "max_domain_classes": len(set(domains)),
        "max_raw_assignments": raw,
    })
    return count_vector, count_kinds, _measurement_layers(source, geometry, support)


def _segment_pair_hit(
    left: list[list[int]], right: list[list[int]]
) -> tuple[Fraction, Fraction] | None:
    p = tuple(left[0])
    q = tuple(right[0])
    r = left[1][0] - left[0][0], left[1][1] - left[0][1]
    s = right[1][0] - right[0][0], right[1][1] - right[0][1]
    denominator = r[0] * s[1] - r[1] * s[0]
    if denominator == 0:
        shared = {tuple(item) for item in left}.intersection(tuple(item) for item in right)
        return tuple(Fraction(value) for value in next(iter(shared))) if len(shared) == 1 else None
    qp = q[0] - p[0], q[1] - p[1]
    t = Fraction(qp[0] * s[1] - qp[1] * s[0], denominator)
    u = Fraction(qp[0] * r[1] - qp[1] * r[0], denominator)
    if not (0 <= t <= 1 and 0 <= u <= 1):
        return None
    return Fraction(p[0]) + t * r[0], Fraction(p[1]) + t * r[1]


def _measurement_layers(
    source: dict[str, Any], geometry: dict[str, Any], support: dict[str, Any]
) -> dict[str, int]:
    strokes = source["geometry_root"]["base_strokes"]
    raw_hits = []
    for left_index, left in enumerate(strokes):
        for right in strokes[left_index + 1:]:
            hit = _segment_pair_hit(left["segment_q4"], right["segment_q4"])
            if hit is not None:
                raw_hits.append(hit)
    raw_parent_runs = [
        (item["base_stroke_id"], *run)
        for item in source["parent_supports"] for run in item["support_runs"]
    ]
    child_runs = [
        (item["atomic_fragment_id"], *run)
        for item in support["child_supports"] for run in item["support_runs"]
    ]
    raw_memberships = [
        (item["base_stroke_id"], row, column)
        for item in source["parent_supports"]
        for row, start, length in item["support_runs"]
        for column in range(start, start + length)
    ]
    return {
        "carrier_canonical_distinct": len({
            json.dumps(item["segment_q4"], separators=(",", ":")) for item in strokes
        }),
        "carrier_compiled": geometry["base_stroke_count"],
        "carrier_deduplicated": len({item["id"] for item in strokes}),
        "carrier_raw": len(strokes),
        "intersection_canonical_distinct": len(set(raw_hits)),
        "intersection_compiled": geometry["intersection_count"],
        "intersection_deduplicated": len(set(raw_hits)),
        "intersection_raw": len(raw_hits),
        "pixel_canonical_distinct_memberships": len(set(raw_memberships)),
        "pixel_compiled_overlap_variables": support["cross_carrier_overlap_pixel_count"],
        "pixel_deduplicated_overlap_variables": len({
            tuple(item["pixel"]) for item in support["cross_carrier_overlaps"]
        }),
        "pixel_raw_memberships": len(raw_memberships),
        "rle_canonical_distinct_parent": len(set(raw_parent_runs)),
        "rle_compiled_child": support["child_support_run_count_total"],
        "rle_deduplicated_child": len(set(child_runs)),
        "rle_raw_parent": len(raw_parent_runs),
    }


def _source_records(output_byte_upper_bound: int) -> list[dict[str, Any]]:
    records = []
    for axis in sorted(AXIS_SIDES):
        for side in ("L", "H"):
            source = build_source(axis, side)
            encoded = _canonical_bytes(source)
            relative_path = f"witnesses/{axis}_{side}.observation.json"
            png_relative_path = f"witnesses/{axis}_{side}.support.png"
            png_bytes = render_support_png(source["parent_supports"])
            if _sha256_bytes(png_bytes) != source["png_sha256"]:
                raise RuntimeError("SOURCE_PNG_BINDING_MISMATCH")
            _atomic_write(ROOT / relative_path, encoded)
            _atomic_write(ROOT / png_relative_path, png_bytes)
            count_vector, count_kinds, measurement_layers = _measure_source(
                source, output_byte_upper_bound
            )
            records.append({
                "axis": axis,
                "canonical_source_sha256": _sha256_bytes(encoded),
                "count_kinds": count_kinds,
                "count_vector": count_vector,
                "geometry_renderer_binding": False,
                "png_binding": True,
                "png_path": png_relative_path,
                "png_sha256": _sha256_bytes(png_bytes),
                "raw_source_sha256": _sha256_bytes(encoded),
                "renderer_binding": True,
                "renderer_path": str(CAPACITY_RENDERER_PATH.relative_to(REPOSITORY_ROOT)),
                "renderer_sha256": _sha256_path(CAPACITY_RENDERER_PATH),
                "measurement_layers": measurement_layers,
                "side": side,
                "source_kind": "synthetic_capacity",
                "source_path": relative_path,
                "support_union_pixel_count": len(support_union(source["parent_supports"])),
                "target_count": AXIS_SIDES[axis][side],
            })
    return records


def _axis_records(source_records: list[dict[str, Any]], certificate: dict[str, Any]) -> list[dict[str, Any]]:
    source_by_key = {(item["axis"], item["side"]): item for item in source_records}
    reachable = {
        "max_carriers": (
            "reachable_end_to_end", "compiled_fragment_root.base_stroke_count",
            "len(canonical base_strokes)", []
        ),
        "max_intersections": (
            "reachable_rejection_only", "compiled_fragment_root.intersection_count",
            "len(distinct exact rational intersection points)", ["max_carriers"]
        ),
        "max_pixel_variables": (
            "reachable_end_to_end", "compiled_support_partition.cross_carrier_overlap_pixel_count",
            "count(pixel carrier fibers with size > 1)",
            ["max_carriers", "max_intersections", "max_atomic_fragments", "max_total_rle_runs"]
        ),
        "max_total_rle_runs": (
            "reachable_end_to_end", "compiled_support_partition.child_support_run_count_total",
            "sum(len(canonical child support runs))",
            ["max_carriers", "max_intersections", "max_atomic_fragments"]
        ),
    }
    unreachable = {
        "max_adjacency_edges": ("conditional_unreachable_no_go", "8-neighbor unordered edge count"),
        "max_ambiguous_variables": ("conditional_not_isolated_no_go", "count(domain size > 1)"),
        "max_atomic_fragments": ("conditional_unreachable_no_go", "compiled_fragment_root.atomic_fragment_count"),
        "max_constraints": ("conditional_unreachable_no_go", "1 + 3 * owner_option_count"),
        "max_domain_classes": ("conditional_unreachable_no_go", "count(deduplicated nonempty owner domains)"),
        "max_output_bytes": ("unproven_no_go", "len(canonical artifact bytes)"),
        "max_owner_options": ("conditional_unreachable_no_go", "len(intersection owner options after exact-two gate)"),
        "max_raw_assignments": ("conditional_shadowed_no_go", "exact product of domain sizes"),
    }
    records: list[dict[str, Any]] = []
    gate_order = [
        "max_carriers", "max_intersections", "max_atomic_fragments", "max_total_rle_runs",
        "R0_REQUIRES_TWO_INTERSECTIONS", "max_owner_options", "max_pixel_variables",
        "max_domain_classes", "max_adjacency_edges", "max_constraints",
        "max_ambiguous_variables", "max_raw_assignments", "max_output_bytes",
    ]
    for axis in sorted(LIMITS):
        if axis in reachable:
            category, point, formula, predecessors = reachable[axis]
            left = source_by_key[(axis, "L")]
            high = source_by_key[(axis, "H")]
            boundary = {
                "tag": "reachable",
                "L": left["target_count"],
                "L_source_sha256": left["canonical_source_sha256"],
                "L_source_path": left["source_path"],
                "H": high["target_count"],
                "H_source_sha256": high["canonical_source_sha256"],
                "H_source_path": high["source_path"],
            }
        else:
            category, formula = unreachable[axis]
            certificate_record = certificate["axes"][axis]
            point = formula
            predecessors = gate_order[:gate_order.index(axis)]
            boundary = {
                "tag": "conditional_or_unproven_no_go",
                "L_optional": (
                    LIMITS[axis] if axis in {"max_ambiguous_variables", "max_raw_assignments"}
                    else None
                ),
                "H": None,
                "certificate": certificate_record,
            }
        records.append({
            "axis": axis,
            "boundary": boundary,
            "category": category,
            "count_formula": formula,
            "gate_order_index": gate_order.index(axis),
            "limit": LIMITS[axis],
            "measurement_point": point,
            "predecessor_gates": predecessors,
            "training_authorized": False,
        })
    return records


def build_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    certificate = check_certificates()
    output_bound = certificate["axes"]["max_output_bytes"]["upper_bound"]
    sources = _source_records(output_bound)
    for record in sources:
        if record["side"] != "H":
            continue
        target = record["axis"]
        if record["count_vector"][target] <= LIMITS[target]:
            raise RuntimeError("HIGH_TARGET_NOT_OVER_LIMIT")
        for other_axis, value in record["count_vector"].items():
            if other_axis != target and value > LIMITS[other_axis]:
                raise RuntimeError(f"HIGH_SOURCE_NOT_ISOLATED:{target}:{other_axis}")
    source_manifest = {
        "constructor_call_scope": "plan_builder_process_only",
        "constructor_called_by_plan_builder": False,
        "global_preregistration_contaminated": True,
        "record_count": len(sources),
        "records": sources,
        "schema_version": "owner-capacity-source-manifest-v0.5-preregistration",
        "status": "prepared_for_preregistration_commit",
        "training_authorized": False,
    }
    plan = {
        "axes": _axis_records(sources, certificate),
        "axis_key_set": sorted(LIMITS),
        "axis_plan_builder": {
            "path": str(AXIS_PLAN_BUILDER_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(AXIS_PLAN_BUILDER_PATH),
        },
        "certificate_checker": {
            "path": str(CERTIFICATE_CHECKER_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(CERTIFICATE_CHECKER_PATH),
        },
        "capacity_support_renderer": {
            "path": str(CAPACITY_RENDERER_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(CAPACITY_RENDERER_PATH),
        },
        "constructor": {
            "called_by_plan_builder": False,
            "global_preregistration_contaminated": True,
            "path": str(CONSTRUCTOR_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(CONSTRUCTOR_PATH),
        },
        "core_commits": {
            "capacity_clarification": "b531804149499797514ff70eba6252f09c7095c0",
            "capacity_lean": "4972e0853150e6b9de72be4cd4490c554468c39c",
            "observation_hash_hardening": "8e138b7b573b732587757bc73169f3b9dd3da7b3",
            "r0_core": "bb72eff416b8f03dcc6a933dcd5c1c2c1a30bf0e",
        },
        "exploratory_exclusion": {
            "exclusion_id": "owner-capacity-v0.5-exploratory-oracle-001",
            "path": str(EXCLUSION_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(EXCLUSION_PATH),
            "status": "excluded_exploratory",
        },
        "limits": LIMITS,
        "limits_canonical_sha256": _sha256_bytes(_canonical_bytes(LIMITS)),
        "schema_version": "owner-capacity-axis-plan-v0.5-preregistration",
        "source_builder": {
            "path": str(SOURCE_BUILDER_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_path(SOURCE_BUILDER_PATH),
        },
        "source_manifest_sha256": _sha256_bytes(_canonical_bytes(source_manifest)),
        "synthetic_source_policy": {
            "geometry_renderer_binding": False,
            "png_binding": True,
            "renderer_binding": True,
            "source_kind": "synthetic_capacity",
            "support_union_binding": True,
            "semantic_claim_authorized": False,
        },
        "training_authorized": False,
        "unreachable_certificate_status": "conditional_or_unproven_no_go",
        "verdict_if_all_reachable_pass": "PARTIAL/NO_GO_GENERIC",
    }
    if set(item["axis"] for item in plan["axes"]) != set(LIMITS):
        raise RuntimeError("AXIS_PLAN_KEY_SET_MISMATCH")
    return plan, source_manifest


def main() -> None:
    plan, source_manifest = build_plan()
    _atomic_write(SOURCE_MANIFEST_PATH, _canonical_bytes(source_manifest))
    _atomic_write(PLAN_PATH, _canonical_bytes(plan))
    print(_canonical_bytes({
        "axis_plan_sha256": _sha256_path(PLAN_PATH),
        "source_manifest_sha256": _sha256_path(SOURCE_MANIFEST_PATH),
        "status": "prepared_for_preregistration_commit",
        "training_authorized": False,
        "witness_count": source_manifest["record_count"],
    }).decode("ascii"), end="")


if __name__ == "__main__":
    main()
