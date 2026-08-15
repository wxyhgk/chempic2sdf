#!/usr/bin/env python3
"""Build deterministic public fixtures for the v0.3 structural validator."""

from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.blind_carrier_v0_2._common import (
    canonical_grayscale_png,
    canonical_json_bytes,
)

from ._common import PROTOCOL_SHA256, framed_sha256
from .validate_candidate_set import (
    ATOM_SOURCE_SHA256,
    DIRECT_SHA256,
    P1A_SHA256,
    STAGE_NAMES,
    VISION_SHA256,
    _spatial_graph_sha256,
    _stage_chain_sha256,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _component(name: str) -> dict[str, str]:
    return {"id": name, "sha256": _sha(name), "version": "0.3.0"}


def _input_and_png() -> tuple[dict[str, Any], bytes]:
    pixels = bytearray([255]) * (768 * 768)
    for column in range(100, 501):
        pixels[384 * 768 + column] = 0
    png = canonical_grayscale_png(bytes(pixels))
    value = {
        "image": {
            "bytes_base64": base64.b64encode(png).decode("ascii"),
            "format": "png",
            "height": 768,
            "width": 768,
        },
        "schema_version": "chemgraph-carrier-candidate-input-v0.3",
        "training_authorized": False,
    }
    return value, png


def _thresholds() -> dict[str, int]:
    return {
        "atom_proposal_threshold_q24": 167772,
        "atom_set_logit_q20": -262144,
        "attach_max_angle_millidegrees": 12000,
        "attach_max_endpoint_distance_q4": 256,
        "boundary_support_width_q4": 16,
        "direct_presence_threshold_q24": 14260634,
        "direct_uncertain_lower_q24": 8388608,
        "lane_white_gap_max_q4": 256,
        "lane_white_gap_min_q4": 1,
        "p1a_bond_threshold_q24": 8388608,
        "p1a_text_threshold_q24": 8388608,
        "pairing_darkness_threshold_q24": 4194304,
        "pairing_interior_occupancy_ppm": 800000,
        "pairing_max_angle_millidegrees": 3000,
        "pairing_max_separation_q4": 64,
        "pairing_min_overlap_q4": 128,
        "pairing_min_separation_q4": 8,
        "pairing_shorter_overlap_ppm": 700000,
        "white_gap_max_length_q4": 256,
    }


def _provenance() -> dict[str, Any]:
    return {
        "atom_source_checkpoint_sha256": ATOM_SOURCE_SHA256,
        "boundary_extractor": _component("boundary_extractor"),
        "boundary_pairing": _component("boundary_pairing"),
        "candidate_predicate": _component("candidate_predicate"),
        "contract_sha256": PROTOCOL_SHA256,
        "direct_atom_pair_checkpoint_sha256": DIRECT_SHA256,
        "enumerator_id": "finite_enumerator_v0_3",
        "enumerator_sha256": _sha("finite_enumerator_v0_3"),
        "fixed_thresholds": _thresholds(),
        "p1a_checkpoint_sha256": P1A_SHA256,
        "preprocess": _component("preprocess"),
        "runtime_sha256": _sha("runtime"),
        "semantic_ink_extractor": _component("semantic_ink_extractor"),
        "structural_verifier": _component("structural_verifier"),
        "vision_checkpoint_sha256": VISION_SHA256,
    }


def _boundary() -> dict[str, Any]:
    runs = [[384, 100, 401]]
    return {
        "bond_mask_support_pixel_count": 401,
        "bond_mask_support_runs": runs,
        "final_foreground_support_pixel_count": 401,
        "final_foreground_support_runs": runs,
        "id": 0,
        "segment_q4": [[1608, 6152], [8008, 6152]],
        "source_lsd_index": 0,
    }


def _fragment(kind: str) -> dict[str, Any]:
    return {
        "boundary_member_ids": [0],
        "derivation": "unmatched_boundary",
        "id": 0,
        "primitive_kind": kind,
        "segment_q4": [[1608, 6152], [8008, 6152]],
        "support_pixel_count": 401,
        "support_runs": [[384, 100, 401]],
        "width_q4": 16,
    }


def _atom(atom_id: int, proposal_id: int, anchor: list[int], port_id: int) -> dict[str, Any]:
    return {
        "anchor_q4": anchor,
        "glyph_fragment_ids": [],
        "id": atom_id,
        "kind": "implicit_vertex",
        "port_ids": [port_id],
        "presence_score_q24": 16000000,
        "source_cell_q4": anchor,
        "source_proposal_id": proposal_id,
    }


def _port(port_id: int, atom_id: int, endpoint_index: int, center: list[int]) -> dict[str, Any]:
    return {
        "atom_id": atom_id,
        "center_q4": center,
        "fragment_endpoint_refs": [{"endpoint_index": endpoint_index, "fragment_id": 0}],
        "id": port_id,
        "radius_q4": 16,
        "support_fragment_ids": [0],
    }


def _relation(relation_id: int, atom_id: int, port_id: int, endpoint_index: int, point: list[int]) -> dict[str, Any]:
    return {
        "atom_id": atom_id,
        "endpoint_ref": {"endpoint_index": endpoint_index, "fragment_id": 0},
        "geometry_evidence": {
            "absolute_cross_q8": 0,
            "angle_millidegrees": 0,
            "atom_anchor_q4": point,
            "attachment_radius_q4": 256,
            "distance_q4": 0,
            "endpoint_q4": point,
            "outward_dot_q8": 0,
            "predicate_passed": True,
        },
        "id": relation_id,
        "port_id": port_id,
    }


def _single_bond_interpretation() -> dict[str, Any]:
    return {
        "atoms": [
            _atom(0, 0, [1608, 6152], 0),
            _atom(1, 1, [8008, 6152], 1),
        ],
        "boundary_matching": {
            "compatible_pair_ids": [],
            "matching_index": 0,
            "unmatched_boundary_ids": [0],
        },
        "bonds": [{
            "atom_pair": [0, 1],
            "atom_pair_evidence": {
                "pair_id": 0,
                "presence_decision": "positive",
                "presence_score_q24": 16000000,
                "source": "direct_atom_pair_v0103",
            },
            "begin_port_id": 0,
            "end_port_id": 1,
            "id": 0,
            "lane_endpoint_alignment": [{"begin_port_id": 0, "end_port_id": 1, "lane_id": 0}],
            "lane_ids": [0],
            "order": 1,
        }],
        "boundaries": [_boundary()],
        "endpoint_atom_relations": [
            _relation(0, 0, 0, 0, [1608, 6152]),
            _relation(1, 1, 1, 1, [8008, 6152]),
        ],
        "id": 0,
        "intersection_regions": [],
        "lane_separation_certificates": [],
        "lanes": [{
            "begin_port_id": 0,
            "end_port_id": 1,
            "fragment_chain": [{"begin_endpoint_index": 0, "end_endpoint_index": 1, "fragment_id": 0}],
            "id": 0,
            "white_gap_certificate_ids": [],
        }],
        "ownership": [{"fragment_id": 0, "owner_id": 0, "owner_kind": "bond_lane"}],
        "ownership_hypothesis_index": 0,
        "ports": [
            _port(0, 0, 0, [1608, 6152]),
            _port(1, 1, 1, [8008, 6152]),
        ],
        "stroke_fragments": [_fragment("straight")],
        "white_gap_certificates": [],
    }


def _glyph_only_interpretation() -> dict[str, Any]:
    return {
        "atoms": [],
        "boundary_matching": {
            "compatible_pair_ids": [],
            "matching_index": 0,
            "unmatched_boundary_ids": [0],
        },
        "bonds": [],
        "boundaries": [_boundary()],
        "endpoint_atom_relations": [],
        "id": 0,
        "intersection_regions": [],
        "lane_separation_certificates": [],
        "lanes": [],
        "ownership": [{"fragment_id": 0, "owner_id": 0, "owner_kind": "text_glyph"}],
        "ownership_hypothesis_index": 0,
        "ports": [],
        "stroke_fragments": [_fragment("glyph")],
        "white_gap_certificates": [],
    }


def _rle_sha256(runs: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runs).to_bytes(4, "big"))
    for row, start, length in runs:
        digest.update(row.to_bytes(4, "big"))
        digest.update(start.to_bytes(4, "big"))
        digest.update(length.to_bytes(4, "big"))
    return digest.hexdigest()


def _finish_candidate(interpretation: dict[str, Any], png: bytes, with_atoms: bool) -> dict[str, Any]:
    runs = [[384, 100, 401]]
    provenance = _provenance()
    boundaries = [_boundary()]
    proposals = (
        [
            {"anchor_q4": [1608, 6152], "heatmap_score_q24": 16000000, "id": 0, "set_decision": "active", "set_logit_q20": 0, "source_cell_q4": [1608, 6152]},
            {"anchor_q4": [8008, 6152], "heatmap_score_q24": 16000000, "id": 1, "set_decision": "active", "set_logit_q20": 0, "source_cell_q4": [8008, 6152]},
        ]
        if with_atoms else []
    )
    direct_pairs = (
        [{
            "atom_proposal_pair": [0, 1],
            "chemical_scores_q24": [16000000, 0, 0, 0],
            "id": 0,
            "presence_decision": "positive",
            "presence_score_q24": 16000000,
        }]
        if with_atoms else []
    )
    interpretations = [interpretation]
    graph_equivalence = {
        "algorithm_id": "spatial_graph_v0_3",
        "algorithm_sha256": _sha("spatial_graph_v0_3"),
        "algorithm_version": "0.3.0",
        "class_count": 1,
        "graph_classes": [{
            "canonical_spatial_graph_sha256": _spatial_graph_sha256(interpretation),
            "id": 0,
            "interpretation_ids": [0],
        }],
    }
    taxonomy = {
        "entries": [],
        "entry_count": 0,
        "taxonomy_id": "ambiguity_v0_3",
        "taxonomy_sha256": _sha("ambiguity_v0_3"),
        "taxonomy_version": "0.3.0",
    }
    matching_records = [interpretation["boundary_matching"]]
    ownership_records = [interpretation["ownership"]]
    relation_records = [{
        "interpretation_id": 0,
        "relations": interpretation["endpoint_atom_relations"],
    }]
    emitted_sha = framed_sha256(interpretations)
    enumeration = {
        "all_boundary_matchings_sha256": framed_sha256(matching_records),
        "all_candidates_emitted": True,
        "all_endpoint_atom_relations_sha256": framed_sha256(relation_records),
        "all_interpretations_sha256": emitted_sha,
        "all_ownership_hypotheses_sha256": framed_sha256(ownership_records),
        "atom_proposal_universe_count": len(proposals),
        "boundary_matching_count": 1,
        "capacity_hit": False,
        "candidate_universe_sha256": emitted_sha,
        "compatible_boundary_pair_count": 0,
        "complete": True,
        "direct_pair_universe_count": len(direct_pairs),
        "emitted_set_sha256": emitted_sha,
        "endpoint_atom_relation_count": len(interpretation["endpoint_atom_relations"]),
        "enumeration_verifier_id": "structural_only_v0_3",
        "enumeration_verifier_sha256": _sha("structural_only_v0_3"),
        "hash_framing": "u32be-length-prefixed-canonical-json-v1",
        "interpretation_count": 1,
        "ownership_component_count": 1,
        "ownership_hypothesis_count": 1,
        "pruned_candidate_count": 0,
        "raw_boundary_count": 1,
    }
    image = {
        "bond_mask_pixel_count": 401,
        "bond_mask_rle_sha256": _rle_sha256(runs),
        "bond_mask_runs": runs,
        "coordinate_format": "Q4-pixel",
        "final_foreground_pixel_count": 401,
        "final_foreground_rle_sha256": _rle_sha256(runs),
        "final_foreground_runs": runs,
        "height": 768,
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "rle_hash_framing": "u32be(run_count)||for_each_run(u32be(row)||u32be(start)||u32be(length))",
        "text_mask_pixel_count": 401 if not with_atoms else 0,
        "text_mask_rle_sha256": _rle_sha256(runs if not with_atoms else []),
        "text_mask_runs": runs if not with_atoms else [],
        "width": 768,
    }
    document = {
        "ambiguity_taxonomy": taxonomy,
        "atom_proposal_universe": proposals,
        "boundary_universe": boundaries,
        "compatible_boundary_pairs": [],
        "direct_pair_universe": direct_pairs,
        "enumeration": enumeration,
        "graph_equivalence": graph_equivalence,
        "image": image,
        "interpretations": interpretations,
        "provenance": provenance,
        "schema_version": "chemgraph-carrier-candidate-set-v0.3",
        "stage_trace": {},
        "status": "candidate_set",
        "training_authorized": False,
    }
    stage_outputs = (
        framed_sha256([runs]),
        framed_sha256([runs, image["text_mask_runs"]]),
        framed_sha256(boundaries),
        framed_sha256([]),
        framed_sha256(matching_records),
        framed_sha256(proposals),
        framed_sha256(direct_pairs),
        framed_sha256(ownership_records),
        framed_sha256(relation_records),
        framed_sha256(interpretations),
        framed_sha256([graph_equivalence]),
        framed_sha256([taxonomy]),
    )
    component_hashes = (
        provenance["preprocess"]["sha256"],
        provenance["semantic_ink_extractor"]["sha256"],
        provenance["boundary_extractor"]["sha256"],
        provenance["boundary_pairing"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["atom_source_checkpoint_sha256"],
        provenance["direct_atom_pair_checkpoint_sha256"],
        provenance["enumerator_sha256"],
        provenance["candidate_predicate"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["structural_verifier"]["sha256"],
        provenance["structural_verifier"]["sha256"],
    )
    stage_counts = (
        image["final_foreground_pixel_count"],
        image["final_foreground_pixel_count"],
        1,
        0,
        1,
        len(proposals),
        len(direct_pairs),
        1,
        len(interpretation["endpoint_atom_relations"]),
        1,
        1,
        0,
    )
    stages = []
    previous = hashlib.sha256(png).hexdigest()
    for index, name in enumerate(STAGE_NAMES):
        stages.append({
            "component_sha256": component_hashes[index],
            "index": index,
            "input_sha256": previous,
            "output_sha256": stage_outputs[index],
            "record_count": stage_counts[index],
            "stage": name,
        })
        previous = stage_outputs[index]
    document["stage_trace"] = {
        "framing": "u32be-index||u32be-stage-bytes-length||stage-bytes||component-sha256-bytes||input-sha256-bytes||output-sha256-bytes||u64be-record-count",
        "ordered_chain_sha256": _stage_chain_sha256(stages),
        "stages": stages,
    }
    return document


def _refresh_collision_document(
    document: dict[str, Any], png: bytes, *, taxonomy_code: str = "A12",
    taxonomy_kind: str = "glyph_vs_bond", evidence_kind: str = "fragment",
) -> None:
    interpretations = document["interpretations"]
    for interpretation in interpretations:
        interpretation.pop("id", None)
    interpretations.sort(key=canonical_json_bytes)
    for index, interpretation in enumerate(interpretations):
        interpretation["id"] = index

    graph_groups: dict[str, list[int]] = {}
    for interpretation in interpretations:
        graph_groups.setdefault(_spatial_graph_sha256(interpretation), []).append(interpretation["id"])
    graph_classes = [
        {"canonical_spatial_graph_sha256": digest, "id": index, "interpretation_ids": ids}
        for index, (digest, ids) in enumerate(sorted(graph_groups.items()))
    ]
    document["graph_equivalence"]["class_count"] = len(graph_classes)
    document["graph_equivalence"]["graph_classes"] = graph_classes
    document["ambiguity_taxonomy"]["entries"] = [{
        "evidence_region_refs": [
            {"interpretation_id": interpretation["id"], "object_id": 0, "object_kind": evidence_kind}
            for interpretation in interpretations
        ],
        "graph_class_ids": list(range(len(graph_classes))),
        "id": 0,
        "interpretation_ids": list(range(len(interpretations))),
        "kind": taxonomy_kind,
        "taxonomy_code": taxonomy_code,
    }]
    document["ambiguity_taxonomy"]["entry_count"] = 1

    matching_by_index = {
        interpretation["boundary_matching"]["matching_index"]: interpretation["boundary_matching"]
        for interpretation in interpretations
    }
    ownership_by_index = {
        interpretation["ownership_hypothesis_index"]: interpretation["ownership"]
        for interpretation in interpretations
    }
    matching_records = [matching_by_index[index] for index in sorted(matching_by_index)]
    ownership_records = [ownership_by_index[index] for index in sorted(ownership_by_index)]
    relation_records = [
        {"interpretation_id": interpretation["id"], "relations": interpretation["endpoint_atom_relations"]}
        for interpretation in interpretations
    ]
    emitted_sha = framed_sha256(interpretations)
    enumeration = document["enumeration"]
    enumeration.update({
        "all_boundary_matchings_sha256": framed_sha256(matching_records),
        "all_endpoint_atom_relations_sha256": framed_sha256(relation_records),
        "all_interpretations_sha256": emitted_sha,
        "all_ownership_hypotheses_sha256": framed_sha256(ownership_records),
        "boundary_matching_count": len(matching_records),
        "candidate_universe_sha256": emitted_sha,
        "emitted_set_sha256": emitted_sha,
        "endpoint_atom_relation_count": sum(len(item["relations"]) for item in relation_records),
        "interpretation_count": len(interpretations),
        "ownership_component_count": len({
            item["fragment_id"]
            for interpretation in interpretations
            for item in interpretation["ownership"]
        }),
        "ownership_hypothesis_count": len(ownership_records),
    })

    image = document["image"]
    provenance = document["provenance"]
    stage_outputs = (
        framed_sha256([image["final_foreground_runs"]]),
        framed_sha256([image["bond_mask_runs"], image["text_mask_runs"]]),
        framed_sha256(document["boundary_universe"]),
        framed_sha256(document["compatible_boundary_pairs"]),
        framed_sha256(matching_records),
        framed_sha256(document["atom_proposal_universe"]),
        framed_sha256(document["direct_pair_universe"]),
        framed_sha256(ownership_records),
        framed_sha256(relation_records),
        framed_sha256(interpretations),
        framed_sha256([document["graph_equivalence"]]),
        framed_sha256([document["ambiguity_taxonomy"]]),
    )
    component_hashes = (
        provenance["preprocess"]["sha256"],
        provenance["semantic_ink_extractor"]["sha256"],
        provenance["boundary_extractor"]["sha256"],
        provenance["boundary_pairing"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["atom_source_checkpoint_sha256"],
        provenance["direct_atom_pair_checkpoint_sha256"],
        provenance["enumerator_sha256"],
        provenance["candidate_predicate"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["structural_verifier"]["sha256"],
        provenance["structural_verifier"]["sha256"],
    )
    stage_counts = (
        image["final_foreground_pixel_count"],
        image["final_foreground_pixel_count"],
        len(document["boundary_universe"]),
        len(document["compatible_boundary_pairs"]),
        len(matching_records),
        len(document["atom_proposal_universe"]),
        len(document["direct_pair_universe"]),
        len(ownership_records),
        sum(len(item["relations"]) for item in relation_records),
        len(interpretations),
        len(graph_classes),
        1,
    )
    stages = []
    previous = hashlib.sha256(png).hexdigest()
    for index, name in enumerate(STAGE_NAMES):
        stages.append({
            "component_sha256": component_hashes[index],
            "index": index,
            "input_sha256": previous,
            "output_sha256": stage_outputs[index],
            "record_count": stage_counts[index],
            "stage": name,
        })
        previous = stage_outputs[index]
    document["stage_trace"]["stages"] = stages
    document["stage_trace"]["ordered_chain_sha256"] = _stage_chain_sha256(stages)


def _glyph_bond_collision(png: bytes) -> dict[str, Any]:
    document = _finish_candidate(_single_bond_interpretation(), png, True)
    runs = [[384, 100, 401]]
    document["image"]["text_mask_runs"] = runs
    document["image"]["text_mask_pixel_count"] = 401
    document["image"]["text_mask_rle_sha256"] = _rle_sha256(runs)
    glyph = _glyph_only_interpretation()
    glyph["ownership_hypothesis_index"] = 1
    document["interpretations"].append(glyph)
    _refresh_collision_document(document, png)
    return document


def _crossing_input_and_png() -> tuple[dict[str, Any], bytes]:
    pixels = bytearray([255]) * (768 * 768)
    for column in range(100, 501):
        pixels[384 * 768 + column] = 0
    for row in range(284, 485):
        pixels[row * 768 + 300] = 0
    png = canonical_grayscale_png(bytes(pixels))
    return ({
        "image": {
            "bytes_base64": base64.b64encode(png).decode("ascii"),
            "format": "png",
            "height": 768,
            "width": 768,
        },
        "schema_version": "chemgraph-carrier-candidate-input-v0.3",
        "training_authorized": False,
    }, png)


def _crossing_runs() -> list[list[int]]:
    return (
        [[row, 300, 1] for row in range(284, 384)]
        + [[384, 100, 401]]
        + [[row, 300, 1] for row in range(385, 485)]
    )


def _crossing_boundaries() -> list[dict[str, Any]]:
    horizontal = [[384, 100, 401]]
    vertical = [[row, 300, 1] for row in range(284, 485)]
    return [
        {
            "bond_mask_support_pixel_count": 401,
            "bond_mask_support_runs": horizontal,
            "final_foreground_support_pixel_count": 401,
            "final_foreground_support_runs": horizontal,
            "id": 0,
            "segment_q4": [[1608, 6152], [8008, 6152]],
            "source_lsd_index": 0,
        },
        {
            "bond_mask_support_pixel_count": 201,
            "bond_mask_support_runs": vertical,
            "final_foreground_support_pixel_count": 201,
            "final_foreground_support_runs": vertical,
            "id": 1,
            "segment_q4": [[4808, 4552], [4808, 7752]],
            "source_lsd_index": 1,
        },
    ]


def _general_atom(atom_id: int, proposal_id: int, anchor: list[int], port_id: int) -> dict[str, Any]:
    return _atom(atom_id, proposal_id, anchor, port_id)


def _general_port(
    port_id: int, atom_id: int, refs: list[tuple[int, int]], center: list[int],
) -> dict[str, Any]:
    return {
        "atom_id": atom_id,
        "center_q4": center,
        "fragment_endpoint_refs": [
            {"endpoint_index": endpoint_index, "fragment_id": fragment_id}
            for fragment_id, endpoint_index in sorted(refs)
        ],
        "id": port_id,
        "radius_q4": 16,
        "support_fragment_ids": sorted({fragment_id for fragment_id, _ in refs}),
    }


def _general_relation(
    relation_id: int, atom_id: int, port_id: int, fragment_id: int,
    endpoint_index: int, point: list[int],
) -> dict[str, Any]:
    value = _relation(relation_id, atom_id, port_id, endpoint_index, point)
    value["endpoint_ref"]["fragment_id"] = fragment_id
    return value


def _cross_fragment(
    fragment_id: int, boundary_id: int, segment: list[list[int]], runs: list[list[int]],
    derivation: str,
) -> dict[str, Any]:
    return {
        "boundary_member_ids": [boundary_id],
        "derivation": derivation,
        "id": fragment_id,
        "primitive_kind": "straight",
        "segment_q4": segment,
        "support_pixel_count": sum(run[2] for run in runs),
        "support_runs": runs,
        "width_q4": 16,
    }


def _cross_region(
    fragment_ids: list[int], *, kind: str, port_ids: list[int],
) -> dict[str, Any]:
    return {
        "boundary_member_ids": [0, 1],
        "center_q4": [4808, 6152],
        "id": 0,
        "kind": kind,
        "port_member_ids": port_ids,
        "stroke_fragment_member_ids": fragment_ids,
        "support_pixel_count": 1,
        "support_runs": [[384, 300, 1]],
    }


def _through_interpretation() -> dict[str, Any]:
    fragments = [
        _cross_fragment(0, 0, [[1608, 6152], [8008, 6152]], [[384, 100, 401]], "unmatched_boundary"),
        _cross_fragment(1, 1, [[4808, 4552], [4808, 7752]], [[row, 300, 1] for row in range(284, 485)], "unmatched_boundary"),
    ]
    anchors = [[1608, 6152], [4808, 4552], [4808, 7752], [8008, 6152]]
    source_ids = [0, 1, 3, 4]
    ports = [
        _general_port(0, 0, [(0, 0)], anchors[0]),
        _general_port(1, 1, [(1, 0)], anchors[1]),
        _general_port(2, 2, [(1, 1)], anchors[2]),
        _general_port(3, 3, [(0, 1)], anchors[3]),
    ]
    relations = [
        _general_relation(0, 0, 0, 0, 0, anchors[0]),
        _general_relation(1, 1, 1, 1, 0, anchors[1]),
        _general_relation(2, 2, 2, 1, 1, anchors[2]),
        _general_relation(3, 3, 3, 0, 1, anchors[3]),
    ]
    lanes = [
        {"begin_port_id": 0, "end_port_id": 3, "fragment_chain": [{"begin_endpoint_index": 0, "end_endpoint_index": 1, "fragment_id": 0}], "id": 0, "white_gap_certificate_ids": []},
        {"begin_port_id": 1, "end_port_id": 2, "fragment_chain": [{"begin_endpoint_index": 0, "end_endpoint_index": 1, "fragment_id": 1}], "id": 1, "white_gap_certificate_ids": []},
    ]
    bonds = [
        {"atom_pair": [0, 3], "atom_pair_evidence": {"pair_id": 3, "presence_decision": "positive", "presence_score_q24": 16000000, "source": "direct_atom_pair_v0103"}, "begin_port_id": 0, "end_port_id": 3, "id": 0, "lane_endpoint_alignment": [{"begin_port_id": 0, "end_port_id": 3, "lane_id": 0}], "lane_ids": [0], "order": 1},
        {"atom_pair": [1, 2], "atom_pair_evidence": {"pair_id": 5, "presence_decision": "positive", "presence_score_q24": 16000000, "source": "direct_atom_pair_v0103"}, "begin_port_id": 1, "end_port_id": 2, "id": 1, "lane_endpoint_alignment": [{"begin_port_id": 1, "end_port_id": 2, "lane_id": 1}], "lane_ids": [1], "order": 1},
    ]
    return {
        "atoms": [_general_atom(index, source_ids[index], anchors[index], index) for index in range(4)],
        "boundary_matching": {"compatible_pair_ids": [], "matching_index": 0, "unmatched_boundary_ids": [0, 1]},
        "bonds": bonds,
        "boundaries": _crossing_boundaries(),
        "endpoint_atom_relations": relations,
        "id": 0,
        "intersection_regions": [_cross_region([0, 1], kind="crossing", port_ids=[])],
        "lane_separation_certificates": [],
        "lanes": lanes,
        "ownership": [
            {"fragment_id": 0, "owner_id": 0, "owner_kind": "bond_lane"},
            {"fragment_id": 1, "owner_id": 1, "owner_kind": "bond_lane"},
        ],
        "ownership_hypothesis_index": 0,
        "ports": ports,
        "stroke_fragments": fragments,
        "white_gap_certificates": [],
    }


def _junction_interpretation() -> dict[str, Any]:
    fragments = [
        _cross_fragment(0, 0, [[1608, 6152], [4808, 6152]], [[384, 100, 201]], "split_at_intersection"),
        _cross_fragment(1, 1, [[4808, 4552], [4808, 6152]], [[row, 300, 1] for row in range(284, 385)], "split_at_intersection"),
        _cross_fragment(2, 1, [[4808, 6152], [4808, 7752]], [[row, 300, 1] for row in range(384, 485)], "split_at_intersection"),
        _cross_fragment(3, 0, [[4808, 6152], [8008, 6152]], [[384, 300, 201]], "split_at_intersection"),
    ]
    anchors = [[1608, 6152], [4808, 4552], [4808, 6152], [4808, 7752], [8008, 6152]]
    center_refs = [(0, 1), (1, 1), (2, 0), (3, 0)]
    ports = [
        _general_port(0, 0, [(0, 0)], anchors[0]),
        _general_port(1, 1, [(1, 0)], anchors[1]),
        _general_port(2, 2, center_refs, anchors[2]),
        _general_port(3, 3, [(2, 1)], anchors[3]),
        _general_port(4, 4, [(3, 1)], anchors[4]),
    ]
    relation_specs = [
        (0, 0, 0, 0, 0), (1, 1, 1, 1, 0),
        (2, 2, 2, 0, 1), (3, 2, 2, 1, 1), (4, 2, 2, 2, 0), (5, 2, 2, 3, 0),
        (6, 3, 3, 2, 1), (7, 4, 4, 3, 1),
    ]
    relation_points = [anchors[0], anchors[1], anchors[2], anchors[2], anchors[2], anchors[2], anchors[3], anchors[4]]
    relations = [
        _general_relation(*spec, point)
        for spec, point in zip(relation_specs, relation_points)
    ]
    lane_specs = [(0, 2, 0, 0, 1), (1, 2, 1, 0, 1), (2, 3, 2, 0, 1), (2, 4, 3, 0, 1)]
    lanes = [
        {"begin_port_id": begin, "end_port_id": end, "fragment_chain": [{"begin_endpoint_index": begin_ep, "end_endpoint_index": end_ep, "fragment_id": fragment_id}], "id": lane_id, "white_gap_certificate_ids": []}
        for lane_id, (begin, end, fragment_id, begin_ep, end_ep) in enumerate(lane_specs)
    ]
    direct_ids = [1, 4, 7, 8]
    atom_pairs = [[0, 2], [1, 2], [2, 3], [2, 4]]
    bonds = [
        {"atom_pair": pair, "atom_pair_evidence": {"pair_id": direct_ids[index], "presence_decision": "positive", "presence_score_q24": 16000000, "source": "direct_atom_pair_v0103"}, "begin_port_id": lanes[index]["begin_port_id"], "end_port_id": lanes[index]["end_port_id"], "id": index, "lane_endpoint_alignment": [{"begin_port_id": lanes[index]["begin_port_id"], "end_port_id": lanes[index]["end_port_id"], "lane_id": index}], "lane_ids": [index], "order": 1}
        for index, pair in enumerate(atom_pairs)
    ]
    return {
        "atoms": [_general_atom(index, index, anchors[index], index) for index in range(5)],
        "boundary_matching": {"compatible_pair_ids": [], "matching_index": 0, "unmatched_boundary_ids": [0, 1]},
        "bonds": bonds,
        "boundaries": _crossing_boundaries(),
        "endpoint_atom_relations": relations,
        "id": 1,
        "intersection_regions": [_cross_region([0, 1, 2, 3], kind="junction", port_ids=[2])],
        "lane_separation_certificates": [],
        "lanes": lanes,
        "ownership": [{"fragment_id": index, "owner_id": index, "owner_kind": "bond_lane"} for index in range(4)],
        "ownership_hypothesis_index": 1,
        "ports": ports,
        "stroke_fragments": fragments,
        "white_gap_certificates": [],
    }


def _crossing_candidate(png: bytes) -> dict[str, Any]:
    document = _finish_candidate(_single_bond_interpretation(), png, True)
    runs = _crossing_runs()
    document["image"].update({
        "bond_mask_pixel_count": 601,
        "bond_mask_rle_sha256": _rle_sha256(runs),
        "bond_mask_runs": runs,
        "final_foreground_pixel_count": 601,
        "final_foreground_rle_sha256": _rle_sha256(runs),
        "final_foreground_runs": runs,
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "text_mask_pixel_count": 0,
        "text_mask_rle_sha256": _rle_sha256([]),
        "text_mask_runs": [],
    })
    document["boundary_universe"] = _crossing_boundaries()
    anchors = [[1608, 6152], [4808, 4552], [4808, 6152], [4808, 7752], [8008, 6152]]
    document["atom_proposal_universe"] = [
        {"anchor_q4": anchor, "heatmap_score_q24": 16000000, "id": index, "set_decision": "active", "set_logit_q20": 0, "source_cell_q4": anchor}
        for index, anchor in enumerate(anchors)
    ]
    document["direct_pair_universe"] = [
        {"atom_proposal_pair": [left, right], "chemical_scores_q24": [16000000, 0, 0, 0], "id": pair_id, "presence_decision": "positive", "presence_score_q24": 16000000}
        for pair_id, (left, right) in enumerate(
            (pair for left in range(5) for pair in ((left, right) for right in range(left + 1, 5)))
        )
    ]
    document["interpretations"] = [_through_interpretation(), _junction_interpretation()]
    document["enumeration"].update({
        "atom_proposal_universe_count": 5,
        "direct_pair_universe_count": 10,
        "raw_boundary_count": 2,
    })
    _refresh_collision_document(
        document, png, taxonomy_code="A10", taxonomy_kind="crossing_vs_junction",
        evidence_kind="intersection",
    )
    return document


def build_documents() -> dict[str, dict[str, Any]]:
    input_value, png = _input_and_png()
    crossing_input, crossing_png = _crossing_input_and_png()
    return {
        "valid_input.json": input_value,
        "valid_crossing_input.json": crossing_input,
        "valid_single_bond.json": _finish_candidate(_single_bond_interpretation(), png, True),
        "valid_glyph_only.json": _finish_candidate(_glyph_only_interpretation(), png, False),
        "valid_glyph_bond_collision.json": _glyph_bond_collision(png),
        "valid_crossing_junction_collision.json": _crossing_candidate(crossing_png),
        "valid_reject.json": {
            "reason": "unsupported_primitive",
            "schema_version": "chemgraph-carrier-candidate-set-v0.3",
            "stage": "boundary_enumeration",
            "status": "reject",
            "training_authorized": False,
        },
    }


def write_fixtures(root: Path = FIXTURE_ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, value in build_documents().items():
        (root / name).write_bytes(canonical_json_bytes(copy.deepcopy(value)))


def main() -> None:
    write_fixtures()


if __name__ == "__main__":
    main()
