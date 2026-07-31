#!/usr/bin/env python3
"""Run the v0.8 ChemScene pipeline with predicted junctions, OCR, and bonds."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import torch

from Data.sdf.chemscene_v0_1 import (
    chemsvg_from_chemscene_v0_1,
    parse_chemscene_v0_1,
    sdf_from_chemscene_v0_1,
)
from chemvector.benchmark_mr_tadf import write_invalid_prediction_sdf
from chemvector.bond_endpoint_junction import (
    BondEndpointJunctionResult,
    junctions_from_bond_endpoints,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.bond_expert_utility import combined_utility_features
from chemvector.bond_utility_router import BondUtilityRouter
from chemvector.chemscene_graph_corrector import (
    ELEMENT_TO_ID,
    ChemSceneGraphCorrector,
    ChemSceneGraphCorrectorConfig,
)
from chemvector.ctc_ocr import (
    PretrainedPixelCTCConfig,
    PretrainedPixelCTCOCR,
    ctc_text_log_score,
    decode_ctc_with_score,
    detected_text_crops,
)
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
    decode_dense_bond_objects,
)
from chemvector.dual_vertex_selector import (
    IMPLICIT_ROLE,
    TEXT_ROLE,
    dual_candidate_auxiliary,
    merge_dual_candidates,
    padded_candidate_roles,
    role_suppressed_role_threshold_indices,
)
from chemvector.endpoint_evidence import endpoint_ray_features
from chemvector.evaluate_group_selector_bond import (
    load_evaluation_selector,
    outside_text_objects,
    resolved_real_thresholds,
    selector_candidate_protocol,
)
from chemvector.grounded_selector import (
    fuse_dense_candidates_with_queries,
    sample_candidate_features,
    sample_candidate_ink_patches,
)
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.junction_bond_assembler import (
    JunctionBondAssembler,
    JunctionBondAssemblerConfig,
    decode_junction_assignment_rescue,
    decode_junction_assignments,
)
from chemvector.junction_aware_bond_selector import (
    load_bond_set_selector,
    load_endpoint_grouper,
    select_and_group_bonds,
)
from chemvector.ocr import OCRTextCodec, atom_display_text, parse_atom_display_text
from chemvector.ocr_utility_router import (
    OCRUtilityRouter,
    add_ocr_cross_score_features,
    add_ocr_visual_features,
    ocr_utility_features,
)
from chemvector.pretrain_dense_bond_object import _apply_detector
from chemvector.pretrain_bond_set_selector import (
    load_line_band_reader,
    read_line_band_features,
)
from chemvector.pretrain_dual_vertex_selector import (
    load_dual_pipeline,
    selector_forward,
)
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelation,
    SemanticBondRelationConfig,
    apply_aromatic_ring_semantics,
)
from chemvector.structured_atom_decoder import (
    AtomStateCandidate,
    atom_state_candidate,
    merge_ctc_atom_candidates,
    repair_impossible_explicit_h as structured_repair_impossible_explicit_h,
    rerank_invalid_scene_atom_states,
)
from chemvector.structure_gate import StructureStyleGate
from chemvector.text_object import TextObjectBBoxHead, decode_text_objects
from chemvector.vertex import decode_vertices


def _quantize(point: torch.Tensor) -> tuple[int, int]:
    return tuple(
        max(0, min(1023, round(float(value) * 1023.0))) for value in point
    )


def attach_text_to_bond_junctions(
    bond_points: torch.Tensor,
    detected_points: torch.Tensor,
    detected_roles: torch.Tensor,
    detected_labels: list[str],
    *,
    image_size: int,
    maximum_distance_pixels: float,
) -> tuple[torch.Tensor, list[str]]:
    """Transfer OCR labels without allowing text detections to create Junctions."""

    if bond_points.ndim != 2 or bond_points.size(-1) != 2:
        raise ValueError("Bond-derived Junction points must have shape [points, 2]")
    if detected_points.shape != (len(detected_roles), 2):
        raise ValueError("Detected Junction points and roles differ")
    if len(detected_labels) != len(detected_points):
        raise ValueError("Detected Junction labels and points differ")
    if maximum_distance_pixels <= 0.0:
        raise ValueError("Text attachment distance must be positive")
    roles = torch.full(
        (len(bond_points),),
        IMPLICIT_ROLE,
        device=bond_points.device,
        dtype=torch.long,
    )
    labels = ["C"] * len(bond_points)
    text_indices = torch.where(detected_roles == TEXT_ROLE)[0]
    if not len(text_indices) or not len(bond_points):
        return roles, labels
    text_points = detected_points[text_indices].to(bond_points.device)
    distances = torch.cdist(text_points, bond_points) * float(image_size)
    assignments = []
    for flat_index in distances.flatten().argsort().tolist():
        text_row = flat_index // len(bond_points)
        bond_index = flat_index % len(bond_points)
        distance = float(distances[text_row, bond_index])
        if distance > maximum_distance_pixels:
            break
        assignments.append((distance, text_row, bond_index))
    used_text = set()
    used_bonds = set()
    for _, text_row, bond_index in assignments:
        if text_row in used_text or bond_index in used_bonds:
            continue
        source_index = int(text_indices[text_row])
        roles[bond_index] = TEXT_ROLE
        labels[bond_index] = detected_labels[source_index]
        used_text.add(text_row)
        used_bonds.add(bond_index)
    return roles, labels


def _atom_from_label(
    label: str,
    role: int,
    *,
    unresolved_as_dummy: bool = False,
) -> dict:
    parsed = parse_atom_display_text(label) if role == TEXT_ROLE else None
    if parsed is None:
        parsed = {
            "element": (
                "*" if role == TEXT_ROLE and unresolved_as_dummy else "C"
            ),
            "formal_charge": 0,
            "isotope": 0,
            "explicit_h_count": 0,
            "radical_electrons": 0,
        }
    explicit_h_count = int(parsed["explicit_h_count"])
    return {
        "element": str(parsed["element"]),
        "formal_charge": int(parsed["formal_charge"]),
        "isotope": int(parsed["isotope"]),
        "aromatic": False,
        "explicit_h_count": explicit_h_count,
        "no_implicit": bool(explicit_h_count) or parsed["element"] == "*",
        "radical_electrons": int(parsed["radical_electrons"]),
        "canonical_chiral_tag": "CHI_UNSPECIFIED",
    }


def _element_id_from_label(label: str, role: int) -> int:
    if role != TEXT_ROLE:
        return ELEMENT_TO_ID["C"]
    parsed = parse_atom_display_text(label)
    element = str(parsed["element"]) if parsed is not None else "OTHER"
    return ELEMENT_TO_ID.get(element, ELEMENT_TO_ID["OTHER"])


def apply_graph_correction(
    bond_objects: list[dict],
    output: dict[str, torch.Tensor],
    *,
    mode: str,
    keep_threshold: float,
) -> tuple[list[dict], dict]:
    if mode not in {"type_only", "full"}:
        raise ValueError(f"Unknown graph correction mode: {mode}")
    if not 0.0 < keep_threshold < 1.0:
        raise ValueError("Graph correction keep threshold must lie in (0, 1)")
    display = output["display_logits"].argmax(-1).tolist()
    chemical = output["chemical_logits"].argmax(-1).tolist()
    keep = output["edge_keep_logits"].sigmoid().ge(keep_threshold).tolist()
    corrected = []
    display_changes = chemical_changes = removed = 0
    for index, item in enumerate(bond_objects):
        if mode == "full" and not keep[index]:
            removed += 1
            continue
        updated = dict(item)
        display_changes += int(int(item["display"]) != int(display[index]))
        chemical_changes += int(int(item["chemical"]) != int(chemical[index]))
        updated["display"] = int(display[index])
        updated["chemical"] = int(chemical[index])
        corrected.append(updated)
    return corrected, {
        "graph_correction_mode": mode,
        "graph_removed_bond_objects": removed,
        "graph_display_type_changes": display_changes,
        "graph_chemical_type_changes": chemical_changes,
        "graph_keep_threshold": keep_threshold,
    }


def repair_impossible_explicit_h(scene: dict) -> tuple[dict, list[dict]]:
    return structured_repair_impossible_explicit_h(scene)


def assemble_chemscene(
    molecule_id: str,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    *,
    width: int = 768,
    height: int = 768,
    assignment_mode: str = "nearest",
    junction_assignments: list[tuple[int, int] | None] | None = None,
    unresolved_as_dummy: bool = False,
) -> tuple[dict, dict]:
    """Map complete predicted bond endpoints to predicted Junction objects."""

    if points.shape != (len(roles), 2) or len(labels) != len(points):
        raise ValueError("Predicted Junction points, roles, and OCR labels differ")
    if not len(points):
        raise ValueError("No Junction was predicted")
    coordinates = [_quantize(point) for point in points]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Predicted Junctions collide after q1023 quantization")

    atoms = [
        _atom_from_label(
            label,
            int(role),
            unresolved_as_dummy=unresolved_as_dummy,
        )
        for label, role in zip(labels, roles)
    ]
    learned_modes = {"learned", "learned_rescue", "explicit"}
    if assignment_mode not in {"nearest", "global_unique", *learned_modes}:
        raise ValueError(f"Unknown Junction assignment mode: {assignment_mode}")
    if assignment_mode in learned_modes:
        if junction_assignments is None or len(junction_assignments) != len(bond_objects):
            raise ValueError("Learned Junction assignments and BondObjects differ")
    elif junction_assignments is not None:
        raise ValueError("Explicit Junction assignments require learned mode")
    pair_objects: dict[tuple[int, int], dict] = {}
    self_loops = 0
    collisions = 0
    reassigned = 0
    endpoint_distances = []
    available_pairs = (
        torch.combinations(torch.arange(len(points)), r=2)
        if assignment_mode == "global_unique"
        else None
    )
    used_pairs: set[tuple[int, int]] = set()
    ordered_objects = sorted(
        enumerate(bond_objects),
        key=lambda item: float(item[1]["score"]),
        reverse=True,
    )
    for object_index, obj in ordered_objects:
        endpoints = obj["endpoints"].float().cpu()
        distances = torch.cdist(endpoints, points.float().cpu())
        nearest_begin = int(distances[0].argmin())
        nearest_end = int(distances[1].argmin())
        nearest_pair = tuple(sorted((nearest_begin, nearest_end)))
        if assignment_mode in learned_modes:
            learned_pair = junction_assignments[object_index]
            if learned_pair is None:
                continue
            begin, end = learned_pair
            pair = tuple(sorted((int(begin), int(end))))
            forward_distances = (
                float(distances[0, pair[0]]),
                float(distances[1, pair[1]]),
            )
            reverse_distances = (
                float(distances[0, pair[1]]),
                float(distances[1, pair[0]]),
            )
            endpoint_distances.extend(
                reverse_distances
                if sum(reverse_distances) < sum(forward_distances)
                else forward_distances
            )
        elif assignment_mode == "global_unique":
            if available_pairs is None or not len(available_pairs):
                raise ValueError("Not enough Junctions for BondObject assignment")
            forward = distances[0, available_pairs[:, 0]] + distances[
                1, available_pairs[:, 1]
            ]
            reverse = distances[0, available_pairs[:, 1]] + distances[
                1, available_pairs[:, 0]
            ]
            costs = torch.minimum(forward, reverse)
            for pair_index, candidate in enumerate(available_pairs.tolist()):
                if tuple(candidate) in used_pairs:
                    costs[pair_index] = float("inf")
            selected_pair_index = int(costs.argmin())
            if not torch.isfinite(costs[selected_pair_index]):
                raise ValueError("Unique Junction-pair capacity is exhausted")
            begin, end = available_pairs[selected_pair_index].tolist()
            pair = (int(begin), int(end))
            used_pairs.add(pair)
            use_reverse = bool(reverse[selected_pair_index] < forward[selected_pair_index])
            mapped_begin, mapped_end = (end, begin) if use_reverse else (begin, end)
            endpoint_distances.extend(
                (
                    float(distances[0, mapped_begin]),
                    float(distances[1, mapped_end]),
                )
            )
            reassigned += int(nearest_pair != pair)
        else:
            begin, end = nearest_begin, nearest_end
            endpoint_distances.extend(
                (float(distances[0, begin]), float(distances[1, end]))
            )
            if begin == end:
                self_loops += 1
                continue
            pair = tuple(sorted((begin, end)))
        previous = pair_objects.get(pair)
        if previous is not None:
            collisions += 1
        if previous is None or float(obj["score"]) > float(previous["score"]):
            pair_objects[pair] = obj
    if not pair_objects:
        raise ValueError("No BondObject connected two distinct predicted Junctions")

    bonds = []
    for (begin, end), obj in sorted(pair_objects.items()):
        chemical_type = BOND_TYPES[int(obj["chemical"])]
        if chemical_type == "AROMATIC":
            atoms[begin]["aromatic"] = True
            atoms[end]["aromatic"] = True
        bonds.append(
            {
                "endpoints": [list(coordinates[begin]), list(coordinates[end])],
                "display_type": BOND_TYPES[int(obj["display"])],
                "chemical_type": chemical_type,
                "direction": DIRECTIONS[int(obj["direction"])],
                "stereo": "STEREONONE",
                "bbox": None,
            }
        )

    junctions = []
    text_objects = []
    for coordinate, role, label, atom in zip(coordinates, roles, labels, atoms):
        role_name = "TEXT" if int(role) == TEXT_ROLE else "IMPLICIT"
        junctions.append(
            {"coordinate": list(coordinate), "role": role_name, "atom": atom}
        )
        if role_name == "TEXT":
            text_objects.append(
                {
                    "anchor": list(coordinate),
                    "center": list(coordinate),
                    "text": label.strip() or atom["element"],
                    "bbox": None,
                }
            )
    scene = {
        "schema_version": "chemscene-0.1",
        "molecule_id": molecule_id,
        "units": "q1023",
        "canvas": {"width": width, "height": height},
        "junctions": junctions,
        "text_objects": text_objects,
        "bond_objects": bonds,
    }
    parse_chemscene_v0_1(scene)
    diagnostics = {
        "predicted_junctions": len(junctions),
        "decoded_bond_objects": len(bond_objects),
        "assembled_bonds": len(bonds),
        "self_loop_objects": self_loops,
        "duplicate_pair_objects": collisions,
        "reassigned_bond_objects": reassigned,
        "junction_assignment_mode": assignment_mode,
        "endpoint_to_junction_distance_mean_pixels": (
            sum(endpoint_distances) / max(len(endpoint_distances), 1) * max(width, height)
        ),
        "endpoint_to_junction_distance_max_pixels": (
            max(endpoint_distances, default=0.0) * max(width, height)
        ),
    }
    return scene, diagnostics


def compile_scene_with_aromatic_fallback(scene: dict) -> tuple[str, dict, bool]:
    """Compile the predicted scene, falling back only for invalid aromatic sets."""

    scene, _ = repair_impossible_explicit_h(scene)
    try:
        return sdf_from_chemscene_v0_1(scene), scene, False
    except Exception as error:
        message = str(error).lower()
        if "kekul" not in message and "aromatic" not in message:
            raise
    fallback = copy.deepcopy(scene)
    for junction in fallback["junctions"]:
        junction["atom"]["aromatic"] = False
    replaced = 0
    for bond in fallback["bond_objects"]:
        if bond["chemical_type"] != "AROMATIC":
            continue
        visible = bond["display_type"]
        bond["chemical_type"] = (
            visible if visible in {"SINGLE", "DOUBLE", "TRIPLE"} else "SINGLE"
        )
        replaced += 1
    if not replaced:
        raise ValueError("Aromatic fallback had no aromatic bonds to replace")
    parse_chemscene_v0_1(fallback)
    return sdf_from_chemscene_v0_1(fallback), fallback, True


def suppress_text_interior_bond_objects(
    bond_objects: list[dict],
    text_objects: list[dict],
    *,
    image_size: int,
    interior_margin_pixels: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    """Remove BondObjects whose two semantic endpoints are inside one text box."""
    if image_size < 1:
        raise ValueError("Image size must be positive")
    if interior_margin_pixels < 0:
        raise ValueError("Text interior margin cannot be negative")
    margin = interior_margin_pixels / image_size
    kept = []
    removed = []
    for bond_index, bond in enumerate(bond_objects):
        endpoints = bond["endpoints"].detach().float().cpu()
        containing_text = None
        for text_index, text in enumerate(text_objects):
            half_width = float(text["half_width"]) - margin
            half_height = float(text["half_height"]) - margin
            if min(half_width, half_height) <= 0:
                continue
            center_x = float(text.get("bbox_x", text["x"]))
            center_y = float(text.get("bbox_y", text["y"]))
            inside = (
                (endpoints[:, 0] > center_x - half_width)
                & (endpoints[:, 0] < center_x + half_width)
                & (endpoints[:, 1] > center_y - half_height)
                & (endpoints[:, 1] < center_y + half_height)
            )
            if bool(inside.all()):
                containing_text = text_index
                break
        if containing_text is None:
            kept.append(bond)
            continue
        text = text_objects[containing_text]
        removed.append(
            {
                "bond_index": bond_index,
                "text_index": containing_text,
                "score": float(bond["score"]),
                "endpoints": endpoints.tolist(),
                "text_bbox": {
                    "x": float(text.get("bbox_x", text["x"])),
                    "y": float(text.get("bbox_y", text["y"])),
                    "half_width": float(text["half_width"]),
                    "half_height": float(text["half_height"]),
                },
            }
        )
    return kept, removed


class ChemSceneHybridPipeline:
    def __init__(
        self,
        selector_checkpoint: Path,
        bond_checkpoint: Path,
        device: torch.device,
        assembler_checkpoint: Path | None = None,
        graph_corrector_checkpoint: Path | None = None,
        ocr_checkpoint: Path | None = None,
        ctc_ocr_checkpoint: Path | None = None,
        text_bbox_checkpoint: Path | None = None,
        ctc_anchor_fallback_margin: float | None = None,
        bond_set_selector_checkpoint: Path | None = None,
        endpoint_grouper_checkpoint: Path | None = None,
        line_band_checkpoint: Path | None = None,
        junction_projection_score_source: str = "detector",
        junction_bond_type_source: str = "detector",
        semantic_bond_relation_checkpoint: Path | None = None,
        semantic_bond_relation_mode: str = "chemical_all",
        semantic_aromatic_mean_threshold: float = 0.72,
        semantic_aromatic_minimum_threshold: float = 0.25,
        semantic_bond_existence_checkpoint: Path | None = None,
        semantic_bond_existence_threshold: float = 0.60,
        semantic_bond_existence_chunk_size: int = 256,
        expert_bond_checkpoint: Path | None = None,
        style_gate_checkpoint: Path | None = None,
        style_gate_threshold: float = 0.90,
        bond_utility_router_checkpoint: Path | None = None,
        bond_utility_break_budget: str = "0",
        text_interior_bond_veto: bool = False,
        expert_ctc_ocr_checkpoint: Path | None = None,
        ocr_utility_router_checkpoint: Path | None = None,
        ocr_utility_router_threshold: float = 0.95,
    ) -> None:
        if junction_projection_score_source not in {"detector", "selector"}:
            raise ValueError(
                "Junction projection score source must be detector or selector"
            )
        self.device = device
        self.text_interior_bond_veto = text_interior_bond_veto
        self.junction_projection_score_source = junction_projection_score_source
        if junction_bond_type_source not in {"detector", "line_band"}:
            raise ValueError("Junction Bond type source must be detector or line_band")
        self.junction_bond_type_source = junction_bond_type_source
        self.semantic_bond_relation = None
        self.semantic_bond_relation_checkpoint = semantic_bond_relation_checkpoint
        if semantic_bond_relation_mode not in {"chemical_all", "aromatic_ring_only"}:
            raise ValueError("Unknown Semantic Bond relation mode")
        self.semantic_bond_relation_mode = semantic_bond_relation_mode
        self.semantic_aromatic_mean_threshold = semantic_aromatic_mean_threshold
        self.semantic_aromatic_minimum_threshold = semantic_aromatic_minimum_threshold
        if not 0.0 < semantic_bond_existence_threshold < 1.0:
            raise ValueError("Semantic Bond existence threshold must lie in (0, 1)")
        if semantic_bond_existence_chunk_size < 1:
            raise ValueError("Semantic Bond existence chunk size must be positive")
        self.semantic_bond_existence = None
        self.semantic_bond_existence_checkpoint = semantic_bond_existence_checkpoint
        self.semantic_bond_existence_threshold = semantic_bond_existence_threshold
        self.semantic_bond_existence_chunk_size = semantic_bond_existence_chunk_size
        if semantic_bond_existence_checkpoint is not None:
            existence_payload = torch.load(
                semantic_bond_existence_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            existence_config = SemanticBondRelationConfig(
                **existence_payload["config"]
            )
            self.semantic_bond_existence = SemanticBondExistence(
                existence_config
            ).to(device)
            self.semantic_bond_existence.load_state_dict(
                existence_payload["semantic_bond_existence"]
            )
            self.semantic_bond_existence.eval().requires_grad_(False)
        if semantic_bond_relation_checkpoint is not None:
            semantic_payload = torch.load(
                semantic_bond_relation_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            semantic_config = SemanticBondRelationConfig(**semantic_payload["config"])
            self.semantic_bond_relation = SemanticBondRelation(semantic_config).to(
                device
            )
            self.semantic_bond_relation.load_state_dict(
                semantic_payload["semantic_bond_relation"]
            )
            self.semantic_bond_relation.eval().requires_grad_(False)
        (
            self.selector,
            self.selector_config,
            selector_payload,
            _,
        ) = load_evaluation_selector(str(selector_checkpoint), device)
        (
            self.base,
            self.query_detector,
            self.grounded_selector,
            self.graph,
            self.text_head,
            self.implicit_head,
        ) = load_dual_pipeline(
            selector_payload["text_object_checkpoint"],
            selector_payload["implicit_vertex_checkpoint"],
            device,
            maximum_candidates=self.selector_config.maximum_candidates,
        )
        self.protocol = selector_candidate_protocol(selector_payload)
        self.text_bbox_checkpoint = text_bbox_checkpoint
        if text_bbox_checkpoint is not None:
            text_bbox_payload = torch.load(
                text_bbox_checkpoint, map_location="cpu", weights_only=False
            )
            if text_bbox_payload.get("head_type") != "bbox":
                raise ValueError("Text bbox checkpoint does not contain a bbox head")
            text_head = TextObjectBBoxHead(text_bbox_payload["input_channels"]).to(
                device
            )
            text_head.load_state_dict(text_bbox_payload["text_object_head"])
            self.text_head = text_head.eval().requires_grad_(False)
        payload = torch.load(bond_checkpoint, map_location="cpu", weights_only=False)
        config = DenseBondObjectConfig(**payload["config"])
        state = payload["dense_bond_object"]
        if any(key.startswith("base.") for key in state):
            self.bond_detector = MultiScaleDenseBondObjectDetector(config).to(device)
        else:
            self.bond_detector = DenseBondObjectDetector(config).to(device)
        self.bond_detector.load_state_dict(state)
        self.bond_detector.eval().requires_grad_(False)
        self.expert_bond_detector = None
        self.style_gate = None
        self.style_gate_threshold = style_gate_threshold
        self.bond_utility_router = None
        self.bond_utility_feature_names = None
        self.bond_utility_feature_mean = None
        self.bond_utility_feature_std = None
        self.bond_utility_threshold = None
        if bool(expert_bond_checkpoint) != bool(style_gate_checkpoint):
            raise ValueError(
                "Expert BondObject checkpoint and style gate must be used together"
            )
        if bond_utility_router_checkpoint is not None and expert_bond_checkpoint is None:
            raise ValueError("Bond utility router requires an expert BondObject checkpoint")
        if expert_bond_checkpoint is not None:
            if not 0.0 <= style_gate_threshold <= 1.0:
                raise ValueError("Style gate threshold must be in [0, 1]")
            expert_payload = torch.load(
                expert_bond_checkpoint, map_location="cpu", weights_only=False
            )
            if Path(expert_payload["grounded_graph_checkpoint"]).resolve() != Path(
                payload["grounded_graph_checkpoint"]
            ).resolve():
                raise ValueError(
                    "Base and style expert use different grounded vision checkpoints"
                )
            expert_config = DenseBondObjectConfig(**expert_payload["config"])
            expert_state = expert_payload["dense_bond_object"]
            self.expert_bond_detector = (
                MultiScaleDenseBondObjectDetector(expert_config).to(device)
                if any(key.startswith("base.") for key in expert_state)
                else DenseBondObjectDetector(expert_config).to(device)
            )
            if type(self.expert_bond_detector) is not type(self.bond_detector):
                raise ValueError("Base and style expert BondObject types differ")
            self.expert_bond_detector.load_state_dict(expert_state)
            self.expert_bond_detector.eval().requires_grad_(False)
            gate_payload = torch.load(
                style_gate_checkpoint, map_location="cpu", weights_only=False
            )
            if Path(gate_payload["base_bond_checkpoint"]).resolve() != Path(
                bond_checkpoint
            ).resolve():
                raise ValueError("Style gate base BondObject checkpoint does not match")
            if Path(gate_payload["expert_bond_checkpoint"]).resolve() != Path(
                expert_bond_checkpoint
            ).resolve():
                raise ValueError(
                    "Style gate expert BondObject checkpoint does not match"
                )
            self.style_gate = StructureStyleGate(
                gate_payload["input_channels"], hidden_dim=gate_payload["hidden_dim"]
            ).to(device)
            self.style_gate.load_state_dict(gate_payload["gate"])
            self.style_gate.eval().requires_grad_(False)
            if bond_utility_router_checkpoint is not None:
                utility_payload = torch.load(
                    bond_utility_router_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                if bond_utility_break_budget not in utility_payload["validation_frontier"]:
                    raise ValueError("Unknown Bond utility break budget")
                self.bond_utility_router = BondUtilityRouter(
                    utility_payload["input_dim"], utility_payload["hidden_dim"]
                ).to(device)
                self.bond_utility_router.load_state_dict(utility_payload["router"])
                self.bond_utility_router.eval().requires_grad_(False)
                self.bond_utility_feature_names = utility_payload["feature_names"]
                self.bond_utility_feature_mean = utility_payload["feature_mean"].to(
                    device
                )
                self.bond_utility_feature_std = utility_payload["feature_std"].to(
                    device
                )
                self.bond_utility_threshold = float(
                    utility_payload["validation_frontier"][bond_utility_break_budget][
                        "threshold"
                    ]
                )
        self.assembler = None
        self.assembler_checkpoint = assembler_checkpoint
        if assembler_checkpoint is not None:
            assembler_payload = torch.load(
                assembler_checkpoint, map_location="cpu", weights_only=False
            )
            assembler_config = JunctionBondAssemblerConfig(
                **assembler_payload["config"]
            )
            self.assembler = JunctionBondAssembler(assembler_config).to(device)
            self.assembler.load_state_dict(
                assembler_payload["junction_bond_assembler"]
            )
            self.assembler.eval().requires_grad_(False)
        self.graph_corrector = None
        self.graph_corrector_checkpoint = graph_corrector_checkpoint
        self.graph_corrector_keep_threshold = 0.3
        self.graph_corrector_uses_elements = False
        if graph_corrector_checkpoint is not None:
            graph_payload = torch.load(
                graph_corrector_checkpoint, map_location="cpu", weights_only=False
            )
            graph_config = ChemSceneGraphCorrectorConfig(**graph_payload["config"])
            self.graph_corrector = ChemSceneGraphCorrector(graph_config).to(device)
            self.graph_corrector.load_state_dict(
                graph_payload["chemscene_graph_corrector"]
            )
            self.graph_corrector.eval().requires_grad_(False)
            self.graph_corrector_keep_threshold = float(
                graph_payload.get("validation", {}).get("keep_threshold", 0.3)
            )
            self.graph_corrector_uses_elements = bool(
                graph_payload.get("protocol", {}).get("include_ocr_elements", False)
            )
        self.ocr_checkpoint = ocr_checkpoint
        if ocr_checkpoint is not None:
            ocr_payload = torch.load(
                ocr_checkpoint, map_location="cpu", weights_only=False
            )
            prefix = "ocr_head."
            ocr_state = {
                key[len(prefix) :]: value
                for key, value in ocr_payload["model"].items()
                if key.startswith(prefix)
            }
            self.base.ocr_head.load_state_dict(ocr_state, strict=True)
            self.base.ocr_head.eval().requires_grad_(False)
        self.ctc_ocr_checkpoint = ctc_ocr_checkpoint
        self.ctc_anchor_fallback_margin = ctc_anchor_fallback_margin
        self.ctc_ocr = None
        if ctc_ocr_checkpoint is not None:
            ctc_payload = torch.load(
                ctc_ocr_checkpoint, map_location="cpu", weights_only=False
            )
            ctc_config = PretrainedPixelCTCConfig(**ctc_payload["config"])
            self.ctc_ocr = PretrainedPixelCTCOCR(ctc_config).to(device)
            self.ctc_ocr.load_state_dict(ctc_payload["pretrained_ctc_ocr"])
            self.ctc_ocr.pixel_encoder_loaded = True
            self.ctc_ocr.eval().requires_grad_(False)
        self.expert_ctc_ocr_checkpoint = expert_ctc_ocr_checkpoint
        self.expert_ctc_ocr = None
        self.ocr_utility_router = None
        self.ocr_utility_feature_names = None
        self.ocr_utility_feature_mean = None
        self.ocr_utility_feature_std = None
        self.ocr_utility_router_threshold = ocr_utility_router_threshold
        if bool(expert_ctc_ocr_checkpoint) != bool(ocr_utility_router_checkpoint):
            raise ValueError("Expert CTC OCR and OCR utility router must be used together")
        if expert_ctc_ocr_checkpoint is not None:
            if self.ctc_ocr is None:
                raise ValueError("Expert CTC OCR requires a primary CTC checkpoint")
            if not 0.0 <= ocr_utility_router_threshold <= 1.0:
                raise ValueError("OCR utility router threshold must lie in [0, 1]")
            expert_payload = torch.load(
                expert_ctc_ocr_checkpoint, map_location="cpu", weights_only=False
            )
            expert_config = PretrainedPixelCTCConfig(**expert_payload["config"])
            if expert_config != self.ctc_ocr.config:
                raise ValueError("Primary and expert CTC OCR configs differ")
            self.expert_ctc_ocr = PretrainedPixelCTCOCR(expert_config).to(device)
            self.expert_ctc_ocr.load_state_dict(
                expert_payload["pretrained_ctc_ocr"]
            )
            self.expert_ctc_ocr.pixel_encoder_loaded = True
            self.expert_ctc_ocr.eval().requires_grad_(False)
            router_payload = torch.load(
                ocr_utility_router_checkpoint, map_location="cpu", weights_only=False
            )
            if router_payload.get("checkpoint_schema_version") not in {
                "ocr-utility-router-0.1",
                "ocr-utility-router-0.2",
            }:
                raise ValueError("OCR utility router checkpoint schema differs")
            self.ocr_utility_router = OCRUtilityRouter(
                router_payload["input_dim"], router_payload["hidden_dim"]
            ).to(device)
            self.ocr_utility_router.load_state_dict(router_payload["router"])
            self.ocr_utility_router.eval().requires_grad_(False)
            self.ocr_utility_feature_names = router_payload["feature_names"]
            self.ocr_utility_feature_mean = router_payload["feature_mean"].to(device)
            self.ocr_utility_feature_std = router_payload["feature_std"].to(device)
        self.codec = OCRTextCodec(max_length=self.base.config.ocr_max_length)
        self.bond_set_selector = None
        self.bond_set_selector_payload = None
        self.endpoint_grouper = None
        self.endpoint_grouper_payload = None
        self.line_band_reader = None
        junction_aware_paths = (
            bond_set_selector_checkpoint,
            endpoint_grouper_checkpoint,
            line_band_checkpoint,
        )
        if any(path is not None for path in junction_aware_paths):
            if not all(path is not None for path in junction_aware_paths):
                raise ValueError(
                    "Junction-aware inference requires selector, grouper, and line-band checkpoints"
                )
            self.bond_set_selector, self.bond_set_selector_payload = (
                load_bond_set_selector(bond_set_selector_checkpoint, device)
            )
            self.endpoint_grouper, self.endpoint_grouper_payload = (
                load_endpoint_grouper(endpoint_grouper_checkpoint, device)
            )
            self.line_band_reader = load_line_band_reader(
                line_band_checkpoint, device
            )
        self.component_diagnostics: dict = {}

    def _correct_bond_objects(
        self,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
        bond_objects: list[dict],
        *,
        mode: str,
        graph_keep_threshold: float | None,
    ) -> tuple[list[dict], dict]:
        if self.graph_corrector is None:
            raise ValueError("Graph correction requires a graph corrector checkpoint")
        if not bond_objects:
            return bond_objects, {
                "graph_correction_mode": mode,
                "graph_removed_bond_objects": 0,
                "graph_display_type_changes": 0,
                "graph_chemical_type_changes": 0,
                "graph_keep_threshold": (
                    self.graph_corrector_keep_threshold
                    if graph_keep_threshold is None
                    else graph_keep_threshold
                ),
            }
        graph_output = self.graph_corrector(
            points,
            roles,
            torch.tensor(
                [
                    (
                        _element_id_from_label(label, int(role))
                        if self.graph_corrector_uses_elements
                        else ELEMENT_TO_ID["C"]
                    )
                    for label, role in zip(labels, roles)
                ],
                device=self.device,
            ),
            torch.stack([item["endpoints"].to(self.device) for item in bond_objects]),
            torch.tensor(
                [float(item["score"]) for item in bond_objects], device=self.device
            ),
            torch.tensor(
                [int(item["display"]) for item in bond_objects], device=self.device
            ),
            torch.tensor(
                [int(item["chemical"]) for item in bond_objects], device=self.device
            ),
        )
        return apply_graph_correction(
            bond_objects,
            graph_output,
            mode=mode,
            keep_threshold=(
                self.graph_corrector_keep_threshold
                if graph_keep_threshold is None
                else graph_keep_threshold
            ),
        )

    def _apply_semantic_bond_types(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        bond_objects: list[dict] | tuple[dict, ...],
    ) -> tuple[list[dict], dict]:
        if self.semantic_bond_relation is None or not bond_objects:
            return list(bond_objects), {
                "semantic_bond_relation_enabled": self.semantic_bond_relation
                is not None,
                "semantic_bond_relation_mode": self.semantic_bond_relation_mode,
                "semantic_chemical_type_changes": 0,
            }
        pair_indices = torch.tensor(
            [[item["junction_pair"] for item in bond_objects]],
            device=self.device,
            dtype=torch.long,
        )
        pair_mask = torch.ones(
            pair_indices.shape[:2], device=self.device, dtype=torch.bool
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            semantic_output = self.semantic_bond_relation(
                pixels,
                points.to(self.device)[None],
                pair_indices,
                pair_mask,
            )
        if self.semantic_bond_relation_mode == "aromatic_ring_only":
            updated, diagnostics = apply_aromatic_ring_semantics(
                bond_objects,
                semantic_output["chemical_logits"],
                mean_threshold=self.semantic_aromatic_mean_threshold,
                minimum_threshold=self.semantic_aromatic_minimum_threshold,
            )
        else:
            chemical = semantic_output["chemical_logits"].argmax(-1).tolist()
            updated = []
            changes = 0
            for item, chemical_type in zip(bond_objects, chemical):
                value = dict(item)
                changes += int(int(value["chemical"]) != chemical_type)
                value["chemical"] = int(chemical_type)
                updated.append(value)
            diagnostics = {"semantic_chemical_type_changes": changes}
        return updated, {
            "semantic_bond_relation_enabled": True,
            "semantic_bond_relation_mode": self.semantic_bond_relation_mode,
            **diagnostics,
        }

    def _junction_aware_bonds(
        self,
        pixels: torch.Tensor,
        bond_objects: list[dict],
    ):
        if (
            self.bond_set_selector is None
            or self.bond_set_selector_payload is None
            or self.endpoint_grouper is None
            or self.endpoint_grouper_payload is None
            or self.line_band_reader is None
        ):
            raise ValueError(
                "Junction-aware Bond source requires all v0.8.21 checkpoints"
            )
        line_band = read_line_band_features(
            self.line_band_reader, pixels, [bond_objects]
        )[0]
        result, diagnostics = select_and_group_bonds(
            pixels,
            bond_objects,
            line_band,
            self.bond_set_selector,
            self.bond_set_selector_payload,
            self.endpoint_grouper,
            self.endpoint_grouper_payload,
            projection_score_source=self.junction_projection_score_source,
            bond_type_source=self.junction_bond_type_source,
        )
        updated_objects, semantic_diagnostics = self._apply_semantic_bond_types(
            pixels, result.points, result.bond_objects
        )
        if self.semantic_bond_relation is not None and result.bond_objects:
            result = BondEndpointJunctionResult(
                points=result.points,
                bond_objects=tuple(updated_objects),
                raw_clusters=result.raw_clusters,
                self_loops=result.self_loops,
                duplicate_pairs=result.duplicate_pairs,
            )
        self.component_diagnostics.update(
            {
                **asdict(diagnostics),
                **semantic_diagnostics,
            }
        )
        return result

    @torch.inference_mode()
    def _semantic_relation_bonds(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
    ) -> list[dict]:
        if self.semantic_bond_existence is None:
            raise ValueError(
                "Semantic relation Junction source requires an existence checkpoint"
            )
        if len(points) < 2:
            raise ValueError("Semantic relation graph requires at least two Junctions")
        pairs = torch.combinations(
            torch.arange(len(points), device=self.device), r=2
        )
        selected_objects = []
        selected_logits = []
        scored_pairs = 0
        for start in range(0, len(pairs), self.semantic_bond_existence_chunk_size):
            current_pairs = pairs[
                start : start + self.semantic_bond_existence_chunk_size
            ]
            pair_indices = current_pairs[None]
            pair_mask = torch.ones(
                pair_indices.shape[:2], device=self.device, dtype=torch.bool
            )
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.semantic_bond_existence(
                    pixels,
                    points.to(self.device)[None],
                    pair_indices,
                    pair_mask,
                )
            probabilities = output["edge_logits"].float().sigmoid()
            display = output["display_logits"].float().argmax(-1)
            chemical_logits = output["chemical_logits"].float()
            for index in torch.where(
                probabilities >= self.semantic_bond_existence_threshold
            )[0].tolist():
                pair = tuple(int(value) for value in current_pairs[index].tolist())
                display_type = int(display[index])
                selected_objects.append(
                    {
                        "score": float(probabilities[index]),
                        "feature": torch.empty(0),
                        "endpoints": points[list(pair)].detach().float().cpu(),
                        "junction_pair": pair,
                        "display": display_type,
                        "chemical": display_type,
                        "direction": 0,
                    }
                )
                selected_logits.append(chemical_logits[index].cpu())
            scored_pairs += len(current_pairs)
        if not selected_objects:
            raise ValueError("Semantic Bond existence rejected every Junction pair")
        updated, ring_diagnostics = apply_aromatic_ring_semantics(
            selected_objects,
            torch.stack(selected_logits),
            mean_threshold=self.semantic_aromatic_mean_threshold,
            minimum_threshold=self.semantic_aromatic_minimum_threshold,
        )
        self.component_diagnostics.update(
            {
                "semantic_bond_existence_enabled": True,
                "semantic_bond_existence_threshold": self.semantic_bond_existence_threshold,
                "semantic_bond_pairs_scored": scored_pairs,
                "semantic_bond_pairs_selected": len(updated),
                **ring_diagnostics,
            }
        )
        return updated

    @torch.inference_mode()
    def _structured_atom_candidates(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
    ) -> list[list[AtomStateCandidate]]:
        if self.ctc_ocr is None:
            raise ValueError("Structured atom decoding requires a CTC OCR checkpoint")
        if pixels.size(0) != 1:
            raise ValueError("Structured atom decoding currently expects one image")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            features = self.base.vision_encoder.forward_stroke_resolution(pixels)
            text_output = self.text_head(features)
        text_objects = decode_text_objects(
            text_output,
            threshold=float(self.protocol["text_threshold"]),
            maximum_objects=self.selector_config.maximum_candidates,
        )[0]
        shifted_crops, point_indices = detected_text_crops(
            pixels[0], points, roles, text_objects, text_role=TEXT_ROLE
        )
        anchor_objects = [
            {
                key: value
                for key, value in item.items()
                if key not in ("bbox_x", "bbox_y")
            }
            for item in text_objects
        ]
        anchor_crops, anchor_indices = detected_text_crops(
            pixels[0], points, roles, anchor_objects, text_role=TEXT_ROLE
        )
        if point_indices != anchor_indices:
            raise ValueError("Structured shifted and anchor crop assignments differ")
        candidates: list[list[AtomStateCandidate]] = [
            [] for _ in range(len(points))
        ]
        if len(shifted_crops):
            combined = torch.cat((shifted_crops, anchor_crops), dim=0)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                combined_logits = self.ctc_ocr(combined).float()
            shifted_logits, anchor_logits = combined_logits.chunk(2)
            for crop_index, point_index in enumerate(point_indices):
                candidates[point_index] = merge_ctc_atom_candidates(
                    (
                        ("shifted", shifted_logits[crop_index]),
                        ("anchor", anchor_logits[crop_index]),
                    ),
                    current_text=labels[point_index],
                    beam_size=12,
                    top_paths_per_source=8,
                    maximum_candidates=8,
                    maximum_bytes=self.ctc_ocr.config.maximum_bytes,
                    token_beam=24,
                )
        for index, role in enumerate(roles.tolist()):
            if role != TEXT_ROLE or candidates[index]:
                continue
            candidates[index] = [
                atom_state_candidate(labels[index], 0.0, source="current")
            ]
        return candidates

    def _junctions(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        *,
        real_threshold: float,
        bbox_expansion_pixels: float,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        results = self._junctions_batch(
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=True,
        )
        if len(results) != 1:
            raise ValueError("Single-image Junction inference received a batch")
        return results[0]

    def _decode_ctc_crops(
        self,
        model: PretrainedPixelCTCOCR,
        shifted_crops: torch.Tensor,
        anchor_crops: torch.Tensor | None,
        *,
        include_visual: bool = False,
    ) -> tuple[list[tuple[str, float]], torch.Tensor | None, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if include_visual:
                shifted_logits, shifted_visual = model.forward_with_visual_descriptor(
                    shifted_crops
                )
            else:
                shifted_logits = model(shifted_crops)
                shifted_visual = None
        shifted = [decode_ctc_with_score(row.float()) for row in shifted_logits]
        if anchor_crops is None:
            return shifted, shifted_visual, shifted_logits
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if include_visual:
                anchor_logits, anchor_visual = model.forward_with_visual_descriptor(
                    anchor_crops
                )
            else:
                anchor_logits = model(anchor_crops)
                anchor_visual = None
        anchor = [decode_ctc_with_score(row.float()) for row in anchor_logits]
        use_shifted = [
            shifted_item[1]
            >= anchor_item[1] + float(self.ctc_anchor_fallback_margin)
            for shifted_item, anchor_item in zip(shifted, anchor)
        ]
        selected = [
            shifted_item
            if choose_shifted
            else anchor_item
            for shifted_item, anchor_item, choose_shifted in zip(
                shifted, anchor, use_shifted
            )
        ]
        if not include_visual:
            selected_logits = torch.stack(
                [
                    shifted_logits[index]
                    if choose_shifted
                    else anchor_logits[index]
                    for index, choose_shifted in enumerate(use_shifted)
                ]
            )
            return selected, None, selected_logits
        selected_visual = torch.stack(
            [
                shifted_visual[index] if choose_shifted else anchor_visual[index]
                for index, choose_shifted in enumerate(use_shifted)
            ]
        )
        selected_logits = torch.stack(
            [
                shifted_logits[index] if choose_shifted else anchor_logits[index]
                for index, choose_shifted in enumerate(use_shifted)
            ]
        )
        return selected, selected_visual, selected_logits

    def _route_ocr_predictions(
        self,
        primary: list[tuple[str, float]],
        expert: list[tuple[str, float]],
        primary_visual: torch.Tensor | None = None,
        expert_visual: torch.Tensor | None = None,
        primary_logits: torch.Tensor | None = None,
        expert_logits: torch.Tensor | None = None,
    ) -> tuple[list[tuple[str, float]], int, int]:
        if len(primary) != len(expert) or self.ocr_utility_router is None:
            raise ValueError("OCR utility routing inputs differ")
        selected = list(primary)
        disagreements = [
            index
            for index, (old, new) in enumerate(zip(primary, expert))
            if old[0] != new[0]
        ]
        if not disagreements:
            return selected, 0, 0
        rows = []
        uses_visual = any(
            name.startswith("old_visual_")
            for name in self.ocr_utility_feature_names
        )
        uses_cross_scores = "old_model_new_text_score" in self.ocr_utility_feature_names
        for index in disagreements:
            row = ocr_utility_features(*primary[index], *expert[index])
            if uses_visual:
                if primary_visual is None or expert_visual is None:
                    raise ValueError("Visual OCR router requires paired descriptors")
                row = add_ocr_visual_features(
                    row, primary_visual[index], expert_visual[index]
                )
            if uses_cross_scores:
                if primary_logits is None or expert_logits is None:
                    raise ValueError("Cross-score OCR router requires paired logits")
                row = add_ocr_cross_score_features(
                    row,
                    old_model_new_text_score=ctc_text_log_score(
                        primary_logits[index],
                        expert[index][0],
                        normalize_by_bytes=True,
                    ),
                    new_model_old_text_score=ctc_text_log_score(
                        expert_logits[index],
                        primary[index][0],
                        normalize_by_bytes=True,
                    ),
                )
            rows.append(row)
        features = torch.tensor(
            [
                [row[name] for name in self.ocr_utility_feature_names]
                for row in rows
            ],
            device=self.device,
            dtype=torch.float32,
        )
        features = (
            features - self.ocr_utility_feature_mean
        ) / self.ocr_utility_feature_std
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            probabilities = self.ocr_utility_router(features).float().sigmoid()
        routes = probabilities >= self.ocr_utility_router_threshold
        for row, route in zip(disagreements, routes.tolist()):
            if route:
                selected[row] = expert[row]
        return selected, len(disagreements), int(routes.sum())

    def _junctions_batch(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        *,
        real_threshold: float,
        bbox_expansion_pixels: float,
        include_ocr: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            query_output = self.query_detector(features)
            text_output = self.text_head(features)
            implicit_output = self.implicit_head(features)
        text_batches = decode_text_objects(
            text_output,
            threshold=float(self.protocol["text_threshold"]),
            maximum_objects=self.selector_config.maximum_candidates,
        )
        filter_text_batches = decode_text_objects(
            text_output,
            threshold=float(self.protocol["text_filter_threshold"]),
            maximum_objects=self.selector_config.maximum_candidates,
        )
        implicit_batches = decode_vertices(
            implicit_output,
            threshold=float(self.protocol["implicit_threshold"]),
            maximum_vertices=self.selector_config.maximum_candidates,
            nms_kernel_size=1,
        )
        merged_batches = []
        role_batches = []
        for text_items, filter_text, implicit_items in zip(
            text_batches, filter_text_batches, implicit_batches
        ):
            filtered_implicit = [
                item
                for item in implicit_items
                if outside_text_objects(item, filter_text, bbox_expansion_pixels)
            ]
            merged, roles = merge_dual_candidates(
                text_items,
                filtered_implicit,
                maximum_candidates=self.selector_config.maximum_candidates,
            )
            merged_batches.append(merged)
            role_batches.append(roles)
        grounded = fuse_dense_candidates_with_queries(
            query_output,
            merged_batches,
            maximum_candidates=self.selector_config.maximum_candidates,
            image_size=pixels.size(-1),
        )
        candidate_roles = padded_candidate_roles(
            role_batches, grounded["candidate_mask"]
        )
        candidate_auxiliary = dual_candidate_auxiliary(
            grounded["candidate_auxiliary"],
            merged_batches,
            grounded["candidate_mask"],
        )
        candidate_visual = sample_candidate_features(
            features, grounded["grounded_centers"], grounded["candidate_mask"]
        )
        candidate_pixel_patches = None
        if getattr(self.selector.config, "patch_mode", "none") in {
            "pixel",
            "semantic_pixel",
        }:
            candidate_pixel_patches = sample_candidate_ink_patches(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                patch_size=self.selector.config.patch_size,
            )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            hidden = self.grounded_selector.encode(
                grounded["query_features"],
                grounded["scalar_features"],
                grounded["grounded_centers"],
                grounded["candidate_mask"],
            )
            graph_output = self.graph(
                hidden,
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                grounded["base_logits"],
            )
            edge_auxiliary = None
            if getattr(self.selector.config, "edge_auxiliary_dim", 0):
                edge_auxiliary = endpoint_ray_features(
                    pixels,
                    grounded["grounded_centers"],
                    grounded["candidate_mask"],
                    graph_output["pair_indices"],
                )
            logits = selector_forward(
                self.selector,
                {
                    "candidate_hidden": hidden,
                    "base_logits": grounded["base_logits"],
                    "points": grounded["grounded_centers"],
                    "candidate_mask": grounded["candidate_mask"],
                    "edge_logits": graph_output["edge_logits"],
                    "candidate_auxiliary": candidate_auxiliary,
                    "candidate_visual": candidate_visual,
                    "candidate_roles": candidate_roles,
                    "edge_auxiliary": edge_auxiliary,
                    "candidate_pixel_patches": candidate_pixel_patches,
                },
                graph_output["pair_indices"],
            )
        thresholds = resolved_real_thresholds(real_threshold, None, None)
        point_batches = []
        selected_role_batches = []
        for batch_index in range(len(pixels)):
            selected = role_suppressed_role_threshold_indices(
                logits[batch_index],
                grounded["grounded_centers"][batch_index],
                grounded["candidate_mask"][batch_index],
                candidate_roles[batch_index],
                probability_thresholds=thresholds,
                suppression_radius_pixels=float(
                    self.protocol["suppression_radius_pixels"]
                ),
                image_size=pixels.size(-1),
            )
            points = grounded["grounded_centers"][batch_index, selected]
            if not len(points):
                raise ValueError(
                    f"Junction selector returned an empty set for batch item {batch_index}"
                )
            point_batches.append(points)
            selected_role_batches.append(candidate_roles[batch_index, selected])
        maximum_points = max(len(points) for points in point_batches)
        point_batch = torch.zeros(
            (len(pixels), maximum_points, 2),
            device=self.device,
            dtype=point_batches[0].dtype,
        )
        point_mask = torch.zeros(
            (len(pixels), maximum_points), device=self.device, dtype=torch.bool
        )
        for batch_index, points in enumerate(point_batches):
            point_batch[batch_index, : len(points)] = points
            point_mask[batch_index, : len(points)] = True
        if include_ocr and self.ctc_ocr is not None:
            label_batches = []
            utility_disagreements = 0
            utility_routes = 0
            for batch_index, (points, selected_roles) in enumerate(
                zip(point_batches, selected_role_batches)
            ):
                labels = ["C"] * len(points)
                crops, label_indices = detected_text_crops(
                    pixels[batch_index],
                    points,
                    selected_roles,
                    text_batches[batch_index],
                    text_role=TEXT_ROLE,
                )
                if len(crops):
                    anchor_crops = None
                    if self.ctc_anchor_fallback_margin is not None:
                        anchor_objects = [
                            {
                                key: value
                                for key, value in item.items()
                                if key not in ("bbox_x", "bbox_y")
                            }
                            for item in text_batches[batch_index]
                        ]
                        anchor_crops, anchor_indices = detected_text_crops(
                            pixels[batch_index],
                            points,
                            selected_roles,
                            anchor_objects,
                            text_role=TEXT_ROLE,
                        )
                        if anchor_indices != label_indices:
                            raise ValueError(
                                "Anchor and shifted OCR crop assignments differ"
                            )
                    include_visual = bool(
                        self.ocr_utility_feature_names
                        and any(
                            name.startswith("old_visual_")
                            for name in self.ocr_utility_feature_names
                        )
                    )
                    predictions, primary_visual, primary_logits = self._decode_ctc_crops(
                        self.ctc_ocr,
                        crops,
                        anchor_crops,
                        include_visual=include_visual,
                    )
                    if self.expert_ctc_ocr is not None:
                        expert_predictions, expert_visual, expert_logits = self._decode_ctc_crops(
                            self.expert_ctc_ocr,
                            crops,
                            anchor_crops,
                            include_visual=include_visual,
                        )
                        predictions, disagreements, routes = (
                            self._route_ocr_predictions(
                                predictions,
                                expert_predictions,
                                primary_visual,
                                expert_visual,
                                primary_logits,
                                expert_logits,
                            )
                        )
                        utility_disagreements += disagreements
                        utility_routes += routes
                    for point_index, (label, _) in zip(label_indices, predictions):
                        labels[point_index] = label
                label_batches.append(labels)
            self.component_diagnostics.update(
                {
                    "ocr_utility_router_enabled": self.ocr_utility_router is not None,
                    "ocr_utility_disagreements": utility_disagreements,
                    "ocr_utility_routes": utility_routes,
                    "ocr_utility_threshold": (
                        self.ocr_utility_router_threshold
                        if self.ocr_utility_router is not None
                        else None
                    ),
                }
            )
            flat_labels = [label for labels in label_batches for label in labels]
        elif include_ocr:
            ocr_features = self.base.ocr_source(pixels, features)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                token_ids = self.base.ocr_head.generate(
                    ocr_features,
                    point_batch,
                    point_mask,
                    pixels.shape[-2:],
                    roi_radius=20.0,
                )
            flat_labels = [self.codec.decode(row.tolist()) for row in token_ids.cpu()]
        else:
            flat_labels = ["C"] * int(point_mask.sum())
        results = []
        offset = 0
        for points, selected_roles in zip(point_batches, selected_role_batches):
            labels = flat_labels[offset : offset + len(points)]
            offset += len(points)
            labels = [
                label if int(role) == TEXT_ROLE else "C"
                for label, role in zip(labels, selected_roles)
            ]
            results.append((points, selected_roles, labels))
        if offset != len(flat_labels):
            raise ValueError("OCR output count differs from selected Junction count")
        return results

    def _bond_detector_features(self, stride4: torch.Tensor):
        if not isinstance(self.bond_detector, MultiScaleDenseBondObjectDetector):
            return (stride4,)
        stride8 = self.base.vision_encoder.forward_high_resolution_from_stroke(
            stride4
        )
        stride16 = stride8
        for stage in self.base.vision_encoder.stages[3:5]:
            stride16 = stage(stride16)
        return stride4, stride8, stride16

    def _decode_routed_bond_objects(
        self,
        stride4: torch.Tensor,
        *,
        threshold: float,
    ) -> list[list[dict]]:
        detector_features = self._bond_detector_features(stride4)
        base_ranked_batches = decode_dense_bond_objects(
            _apply_detector(self.bond_detector, detector_features),
            threshold=0.0,
            maximum_objects=128,
        )
        if self.expert_bond_detector is None:
            self.component_diagnostics["bond_style_expert_probability"] = 0.0
            self.component_diagnostics["selected_bond_expert"] = "base"
            return [
                [item for item in ranked if float(item["score"]) >= threshold]
                for ranked in base_ranked_batches
            ]
        probabilities = self.style_gate(stride4).float().sigmoid()
        expert_ranked_batches = decode_dense_bond_objects(
            _apply_detector(self.expert_bond_detector, detector_features),
            threshold=0.0,
            maximum_objects=128,
        )
        if self.bond_utility_router is None:
            route_probabilities = probabilities
            route_threshold = self.style_gate_threshold
            route_kind = "style"
        else:
            feature_rows = []
            for base_ranked, expert_ranked, style_probability in zip(
                base_ranked_batches, expert_ranked_batches, probabilities
            ):
                values = combined_utility_features(
                    base_ranked,
                    expert_ranked,
                    style_probability=float(style_probability),
                )
                feature_rows.append(
                    [values[name] for name in self.bond_utility_feature_names]
                )
            feature_tensor = torch.tensor(
                feature_rows, device=self.device, dtype=torch.float32
            )
            feature_tensor = (
                feature_tensor - self.bond_utility_feature_mean
            ) / self.bond_utility_feature_std
            route_probabilities = self.bond_utility_router(
                feature_tensor
            ).float().sigmoid()
            route_threshold = self.bond_utility_threshold
            route_kind = "utility"
        routes = [float(value) >= route_threshold for value in route_probabilities]
        selected_ranked = [
            expert if use_expert else base
            for base, expert, use_expert in zip(
                base_ranked_batches, expert_ranked_batches, routes
            )
        ]
        selected = [
            [item for item in ranked if float(item["score"]) >= threshold]
            for ranked in selected_ranked
        ]
        self.component_diagnostics["bond_style_expert_probability"] = (
            float(probabilities[0]) if len(probabilities) == 1 else None
        )
        self.component_diagnostics["bond_utility_expert_probability"] = (
            float(route_probabilities[0])
            if route_kind == "utility" and len(route_probabilities) == 1
            else None
        )
        self.component_diagnostics["selected_bond_expert"] = (
            "style_expert" if routes[0] else "base"
        ) if len(routes) == 1 else None
        self.component_diagnostics["bond_style_expert_routes"] = sum(routes)
        self.component_diagnostics["bond_expert_route_kind"] = route_kind
        self.component_diagnostics["bond_expert_route_threshold"] = route_threshold
        return selected

    @torch.inference_mode()
    def predict_components(
        self,
        pixels: torch.Tensor,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        include_ocr: bool = True,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
        self.component_diagnostics = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            features = self.base.vision_encoder.forward_stroke_resolution(pixels)
            bond_batches = (
                [[]]
                if junction_source == "semantic_relations"
                else self._decode_routed_bond_objects(
                    features,
                    threshold=(
                        0.0
                        if junction_source
                        in {"junction_aware_bonds", "junction_semantic_relations"}
                        else bond_threshold
                    ),
                )
            )
        bond_objects = bond_batches[0]
        if self.text_interior_bond_veto and bond_objects:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                text_output = self.text_head(features)
            text_objects = decode_text_objects(
                text_output,
                threshold=float(self.protocol["text_threshold"]),
                maximum_objects=self.selector_config.maximum_candidates,
            )[0]
            bond_objects, removed = suppress_text_interior_bond_objects(
                bond_objects,
                text_objects,
                image_size=pixels.size(-1),
            )
            self.component_diagnostics.update(
                {
                    "text_interior_bond_veto_enabled": True,
                    "text_interior_bond_veto_text_objects": len(text_objects),
                    "text_interior_bond_veto_removed": len(removed),
                    "text_interior_bond_veto_details": removed,
                }
            )
        else:
            self.component_diagnostics.update(
                {
                    "text_interior_bond_veto_enabled": self.text_interior_bond_veto,
                    "text_interior_bond_veto_text_objects": 0,
                    "text_interior_bond_veto_removed": 0,
                    "text_interior_bond_veto_details": [],
                }
            )
        if junction_source in {
            "bond_endpoints",
            "junction_aware_bonds",
            "junction_semantic_relations",
        }:
            endpoint_result = (
                self._junction_aware_bonds(pixels, bond_objects)
                if junction_source
                in {"junction_aware_bonds", "junction_semantic_relations"}
                else junctions_from_bond_endpoints(
                    bond_objects,
                    width=pixels.size(-1),
                    height=pixels.size(-2),
                    radius_pixels=endpoint_cluster_radius_pixels,
                )
            )
            points = endpoint_result.points.to(self.device)
            bond_objects = list(endpoint_result.bond_objects)
            if not len(points):
                raise ValueError("Bond endpoint clustering returned no Junctions")
            if include_ocr:
                detected_points, detected_roles, detected_labels = self._junctions(
                    pixels,
                    features,
                    real_threshold=real_threshold,
                    bbox_expansion_pixels=bbox_expansion_pixels,
                )
                roles, labels = attach_text_to_bond_junctions(
                    points,
                    detected_points,
                    detected_roles,
                    detected_labels,
                    image_size=pixels.size(-1),
                    maximum_distance_pixels=text_attach_radius_pixels,
                )
            else:
                roles = torch.full(
                    (len(points),),
                    IMPLICIT_ROLE,
                    device=self.device,
                    dtype=torch.long,
                )
                labels = ["C"] * len(points)
            if junction_source == "junction_semantic_relations":
                bond_objects = self._semantic_relation_bonds(pixels, points)
        elif junction_source in {"vertex", "semantic_relations"}:
            if include_ocr:
                points, roles, labels = self._junctions(
                    pixels,
                    features,
                    real_threshold=real_threshold,
                    bbox_expansion_pixels=bbox_expansion_pixels,
                )
            else:
                points, roles, labels = self._junctions_batch(
                    pixels,
                    features,
                    real_threshold=real_threshold,
                    bbox_expansion_pixels=bbox_expansion_pixels,
                    include_ocr=False,
                )[0]
            if junction_source == "semantic_relations":
                bond_objects = self._semantic_relation_bonds(pixels, points)
        else:
            raise ValueError(f"Unknown Junction source: {junction_source}")
        return points, roles, labels, bond_objects

    @torch.inference_mode()
    def predict_components_batch(
        self,
        pixels: torch.Tensor,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        include_ocr: bool = True,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]]:
        self.component_diagnostics = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            features = self.base.vision_encoder.forward_stroke_resolution(pixels)
            bond_batches = self._decode_routed_bond_objects(
                features,
                threshold=(
                    0.0
                    if junction_source == "junction_aware_bonds"
                    else bond_threshold
                ),
            )
        detected_junctions = self._junctions_batch(
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=include_ocr,
        )
        if junction_source == "vertex":
            return [
                (points, roles, labels, bonds)
                for (points, roles, labels), bonds in zip(
                    detected_junctions, bond_batches
                )
            ]
        if junction_source not in {"bond_endpoints", "junction_aware_bonds"}:
            raise ValueError(f"Unknown Junction source: {junction_source}")
        results = []
        for batch_index, (
            (detected_points, detected_roles, detected_labels),
            bonds,
        ) in enumerate(zip(
            detected_junctions, bond_batches
        )):
            endpoint_result = (
                self._junction_aware_bonds(
                    pixels[batch_index : batch_index + 1], bonds
                )
                if junction_source == "junction_aware_bonds"
                else junctions_from_bond_endpoints(
                    bonds,
                    width=pixels.size(-1),
                    height=pixels.size(-2),
                    radius_pixels=endpoint_cluster_radius_pixels,
                )
            )
            points = endpoint_result.points.to(self.device)
            if not len(points):
                raise ValueError("Bond endpoint clustering returned no Junctions")
            if include_ocr:
                roles, labels = attach_text_to_bond_junctions(
                    points,
                    detected_points,
                    detected_roles,
                    detected_labels,
                    image_size=pixels.size(-1),
                    maximum_distance_pixels=text_attach_radius_pixels,
                )
            else:
                roles = torch.full(
                    (len(points),),
                    IMPLICIT_ROLE,
                    device=self.device,
                    dtype=torch.long,
                )
                labels = ["C"] * len(points)
            results.append((points, roles, labels, list(endpoint_result.bond_objects)))
        return results

    @torch.inference_mode()
    def predict(
        self,
        pixels: torch.Tensor,
        molecule_id: str,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        assignment_mode: str = "nearest",
        graph_correction_mode: str = "none",
        graph_keep_threshold: float | None = None,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
        atom_decode_mode: str = "greedy",
    ) -> tuple[dict, dict]:
        if atom_decode_mode not in {"greedy", "structured_invalid_only"}:
            raise ValueError(f"Unknown atom decode mode: {atom_decode_mode}")
        points, roles, labels, bond_objects = self.predict_components(
            pixels,
            real_threshold=real_threshold,
            bond_threshold=bond_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            junction_source=junction_source,
            endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
            text_attach_radius_pixels=text_attach_radius_pixels,
        )
        graph_diagnostics = {}
        if graph_correction_mode in {"type_only", "full"}:
            bond_objects, graph_diagnostics = self._correct_bond_objects(
                points,
                roles,
                labels,
                bond_objects,
                mode=graph_correction_mode,
                graph_keep_threshold=graph_keep_threshold,
            )
        explicit_sources = {
            "bond_endpoints",
            "junction_aware_bonds",
            "semantic_relations",
            "junction_semantic_relations",
        }
        junction_assignments = (
            [item["junction_pair"] for item in bond_objects]
            if junction_source in explicit_sources
            else None
        )
        effective_assignment_mode = (
            "explicit"
            if junction_source in explicit_sources
            else assignment_mode
        )
        assembler_diagnostics = {}
        if effective_assignment_mode in {"learned", "learned_rescue"}:
            if self.assembler is None:
                raise ValueError("Learned assignment requires an assembler checkpoint")
            if bond_objects:
                bond_endpoints = torch.stack(
                    [item["endpoints"].to(self.device) for item in bond_objects]
                )
                bond_scores = torch.tensor(
                    [float(item["score"]) for item in bond_objects],
                    device=self.device,
                )
                logits = self.assembler(
                    points, roles, bond_endpoints, bond_scores
                )
                if effective_assignment_mode == "learned_rescue":
                    junction_assignments, assembler_diagnostics = (
                        decode_junction_assignment_rescue(
                            logits,
                            bond_scores,
                            bond_endpoints,
                            points,
                        )
                    )
                else:
                    junction_assignments, assembler_diagnostics = (
                        decode_junction_assignments(logits, bond_scores)
                    )
            else:
                junction_assignments = []
        scene, diagnostics = assemble_chemscene(
            molecule_id,
            points,
            roles,
            labels,
            bond_objects,
            width=pixels.size(-1),
            height=pixels.size(-2),
            assignment_mode=effective_assignment_mode,
            junction_assignments=junction_assignments,
            unresolved_as_dummy=atom_decode_mode != "greedy",
        )
        atom_diagnostics = {"atom_decode_mode": atom_decode_mode}
        atom_candidates = None
        if atom_decode_mode == "structured_invalid_only":
            atom_candidates = self._structured_atom_candidates(
                pixels, points, roles, labels
            )
            scene, atom_diagnostics = rerank_invalid_scene_atom_states(
                scene, atom_candidates
            )
        if graph_correction_mode == "invalid_only":
            graph_diagnostics = {
                "graph_correction_mode": "invalid_only",
                "graph_correction_triggered": False,
                "graph_removed_bond_objects": 0,
                "graph_display_type_changes": 0,
                "graph_chemical_type_changes": 0,
                "graph_keep_threshold": (
                    self.graph_corrector_keep_threshold
                    if graph_keep_threshold is None
                    else graph_keep_threshold
                ),
            }
            try:
                compile_scene_with_aromatic_fallback(scene)
            except Exception as base_error:
                corrected_bonds, correction = self._correct_bond_objects(
                    points,
                    roles,
                    labels,
                    bond_objects,
                    mode="type_only",
                    graph_keep_threshold=graph_keep_threshold,
                )
                corrected_bonds, post_semantic = self._apply_semantic_bond_types(
                    pixels, points, corrected_bonds
                )
                scene, diagnostics = assemble_chemscene(
                    molecule_id,
                    points,
                    roles,
                    labels,
                    corrected_bonds,
                    width=pixels.size(-1),
                    height=pixels.size(-2),
                    assignment_mode=effective_assignment_mode,
                    junction_assignments=junction_assignments,
                    unresolved_as_dummy=atom_decode_mode != "greedy",
                )
                if atom_candidates is not None:
                    scene, atom_diagnostics = rerank_invalid_scene_atom_states(
                        scene, atom_candidates
                    )
                graph_diagnostics = {
                    **correction,
                    "graph_correction_mode": "invalid_only",
                    "graph_correction_triggered": True,
                    "graph_correction_base_error": str(base_error),
                    **{
                        f"post_graph_{key}": value
                        for key, value in post_semantic.items()
                    },
                }
        diagnostics.update(assembler_diagnostics)
        diagnostics.update(graph_diagnostics)
        diagnostics.update(atom_diagnostics)
        diagnostics.update(self.component_diagnostics)
        diagnostics["ocr_labels"] = labels
        diagnostics["real_threshold"] = real_threshold
        diagnostics["bond_threshold"] = bond_threshold
        diagnostics["junction_source"] = junction_source
        diagnostics["endpoint_cluster_radius_pixels"] = (
            endpoint_cluster_radius_pixels
        )
        diagnostics["text_attach_radius_pixels"] = text_attach_radius_pixels
        return scene, diagnostics


def load_compatible_fallback_ctc_ocr(
    checkpoint: str | Path,
    primary: PretrainedPixelCTCOCR,
    device: torch.device,
) -> PretrainedPixelCTCOCR:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = PretrainedPixelCTCConfig(**payload["config"])
    if config != primary.config:
        raise ValueError("Primary and fallback CTC OCR configs differ")
    model = PretrainedPixelCTCOCR(config).to(device)
    model.load_state_dict(payload["pretrained_ctc_ocr"])
    model.pixel_encoder_loaded = True
    return model.eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
    )
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--expert-bond-checkpoint")
    parser.add_argument("--style-gate-checkpoint")
    parser.add_argument("--style-gate-threshold", type=float, default=0.90)
    parser.add_argument("--bond-utility-router-checkpoint")
    parser.add_argument(
        "--bond-utility-break-budget", choices=("0", "1", "2", "5", "10"), default="0"
    )
    parser.add_argument("--text-interior-bond-veto", action="store_true")
    parser.add_argument("--assembler-checkpoint")
    parser.add_argument("--graph-corrector-checkpoint")
    parser.add_argument("--ocr-checkpoint")
    parser.add_argument("--ctc-ocr-checkpoint")
    parser.add_argument("--expert-ctc-ocr-checkpoint")
    parser.add_argument("--invalid-fallback-ctc-ocr-checkpoint")
    parser.add_argument("--ocr-utility-router-checkpoint")
    parser.add_argument("--ocr-utility-router-threshold", type=float, default=0.95)
    parser.add_argument("--text-bbox-checkpoint")
    parser.add_argument("--ctc-anchor-fallback-margin", type=float)
    parser.add_argument("--bond-set-selector-checkpoint")
    parser.add_argument("--endpoint-grouper-checkpoint")
    parser.add_argument("--line-band-checkpoint")
    parser.add_argument("--semantic-bond-relation-checkpoint")
    parser.add_argument("--semantic-bond-existence-checkpoint")
    parser.add_argument("--semantic-bond-existence-threshold", type=float, default=0.60)
    parser.add_argument("--semantic-bond-existence-chunk-size", type=int, default=256)
    parser.add_argument(
        "--semantic-bond-relation-mode",
        choices=("chemical_all", "aromatic_ring_only"),
        default="chemical_all",
    )
    parser.add_argument("--semantic-aromatic-mean-threshold", type=float, default=0.72)
    parser.add_argument(
        "--semantic-aromatic-minimum-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--junction-projection-score-source",
        choices=("detector", "selector"),
        default="detector",
    )
    parser.add_argument(
        "--junction-bond-type-source",
        choices=("detector", "line_band"),
        default="detector",
    )
    parser.add_argument(
        "--graph-correction-mode",
        choices=("none", "type_only", "full", "invalid_only"),
        default="none",
    )
    parser.add_argument("--graph-keep-threshold", type=float)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--image-is-prepared", action="store_true")
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument(
        "--junction-source",
        choices=(
            "vertex",
            "bond_endpoints",
            "junction_aware_bonds",
            "semantic_relations",
            "junction_semantic_relations",
        ),
        default="vertex",
    )
    parser.add_argument("--endpoint-cluster-radius-pixels", type=float, default=12.0)
    parser.add_argument("--text-attach-radius-pixels", type=float, default=24.0)
    parser.add_argument(
        "--atom-decode-mode",
        choices=("greedy", "structured_invalid_only"),
        default="greedy",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("nearest", "global_unique", "learned", "learned_rescue"),
        default="nearest",
    )
    args = parser.parse_args()

    if args.invalid_fallback_ctc_ocr_checkpoint and (
        args.expert_ctc_ocr_checkpoint or args.ocr_utility_router_checkpoint
    ):
        raise ValueError(
            "Invalid-only OCR fallback cannot be combined with utility OCR routing"
        )

    device = torch.device(args.device)
    started = time.perf_counter()
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        Path(args.assembler_checkpoint) if args.assembler_checkpoint else None,
        Path(args.graph_corrector_checkpoint)
        if args.graph_corrector_checkpoint
        else None,
        Path(args.ocr_checkpoint) if args.ocr_checkpoint else None,
        Path(args.ctc_ocr_checkpoint) if args.ctc_ocr_checkpoint else None,
        Path(args.text_bbox_checkpoint) if args.text_bbox_checkpoint else None,
        args.ctc_anchor_fallback_margin,
        Path(args.bond_set_selector_checkpoint)
        if args.bond_set_selector_checkpoint
        else None,
        Path(args.endpoint_grouper_checkpoint)
        if args.endpoint_grouper_checkpoint
        else None,
        Path(args.line_band_checkpoint) if args.line_band_checkpoint else None,
        args.junction_projection_score_source,
        args.junction_bond_type_source,
        Path(args.semantic_bond_relation_checkpoint)
        if args.semantic_bond_relation_checkpoint
        else None,
        args.semantic_bond_relation_mode,
        args.semantic_aromatic_mean_threshold,
        args.semantic_aromatic_minimum_threshold,
        Path(args.semantic_bond_existence_checkpoint)
        if args.semantic_bond_existence_checkpoint
        else None,
        args.semantic_bond_existence_threshold,
        args.semantic_bond_existence_chunk_size,
        expert_bond_checkpoint=(
            Path(args.expert_bond_checkpoint)
            if args.expert_bond_checkpoint
            else None
        ),
        style_gate_checkpoint=(
            Path(args.style_gate_checkpoint) if args.style_gate_checkpoint else None
        ),
        style_gate_threshold=args.style_gate_threshold,
        bond_utility_router_checkpoint=(
            Path(args.bond_utility_router_checkpoint)
            if args.bond_utility_router_checkpoint
            else None
        ),
        bond_utility_break_budget=args.bond_utility_break_budget,
        text_interior_bond_veto=args.text_interior_bond_veto,
        expert_ctc_ocr_checkpoint=(
            Path(args.expert_ctc_ocr_checkpoint)
            if args.expert_ctc_ocr_checkpoint
            else None
        ),
        ocr_utility_router_checkpoint=(
            Path(args.ocr_utility_router_checkpoint)
            if args.ocr_utility_router_checkpoint
            else None
        ),
        ocr_utility_router_threshold=args.ocr_utility_router_threshold,
    )
    fallback_ctc_ocr = None
    if args.invalid_fallback_ctc_ocr_checkpoint:
        if pipeline.ctc_ocr is None:
            raise ValueError("Invalid-only OCR fallback requires a primary CTC OCR")
        fallback_ctc_ocr = load_compatible_fallback_ctc_ocr(
            args.invalid_fallback_ctc_ocr_checkpoint,
            pipeline.ctc_ocr,
            device,
        )
    load_seconds = time.perf_counter() - started
    source = Image.open(args.image)
    image = (
        source.convert("RGB")
        if args.image_is_prepared
        else prepare_input_image(
            source, image_size=args.image_size, content_size=args.content_size
        )
    )
    if image.size != (args.image_size, args.image_size):
        image = image.resize(
            (args.image_size, args.image_size), Image.Resampling.BICUBIC
        )
    pixels = image_tensor(image).unsqueeze(0).to(device)
    molecule_id = Path(args.image).stem
    started = time.perf_counter()
    scene, diagnostics = pipeline.predict(
        pixels,
        molecule_id,
        real_threshold=args.real_threshold,
        bond_threshold=args.bond_threshold,
        assignment_mode=args.assignment_mode,
        graph_correction_mode=args.graph_correction_mode,
        graph_keep_threshold=args.graph_keep_threshold,
        junction_source=args.junction_source,
        endpoint_cluster_radius_pixels=args.endpoint_cluster_radius_pixels,
        text_attach_radius_pixels=args.text_attach_radius_pixels,
        atom_decode_mode=args.atom_decode_mode,
    )
    primary_scene = scene
    primary_diagnostics = diagnostics
    primary_compile = None
    primary_error = None
    fallback_scene = None
    fallback_diagnostics = None
    fallback_compile = None
    fallback_error = None
    fallback_triggered = False
    fallback_selected = False
    if fallback_ctc_ocr is not None:
        try:
            primary_compile = compile_scene_with_aromatic_fallback(primary_scene)
        except Exception as error:
            primary_error = str(error)
            fallback_triggered = True
            primary_ctc_ocr = pipeline.ctc_ocr
            pipeline.ctc_ocr = fallback_ctc_ocr
            try:
                fallback_scene, fallback_diagnostics = pipeline.predict(
                    pixels,
                    molecule_id,
                    real_threshold=args.real_threshold,
                    bond_threshold=args.bond_threshold,
                    assignment_mode=args.assignment_mode,
                    graph_correction_mode=args.graph_correction_mode,
                    graph_keep_threshold=args.graph_keep_threshold,
                    junction_source=args.junction_source,
                    endpoint_cluster_radius_pixels=args.endpoint_cluster_radius_pixels,
                    text_attach_radius_pixels=args.text_attach_radius_pixels,
                    atom_decode_mode=args.atom_decode_mode,
                )
            finally:
                pipeline.ctc_ocr = primary_ctc_ocr
            try:
                fallback_compile = compile_scene_with_aromatic_fallback(
                    fallback_scene
                )
            except Exception as error:
                fallback_error = str(error)
            if fallback_compile is not None:
                scene = fallback_scene
                diagnostics = fallback_diagnostics
                fallback_selected = True
        diagnostics = {
            **diagnostics,
            "invalid_ocr_fallback": {
                "enabled": True,
                "triggered": fallback_triggered,
                "selected": fallback_selected,
                "reason": (
                    "primary_invalid_fallback_valid"
                    if fallback_selected
                    else "primary_valid"
                    if not fallback_triggered
                    else "both_invalid"
                ),
                "primary_valid": primary_compile is not None,
                "fallback_valid": fallback_compile is not None,
                "primary_error": primary_error,
                "fallback_error": fallback_error,
                "checkpoint": args.invalid_fallback_ctc_ocr_checkpoint,
            },
        }
    inference_seconds = time.perf_counter() - started

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "input-prepared.png")
    if fallback_ctc_ocr is not None:
        (output / "candidate-primary.chemscene.json").write_text(
            json.dumps(primary_scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if primary_compile is not None:
            (output / "candidate-primary.sdf").write_text(
                primary_compile[0], encoding="utf-8"
            )
        if fallback_scene is not None:
            (output / "candidate-fallback.chemscene.json").write_text(
                json.dumps(fallback_scene, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if fallback_compile is not None:
            (output / "candidate-fallback.sdf").write_text(
                fallback_compile[0], encoding="utf-8"
            )
    (output / "prediction-raw.chemscene.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        sdf, compiled_scene, aromatic_fallback = compile_scene_with_aromatic_fallback(
            scene
        )
    except Exception as error:
        (output / "prediction.chemscene.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        diagnostic_sdf = output / "prediction-repaired.sdf"
        atoms, bonds = write_invalid_prediction_sdf(
            chemsvg_from_chemscene_v0_1(scene), diagnostic_sdf
        )
        diagnostic_rendered = False
        diagnostic_mol = Chem.MolFromMolFile(
            str(diagnostic_sdf), removeHs=False, sanitize=False, strictParsing=False
        )
        if diagnostic_mol is not None:
            try:
                diagnostic_mol.UpdatePropertyCache(strict=False)
                drawer = rdMolDraw2D.MolDraw2DCairo(args.image_size, args.image_size)
                drawer.drawOptions().padding = 0.08
                drawer.DrawMolecule(diagnostic_mol)
                drawer.FinishDrawing()
                (output / "prediction-repaired.png").write_bytes(
                    drawer.GetDrawingText()
                )
                diagnostic_rendered = True
            except Exception:
                diagnostic_rendered = False
        metrics = {
            "model_version": args.model_version or (
                "v0.8.23"
                if args.atom_decode_mode == "structured_invalid_only"
                else "v0.8.22"
                if args.junction_source == "junction_aware_bonds"
                else "v0.8.16"
                if args.junction_source == "bond_endpoints"
                else "v0.8.x"
            ),
            "protocol": "ChemScene diagnostic invalid SDF",
            "device": args.device,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "vertices": atoms,
            "candidate_strokes": diagnostics["decoded_bond_objects"],
            "repaired_strokes": bonds,
            "sdf_valid": False,
            "invalid_fallback_ctc_ocr_checkpoint": (
                args.invalid_fallback_ctc_ocr_checkpoint
            ),
            "diagnostic_rendered": diagnostic_rendered,
            "error": str(error),
            "junction_source": args.junction_source,
            "atom_decode_mode": args.atom_decode_mode,
            "diagnostics": diagnostics,
        }
        if args.invalid_fallback_ctc_ocr_checkpoint:
            metrics["protocol"] += " + invalid-only dual OCR fallback"
        (output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    (output / "prediction.chemscene.json").write_text(
        json.dumps(compiled_scene, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sdf_path = output / "prediction-repaired.sdf"
    sdf_path.write_text(sdf, encoding="utf-8")
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise RuntimeError("Cannot read compiled ChemScene SDF")
    drawer = rdMolDraw2D.MolDraw2DCairo(args.image_size, args.image_size)
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    (output / "prediction-repaired.png").write_bytes(drawer.GetDrawingText())
    metrics = {
        "model_version": args.model_version or (
            "v0.8.23"
            if args.atom_decode_mode == "structured_invalid_only"
            else "v0.8.22"
            if args.junction_source == "junction_aware_bonds"
            else "v0.8.16"
            if args.junction_source == "bond_endpoints"
            else "v0.8.9"
            if args.text_bbox_checkpoint
            else "v0.8.8"
            if args.ctc_ocr_checkpoint
            else "v0.8.6"
            if args.ocr_checkpoint
            else "v0.8.5"
            if args.graph_correction_mode != "none"
            else "v0.8.4" if args.assembler_checkpoint else "v0.8.3"
        ),
        "protocol": (
            "top-128 Bond selector -> visual endpoint grouping -> OCR label attachment + "
            + (
                "CTC top-k fixed-topology atom decode + "
                if args.atom_decode_mode == "structured_invalid_only"
                else ""
            )
            + f"graph {args.graph_correction_mode} + ChemScene"
            if args.junction_source == "junction_aware_bonds"
            else "semantic BondObject endpoints -> Junction + OCR label attachment + "
            f"graph {args.graph_correction_mode} + ChemScene"
            if args.junction_source == "bond_endpoints"
            else "predicted Junction + asymmetric TextObject bbox + conservative "
            "pretrained CTC OCR + multiscale "
            f"BondObject + graph {args.graph_correction_mode} + ChemScene"
            if args.text_bbox_checkpoint
            else "predicted Junction + bbox pretrained CTC OCR + multiscale "
            f"BondObject + graph {args.graph_correction_mode} + ChemScene"
            if args.ctc_ocr_checkpoint
            else "predicted Junction + Pixel OCR + multiscale BondObject + "
            f"graph {args.graph_correction_mode} + ChemScene"
            if args.graph_correction_mode != "none"
            else "predicted Junction + Pixel OCR + multiscale BondObject + ChemScene"
        ),
        "device": args.device,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "vertices": mol.GetNumAtoms(),
        "candidate_strokes": diagnostics["decoded_bond_objects"],
        "repaired_strokes": mol.GetNumBonds(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "aromatic_fallback": aromatic_fallback,
        "explicit_h_valence_repairs": sum(
            int(raw["atom"].get("explicit_h_count", 0))
            != int(compiled["atom"].get("explicit_h_count", 0))
            for raw, compiled in zip(scene["junctions"], compiled_scene["junctions"])
        ),
        "assembler_checkpoint": args.assembler_checkpoint,
        "graph_corrector_checkpoint": args.graph_corrector_checkpoint,
        "ocr_checkpoint": args.ocr_checkpoint,
        "ctc_ocr_checkpoint": args.ctc_ocr_checkpoint,
        "expert_ctc_ocr_checkpoint": args.expert_ctc_ocr_checkpoint,
        "invalid_fallback_ctc_ocr_checkpoint": (
            args.invalid_fallback_ctc_ocr_checkpoint
        ),
        "ocr_utility_router_checkpoint": args.ocr_utility_router_checkpoint,
        "ocr_utility_router_threshold": args.ocr_utility_router_threshold,
        "text_bbox_checkpoint": args.text_bbox_checkpoint,
        "atom_decode_mode": args.atom_decode_mode,
        "bond_set_selector_checkpoint": args.bond_set_selector_checkpoint,
        "endpoint_grouper_checkpoint": args.endpoint_grouper_checkpoint,
        "line_band_checkpoint": args.line_band_checkpoint,
        "semantic_bond_relation_checkpoint": args.semantic_bond_relation_checkpoint,
        "semantic_bond_relation_mode": args.semantic_bond_relation_mode,
        "semantic_aromatic_mean_threshold": args.semantic_aromatic_mean_threshold,
        "semantic_aromatic_minimum_threshold": args.semantic_aromatic_minimum_threshold,
        "ctc_anchor_fallback_margin": args.ctc_anchor_fallback_margin,
        "graph_correction_mode": args.graph_correction_mode,
        "junction_source": args.junction_source,
        "junction_bond_type_source": args.junction_bond_type_source,
        "endpoint_cluster_radius_pixels": args.endpoint_cluster_radius_pixels,
        "text_attach_radius_pixels": args.text_attach_radius_pixels,
        "diagnostics": diagnostics,
    }
    if args.invalid_fallback_ctc_ocr_checkpoint:
        metrics["protocol"] += " + invalid-only dual OCR fallback"
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
