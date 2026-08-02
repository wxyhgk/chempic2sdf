"""Shared visual vertex detection and anchor-local OCR inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from Data.sdf.chemsvg_sequence_v4 import VertexRecordV4
from chemvector.model import ChemVectorModel, OCR_ROLE_LABELS
from chemvector.ocr import OCRTextCodec, parse_atom_display_text
from chemvector.proposal_keep import (
    ProposalKeepHead,
    decode_vertices_with_proposal_keep,
)
from chemvector.proposal_graph import (
    ProposalGraphRefiner,
    decode_proposal_graph,
)
from chemvector.vertex import decode_vertices


Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class VertexDetectionResult:
    vertices: list[VertexRecordV4]
    stroke_features: torch.Tensor
    proposal_graph_edges: dict[tuple[Coordinate, Coordinate], float]


@torch.inference_mode()
def detect_vertex_records(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    *,
    threshold: float = 0.3,
    maximum_vertices: int = 128,
    ocr_roi_radius: float = 20.0,
    nms_kernel_size: int = 3,
    proposal_keep_head: ProposalKeepHead | None = None,
    proposal_keep_threshold: float = 0.5,
    proposal_graph_head: ProposalGraphRefiner | None = None,
    proposal_graph_threshold: float = 0.5,
) -> tuple[list[VertexRecordV4], torch.Tensor]:
    """Detect visual vertices, recognize labels, and return shared stroke features."""

    result = detect_vertex_records_with_evidence(
        model,
        pixel_values,
        threshold=threshold,
        maximum_vertices=maximum_vertices,
        ocr_roi_radius=ocr_roi_radius,
        nms_kernel_size=nms_kernel_size,
        proposal_keep_head=proposal_keep_head,
        proposal_keep_threshold=proposal_keep_threshold,
        proposal_graph_head=proposal_graph_head,
        proposal_graph_threshold=proposal_graph_threshold,
    )
    return result.vertices, result.stroke_features


@torch.inference_mode()
def detect_vertex_records_with_evidence(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    *,
    threshold: float = 0.3,
    maximum_vertices: int = 128,
    ocr_roi_radius: float = 20.0,
    nms_kernel_size: int = 3,
    proposal_keep_head: ProposalKeepHead | None = None,
    proposal_keep_threshold: float = 0.5,
    proposal_graph_head: ProposalGraphRefiner | None = None,
    proposal_graph_threshold: float = 0.5,
) -> VertexDetectionResult:
    """Detect vertices and retain graph edge evidence for selected proposals."""

    if model.vertex_head is None or model.ocr_head is None:
        raise ValueError("Vertex detection requires vertex and OCR heads")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if maximum_vertices < 1:
        raise ValueError("maximum_vertices must be positive")
    if ocr_roi_radius <= 0:
        raise ValueError("ocr_roi_radius must be positive")
    stroke_features = model.vision_encoder.forward_stroke_resolution(pixel_values)
    dense_output = model.vertex_head(stroke_features)
    keep_logits = None
    if proposal_keep_head is not None:
        if proposal_keep_head.config.input_channels != stroke_features.size(1):
            raise ValueError("Proposal keep head and visual features are incompatible")
        keep_logits = proposal_keep_head(stroke_features, dense_output)
    graph_output = None
    if proposal_graph_head is not None:
        if proposal_graph_head.config.input_channels != stroke_features.size(1):
            raise ValueError("Proposal graph head and visual features are incompatible")
        graph_output = proposal_graph_head(
            stroke_features,
            pixel_values,
            dense_output,
            base_keep_logits=keep_logits,
        )
        detected = decode_proposal_graph(
            graph_output,
            threshold=proposal_graph_threshold,
            maximum_vertices=maximum_vertices,
        )[0]
    elif proposal_keep_head is None:
        detected = decode_vertices(
            dense_output,
            threshold=threshold,
            maximum_vertices=maximum_vertices,
            nms_kernel_size=nms_kernel_size,
        )[0]
    else:
        detected = decode_vertices_with_proposal_keep(
            dense_output,
            keep_logits,
            proposal_keep_head.config,
            keep_threshold=proposal_keep_threshold,
        )[0][:maximum_vertices]
    unique: dict[Coordinate, dict[str, float | bool | int]] = {}
    for item in detected:
        coordinate = (
            max(0, min(1023, round(float(item["x"]) * 1023))),
            max(0, min(1023, round(float(item["y"]) * 1023))),
        )
        if coordinate not in unique or float(item["score"]) > float(
            unique[coordinate]["score"]
        ):
            unique[coordinate] = item
    ordered = sorted(unique.items(), key=lambda item: (item[0][1], item[0][0]))
    points = torch.tensor(
        [[
            [coordinate[0] / 1023.0, coordinate[1] / 1023.0]
            for coordinate, _ in ordered
        ]],
        device=pixel_values.device,
        dtype=torch.float32,
    )
    mask = torch.ones((1, len(ordered)), device=pixel_values.device, dtype=torch.bool)
    ocr_features = model.ocr_source(pixel_values, stroke_features)
    if (
        model.ocr_head.role_classifier is not None
        and model.config.ocr_role_routing
    ):
        ocr_ids, role_logits = model.ocr_head.generate_with_roles(
            ocr_features,
            points,
            mask,
            pixel_values.shape[-2:],
            roi_radius=ocr_roi_radius,
        )
        roles = [OCR_ROLE_LABELS[index] for index in role_logits.argmax(dim=-1).tolist()]
    else:
        ocr_ids = model.ocr_head.generate(
            ocr_features,
            points,
            mask,
            pixel_values.shape[-2:],
            roi_radius=ocr_roi_radius,
        )
        roles = [None] * len(ordered)
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    labels = [codec.decode(row.tolist()) for row in ocr_ids.cpu()]
    vertices = []
    retained_proposals: list[tuple[Coordinate, int | None]] = []
    for (coordinate, detection), label, role in zip(ordered, labels, roles):
        if role == "non_molecule_text":
            continue
        parsed = None if role == "implicit_c" else parse_atom_display_text(label)
        if parsed is None or parsed["element"] == "C":
            parsed = {
                "element": "C",
                "formal_charge": 0,
                "isotope": 0,
                "explicit_h_count": 0,
                "radical_electrons": 0,
            }
        explicit_h_count = parsed["explicit_h_count"]
        vertices.append(
            VertexRecordV4(
                x=coordinate[0],
                y=coordinate[1],
                element=parsed["element"],
                formal_charge=parsed["formal_charge"],
                isotope=parsed["isotope"],
                aromatic=bool(detection["aromatic"]),
                explicit_h_count=explicit_h_count,
                no_implicit=bool(explicit_h_count),
                radical_electrons=parsed["radical_electrons"],
                canonical_chiral_tag="CHI_UNSPECIFIED",
            )
        )
        proposal_id = detection.get("proposal_id")
        retained_proposals.append(
            (coordinate, int(proposal_id) if proposal_id is not None else None)
        )
    graph_edges: dict[tuple[Coordinate, Coordinate], float] = {}
    if graph_output is not None:
        proposal_to_coordinate = {
            proposal_id: coordinate
            for coordinate, proposal_id in retained_proposals
            if proposal_id is not None
        }
        graph_pairs = graph_output["pair_indices"].cpu().tolist()
        graph_scores = graph_output["edge_logits"][0].sigmoid().cpu().tolist()
        for (begin, end), graph_score in zip(graph_pairs, graph_scores):
            if begin not in proposal_to_coordinate or end not in proposal_to_coordinate:
                continue
            coordinate_pair = tuple(
                sorted((proposal_to_coordinate[begin], proposal_to_coordinate[end]))
            )
            graph_edges[coordinate_pair] = float(graph_score)
    return VertexDetectionResult(vertices, stroke_features, graph_edges)
