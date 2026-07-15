"""Parallel vertex, OCR, and pair inference without the autoregressive decoder."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    StrokeRecordV4,
    VertexRecordV4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.detection import detect_vertex_records
from chemvector.model import ChemVectorModel
from chemvector.relation import factorized_relation_probabilities
from chemvector.stroke import STROKE_CLASSES, all_pair_indices


@dataclass(frozen=True, slots=True)
class ParallelChemSVGResult:
    sequence: str
    vertices: tuple[VertexRecordV4, ...]
    strokes: tuple[StrokeRecordV4, ...]
    pair_predictions: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[float, str],
    ]
    vertex_ocr_seconds: float
    pair_seconds: float


def strip_autoregressive_modules(model: ChemVectorModel) -> ChemVectorModel:
    """Remove modules that are not used by parallel deployment inference."""

    model.token_embedding = None
    model.position_embedding = None
    model.decoder = None
    model.output_projection = None
    model.visual_aux_head = None
    return model


def deployment_parameter_count(model: ChemVectorModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def score_vertex_pairs_parallel(
    model: ChemVectorModel,
    stroke_features: torch.Tensor,
    vertices: list[VertexRecordV4] | tuple[VertexRecordV4, ...],
    pixel_size: tuple[int, int],
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    if model.stroke_head is None:
        raise ValueError("Parallel inference requires a stroke head")
    pairs = all_pair_indices(len(vertices))
    if not pairs:
        return [], torch.empty(
            (0, len(STROKE_CLASSES)),
            device=stroke_features.device,
        )
    anchors = torch.tensor(
        [[[vertex.x / 1023.0, vertex.y / 1023.0] for vertex in vertices]],
        device=stroke_features.device,
        dtype=torch.float32,
    )
    pair_indices = torch.tensor(
        [pairs],
        device=stroke_features.device,
        dtype=torch.long,
    )
    pair_mask = torch.ones(
        (1, len(pairs)),
        device=stroke_features.device,
        dtype=torch.bool,
    )
    pair_features = model.stroke_head.encode_pairs(
        stroke_features,
        anchors,
        pair_indices,
        pair_mask,
        pixel_size,
    )
    if model.relation_head is None:
        logits = model.stroke_head.classifier[3](pair_features)
        probabilities = logits.float().softmax(dim=-1)
    else:
        output = model.relation_head(pair_features)
        probabilities = factorized_relation_probabilities(
            output["edge_logits"],
            output["bond_logits"],
        )
    return pairs, probabilities


def build_parallel_sequence(
    vertices: list[VertexRecordV4] | tuple[VertexRecordV4, ...],
    pairs: list[tuple[int, int]],
    probabilities: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
) -> tuple[str, tuple[StrokeRecordV4, ...]]:
    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be in [0, 1]")
    if probabilities.shape != (len(pairs), len(STROKE_CLASSES)):
        raise ValueError("Pair probabilities do not match the stroke vocabulary")
    strokes = []
    for (begin_index, end_index), row in zip(pairs, probabilities.cpu()):
        edge_score = 1.0 - float(row[0])
        if edge_score < edge_threshold:
            continue
        class_id = int(row[1:].argmax()) + 1
        bond_type = STROKE_CLASSES[class_id]
        begin = vertices[begin_index]
        end = vertices[end_index]
        strokes.append(
            StrokeRecordV4(
                begin_x=begin.x,
                begin_y=begin.y,
                end_x=end.x,
                end_y=end.y,
                display_type=bond_type,
                bond_type=bond_type,
                direction="NONE",
                stereo="STEREONONE",
            )
        )
    parsed = ParsedChemSVGSequenceV4(tuple(vertices), tuple(strokes))
    return serialize_parsed_chemsvg_v4(parsed), tuple(strokes)


@torch.inference_mode()
def generate_parallel_chemsvg_v6(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    *,
    vertex_threshold: float = 0.3,
    maximum_vertices: int = 128,
    edge_threshold: float = 0.5,
    ocr_roi_radius: float = 20.0,
) -> ParallelChemSVGResult:
    if pixel_values.size(0) != 1:
        raise ValueError("Parallel inference currently supports batch size 1")
    if model.vertex_head is None or model.stroke_head is None or model.ocr_head is None:
        raise ValueError("Parallel inference requires vertex, stroke, and OCR heads")
    device = pixel_values.device
    _synchronize(device)
    started = time.perf_counter()
    vertices, stroke_features = detect_vertex_records(
        model,
        pixel_values,
        threshold=vertex_threshold,
        maximum_vertices=maximum_vertices,
        ocr_roi_radius=ocr_roi_radius,
    )
    _synchronize(device)
    vertex_ocr_seconds = time.perf_counter() - started

    started = time.perf_counter()
    pairs, probabilities = score_vertex_pairs_parallel(
        model,
        stroke_features,
        vertices,
        pixel_values.shape[-2:],
    )
    sequence, strokes = build_parallel_sequence(
        vertices,
        pairs,
        probabilities,
        edge_threshold=edge_threshold,
    )
    pair_predictions = {}
    for (begin_index, end_index), row in zip(pairs, probabilities.cpu()):
        begin = vertices[begin_index]
        end = vertices[end_index]
        coordinate_pair = tuple(
            sorted(((begin.x, begin.y), (end.x, end.y)))
        )
        class_id = int(row[1:].argmax()) + 1
        pair_predictions[coordinate_pair] = (
            1.0 - float(row[0]),
            STROKE_CLASSES[class_id],
        )
    _synchronize(device)
    pair_seconds = time.perf_counter() - started
    return ParallelChemSVGResult(
        sequence=sequence,
        vertices=tuple(vertices),
        strokes=strokes,
        pair_predictions=pair_predictions,
        vertex_ocr_seconds=vertex_ocr_seconds,
        pair_seconds=pair_seconds,
    )
