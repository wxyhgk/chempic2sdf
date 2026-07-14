"""Shared visual vertex detection and anchor-local OCR inference."""

from __future__ import annotations

import torch

from Data.sdf.chemsvg_sequence_v4 import VertexRecordV4
from chemvector.model import ChemVectorModel
from chemvector.ocr import OCRTextCodec, parse_atom_display_text
from chemvector.vertex import decode_vertices


Coordinate = tuple[int, int]


@torch.inference_mode()
def detect_vertex_records(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    *,
    threshold: float = 0.3,
    maximum_vertices: int = 128,
    ocr_roi_radius: float = 20.0,
) -> tuple[list[VertexRecordV4], torch.Tensor]:
    """Detect visual vertices, recognize labels, and return shared stroke features."""

    if model.vertex_head is None or model.ocr_head is None:
        raise ValueError("Vertex detection requires vertex and OCR heads")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if maximum_vertices < 1:
        raise ValueError("maximum_vertices must be positive")
    if ocr_roi_radius <= 0:
        raise ValueError("ocr_roi_radius must be positive")
    stroke_features = model.vision_encoder.forward_stroke_resolution(pixel_values)
    detected = decode_vertices(
        model.vertex_head(stroke_features),
        threshold=threshold,
        maximum_vertices=maximum_vertices,
    )[0]
    unique: dict[Coordinate, dict[str, float | bool]] = {}
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
    ocr_features = model.vision_encoder.forward_high_resolution_from_stroke(
        stroke_features
    )
    ocr_ids = model.ocr_head.generate(
        ocr_features,
        points,
        mask,
        pixel_values.shape[-2:],
        roi_radius=ocr_roi_radius,
    )
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    labels = [codec.decode(row.tolist()) for row in ocr_ids.cpu()]
    vertices = []
    for (coordinate, detection), label in zip(ordered, labels):
        parsed = parse_atom_display_text(label)
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
    return vertices, stroke_features
