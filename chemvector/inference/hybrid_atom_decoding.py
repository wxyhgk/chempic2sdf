"""Build structured atom-state candidates for invalid-scene recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.ctc_ocr import detected_text_crops
from chemvector.dual_vertex_selector import TEXT_ROLE
from chemvector.structured_atom_decoder import (
    AtomStateCandidate,
    atom_state_candidate,
    merge_ctc_atom_candidates,
)
from chemvector.text_object import decode_text_objects


@dataclass(frozen=True, slots=True)
class HybridAtomDecodingRuntime:
    device: torch.device
    base: Any
    text_head: Any
    protocol: Any
    maximum_text_objects: int
    ctc_ocr: Any


def structured_atom_candidates(
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    *,
    runtime: HybridAtomDecodingRuntime,
) -> list[list[AtomStateCandidate]]:
    """Generate alternative atom states without changing nodes or bonds."""

    if runtime.ctc_ocr is None:
        raise ValueError("Structured atom decoding requires a CTC OCR checkpoint")
    if pixels.size(0) != 1:
        raise ValueError("Structured atom decoding currently expects one image")
    with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
        features = runtime.base.vision_encoder.forward_stroke_resolution(pixels)
        text_output = runtime.text_head(features)
    text_objects = decode_text_objects(
        text_output,
        threshold=float(runtime.protocol["text_threshold"]),
        maximum_objects=runtime.maximum_text_objects,
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
    candidates: list[list[AtomStateCandidate]] = [[] for _ in range(len(points))]
    if len(shifted_crops):
        combined = torch.cat((shifted_crops, anchor_crops), dim=0)
        with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
            combined_logits = runtime.ctc_ocr(combined).float()
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
                maximum_bytes=runtime.ctc_ocr.config.maximum_bytes,
                token_beam=24,
            )
    for index, role in enumerate(roles.tolist()):
        if role != TEXT_ROLE or candidates[index]:
            continue
        candidates[index] = [
            atom_state_candidate(labels[index], 0.0, source="current")
        ]
    return candidates
