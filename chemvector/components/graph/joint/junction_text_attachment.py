"""Attach OCR text to an already selected structural Junction set."""

from __future__ import annotations

import torch

from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE


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


def components_from_external_junctions(
    points: torch.Tensor,
    bond_objects: list[dict],
    *,
    include_ocr: bool,
    detected_points: torch.Tensor | None = None,
    detected_roles: torch.Tensor | None = None,
    detected_labels: list[str] | None = None,
    image_size: int = 768,
    text_attach_radius_pixels: float = 24.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    """Attach frozen OCR results to an externally selected Junction set."""

    if points.ndim != 2 or points.size(1) != 2 or not len(points):
        raise ValueError("External Junctions must have non-empty shape [N,2]")
    if not bool(torch.isfinite(points).all()) or not bool(
        ((points >= 0.0) & (points <= 1.0)).all()
    ):
        raise ValueError("External Junction is outside the normalized image")
    if image_size < 1 or text_attach_radius_pixels <= 0:
        raise ValueError("External Junction attachment dimensions are invalid")
    if include_ocr:
        if (
            detected_points is None
            or detected_roles is None
            or detected_labels is None
        ):
            raise ValueError("External Junction OCR requires detected text objects")
        roles, labels = attach_text_to_bond_junctions(
            points,
            detected_points,
            detected_roles,
            detected_labels,
            image_size=image_size,
            maximum_distance_pixels=text_attach_radius_pixels,
        )
    else:
        roles = torch.full(
            (len(points),),
            IMPLICIT_ROLE,
            device=points.device,
            dtype=torch.long,
        )
        labels = ["C"] * len(points)
    return points, roles, labels, list(bond_objects)
