"""Reject BondObjects whose endpoints are both inside one text box."""

from __future__ import annotations


def suppress_text_interior_bond_objects(
    bond_objects: list[dict],
    text_objects: list[dict],
    *,
    image_size: int,
    interior_margin_pixels: float = 2.0,
) -> tuple[list[dict], list[dict]]:
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
