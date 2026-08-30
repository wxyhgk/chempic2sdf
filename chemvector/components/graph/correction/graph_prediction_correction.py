"""Apply graph-corrector logits to decoded BondObjects."""

from __future__ import annotations

import torch


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
