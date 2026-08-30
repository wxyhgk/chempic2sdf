"""Deterministic ChemScene assembly and SDF compilation."""

from __future__ import annotations

import copy

import torch

from Data.sdf.chemscene_v0_1 import parse_chemscene_v0_1, sdf_from_chemscene_v0_1
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.chemscene_graph_corrector import ELEMENT_TO_ID
from chemvector.dual_vertex_selector import TEXT_ROLE
from chemvector.ocr import parse_atom_display_text
from chemvector.structured_atom_decoder import (
    repair_impossible_explicit_h as structured_repair_impossible_explicit_h,
)


def _quantize(point: torch.Tensor) -> tuple[int, int]:
    return tuple(
        max(0, min(1023, round(float(value) * 1023.0))) for value in point
    )


def _atom_from_label(
    label: str,
    role: int,
    *,
    unresolved_as_dummy: bool = False,
) -> dict:
    parsed = parse_atom_display_text(label) if role == TEXT_ROLE else None
    if parsed is None:
        parsed = {
            "element": "*" if role == TEXT_ROLE and unresolved_as_dummy else "C",
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


def element_id_from_label(label: str, role: int) -> int:
    if role != TEXT_ROLE:
        return ELEMENT_TO_ID["C"]
    parsed = parse_atom_display_text(label)
    element = str(parsed["element"]) if parsed is not None else "OTHER"
    return ELEMENT_TO_ID.get(element, ELEMENT_TO_ID["OTHER"])


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
