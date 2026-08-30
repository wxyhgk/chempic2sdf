"""Shared placeholder canvas, image preparation, and carbon fallback chemistry."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2


def placeholder_scene(molecule_id: str, width: int, height: int) -> dict:
    """Provide only a canvas; compiled atoms intentionally fall back to carbon."""

    return {
        "schema_version": "chemscene-0.2",
        "molecule_id": molecule_id,
        "units": "q1023",
        "canvas": {"width": width, "height": height},
        "atom_objects": [],
        "bond_objects": [],
    }


def prepare_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        rgba = opened.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    image = Image.alpha_composite(white, rgba).convert("RGB")
    return image.resize((768, 768), Image.Resampling.BICUBIC)


def compile_carbon_placeholder(scene: dict) -> tuple[dict, object, int]:
    """Demote aromatic chemistry when placeholder carbon cannot be kekulized."""

    repaired = deepcopy(scene)
    for atom in repaired.get("atom_objects", []):
        atom["chemical"]["aromatic"] = False
        atom["incident_ports"] = []
    demoted = 0
    for bond in repaired.get("bond_objects", []):
        if bond.get("chemical_type") != "AROMATIC":
            pass
        else:
            display = bond.get("display_type")
            bond["chemical_type"] = (
                display if display in {"SINGLE", "DOUBLE", "TRIPLE"} else "SINGLE"
            )
            demoted += 1
        for endpoint, atom_id in enumerate(bond["atom_refs"]):
            atom = repaired["atom_objects"][atom_id]
            coordinate = list(atom["anchor"])
            bond["ports"][endpoint] = coordinate
            atom["incident_ports"].append(
                {
                    "bond_id": bond["id"],
                    "endpoint": endpoint,
                    "coordinate": coordinate,
                    "source": "ANCHOR",
                }
            )
    return repaired, mol_from_chemscene_v0_2(repaired), demoted