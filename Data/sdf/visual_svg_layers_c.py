#!/usr/bin/env python3
"""Single-SVG assembly primitives for Visual SVG Layers C v0.1."""

from __future__ import annotations

import io
import json
from typing import Any, Mapping

from rdkit import Chem

try:
    from . import build_chemvision_layers as layer_builder
    from .build_visual_svg_capacity_100 import (
        binary_bond_iou,
        group_iou_metrics,
        group_structural_metrics,
        json_bytes,
        sha256_bytes,
    )
    from .build_webdataset_sample import mol_from_chemmol
    from .visual_primitives_v0_1 import (
        extract_visual_primitives_v0_1,
        parsed_visual_svg_from_primitives,
        primitive_counts,
        primitives_to_bond_svg,
    )
    from .visual_svg_sequence_v0_1 import (
        parse_visual_svg_v0_1,
        serialize_visual_svg_v0_1,
    )
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from .visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_visual_svg_v0_2,
    )
except ImportError:
    import build_chemvision_layers as layer_builder
    from build_visual_svg_capacity_100 import (
        binary_bond_iou,
        group_iou_metrics,
        group_structural_metrics,
        json_bytes,
        sha256_bytes,
    )
    from build_webdataset_sample import mol_from_chemmol
    from visual_primitives_v0_1 import (
        extract_visual_primitives_v0_1,
        parsed_visual_svg_from_primitives,
        primitive_counts,
        primitives_to_bond_svg,
    )
    from visual_svg_sequence_v0_1 import (
        parse_visual_svg_v0_1,
        serialize_visual_svg_v0_1,
    )
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_visual_svg_v0_2,
    )


META_SCHEMA_VERSION = "visual-svg-layers-c-meta-0.1"
GROUP_DIAGNOSTIC_MODES = ("full_raster", "structural")
MEMBER_SUFFIXES = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".primitives.json",
    ".visualsvg.txt",
    ".visualsvg-v02.json",
    ".layers.json",
    ".render.json",
    ".chemmol.json",
    ".sdf",
    ".meta.json",
)
HASHED_SUFFIXES = MEMBER_SUFFIXES[:-1]
LAYER_SUFFIXES = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".layers.json",
    ".render.json",
)


def parse_single_sdf(data: bytes, *, molecule_id: str) -> tuple[Chem.Mol, str]:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"SDF for {molecule_id} must contain one valid molecule")
    molecule = molecules[0]
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise ValueError(f"SDF for {molecule_id} lacks a valid InChIKey")
    return molecule, inchikey


def _validate_source_bond_groups(
    primitives: Mapping[str, Any],
    parsed_v02: Any,
) -> None:
    source_groups = primitives.get("bond_groups")
    if not isinstance(source_groups, list) or not source_groups:
        raise ValueError("Visual primitives must contain source bond groups")
    for group in source_groups:
        if set(("begin_atom", "end_atom")) - set(group):
            raise ValueError("A source bond group lacks its two chemical endpoints")
        if int(group["begin_atom"]) == int(group["end_atom"]):
            raise ValueError("A source bond group has identical chemical endpoints")
    if len(parsed_v02.bond_groups) != len(source_groups):
        raise ValueError("Visual SVG v0.2 group count differs from source bond groups")
    expected_members = {
        "single_line": 1,
        "parallel_2": 2,
        "parallel_3": 3,
        "solid_triangle": 1,
        "solid_quad": 1,
    }
    for group in parsed_v02.bond_groups:
        expected = expected_members.get(group.kind)
        if expected is not None and len(group.members) != expected:
            raise ValueError(
                f"Visual SVG v0.2 {group.kind} has {len(group.members)} members"
            )


def assemble_c_sample(
    task: layer_builder.RenderTask,
    chemmol_bytes: bytes,
    sdf_bytes: bytes,
    source_render: Mapping[str, Any],
    *,
    primitive_validator,
    tokenizer: VisualSVGTokenizerV01,
    output_split: str,
    source_dataset: str,
    structure_ref: str,
    fresh_audit_eligible: bool,
    expected_layer_members: Mapping[str, bytes] | None = None,
    max_tokens: int = 4096,
    group_diagnostics_mode: str = "full_raster",
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Create all 12 members from one semantic SVG generation."""

    if group_diagnostics_mode not in GROUP_DIAGNOSTIC_MODES:
        raise ValueError(
            "Unsupported group diagnostics mode: "
            f"{group_diagnostics_mode!r}; expected one of {GROUP_DIAGNOSTIC_MODES}"
        )

    chemmol = json.loads(chemmol_bytes)
    generated, layer_metrics, semantic_svg = (
        layer_builder.render_and_layers_with_svg(
            task,
            chemmol,
            dict(source_render),
        )
    )
    if expected_layer_members is not None:
        for suffix in LAYER_SUFFIXES:
            expected = expected_layer_members.get(suffix)
            if expected is None or generated[suffix] != expected:
                raise ValueError(
                    f"Deterministic single-SVG redraw mismatch for {task.sample_key}{suffix}"
                )

    layers = json.loads(generated[".layers.json"])
    primitives = extract_visual_primitives_v0_1(semantic_svg, layers)
    primitive_validator(primitives)
    parsed_from_primitives = parsed_visual_svg_from_primitives(primitives)
    visual_v01 = serialize_visual_svg_v0_1(primitives)
    parsed_v01 = parse_visual_svg_v0_1(visual_v01)
    if parsed_v01 != parsed_from_primitives:
        raise ValueError(
            f"Primitive/Visual SVG v0.1 mismatch for {task.sample_key}"
        )
    token_ids = tokenizer.encode(visual_v01)
    if tokenizer.decode(token_ids) != visual_v01:
        raise ValueError(f"Visual SVG tokenizer mismatch for {task.sample_key}")
    if len(token_ids) > max_tokens:
        raise ValueError(
            f"Visual SVG sequence has {len(token_ids)} tokens, limit is {max_tokens}"
        )

    visual_v02 = serialize_visual_svg_v0_2(primitives)
    parsed_v02 = parse_visual_svg_v0_2(visual_v02)
    if flatten_visual_svg_v0_2_to_v0_1(parsed_v02) != parsed_v01:
        raise ValueError(f"Visual SVG v0.2 flatten mismatch for {task.sample_key}")
    _validate_source_bond_groups(primitives, parsed_v02)

    bond_iou = binary_bond_iou(
        generated[".bond.png"],
        primitives_to_bond_svg(primitives),
    )
    if bond_iou < 0.90:
        raise ValueError(
            f"Primitive bond IoU is too low for {task.sample_key}: {bond_iou:.6f}"
        )
    structural_metrics = group_structural_metrics(primitives)
    group_metrics = None
    if group_diagnostics_mode == "full_raster":
        group_metrics = group_iou_metrics(semantic_svg, primitives)
        if group_metrics["group_alignment_f1_min_nonadjusted"] < 0.90:
            raise ValueError(
                "Per-bond primitive alignment is too low for "
                f"{task.sample_key}: "
                f"{group_metrics['group_alignment_f1_min_nonadjusted']:.6f}"
            )
        if group_metrics["polygon_alignment_f1_min"] < 0.95:
            raise ValueError(
                "Polygon primitive alignment is too low for "
                f"{task.sample_key}: {group_metrics['polygon_alignment_f1_min']:.6f}"
            )

    sdf_molecule, sdf_inchikey = parse_single_sdf(
        sdf_bytes,
        molecule_id=task.molecule_id,
    )
    chemmol_inchikey = Chem.MolToInchiKey(mol_from_chemmol(chemmol))
    if chemmol_inchikey != sdf_inchikey:
        raise ValueError(
            f"ChemMol/SDF InChIKey mismatch for {task.sample_key}: "
            f"{chemmol_inchikey} != {sdf_inchikey}"
        )

    primitives_bytes = json_bytes(primitives)
    members = {
        ".png": generated[".png"],
        ".text.png": generated[".text.png"],
        ".bond.png": generated[".bond.png"],
        ".semantic.png": generated[".semantic.png"],
        ".primitives.json": primitives_bytes,
        ".visualsvg.txt": visual_v01.encode("utf-8"),
        ".visualsvg-v02.json": visual_v02.encode("utf-8"),
        ".layers.json": generated[".layers.json"],
        ".render.json": generated[".render.json"],
        ".chemmol.json": chemmol_bytes,
        ".sdf": sdf_bytes,
    }
    hashes = {
        suffix.removeprefix("."): sha256_bytes(members[suffix])
        for suffix in HASHED_SUFFIXES
    }
    meta = {
        "schema_version": META_SCHEMA_VERSION,
        "sample_key": task.sample_key,
        "molecule_id": task.molecule_id,
        "split": output_split,
        "variant": task.variant,
        "source": {
            "dataset": source_dataset,
            "source_sample_key": task.source_sample_key,
            "source_shard": task.source_shard,
            "structure_ref": structure_ref,
        },
        "fresh_audit_eligible": bool(fresh_audit_eligible),
        "semantic_svg_sha256": sha256_bytes(semantic_svg),
        "sdf_inchikey": sdf_inchikey,
        "sequence_tokens": len(token_ids),
        "sha256": hashes,
    }
    members[".meta.json"] = json_bytes(meta)
    counts = primitive_counts(primitives)
    return members, {
        "sample_key": task.sample_key,
        "molecule_id": task.molecule_id,
        "split": output_split,
        "variant": task.variant,
        "sdf_inchikey": sdf_inchikey,
        "sdf_atom_count": sdf_molecule.GetNumAtoms(),
        "sdf_bond_count": sdf_molecule.GetNumBonds(),
        "sequence_tokens": len(token_ids),
        "primitive_count": parsed_v01.primitive_count,
        "primitive_counts": counts,
        "visual_svg_v02_group_count": len(parsed_v02.bond_groups),
        "bond_iou": bond_iou,
        "group_diagnostics_mode": group_diagnostics_mode,
        "group_metrics": group_metrics,
        "group_structural_metrics": structural_metrics,
        "layer_metrics": layer_metrics,
        "semantic_svg_sha256": meta["semantic_svg_sha256"],
        "sha256": hashes,
    }


__all__ = [
    "GROUP_DIAGNOSTIC_MODES",
    "HASHED_SUFFIXES",
    "MEMBER_SUFFIXES",
    "META_SCHEMA_VERSION",
    "assemble_c_sample",
    "parse_single_sdf",
]
