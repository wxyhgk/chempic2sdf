#!/usr/bin/env python3
"""Audit whether rendered GT Atom anchors remain visually separable."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from chemvector.evaluation.audits.atoms import build_g3_p2f_train25k_manifest as p2f
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


OUTPUT = Path("outputs/audits/chemvector-g3-p2g-render-recoverability")
ALIAS_DISTANCE_PIXELS = 6.0


def point_segment_distance(point, start, end) -> tuple[float, float]:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 0:
        return float(np.linalg.norm(point - start)), 0.0
    projection = float(np.dot(point - start, vector) / denominator)
    closest = start + np.clip(projection, 0.0, 1.0) * vector
    return float(np.linalg.norm(point - closest)), projection


def sample_ambiguities(render: dict, layers: dict) -> dict:
    atoms = render["atoms"]
    bonds = render["bonds"]
    visible_text = {
        int(atom["atom_id"]): bool(atom["visible_text"])
        for atom in layers["atoms"]
    }
    render_atom_ids = {int(atom["atom_id"]) for atom in atoms}
    if set(visible_text) != render_atom_ids:
        raise ValueError("Render and semantic-layer Atom IDs differ")
    anchors = np.asarray(
        [[atom["image_x"], atom["image_y"]] for atom in atoms], dtype=np.float64
    )
    alias_pairs = []
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            distance = float(np.linalg.norm(anchors[left] - anchors[right]))
            if distance < ALIAS_DISTANCE_PIXELS:
                alias_pairs.append({
                    "left_atom": int(atoms[left]["atom_id"]),
                    "right_atom": int(atoms[right]["atom_id"]),
                    "distance_pixels": distance,
                    "left_visible_text": visible_text[int(atoms[left]["atom_id"])],
                    "right_visible_text": visible_text[int(atoms[right]["atom_id"])],
                })
    hidden_alias_pairs = [
        pair for pair in alias_pairs
        if not pair["left_visible_text"] and not pair["right_visible_text"]
    ]
    line_width = float(render["render_parameters"]["bond_line_width"])
    occlusion_radius = max(3.0, 1.5 * line_width)
    occlusions = []
    for atom, point in zip(atoms, anchors):
        for bond in bonds:
            start = np.asarray(
                [bond["begin_image_x"], bond["begin_image_y"]], dtype=np.float64
            )
            end = np.asarray(
                [bond["end_image_x"], bond["end_image_y"]], dtype=np.float64
            )
            if np.linalg.norm(point - start) < 1e-4 or np.linalg.norm(point - end) < 1e-4:
                continue
            distance, projection = point_segment_distance(point, start, end)
            if 0.05 < projection < 0.95 and distance < occlusion_radius:
                occlusions.append({
                    "atom_id": int(atom["atom_id"]),
                    "bond_id": int(bond["bond_id"]),
                    "distance_pixels": distance,
                    "projection": projection,
                    "visible_text": visible_text[int(atom["atom_id"])],
                })
    hidden_occlusions = [item for item in occlusions if not item["visible_text"]]
    hidden_occlusion_counts = Counter(item["atom_id"] for item in hidden_occlusions)
    multiply_occluded_hidden_atoms = [
        {"atom_id": atom_id, "nonincident_bond_count": count}
        for atom_id, count in sorted(hidden_occlusion_counts.items())
        if count >= 2
    ]
    has_hidden_alias = bool(hidden_alias_pairs)
    has_multiply_occluded_hidden_atom = bool(multiply_occluded_hidden_atoms)
    if has_hidden_alias and has_multiply_occluded_hidden_atom:
        status = "severe"
    elif has_hidden_alias or has_multiply_occluded_hidden_atom:
        status = "ambiguous"
    else:
        status = "eligible"
    return {
        "alias_pairs": alias_pairs,
        "hidden_alias_pairs": hidden_alias_pairs,
        "nonincident_bond_occlusions": occlusions,
        "hidden_nonincident_bond_occlusions": hidden_occlusions,
        "multiply_occluded_hidden_atoms": multiply_occluded_hidden_atoms,
        "raw_geometry_flag": bool(alias_pairs or occlusions),
        "ambiguous": status != "eligible",
        "recoverability_status": status,
        "bond_line_width": line_width,
        "occlusion_radius": occlusion_radius,
    }


def audit_manifest(manifest) -> dict:
    flagged = []
    raw_geometry_images = 0
    status_counts = Counter()
    for sample in p1a.iter_manifest_samples(manifest):
        result = sample_ambiguities(sample["render"], sample["layers"])
        raw_geometry_images += int(result["raw_geometry_flag"])
        status_counts[result["recoverability_status"]] += 1
        if result["ambiguous"]:
            flagged.append({
                "sample_key": str(sample["key"]),
                "molecule_id": str(sample["chemmol"]["molecule_id"]),
                "variant": str(sample["render"]["variant"]),
                "stress_operation": sample["render"]["render_parameters"].get(
                    "stress_operation"
                ),
                **result,
            })
    return {
        "images": len(manifest),
        "raw_geometry_images": raw_geometry_images,
        "ambiguous_images": len(flagged),
        "severe_images": status_counts["severe"],
        "eligible_images": len(manifest) - len(flagged),
        "status_counts": {
            status: status_counts[status]
            for status in ("eligible", "ambiguous", "severe")
        },
        "flagged": flagged,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-G output is not empty: {args.output_dir}")
    route, route_audit = p1a.load_route_manifest()
    train, train_audit = load_g3_sample_manifest(
        p2f.MANIFEST,
        expected_sha256="b0f10eb0922599636895e19294e55b95ba9c051492b7bd69ae42a639c2846161",
        expected_samples=25_000,
        expected_molecules=6_250,
        expected_variants=p1a.EXPECTED_VARIANTS,
    )
    results = {
        "route100": audit_manifest(route),
        "train25k": audit_manifest(train),
    }
    audit = {
        "schema_version": "g3-p2g-render-recoverability-0.2",
        "criteria": {
            "atom_alias_distance_pixels": ALIAS_DISTANCE_PIXELS,
            "nonincident_bond_occlusion_radius": "max(3px,1.5*bond_line_width)",
            "segment_projection_open_interval": [0.05, 0.95],
            "ambiguous": (
                "two hidden Atom anchors alias, or one hidden Atom anchor is crossed "
                "by at least two nonincident Bond centerlines"
            ),
            "severe": "both ambiguous conditions occur in the same image",
            "visible_text_source": "layers.atoms[].visible_text",
        },
        "manifests": {"route100": route_audit, "train25k": train_audit},
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({
        domain: {
            "images": value["images"],
            "raw_geometry_images": value["raw_geometry_images"],
            "ambiguous_images": value["ambiguous_images"],
            "severe_images": value["severe_images"],
            "eligible_images": value["eligible_images"],
        }
        for domain, value in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
