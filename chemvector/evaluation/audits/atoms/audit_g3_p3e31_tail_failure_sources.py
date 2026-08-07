#!/usr/bin/env python3
"""Stratify and visualize the 33 top128-missed/top192-reachable Atom targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import ImageDraw
import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_gt_candidate_compatibility,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_manifest_samples,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e31-tail-failure-sources"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
OVERLAYS = OUTPUT / "overlays"
P3E29_AUDIT_SHA256 = (
    "dff767d8a61308e566c01c67a7ae96906b3bc3e2bb854337fda88be7b172fc23"
)


def atom_visual_role(*, visible_text: bool, degree: int) -> str:
    if visible_text:
        return "visible_text"
    if degree <= 0:
        return "implicit_isolated"
    if degree == 1:
        return "implicit_endpoint"
    if degree == 2:
        return "implicit_chain"
    return "implicit_junction"


def rank_bucket(rank: int) -> str:
    if not 129 <= rank <= 192:
        raise ValueError("P3-E31 tail candidate rank is outside 129..192")
    lower = 129 + ((rank - 129) // 16) * 16
    return f"{lower}-{min(lower + 15, 192)}"


def crowd_bucket(ratio: float) -> str:
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("P3-E31 crowd ratio is invalid")
    if ratio < 0.5:
        return "lt0.50"
    if ratio < 0.75:
        return "0.50-0.75"
    if ratio < 1.0:
        return "0.75-1.00"
    return "ge1.00"


def direction_from_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("P3-E31 requires at least one tail case")
    roles = Counter(value["role"] for value in cases)
    variants = Counter(value["variant"] for value in cases)
    total = len(cases)
    visible_rate = roles["visible_text"] / total
    junction_rate = roles["implicit_junction"] / total
    variant, variant_count = variants.most_common(1)[0]
    if visible_rate >= 0.5:
        direction = "text_aware_candidate_generator"
        reason = "visible_text_majority"
    elif junction_rate >= 0.5:
        direction = "junction_aware_candidate_generator"
        reason = "implicit_junction_majority"
    elif variant_count / total >= 0.5:
        direction = "variant_conditioned_candidate_generator"
        reason = f"variant_majority:{variant}"
    else:
        direction = "mixed_candidate_generator_ranking"
        reason = "no_single_role_or_variant_majority"
    return {
        "direction": direction,
        "reason": reason,
        "visible_text_rate": visible_rate,
        "implicit_junction_rate": junction_rate,
        "largest_variant": variant,
        "largest_variant_rate": variant_count / total,
    }


def _degree_by_atom(sample: dict) -> dict[int, int]:
    degree = Counter()
    for bond in sample["chemmol"]["bonds"]:
        degree[int(bond["begin_atom"])] += 1
        degree[int(bond["end_atom"])] += 1
    return dict(degree)


def _median_bond_length(sample: dict) -> float:
    positions = {
        int(value["atom_id"]): torch.tensor(
            (float(value["image_x"]), float(value["image_y"]))
        )
        for value in sample["render"]["atoms"]
    }
    lengths = [
        float(
            torch.linalg.vector_norm(
                positions[int(bond["begin_atom"])]
                - positions[int(bond["end_atom"])]
            )
        )
        for bond in sample["chemmol"]["bonds"]
    ]
    if not lengths:
        return 1.0
    return float(torch.tensor(lengths).median())


def _case(
    record: dict,
    sample: dict,
    target_index: int,
    compatibility: torch.Tensor,
) -> dict:
    atoms = sorted(sample["layers"]["atoms"], key=lambda value: int(value["atom_id"]))
    atom = atoms[target_index]
    atom_id = int(atom["atom_id"])
    degree = _degree_by_atom(sample).get(atom_id, 0)
    target = record["atom_anchors"][target_index].float()
    candidate_distance = torch.linalg.vector_norm(
        (record["candidate_anchors"].float() - target) * p3e29.IMAGE_SIZE,
        dim=-1,
    )
    compatible_tail = torch.where(compatibility[target_index, 128:])[0] + 128
    best_tail = compatible_tail[
        torch.argmax(record["raw_scores"][compatible_tail].float())
    ]
    other = record["atom_anchors"][record["full_atom_mask"]].float()
    other_distance = torch.linalg.vector_norm(
        (other - target) * p3e29.IMAGE_SIZE, dim=-1
    )
    other_distance = other_distance[other_distance > 1e-4]
    nearest_other = float(other_distance.min()) if len(other_distance) else 0.0
    median_bond = _median_bond_length(sample)
    parameters = sample["render"]["render_parameters"]
    rank = int(best_tail) + 1
    return {
        "sample_key": str(record["sample_key"]),
        "molecule_id": str(sample["chemmol"]["molecule_id"]),
        "atom_id": atom_id,
        "target_index": target_index,
        "element": str(atom["element"]),
        "text": str(atom["text"]),
        "visible_text": bool(atom["visible_text"]),
        "degree": degree,
        "role": atom_visual_role(
            visible_text=bool(atom["visible_text"]), degree=degree
        ),
        "aromatic": bool(atom["aromatic"]),
        "formal_charge": int(atom["formal_charge"]),
        "variant": str(sample["render"]["variant"]),
        "font_family_id": str(parameters["font_family_id"]),
        "font_size": int(parameters["font_size"]),
        "bond_line_width": float(parameters["bond_line_width"]),
        "stress_operation": parameters.get("stress_operation"),
        "candidate_rank": rank,
        "candidate_rank_bucket": rank_bucket(rank),
        "candidate_raw_score": float(record["raw_scores"][best_tail]),
        "tail_candidate_distance_px": float(candidate_distance[best_tail]),
        "nearest_top128_distance_px": float(candidate_distance[:128].min()),
        "nearest_other_atom_px": nearest_other,
        "median_incident_scale_px": median_bond,
        "crowd_ratio": nearest_other / max(median_bond, 1e-8),
        "crowd_bucket": crowd_bucket(nearest_other / max(median_bond, 1e-8)),
        "target_xy": (float(target[0]), float(target[1])),
        "tail_candidate_xy": tuple(
            float(value) for value in record["candidate_anchors"][best_tail]
        ),
        "nearest_top128_xy": tuple(
            float(value)
            for value in record["candidate_anchors"][candidate_distance[:128].argmin()]
        ),
    }


def _draw_overlay(sample: dict, cases: list[dict], path: Path) -> None:
    image = sample["image"].convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for case in cases:
        points = (
            (case["target_xy"], "red", 10),
            (case["tail_candidate_xy"], "lime", 7),
            (case["nearest_top128_xy"], "orange", 5),
        )
        for point, color, radius in points:
            x = point[0] * image.width
            y = point[1] * image.height
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=3,
            )
        x = case["target_xy"][0] * image.width + 12
        y = case["target_xy"][1] * image.height - 12
        draw.text(
            (x, y),
            f"{case['role']} r{case['candidate_rank']}",
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )
    image.save(path)


def _counts(cases: list[dict], name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(value[name]) for value in cases).items()))


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E31 output is not empty: {OUTPUT}")
    if file_sha256(p3e29.OUTPUT / "audit.json") != P3E29_AUDIT_SHA256:
        raise ValueError("P3-E31 P3-E29 audit SHA256 differs")
    records, input_audit, manifest = p3e29._load_micro160()
    manifest_records = [
        manifest["records"][str(record["sample_key"])] for record in records
    ]
    samples = load_g3_manifest_samples(manifest_records)
    samples_by_key = {str(sample["key"]): sample for sample in samples}
    cases = []
    by_sample = defaultdict(list)
    for record in records:
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        compatibility = g3_gt_candidate_compatibility(
            expected, record["candidate_anchors"].float()
        )
        for target_index in range(len(expected)):
            if bool(compatibility[target_index, :128].any()):
                continue
            if not bool(compatibility[target_index, 128:].any()):
                raise RuntimeError("P3-E31 target is not top192 reachable")
            value = _case(
                record,
                samples_by_key[str(record["sample_key"])],
                target_index,
                compatibility,
            )
            cases.append(value)
            by_sample[value["sample_key"]].append(value)
    if len(cases) != 33 or len(by_sample) != 27:
        raise RuntimeError("P3-E31 tail case/image count differs")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    OVERLAYS.mkdir()
    overlay_digest = hashlib.sha256()
    for key, values in sorted(by_sample.items()):
        path = OVERLAYS / f"{key}.png"
        _draw_overlay(samples_by_key[key], values, path)
        overlay_digest.update(key.encode())
        overlay_digest.update(path.read_bytes())
    case_path = OUTPUT / "cases.json"
    _atomic_json(case_path, cases)
    unique_atoms = {(value["molecule_id"], value["atom_id"]) for value in cases}
    summary = {
        "schema_version": "g3-p3e31-tail-failure-sources-audit-0.1",
        "decision": direction_from_cases(cases),
        "cases": len(cases),
        "images": len(by_sample),
        "unique_molecule_atoms": len(unique_atoms),
        "counts": {
            name: _counts(cases, name)
            for name in (
                "role",
                "degree",
                "element",
                "variant",
                "font_family_id",
                "stress_operation",
                "candidate_rank_bucket",
                "crowd_bucket",
            )
        },
        "continuous": {
            name: {
                "mean": sum(float(value[name]) for value in cases) / len(cases),
                "minimum": min(float(value[name]) for value in cases),
                "maximum": max(float(value[name]) for value in cases),
            }
            for name in (
                "candidate_rank",
                "tail_candidate_distance_px",
                "nearest_top128_distance_px",
                "nearest_other_atom_px",
                "median_incident_scale_px",
                "crowd_ratio",
            )
        },
        "artifacts": {
            "cases": str(case_path.resolve()),
            "cases_sha256": file_sha256(case_path),
            "overlays": str(OVERLAYS.resolve()),
            "overlay_count": len(by_sample),
            "overlays_sha256": overlay_digest.hexdigest(),
            "legend": {
                "red": "tail-only GT Atom",
                "green": "best compatible top129-192 candidate",
                "orange": "nearest top1-128 candidate",
            },
        },
        "input": input_audit,
        "lineage": {
            "p3e29_audit": str((p3e29.OUTPUT / "audit.json").resolve()),
            "p3e29_audit_sha256": P3E29_AUDIT_SHA256,
        },
        "data_access": {
            "training_png_read_for_overlay": True,
            "gt_atom_role_read_for_diagnostic_only": True,
            "gt_bond_read_for_degree_only": True,
            "model_input_created": False,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
    }
    _atomic_json(OUTPUT / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
