#!/usr/bin/env python3
"""Ablate predicted Atom anchors and scores in frozen P0-J composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.audit_g3_p0j_frozen_composition import (
    DEFAULT_OUTPUT_DIR,
    P0G_CHECKPOINT,
    P0I_CHECKPOINT,
    compose_pair_output,
    composition_batch_statistics,
    empty_mode_totals,
    finalize_mode_totals,
    gather_slot_features,
    load_frozen_p0_models,
    maximum_radius_atom_mapping,
    merge_totals,
)
from chemvector.g3_hierarchical_graph import build_g3_bond_targets
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _autocast_context,
    cache_frozen_vision_features,
    load_frozen_vision_checkpoint,
)


OUTPUT_NAME = "composition_residual_ablation.json"
MODES = (
    "pred_anchor_p0i_score",
    "pred_anchor_p0g_score",
    "gt_anchor_p0i_score",
    "gt_anchor_p0g_score",
)


def gather_slot_scores(
    dense: dict[str, torch.Tensor],
    slot_indices: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    logits = dense["presence_logits"]
    scores = logits.new_zeros(atom_mask.shape)
    for batch_index, query_index in atom_mask.nonzero(as_tuple=False).tolist():
        slot, y_index, x_index = (
            int(value) for value in slot_indices[batch_index, query_index].tolist()
        )
        scores[batch_index, query_index] = logits[
            batch_index, slot, y_index, x_index
        ].sigmoid()
    return scores


def replace_with_mapped_gt_anchors(
    atoms: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Oracle ablation only: replace matched anchors after P0-I forward."""

    anchors = atoms["atom_anchors"].clone()
    for batch_index in range(len(anchors)):
        predicted = torch.where(atoms["atom_mask"][batch_index])[0]
        expected = torch.where(targets["atom_mask"][batch_index].bool())[0]
        mapping = maximum_radius_atom_mapping(
            atoms["atom_anchors"][batch_index],
            predicted,
            targets["atom_anchors"][batch_index],
            expected,
        )
        if len(mapping) != len(predicted) or len(mapping) != len(expected):
            raise ValueError("P0-J GT-anchor ablation requires an exact Atom mapping")
        for query, target in mapping.items():
            anchors[batch_index, query] = targets["atom_anchors"][
                batch_index, target
            ]
    return anchors


@torch.no_grad()
def evaluate_residual_modes(
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
) -> dict[str, dict]:
    totals = {mode: empty_mode_totals() for mode in MODES}
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            base_atoms = p0i(current_features)
            p0g_dense = p0g.atom_head(current_features)
            atom_features = gather_slot_features(
                p0g_dense,
                base_atoms["atom_slot_indices"],
                base_atoms["atom_mask"],
            )
            p0g_scores = gather_slot_scores(
                p0g_dense,
                base_atoms["atom_slot_indices"],
                base_atoms["atom_mask"],
            )
            gt_anchors = replace_with_mapped_gt_anchors(
                base_atoms, current_targets
            )
            for mode in MODES:
                atoms = dict(base_atoms)
                if mode.startswith("gt_anchor"):
                    atoms["atom_anchors"] = gt_anchors
                if mode.endswith("p0g_score"):
                    atoms["atom_scores"] = p0g_scores
                pair_output = compose_pair_output(
                    p0g, current_features, atoms, atom_features
                )
                current = composition_batch_statistics(
                    pair_output,
                    atoms,
                    current_targets,
                    build_g3_bond_targets(current_targets),
                )
                merge_totals(totals[mode], current)
    return {mode: finalize_mode_totals(totals[mode]) for mode in MODES}


def residual_interpretation(modes: dict[str, dict]) -> dict:
    exact = {
        mode: values["gt_bond_count"]["typed_graph_exact_images"]
        for mode, values in modes.items()
    }
    score_fixes = exact["pred_anchor_p0g_score"] >= 30
    anchor_fixes = exact["gt_anchor_p0i_score"] >= 30
    both_fixes = exact["gt_anchor_p0g_score"] >= 30
    if anchor_fixes and not score_fixes:
        cause = "predicted_anchor_geometry_shift"
    elif score_fixes and not anchor_fixes:
        cause = "atom_score_distribution_shift"
    elif anchor_fixes and score_fixes:
        cause = "both_single_factor_ablations_pass"
    elif both_fixes:
        cause = "joint_anchor_and_score_shift"
    else:
        cause = "relation_alignment_beyond_anchor_and_score"
    return {
        "typed_graph_exact_images": exact,
        "score_oracle_passed": score_fixes,
        "anchor_oracle_passed": anchor_fixes,
        "joint_oracle_passed": both_fixes,
        "primary_cause": cause,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0i-checkpoint", default=str(P0I_CHECKPOINT))
    parser.add_argument("--p0g-checkpoint", default=str(P0G_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.batch_size, args.vision_batch_size) < 1:
        raise ValueError("P0-J residual audit batch sizes must be positive")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-J residual CPU audit requires fp32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-J residual CUDA audit requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("P0-J residual audit requires the committed output directory")
    if not (output_dir / "composition_audit.json").is_file():
        raise FileNotFoundError("P0-J residual audit requires the primary composition audit")
    output = output_dir / OUTPUT_NAME
    if output.exists():
        raise RuntimeError(f"P0-J residual output already exists: {output}")

    device = torch.device(args.device)
    p0i, p0g, p0i_payload, _, config = load_frozen_p0_models(
        Path(args.p0i_checkpoint), Path(args.p0g_checkpoint), device
    )
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-J residual vision SHA differs from the protocol")
    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    batch = collator(records)
    features = cache_frozen_vision_features(
        vision,
        batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(batch, device)
    modes = evaluate_residual_modes(
        p0i,
        p0g,
        features,
        targets,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    report = {
        "schema_version": "g3-p0j-composition-residual-ablation-0.1",
        "scope": "train-only-overfit32-no-grad-oracle-ablation",
        "warning": (
            "gt_anchor modes inject scoring-only oracle anchors into pair forward; "
            "they are causal diagnostics and not deployable results"
        ),
        "manifest": manifest_audit,
        "vision_sha256": vision_sha,
        "primary_composition_audit_sha256": file_sha256(
            output_dir / "composition_audit.json"
        ),
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "sha256": file_sha256(__file__),
        },
        "modes": modes,
        "interpretation": residual_interpretation(modes),
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
