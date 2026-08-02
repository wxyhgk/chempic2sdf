#!/usr/bin/env python3
"""Audit G3 Atom count, dense-slot ranking, and localization separately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.audit_g3_bond_slot_identity import EXPECTED_STEPS, _config_from_payload
from chemvector.g3_hierarchical_graph import (
    G3HierarchicalGraphPredictor,
    _build_atom_slot_targets,
    select_count_conditioned_atom_slots,
)
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    OVERFIT32_MANIFEST_SHA256,
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


DEFAULT_ARTIFACT_DIR = Path("outputs/smoke/chemvector-g3-p0-overfit32")
ATOM_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0


def validate_p0_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0 checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if any(
        scope.get(name)
        for name in (
            "gt_atom_slot_oracle",
            "shared_confidence_aware_bond_assignment",
            "gt_bond_slot_identity_oracle",
            "fixed_learned_bond_slots",
            "set_learned_bond_slots",
            "direct_gt_atom_pair_bonds",
        )
    ):
        raise ValueError("Atom audit accepts only the original predicted-Atom P0")
    if payload.get("sample_manifest", {}).get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0 checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0 checkpoint uses an unexpected vision source")


def matched_atom_count(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    radius_pixels: float = ATOM_RADIUS_PIXELS,
    image_size: float = IMAGE_SIZE,
) -> int:
    if not len(predicted) or not len(expected):
        return 0
    distances = torch.cdist(predicted.float(), expected.float()) * image_size
    rows, columns = linear_sum_assignment(distances.detach().cpu().numpy())
    return sum(float(distances[row, column]) <= radius_pixels for row, column in zip(rows, columns))


@torch.no_grad()
def audit_checkpoint(
    model: G3HierarchicalGraphPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    variants: list[str],
    batch_size: int,
    precision: str,
) -> dict:
    model.eval()
    totals = {
        "images": 0,
        "target_atoms": 0,
        "deploy_count_exact_images": 0,
        "deploy_count_absolute_error": 0,
        "gt_count_atom_set_exact_images": 0,
        "gt_count_atom_matched": 0,
        "assigned_dense_slot_selected": 0,
        "assigned_dense_slot_within_6px": 0,
    }
    variant_totals: dict[str, dict[str, int]] = {}
    assigned_errors = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            dense = model.atom_head(current_features)
        target_counts = current_targets["atom_mask"].sum(dim=1).long()
        selected = select_count_conditioned_atom_slots(
            dense,
            target_counts,
            maximum_atoms=model.config.maximum_atoms,
        )
        deploy_counts = dense["count_logits"].argmax(dim=-1)
        dense_targets = _build_atom_slot_targets(dense, current_targets)
        positive_rows = dense_targets["positive"].nonzero(as_tuple=False)
        positive_by_sample = [[] for _ in range(len(current_features))]
        for row in positive_rows.tolist():
            batch_index, slot, y_index, x_index = row
            positive_by_sample[batch_index].append((slot, y_index, x_index))
            predicted_anchor = dense["anchors"][
                batch_index, slot, :, y_index, x_index
            ]
            expected_anchor = dense_targets["anchors"][
                batch_index, slot, :, y_index, x_index
            ]
            error = float(
                torch.linalg.vector_norm(
                    predicted_anchor.float() - expected_anchor.float()
                )
                * IMAGE_SIZE
            )
            assigned_errors.append(error)
            totals["assigned_dense_slot_within_6px"] += int(
                error <= ATOM_RADIUS_PIXELS
            )

        for batch_index in range(len(current_features)):
            count = int(target_counts[batch_index])
            expected = current_targets["atom_anchors"][
                batch_index, current_targets["atom_mask"][batch_index]
            ]
            predicted = selected["atom_anchors"][batch_index, :count]
            matched = matched_atom_count(predicted, expected)
            selected_identities = {
                tuple(int(value) for value in row)
                for row in selected["atom_slot_indices"][batch_index, :count].tolist()
            }
            positive_identities = set(positive_by_sample[batch_index])
            selected_positive = len(selected_identities & positive_identities)
            variant = variants[start + batch_index]
            bucket = variant_totals.setdefault(
                variant,
                {"images": 0, "set_exact_images": 0, "count_exact_images": 0},
            )
            count_exact = int(deploy_counts[batch_index]) == count
            set_exact = matched == count
            bucket["images"] += 1
            bucket["set_exact_images"] += int(set_exact)
            bucket["count_exact_images"] += int(count_exact)
            totals["images"] += 1
            totals["target_atoms"] += count
            totals["deploy_count_exact_images"] += int(count_exact)
            totals["deploy_count_absolute_error"] += abs(
                int(deploy_counts[batch_index]) - count
            )
            totals["gt_count_atom_set_exact_images"] += int(set_exact)
            totals["gt_count_atom_matched"] += matched
            totals["assigned_dense_slot_selected"] += selected_positive
    errors = torch.tensor(assigned_errors, dtype=torch.float64)
    totals.update(
        {
            "deploy_count_mae": totals["deploy_count_absolute_error"]
            / max(totals["images"], 1),
            "gt_count_atom_micro_recall": totals["gt_count_atom_matched"]
            / max(totals["target_atoms"], 1),
            "assigned_dense_slot_topk_coverage": totals[
                "assigned_dense_slot_selected"
            ]
            / max(totals["target_atoms"], 1),
            "assigned_dense_slot_6px_coverage": totals[
                "assigned_dense_slot_within_6px"
            ]
            / max(totals["target_atoms"], 1),
            "assigned_dense_slot_error_median_pixels": float(errors.quantile(0.5)),
            "assigned_dense_slot_error_p90_pixels": float(errors.quantile(0.9)),
        }
    )
    return {"totals": totals, "variants": variant_totals}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.batch_size, args.vision_batch_size) < 1:
        raise ValueError("P0 Atom audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0 Atom audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0 Atom CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0 Atom audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(
        first_payload["source_vision_checkpoint"], device
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0 Atom audit vision bytes differ from the frozen protocol")
    config = _config_from_payload(first_payload)
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
    variants = list(batch["variants"])
    del vision, batch

    reports = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0_checkpoint(payload, step=step)
        model = G3HierarchicalGraphPredictor(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        current = audit_checkpoint(
            model,
            features,
            targets,
            variants=variants,
            batch_size=args.batch_size,
            precision=args.precision,
        )
        current.update(
            {
                "step": step,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": file_sha256(path),
            }
        )
        reports.append(current)
        del model, payload

    report = {
        "schema_version": "g3-p0-atom-slot-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "manifest_sha256": hashlib.sha256(
            Path(OVERFIT32_MANIFEST).read_bytes()
        ).hexdigest(),
        "expected_steps": list(EXPECTED_STEPS),
        "atom_radius_pixels": ATOM_RADIUS_PIXELS,
        "checkpoints": reports,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
