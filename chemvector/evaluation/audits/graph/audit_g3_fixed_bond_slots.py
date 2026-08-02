#!/usr/bin/env python3
"""Audit P0-E fixed Bond slots independently of whole-image exact metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.audit_g3_bond_slot_identity import (
    EXPECTED_STEPS,
    _config_from_payload,
    endpoint_error_summary,
    summarize_routed_predictions,
)
from chemvector.g3_hierarchical_graph import (
    G3FixedLearnedBondSlotPredictor,
    build_g3_bond_targets,
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


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0e-fixed-learned-bond-slots"
)


def validate_p0e_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0-E checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if not scope.get("gt_atom_slot_oracle") or not scope.get(
        "fixed_learned_bond_slots"
    ):
        raise ValueError("Audit accepts only P0-E fixed Bond-slot checkpoints")
    if scope.get("gt_bond_slot_identity_oracle") or scope.get(
        "shared_confidence_aware_bond_assignment"
    ):
        raise ValueError("P0-E checkpoint contains an incompatible Bond protocol")
    manifest = payload.get("sample_manifest", {})
    if manifest.get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-E checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-E checkpoint uses an unexpected vision source")


def summarize_fixed_predictions(
    *,
    predicted_endpoints: torch.Tensor,
    target_endpoints: torch.Tensor,
    target_mask: torch.Tensor,
    pointer_logits: torch.Tensor,
    type_logits: torch.Tensor,
    target_atom_refs: torch.Tensor,
    target_types: torch.Tensor,
    image_size: float = 768.0,
) -> tuple[dict[str, int | float], list[float]]:
    """Return micro metrics without treating a topology failure as a type failure."""

    batch, slots = target_mask.shape
    slot_ids = torch.arange(slots, device=target_mask.device)[None, :, None]
    candidate_ids = slot_ids.expand(batch, slots, 3)
    metrics, identities, errors = summarize_routed_predictions(
        router_endpoints=predicted_endpoints,
        target_endpoints=target_endpoints,
        target_mask=target_mask,
        candidate_ids=candidate_ids,
        pointer_logits=pointer_logits,
        type_logits=type_logits,
        target_atom_refs=target_atom_refs,
        target_types=target_types,
        image_size=image_size,
    )
    expected_identities = [
        [(slot, slot, slot) for slot in range(int(mask.sum()))]
        for mask in target_mask
    ]
    if identities != expected_identities:
        raise RuntimeError("P0-E fixed slot identity changed during audit")
    return metrics, errors


@torch.no_grad()
def audit_checkpoint(
    model: G3FixedLearnedBondSlotPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
) -> dict:
    model.eval()
    aggregate: dict[str, int | float] = {}
    all_errors: list[float] = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        normalized = build_g3_bond_targets(current_targets)
        with _autocast_context(features.device, precision):
            output = model(
                current_features,
                atom_counts=current_targets["atom_mask"].sum(dim=1).long(),
                bond_counts=current_targets["bond_mask"].sum(dim=1).long(),
                oracle_atom_anchors=current_targets["atom_anchors"],
                oracle_atom_mask=current_targets["atom_mask"],
            )
        metrics, errors = summarize_fixed_predictions(
            predicted_endpoints=output["bond_endpoints"],
            target_endpoints=normalized["endpoints"],
            target_mask=normalized["mask"],
            pointer_logits=output["endpoint_pointer_logits"],
            type_logits=output["bond_type_logits"],
            target_atom_refs=normalized["atom_refs"],
            target_types=normalized["chemical"],
        )
        for name, value in metrics.items():
            if not name.endswith("accuracy"):
                aggregate[name] = aggregate.get(name, 0) + value
        all_errors.extend(errors)
    aggregate.update(
        {
            "pointer_endpoint_accuracy": aggregate["pointer_endpoint_correct"]
            / max(aggregate["pointer_endpoints"], 1),
            "pointer_both_endpoint_accuracy": aggregate[
                "pointer_both_endpoint_correct"
            ]
            / max(aggregate["pointer_bonds"], 1),
            "type_accuracy": aggregate["type_correct"]
            / max(aggregate["type_bonds"], 1),
            "type_given_pointer_accuracy": aggregate["type_given_pointer_correct"]
            / max(aggregate["type_given_pointer_bonds"], 1),
        }
    )
    return {
        "micro": aggregate,
        "direct_endpoint_error": endpoint_error_summary(all_errors),
    }


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
        raise ValueError("P0-E audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-E audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-E CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0-E audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0e_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(
        first_payload["source_vision_checkpoint"], device
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-E audit vision bytes differ from the frozen protocol")
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
    del vision, batch

    reports = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0e_checkpoint(payload, step=step)
        if _config_from_payload(payload) != config:
            raise ValueError("P0-E checkpoint model configs differ")
        model = G3FixedLearnedBondSlotPredictor(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        current = audit_checkpoint(
            model,
            features,
            targets,
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
        "schema_version": "g3-p0e-fixed-bond-slot-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "manifest_sha256": hashlib.sha256(
            Path(OVERFIT32_MANIFEST).read_bytes()
        ).hexdigest(),
        "expected_steps": list(EXPECTED_STEPS),
        "endpoint_error_definition": (
            "maximum Euclidean endpoint error after unordered endpoint alignment"
        ),
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
