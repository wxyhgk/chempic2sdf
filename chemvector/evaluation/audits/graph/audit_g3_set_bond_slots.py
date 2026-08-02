#!/usr/bin/env python3
"""Audit P0-F set matching, presence selection, and Bond content separately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.audit_g3_bond_slot_identity import (
    EXPECTED_STEPS,
    _config_from_payload,
    _gather_queries,
    candidate_identity_retention,
    endpoint_error_summary,
    summarize_routed_predictions,
)
from chemvector.g3_hierarchical_graph import (
    G3SetLearnedBondSlotPredictor,
    build_g3_bond_targets,
    g3_set_bond_slot_assignments,
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
    "outputs/smoke/chemvector-g3-p0f-set-learned-bond-slots"
)


def validate_p0f_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0-F checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if not scope.get("gt_atom_slot_oracle") or not scope.get(
        "set_learned_bond_slots"
    ):
        raise ValueError("Audit accepts only P0-F set Bond-slot checkpoints")
    if any(
        scope.get(name)
        for name in (
            "fixed_learned_bond_slots",
            "gt_bond_slot_identity_oracle",
            "shared_confidence_aware_bond_assignment",
        )
    ):
        raise ValueError("P0-F checkpoint contains an incompatible Bond protocol")
    if payload.get("sample_manifest", {}).get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-F checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-F checkpoint uses an unexpected vision source")


def order_queries_by_target(
    assignments: list[tuple[torch.Tensor, torch.Tensor]],
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the matched query for every canonical target-prefix position."""

    batch, slots = target_mask.shape
    if len(assignments) != batch:
        raise ValueError("P0-F audit assignments have an incompatible batch")
    ordered = torch.zeros(batch, slots, device=target_mask.device, dtype=torch.long)
    for batch_index, (queries, targets) in enumerate(assignments):
        count = int(target_mask[batch_index].sum())
        if len(queries) != count or set(targets.tolist()) != set(range(count)):
            raise ValueError("P0-F audit assignment does not cover the target prefix")
        ordered[batch_index, targets] = queries
    return ordered


def summarize_set_predictions(
    *,
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    normalized: dict[str, torch.Tensor],
    assignments: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, int | float], list[list[tuple[int, int, int]]], list[float]]:
    """Separate oracle-matched Bond content from deploy presence selection."""

    slots = output["set_bond_slot_output"]
    target_mask = normalized["mask"].bool()
    ordered_queries = order_queries_by_target(assignments, target_mask)
    matched_endpoints = _gather_queries(slots["endpoints"], ordered_queries)
    matched_pointer = _gather_queries(
        slots["endpoint_pointer_logits"], ordered_queries
    )
    matched_types = _gather_queries(slots["chemical_logits"], ordered_queries)
    identities = ordered_queries[..., None].expand(-1, -1, 3)
    metrics, snapshots, errors = summarize_routed_predictions(
        router_endpoints=matched_endpoints,
        target_endpoints=normalized["endpoints"],
        target_mask=target_mask,
        candidate_ids=identities,
        pointer_logits=matched_pointer,
        type_logits=matched_types,
        target_atom_refs=normalized["atom_refs"],
        target_types=normalized["chemical"],
    )

    selected_total = matched_total = intersection_total = 0
    query_set_exact_images = graph_exact_images = 0
    for batch_index in range(target_mask.size(0)):
        count = int(target_mask[batch_index].sum())
        matched = set(ordered_queries[batch_index, :count].tolist())
        selected = set(torch.where(output["bond_mask"][batch_index])[0].tolist())
        selected_total += len(selected)
        matched_total += len(matched)
        intersection_total += len(selected & matched)
        query_set_exact_images += int(selected == matched)
        predicted_refs = matched_pointer[batch_index, :count].argmax(dim=-1)
        expected_refs = normalized["atom_refs"][batch_index, :count].long()
        pointer_exact = torch.equal(
            predicted_refs.sort(dim=-1).values,
            expected_refs.sort(dim=-1).values,
        )
        type_exact = torch.equal(
            matched_types[batch_index, :count].argmax(dim=-1),
            normalized["chemical"][batch_index, :count].long(),
        )
        graph_exact_images += int(pointer_exact and type_exact)
    metrics.update(
        {
            "presence_selected_queries": selected_total,
            "hungarian_matched_queries": matched_total,
            "presence_hungarian_intersection_queries": intersection_total,
            "presence_hungarian_precision": intersection_total
            / max(selected_total, 1),
            "presence_hungarian_recall": intersection_total
            / max(matched_total, 1),
            "presence_hungarian_query_set_exact_images": query_set_exact_images,
            "hungarian_content_graph_exact_images": graph_exact_images,
        }
    )
    return metrics, snapshots, errors


@torch.no_grad()
def audit_checkpoint(
    model: G3SetLearnedBondSlotPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
) -> tuple[dict, list[list[tuple[int, int, int]]]]:
    model.eval()
    aggregate: dict[str, int | float] = {}
    snapshots = []
    errors: list[float] = []
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
        assignments = g3_set_bond_slot_assignments(output, normalized)
        metrics, current_snapshots, current_errors = summarize_set_predictions(
            output=output,
            normalized=normalized,
            assignments=assignments,
        )
        for name, value in metrics.items():
            if not name.endswith("accuracy") and not name.endswith("precision") and not name.endswith("recall"):
                aggregate[name] = aggregate.get(name, 0) + value
        snapshots.extend(current_snapshots)
        errors.extend(current_errors)
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
            "presence_hungarian_precision": aggregate[
                "presence_hungarian_intersection_queries"
            ]
            / max(aggregate["presence_selected_queries"], 1),
            "presence_hungarian_recall": aggregate[
                "presence_hungarian_intersection_queries"
            ]
            / max(aggregate["hungarian_matched_queries"], 1),
        }
    )
    return {
        "micro": aggregate,
        "matched_endpoint_error": endpoint_error_summary(errors),
    }, snapshots


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
        raise ValueError("P0-F audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-F audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-F CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0-F audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0f_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(
        first_payload["source_vision_checkpoint"], device
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-F audit vision bytes differ from the frozen protocol")
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
    checkpoint_snapshots = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0f_checkpoint(payload, step=step)
        if _config_from_payload(payload) != config:
            raise ValueError("P0-F checkpoint model configs differ")
        model = G3SetLearnedBondSlotPredictor(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        current, snapshots = audit_checkpoint(
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
        checkpoint_snapshots.append(snapshots)
        del model, payload

    retention = []
    for index in range(1, len(EXPECTED_STEPS)):
        retention.append(
            {
                "from_step": EXPECTED_STEPS[index - 1],
                "to_step": EXPECTED_STEPS[index],
                **candidate_identity_retention(
                    checkpoint_snapshots[index - 1],
                    checkpoint_snapshots[index],
                ),
            }
        )
    report = {
        "schema_version": "g3-p0f-set-bond-slot-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "manifest_sha256": hashlib.sha256(
            Path(OVERFIT32_MANIFEST).read_bytes()
        ).hexdigest(),
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": reports,
        "adjacent_checkpoint_assignment_retention": retention,
        "first_to_final_assignment_retention": candidate_identity_retention(
            checkpoint_snapshots[0], checkpoint_snapshots[-1]
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
