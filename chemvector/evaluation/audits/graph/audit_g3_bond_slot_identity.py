#!/usr/bin/env python3
"""Audit whether P0-D routes stable, geometrically useful Bond candidates."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from chemvector.g3_hierarchical_graph import (
    G3HierarchicalGraphConfig,
    G3HierarchicalGraphPredictor,
    build_g3_bond_targets,
    extract_g3_dense_bond_queries,
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
    "outputs/smoke/chemvector-g3-p0d-gt-bond-slot-identity-oracle"
)
EXPECTED_STEPS = (400, 800, 1200, 1600, 2000, 2400)
ROUTER_THRESHOLDS_PIXELS = (4.0, 8.0, 16.0)


def _gather_queries(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather = indices.reshape(
        indices.size(0), indices.size(1), *([1] * (value.ndim - 2))
    ).expand(indices.size(0), indices.size(1), *value.shape[2:])
    return value.gather(1, gather)


def _aligned_endpoint_errors_pixels(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    image_size: float,
) -> torch.Tensor:
    """Return the maximum endpoint error for each unordered Bond object."""

    if predicted.shape != expected.shape or predicted.shape[-2:] != (2, 2):
        raise ValueError("Router endpoint tensors must share shape [..., 2, 2]")
    if image_size <= 0.0:
        raise ValueError("Router audit image size must be positive")
    forward = torch.linalg.vector_norm(predicted - expected, dim=-1)
    reverse = torch.linalg.vector_norm(predicted - expected.flip(-2), dim=-1)
    forward_cost = forward.sum(dim=-1)
    reverse_cost = reverse.sum(dim=-1)
    aligned = torch.where((reverse_cost < forward_cost)[..., None], reverse, forward)
    return aligned.max(dim=-1).values.float() * image_size


def summarize_routed_predictions(
    *,
    router_endpoints: torch.Tensor,
    target_endpoints: torch.Tensor,
    target_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    pointer_logits: torch.Tensor,
    type_logits: torch.Tensor,
    target_atom_refs: torch.Tensor,
    target_types: torch.Tensor,
    image_size: float = 768.0,
) -> tuple[dict[str, int | float], list[list[tuple[int, int, int]]], list[float]]:
    """Compute identity-independent micro metrics for one checkpoint."""

    batch, queries = target_mask.shape
    expected_shapes = {
        "router_endpoints": (batch, queries, 2, 2),
        "target_endpoints": (batch, queries, 2, 2),
        "candidate_ids": (batch, queries, 3),
        "target_atom_refs": (batch, queries, 2),
        "target_types": (batch, queries),
    }
    values = {
        "router_endpoints": router_endpoints,
        "target_endpoints": target_endpoints,
        "candidate_ids": candidate_ids,
        "target_atom_refs": target_atom_refs,
        "target_types": target_types,
    }
    for name, expected_shape in expected_shapes.items():
        if values[name].shape != expected_shape:
            raise ValueError(f"{name} has an incompatible shape")
    if pointer_logits.shape[:3] != (batch, queries, 2):
        raise ValueError("Pointer logits have an incompatible shape")
    if type_logits.shape[:2] != (batch, queries):
        raise ValueError("Bond type logits have an incompatible shape")

    identities: list[list[tuple[int, int, int]]] = []
    endpoint_errors: list[float] = []
    endpoint_total = endpoint_correct = 0
    bond_total = bond_correct = 0
    type_total = type_correct = 0
    type_given_pointer_total = type_given_pointer_correct = 0
    for batch_index in range(batch):
        count = int(target_mask[batch_index].sum())
        expected_prefix = torch.arange(queries, device=target_mask.device) < count
        if not torch.equal(target_mask[batch_index].bool(), expected_prefix):
            raise ValueError("Router audit targets must use canonical prefix order")
        identities.append(
            [
                tuple(int(item) for item in row)
                for row in candidate_ids[batch_index, :count].detach().cpu().tolist()
            ]
        )
        errors = _aligned_endpoint_errors_pixels(
            router_endpoints[batch_index, :count].float(),
            target_endpoints[batch_index, :count].float(),
            image_size=image_size,
        )
        endpoint_errors.extend(float(value) for value in errors.cpu())

        predicted_refs = pointer_logits[batch_index, :count].argmax(dim=-1)
        expected_refs = target_atom_refs[batch_index, :count].long()
        pointer_equal = predicted_refs.sort(dim=-1).values == expected_refs.sort(
            dim=-1
        ).values
        both_equal = pointer_equal.all(dim=-1)
        predicted_types = type_logits[batch_index, :count].argmax(dim=-1)
        type_equal = predicted_types == target_types[batch_index, :count].long()
        endpoint_total += int(pointer_equal.numel())
        endpoint_correct += int(pointer_equal.sum())
        bond_total += count
        bond_correct += int(both_equal.sum())
        type_total += count
        type_correct += int(type_equal.sum())
        type_given_pointer_total += int(both_equal.sum())
        type_given_pointer_correct += int((type_equal & both_equal).sum())

    metrics: dict[str, int | float] = {
        "bond_targets": bond_total,
        "pointer_endpoints": endpoint_total,
        "pointer_endpoint_correct": endpoint_correct,
        "pointer_endpoint_accuracy": endpoint_correct / max(endpoint_total, 1),
        "pointer_bonds": bond_total,
        "pointer_both_endpoint_correct": bond_correct,
        "pointer_both_endpoint_accuracy": bond_correct / max(bond_total, 1),
        "type_bonds": type_total,
        "type_correct": type_correct,
        "type_accuracy": type_correct / max(type_total, 1),
        "type_given_pointer_bonds": type_given_pointer_total,
        "type_given_pointer_correct": type_given_pointer_correct,
        "type_given_pointer_accuracy": type_given_pointer_correct
        / max(type_given_pointer_total, 1),
    }
    return metrics, identities, endpoint_errors


def candidate_identity_retention(
    previous: list[list[tuple[int, int, int]]],
    current: list[list[tuple[int, int, int]]],
) -> dict[str, int | float]:
    """Compare routed dense-cell identities target by target."""

    if len(previous) != len(current):
        raise ValueError("Candidate snapshots contain different sample counts")
    comparable = retained = 0
    changed_images = 0
    for previous_sample, current_sample in zip(previous, current):
        if len(previous_sample) != len(current_sample):
            raise ValueError("Candidate snapshots contain different target counts")
        matches = [left == right for left, right in zip(previous_sample, current_sample)]
        comparable += len(matches)
        retained += sum(matches)
        changed_images += int(not all(matches))
    changed = comparable - retained
    return {
        "comparable_targets": comparable,
        "retained_targets": retained,
        "changed_targets": changed,
        "retention": retained / max(comparable, 1),
        "churn": changed / max(comparable, 1),
        "changed_images": changed_images,
    }


def endpoint_error_summary(errors: list[float]) -> dict[str, int | float]:
    if not errors:
        raise ValueError("Router audit requires at least one endpoint error")
    values = torch.tensor(errors, dtype=torch.float64)
    result: dict[str, int | float] = {
        "objects": len(errors),
        "median_pixels": float(values.quantile(0.5)),
        "p90_pixels": float(values.quantile(0.9)),
        "maximum_pixels": float(values.max()),
    }
    for threshold in ROUTER_THRESHOLDS_PIXELS:
        label = str(int(threshold))
        covered = int((values <= threshold).sum())
        result[f"within_{label}px_objects"] = covered
        result[f"within_{label}px_coverage"] = covered / len(errors)
    return result


def validate_p0d_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0-D checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if not scope.get("gt_atom_slot_oracle") or not scope.get(
        "gt_bond_slot_identity_oracle"
    ):
        raise ValueError("Audit accepts only P0-D GT Atom/Bond oracle checkpoints")
    manifest = payload.get("sample_manifest", {})
    if manifest.get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-D checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-D checkpoint uses an unexpected vision source")


def _config_from_payload(payload: dict) -> G3HierarchicalGraphConfig:
    valid = {field.name for field in fields(G3HierarchicalGraphConfig)}
    raw = payload.get("g3_config")
    if not isinstance(raw, dict):
        raise ValueError("P0-D checkpoint has no G3 config")
    return G3HierarchicalGraphConfig(
        **{name: value for name, value in raw.items() if name in valid}
    )


@torch.no_grad()
def audit_checkpoint(
    model: G3HierarchicalGraphPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    sample_keys: list[str],
    batch_size: int,
    precision: str,
) -> tuple[dict, list[list[tuple[int, int, int]]]]:
    model.eval()
    aggregate: dict[str, int | float] = {}
    all_identities: list[list[tuple[int, int, int]]] = []
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
                oracle_bond_endpoints=normalized["endpoints"],
                oracle_bond_mask=normalized["mask"],
            )
        query_output = output["bond_query_output"]
        raw = extract_g3_dense_bond_queries(
            output["bond_dense_output"],
            maximum_queries=model.config.maximum_bonds,
        )
        permutation = query_output["oracle_permutation"]
        router_endpoints = _gather_queries(raw["base_endpoints"], permutation)
        candidate_ids = _gather_queries(raw["indices"], permutation)
        metrics, identities, errors = summarize_routed_predictions(
            router_endpoints=router_endpoints,
            target_endpoints=normalized["endpoints"],
            target_mask=normalized["mask"],
            candidate_ids=candidate_ids,
            pointer_logits=output["endpoint_pointer_logits"],
            type_logits=output["bond_type_logits"],
            target_atom_refs=normalized["atom_refs"],
            target_types=normalized["chemical"],
        )
        for name, value in metrics.items():
            if name.endswith("accuracy"):
                continue
            aggregate[name] = aggregate.get(name, 0) + value
        all_identities.extend(identities)
        all_errors.extend(errors)
    if len(all_identities) != len(sample_keys):
        raise RuntimeError("P0-D audit did not replay every committed sample")
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
    per_sample = [
        {
            "sample_key": key,
            "candidate_ids_by_target": [list(identity) for identity in identities],
        }
        for key, identities in zip(sample_keys, all_identities)
    ]
    return {
        "micro": aggregate,
        "router_endpoint_error": endpoint_error_summary(all_errors),
        "samples": per_sample,
    }, all_identities


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
        raise ValueError("P0-D audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-D audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-D CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0-D audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0d_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision_path = first_payload["source_vision_checkpoint"]
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-D audit vision bytes differ from the frozen protocol")
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
    sample_keys = list(batch["keys"])
    del vision, batch

    reports = []
    snapshots: list[list[list[tuple[int, int, int]]]] = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0d_checkpoint(payload, step=step)
        if _config_from_payload(payload) != config:
            raise ValueError("P0-D checkpoint model configs differ")
        model = G3HierarchicalGraphPredictor(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        current, identities = audit_checkpoint(
            model,
            features,
            targets,
            sample_keys=sample_keys,
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
        snapshots.append(identities)
        del model, payload

    retention = []
    for index in range(1, len(EXPECTED_STEPS)):
        retention.append(
            {
                "from_step": EXPECTED_STEPS[index - 1],
                "to_step": EXPECTED_STEPS[index],
                **candidate_identity_retention(
                    snapshots[index - 1], snapshots[index]
                ),
            }
        )
    report = {
        "schema_version": "g3-p0d-bond-slot-identity-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "manifest_sha256": hashlib.sha256(
            Path(OVERFIT32_MANIFEST).read_bytes()
        ).hexdigest(),
        "expected_steps": list(EXPECTED_STEPS),
        "router_error_definition": (
            "maximum Euclidean endpoint error after unordered endpoint alignment"
        ),
        "checkpoints": reports,
        "adjacent_checkpoint_candidate_retention": retention,
        "first_to_final_candidate_retention": candidate_identity_retention(
            snapshots[0], snapshots[-1]
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
