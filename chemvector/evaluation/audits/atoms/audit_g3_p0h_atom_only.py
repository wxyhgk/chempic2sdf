#!/usr/bin/env python3
"""Audit the train-only G3 P0-H Atom-only capacity checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
    _build_atom_slot_targets,
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
    "outputs/smoke/chemvector-g3-p0h-atom-only-overfit32"
)
EXPECTED_STEPS = (400, 800, 1200, 1600, 2000, 2400)
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0
TOP128 = 128


def validate_p0h_checkpoint(payload: dict, *, step: int) -> None:
    """Reject checkpoints outside the fixed P0-H train-only protocol."""

    if payload.get("global_step") != step:
        raise ValueError("P0-H checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if (
        not scope.get("atom_only_capacity")
        or scope.get("bond_refs_and_types") is not False
        or scope.get("count_teacher_forcing") is not False
    ):
        raise ValueError("Atom audit accepts only the P0-H Atom-only protocol")
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
        raise ValueError("P0-H Atom audit rejects graph or oracle protocols")
    if payload.get("training_args", {}).get("atom_only_capacity") is not True:
        raise ValueError("P0-H checkpoint is missing --atom-only-capacity")
    if payload.get("sample_manifest", {}).get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-H checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-H checkpoint uses an unexpected vision source")


def config_from_payload(payload: dict) -> G3HierarchicalGraphConfig:
    valid = {field.name for field in fields(G3HierarchicalGraphConfig)}
    raw = payload.get("g3_config")
    if not isinstance(raw, dict):
        raise ValueError("P0-H checkpoint has no G3 config")
    return G3HierarchicalGraphConfig(
        **{name: value for name, value in raw.items() if name in valid}
    )


def model_from_checkpoint(
    payload: dict,
    config: G3HierarchicalGraphConfig,
    device: torch.device,
) -> G3AtomOnlyPredictor:
    """Strictly restore the actual P0-H Atom-only checkpoint state."""

    if payload.get("model_type") != G3AtomOnlyPredictor.model_type:
        raise ValueError("P0-H checkpoint has an unexpected model type")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("P0-H checkpoint has no model state")
    model = G3AtomOnlyPredictor(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def hungarian_radius_match_count(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    radius_pixels: float = MATCH_RADIUS_PIXELS,
    image_size: float = IMAGE_SIZE,
) -> int:
    """Return maximum one-to-one matches inside the radius, then minimize distance."""

    if predicted.ndim != 2 or expected.ndim != 2 or predicted.shape[-1] != 2:
        raise ValueError("Atom matching requires [objects, 2] coordinates")
    if expected.shape[-1] != 2 or min(radius_pixels, image_size) <= 0.0:
        raise ValueError("Atom matching dimensions must be positive")
    if not len(predicted) or not len(expected):
        return 0
    distances = torch.cdist(expected.float(), predicted.float()) * image_size
    eligible_columns = (distances <= radius_pixels).any(dim=0)
    eligible = distances[:, eligible_columns]
    targets = len(expected)
    priority = float(targets + 1)
    real_cost = torch.where(
        eligible <= radius_pixels,
        eligible / radius_pixels,
        torch.full_like(eligible, 2.0 * priority),
    )
    dummy = torch.full(
        (targets, targets),
        priority,
        device=distances.device,
        dtype=distances.dtype,
    )
    cost = torch.cat((real_cost, dummy), dim=1)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    real_columns = eligible.size(1)
    return int(
        sum(
            column < real_columns
            and float(eligible[row, column]) <= radius_pixels
            for row, column in zip(rows, columns)
        )
    )


def flatten_atom_candidates(
    dense: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = dense["presence_logits"]
    anchors = dense["anchors"]
    if logits.ndim != 4 or anchors.shape != (*logits.shape[:2], 2, *logits.shape[2:]):
        raise ValueError("P0-H dense Atom tensors have incompatible shapes")
    batch, slots, height, width = logits.shape
    flat_logits = logits.reshape(batch, -1)
    flat_anchors = anchors.permute(0, 1, 3, 4, 2).reshape(batch, -1, 2)
    slot, y_index, x_index = torch.meshgrid(
        torch.arange(slots, device=logits.device),
        torch.arange(height, device=logits.device),
        torch.arange(width, device=logits.device),
        indexing="ij",
    )
    identities = torch.stack((slot, y_index, x_index), dim=-1).reshape(-1, 3)
    return flat_logits, flat_anchors, identities


def supervision_positive_snapshot(
    dense: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    image_size: float = IMAGE_SIZE,
) -> tuple[list[list[tuple[int, int, int]]], list[list[float]]]:
    """Recover the supervised dense slot identity for each canonical GT Atom."""

    assigned = _build_atom_slot_targets(dense, targets)
    target_mask = targets["atom_mask"].bool()
    identities: list[list[tuple[int, int, int]]] = []
    errors: list[list[float]] = []
    for batch_index in range(len(target_mask)):
        positive_rows = assigned["positive"][batch_index].nonzero(as_tuple=False)
        available = set(range(len(positive_rows)))
        sample_identities = []
        sample_errors = []
        for target_index in torch.where(target_mask[batch_index])[0].tolist():
            target_anchor = targets["atom_anchors"][batch_index, target_index]
            matches = [
                row_index
                for row_index in available
                if torch.equal(
                    assigned["anchors"][
                        batch_index,
                        int(positive_rows[row_index, 0]),
                        :,
                        int(positive_rows[row_index, 1]),
                        int(positive_rows[row_index, 2]),
                    ],
                    target_anchor,
                )
            ]
            if len(matches) != 1:
                raise ValueError("P0-H supervision identity is not uniquely recoverable")
            row_index = matches[0]
            available.remove(row_index)
            slot, y_index, x_index = (
                int(value) for value in positive_rows[row_index].tolist()
            )
            predicted = dense["anchors"][batch_index, slot, :, y_index, x_index]
            error = torch.linalg.vector_norm(
                predicted.float() - target_anchor.float()
            ) * image_size
            sample_identities.append((slot, y_index, x_index))
            sample_errors.append(float(error))
        identities.append(sample_identities)
        errors.append(sample_errors)
    return identities, errors


def identity_retention(
    previous: list[list[tuple[int, int, int]]],
    current: list[list[tuple[int, int, int]]],
) -> dict[str, int | float]:
    if len(previous) != len(current):
        raise ValueError("Atom identity snapshots contain different sample counts")
    comparable = retained = changed_images = 0
    for left, right in zip(previous, current):
        if len(left) != len(right):
            raise ValueError("Atom identity snapshots contain different object counts")
        matches = [first == second for first, second in zip(left, right)]
        comparable += len(matches)
        retained += sum(matches)
        changed_images += int(not all(matches))
    changed = comparable - retained
    return {
        "comparable": comparable,
        "retained": retained,
        "changed": changed,
        "retention": retained / max(comparable, 1),
        "churn": changed / max(comparable, 1),
        "changed_images": changed_images,
    }


def ranked_set_retention(
    previous: list[list[tuple[int, int, int]]],
    current: list[list[tuple[int, int, int]]],
) -> dict[str, int | float]:
    if len(previous) != len(current):
        raise ValueError("Atom ranking snapshots contain different sample counts")
    selected = intersection = union = rank_exact = changed_images = 0
    for left, right in zip(previous, current):
        if len(left) != len(right):
            raise ValueError("Atom ranking snapshots contain different selection counts")
        left_set, right_set = set(left), set(right)
        selected += len(left)
        intersection += len(left_set & right_set)
        union += len(left_set | right_set)
        rank_exact += sum(first == second for first, second in zip(left, right))
        changed_images += int(left != right)
    return {
        "selected": selected,
        "set_intersection": intersection,
        "set_union": union,
        "set_retention": intersection / max(selected, 1),
        "set_churn": (selected - intersection) / max(selected, 1),
        "set_jaccard": intersection / max(union, 1),
        "rank_position_retention": rank_exact / max(selected, 1),
        "changed_images": changed_images,
    }


def localization_summary(errors: list[float]) -> dict[str, int | float]:
    if not errors:
        raise ValueError("P0-H audit requires positive localization errors")
    values = torch.tensor(errors, dtype=torch.float64)
    return {
        "positive_atoms": len(errors),
        "median_pixels": float(values.quantile(0.5)),
        "p90_pixels": float(values.quantile(0.9)),
        "within_6px": int((values <= MATCH_RADIUS_PIXELS).sum()),
        "within_6px_coverage": float((values <= MATCH_RADIUS_PIXELS).double().mean()),
    }


def selector_identity_counts(
    selected: list[tuple[int, int, int]],
    positive: list[tuple[int, int, int]],
) -> dict[str, int]:
    """Return the complete GT-count selector/positive identity confusion counts."""

    selected_set = set(selected)
    positive_set = set(positive)
    if len(selected_set) != len(selected) or len(positive_set) != len(positive):
        raise ValueError("P0-H selector identities must be unique within an image")
    intersection = selected_set & positive_set
    return {
        "selected": len(selected_set),
        "positive": len(positive_set),
        "intersection": len(intersection),
        "selected_only": len(selected_set - positive_set),
        "positive_only": len(positive_set - selected_set),
        "whole_set_exact": int(selected_set == positive_set),
    }


def conditional_localization_summary(
    errors: list[float],
) -> dict[str, int | float | None]:
    """Summarize localization only where selector and supervision identity agree."""

    if not errors:
        return {
            "selected_positive_atoms": 0,
            "median_pixels": None,
            "p90_pixels": None,
            "within_6px": 0,
            "within_6px_accuracy": 0.0,
        }
    values = torch.tensor(errors, dtype=torch.float64)
    within = int((values <= MATCH_RADIUS_PIXELS).sum())
    return {
        "selected_positive_atoms": len(errors),
        "median_pixels": float(values.quantile(0.5)),
        "p90_pixels": float(values.quantile(0.9)),
        "within_6px": within,
        "within_6px_accuracy": within / len(errors),
    }


@torch.no_grad()
def audit_checkpoint(
    model: G3AtomOnlyPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    sample_keys: list[str],
    batch_size: int,
    precision: str,
) -> tuple[dict, dict[str, list[list[tuple[int, int, int]]]]]:
    model.eval()
    totals = {
        "images": 0,
        "target_atoms": 0,
        "deploy_count_exact_images": 0,
        "deploy_count_absolute_error": 0,
        "selector_identity_selected": 0,
        "selector_identity_supervision_positive": 0,
        "selector_identity_intersection": 0,
        "selector_identity_selected_only": 0,
        "selector_identity_positive_only": 0,
        "selector_identity_whole_set_exact_images": 0,
    }
    for mode in ("all_dense", "top128", "gt_count_topk"):
        totals[f"{mode}_matched_atoms"] = 0
        totals[f"{mode}_set_exact_images"] = 0
    snapshots: dict[str, list[list[tuple[int, int, int]]]] = {
        "supervision_positive": [],
        "top128_ranking": [],
        "gt_count_topk_ranking": [],
    }
    all_errors: list[float] = []
    selected_positive_errors: list[float] = []
    per_sample = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            dense = model.atom_head(current_features)
        flat_logits, flat_anchors, flat_identities = flatten_atom_candidates(dense)
        positive_ids, positive_errors = supervision_positive_snapshot(
            dense, current_targets
        )
        deploy_counts = dense["count_logits"].argmax(dim=-1)
        for batch_index in range(len(current_features)):
            target_indices = torch.where(current_targets["atom_mask"][batch_index])[0]
            expected = current_targets["atom_anchors"][batch_index, target_indices]
            count = len(expected)
            ranked = flat_logits[batch_index].argsort(descending=True)
            top128_indices = ranked[: min(TOP128, len(ranked))]
            gt_count_indices = ranked[:count]
            mode_anchors = {
                "all_dense": flat_anchors[batch_index],
                "top128": flat_anchors[batch_index, top128_indices],
                "gt_count_topk": flat_anchors[batch_index, gt_count_indices],
            }
            matches = {
                mode: hungarian_radius_match_count(anchors, expected)
                for mode, anchors in mode_anchors.items()
            }
            top128_ids = [
                tuple(int(value) for value in flat_identities[index].tolist())
                for index in top128_indices.tolist()
            ]
            gt_count_ids = [
                tuple(int(value) for value in flat_identities[index].tolist())
                for index in gt_count_indices.tolist()
            ]
            identity_counts = selector_identity_counts(
                gt_count_ids, positive_ids[batch_index]
            )
            selected_set = set(gt_count_ids)
            selected_positive_errors.extend(
                error
                for identity, error in zip(
                    positive_ids[batch_index], positive_errors[batch_index]
                )
                if identity in selected_set
            )
            deploy_count = int(deploy_counts[batch_index])
            totals["images"] += 1
            totals["target_atoms"] += count
            totals["deploy_count_exact_images"] += int(deploy_count == count)
            totals["deploy_count_absolute_error"] += abs(deploy_count - count)
            totals["selector_identity_selected"] += identity_counts["selected"]
            totals["selector_identity_supervision_positive"] += identity_counts[
                "positive"
            ]
            totals["selector_identity_intersection"] += identity_counts[
                "intersection"
            ]
            totals["selector_identity_selected_only"] += identity_counts[
                "selected_only"
            ]
            totals["selector_identity_positive_only"] += identity_counts[
                "positive_only"
            ]
            totals["selector_identity_whole_set_exact_images"] += identity_counts[
                "whole_set_exact"
            ]
            for mode, matched in matches.items():
                totals[f"{mode}_matched_atoms"] += matched
                totals[f"{mode}_set_exact_images"] += int(matched == count)
            snapshots["supervision_positive"].append(positive_ids[batch_index])
            snapshots["top128_ranking"].append(top128_ids)
            snapshots["gt_count_topk_ranking"].append(gt_count_ids)
            all_errors.extend(positive_errors[batch_index])
            per_sample.append(
                {
                    "sample_key": sample_keys[start + batch_index],
                    "target_atoms": count,
                    "deploy_count": deploy_count,
                    "supervision_positive_ids_by_target": [
                        list(identity) for identity in positive_ids[batch_index]
                    ],
                    "top128_candidate_ids_by_rank": [list(value) for value in top128_ids],
                    "gt_count_topk_candidate_ids_by_rank": [
                        list(value) for value in gt_count_ids
                    ],
                    "matches_6px": matches,
                }
            )
    images = max(totals["images"], 1)
    target_atoms = max(totals["target_atoms"], 1)
    totals.update(
        {
            "deploy_count_accuracy": totals["deploy_count_exact_images"] / images,
            "deploy_count_mae": totals["deploy_count_absolute_error"] / images,
            "selector_identity_micro_precision": totals[
                "selector_identity_intersection"
            ]
            / max(totals["selector_identity_selected"], 1),
            "selector_identity_micro_recall": totals[
                "selector_identity_intersection"
            ]
            / max(totals["selector_identity_supervision_positive"], 1),
            "selector_identity_whole_set_exact_rate": totals[
                "selector_identity_whole_set_exact_images"
            ]
            / images,
            "supervision_positive_gt_count_topk_identity_coverage": totals[
                "selector_identity_intersection"
            ]
            / target_atoms,
        }
    )
    for mode in ("all_dense", "top128", "gt_count_topk"):
        totals[f"{mode}_micro_recall_6px"] = (
            totals[f"{mode}_matched_atoms"] / target_atoms
        )
        totals[f"{mode}_set_exact_rate_6px"] = (
            totals[f"{mode}_set_exact_images"] / images
        )
    return {
        "metrics": totals,
        "positive_localization": localization_summary(all_errors),
        "selected_positive_conditional_localization": (
            conditional_localization_summary(selected_positive_errors)
        ),
        "samples": per_sample,
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
        raise ValueError("P0-H audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-H audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-H CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0-H audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0h_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(
        first_payload["source_vision_checkpoint"], device
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-H audit vision bytes differ from the frozen protocol")
    config = config_from_payload(first_payload)
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
    snapshots = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0h_checkpoint(payload, step=step)
        if config_from_payload(payload) != config:
            raise ValueError("P0-H checkpoint model configs differ")
        model = model_from_checkpoint(payload, config, device)
        current, current_snapshots = audit_checkpoint(
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
        snapshots.append(current_snapshots)
        del model, payload

    transitions = []
    for index in range(1, len(EXPECTED_STEPS)):
        transitions.append(
            {
                "from_step": EXPECTED_STEPS[index - 1],
                "to_step": EXPECTED_STEPS[index],
                "supervision_positive_identity": identity_retention(
                    snapshots[index - 1]["supervision_positive"],
                    snapshots[index]["supervision_positive"],
                ),
                "top128_ranking": ranked_set_retention(
                    snapshots[index - 1]["top128_ranking"],
                    snapshots[index]["top128_ranking"],
                ),
                "gt_count_topk_ranking": ranked_set_retention(
                    snapshots[index - 1]["gt_count_topk_ranking"],
                    snapshots[index]["gt_count_topk_ranking"],
                ),
            }
        )
    report = {
        "schema_version": "g3-p0h-atom-only-final-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "vision": {
            "path": str(Path(first_payload["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha256,
        },
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
        },
        "expected_steps": list(EXPECTED_STEPS),
        "matching": {
            "method": "one-to-one Hungarian, maximum radius matches then distance",
            "radius_pixels": MATCH_RADIUS_PIXELS,
            "candidate_modes": ["all_dense", "top128", "gt_count_topk"],
        },
        "checkpoints": reports,
        "adjacent_checkpoint_churn": transitions,
        "first_to_final_churn": {
            "supervision_positive_identity": identity_retention(
                snapshots[0]["supervision_positive"],
                snapshots[-1]["supervision_positive"],
            ),
            "top128_ranking": ranked_set_retention(
                snapshots[0]["top128_ranking"], snapshots[-1]["top128_ranking"]
            ),
            "gt_count_topk_ranking": ranked_set_retention(
                snapshots[0]["gt_count_topk_ranking"],
                snapshots[-1]["gt_count_topk_ranking"],
            ),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
