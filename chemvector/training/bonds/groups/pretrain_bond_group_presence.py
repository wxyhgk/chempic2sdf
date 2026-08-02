#!/usr/bin/env python3
"""Train local-evidence presence ranking on frozen v0.8.32 Bond groups."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.bond_group_presence import (
    BondGroupPresenceConfig,
    BondGroupPresenceSelector,
    bond_group_presence_loss,
    build_geometry_equivalence_labels,
)
from chemvector.bond_group_query import BondGroupQueryConfig, BondGroupQueryDetector
from chemvector.data import ChemVectorIterableDataset
from chemvector.experiment_loop.artifacts import sha256_file
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_bond_group_query import (
    _autocast,
    _frozen_proposals,
    _target_cells,
    capacity_exposure_order,
    load_records,
    mixed_exposure_order,
)
from chemvector.pretrain_chemscene_bond_query import (
    ChemSceneBondCollator,
    stack_targets,
)
from chemvector.pretrain_dense_slot_matching import build_collision_split, load_detector
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline


IMAGE_SIZE = 768.0
DEFAULT_GROUP_CHECKPOINT = (
    "outputs/checkpoints/"
    "chemvector-v0832-bond-group-query-ink-line-local-only-early4k/best.pt"
)
CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0833-bond-group-presence-capacity100"
)
CAPACITY_4K_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0833-bond-group-presence-capacity100-4k"
)
GENERALIZATION_SAFE_SELECT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v0833-bond-group-presence-generalization-safe-select"
)
CAPACITY_SPLIT = (
    "outputs/checkpoints/"
    "chemvector-v0832-bond-group-query-ink-line-local-only-capacity100/split.json"
)
GENERALIZATION_SPLIT = (
    "outputs/checkpoints/"
    "chemvector-v0832-bond-group-query-ink-line-local-only-early4k/split.json"
)


def apply_mode_protocol(args) -> None:
    if args.mode == "capacity100":
        if args.output_dir is None:
            args.output_dir = (
                CAPACITY_4K_OUTPUT_DIR
                if args.capacity_exposure_control_4k
                else CAPACITY_OUTPUT_DIR
            )
        if args.expected_split is None:
            args.expected_split = CAPACITY_SPLIT
        args.collision_train_images = 100
        args.collision_holdout_images = 100
        args.ordinary_replay_images = 100
        args.train_exposures = 4000 if args.capacity_exposure_control_4k else 2000
        args.validate_every = 400 if args.capacity_exposure_control_4k else 200
        return
    if args.capacity_exposure_control_4k:
        raise ValueError("The 4k exposure control is only valid in capacity100 mode")
    if args.output_dir is None:
        args.output_dir = GENERALIZATION_SAFE_SELECT_OUTPUT_DIR
    if args.expected_split is None:
        args.expected_split = GENERALIZATION_SPLIT
    args.collision_train_images = 400
    args.collision_holdout_images = 100
    args.ordinary_replay_images = 400
    args.train_exposures = 4000
    args.validate_every = 400


def _prepare_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "training_args.json",
        "config.json",
        "split.json",
        "zero_step_parity.json",
        "initial.pt",
        "best.pt",
        "last.pt",
        "summary.json",
        "frozen_state_audit.json",
    }
    existing = sorted(
        item.name
        for item in path.iterdir()
        if item.name in artifacts or item.name.startswith("checkpoint-")
    )
    if existing:
        raise FileExistsError(
            f"Bond Group presence output already contains artifacts: {existing[:8]}"
        )


def _state_dict_sha256(module: torch.nn.Module) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    tensors = parameters = 0
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensors += 1
        parameters += tensor.numel()
    return digest.hexdigest(), tensors, parameters


def _load_frozen_pipeline(
    group_checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module, BondGroupQueryDetector, dict]:
    group_payload = torch.load(
        group_checkpoint, map_location="cpu", weights_only=False
    )
    if group_payload.get("schema_version") != "bond-group-query-checkpoint-0.1":
        raise ValueError("Unsupported Bond Group Query checkpoint")
    group = BondGroupQueryDetector(
        BondGroupQueryConfig(**group_payload["config"])
    ).to(device)
    group.load_state_dict(group_payload["bond_group_query"], strict=True)
    group.requires_grad_(False).eval()

    dense_path = Path(group_payload["dense_bond_checkpoint"])
    dense_payload = torch.load(dense_path, map_location="cpu", weights_only=False)
    _, grounded_payload = load_grounded_graph_checkpoint(
        dense_payload["grounded_graph_checkpoint"], device
    )
    vision, _, _, _ = load_frozen_grounded_pipeline(
        Path(grounded_payload["grounded_selector_checkpoint"]), device
    )
    vision.requires_grad_(False).eval()
    dense = load_detector(dense_payload, device)
    dense.requires_grad_(False).eval()
    return vision, dense, group, group_payload


@torch.no_grad()
def _frozen_group_outputs(
    vision,
    dense,
    group,
    pixels: torch.Tensor,
    *,
    queries: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stride4, _, proposals = _frozen_proposals(
        vision, dense, pixels, queries=queries
    )
    with _autocast(pixels.device):
        output = group(stride4, proposals, pixel_values=pixels)
    return output, proposals


def _unordered_max_error_pixels(
    predicted: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    forward = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None]) * IMAGE_SIZE, dim=-1
    ).max(dim=-1).values
    reverse = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None].flip(2)) * IMAGE_SIZE, dim=-1
    ).max(dim=-1).values
    return torch.minimum(forward, reverse)


def _selected_target_indices(
    endpoints: torch.Tensor,
    selected: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance_pixels: float,
) -> set[int]:
    if not len(selected) or not len(expected):
        return set()
    cost = _unordered_max_error_pixels(endpoints[selected].float(), expected.float())
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    return {
        int(column)
        for row, column in zip(rows.tolist(), columns.tolist())
        if float(cost[row, column]) <= tolerance_pixels + 1e-6
    }


def _topk_valid(
    logits: torch.Tensor, mask: torch.Tensor, count: int
) -> torch.Tensor:
    valid = torch.where(mask)[0]
    if count <= 0 or not len(valid):
        return valid[:0]
    count = min(count, len(valid))
    return valid[logits[valid].topk(count).indices]


def _update_representable_selection(
    totals: Counter,
    logits: torch.Tensor,
    labels: dict[str, torch.Tensor],
    batch_index: int,
) -> None:
    desired = int(labels["representable_counts"][batch_index])
    selected = _topk_valid(
        logits[batch_index], labels["loss_mask"][batch_index], desired
    )
    positive_groups = set(
        torch.where(
            labels["group_mask"][batch_index]
            & labels["group_targets"][batch_index]
        )[0].tolist()
    )
    counts: Counter = Counter(
        int(labels["group_ids"][batch_index, index])
        for index in selected.tolist()
    )
    recalled = sum(counts[group] > 0 for group in positive_groups)
    exactly_one = sum(counts[group] == 1 for group in positive_groups)
    selected_positive = {group for group in counts if group in positive_groups}
    selected_negative = {group for group in counts if group not in positive_groups}
    totals["representable_groups"] += len(positive_groups)
    totals["representable_recalled"] += recalled
    totals["representable_exactly_one"] += exactly_one
    totals["representable_set_exact"] += int(
        selected_positive == positive_groups
        and not selected_negative
        and exactly_one == len(positive_groups)
    )


def _update_geometry_selection(
    totals: Counter,
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    logits: torch.Tensor,
    batch_index: int,
    *,
    grid_size: int,
    tolerance_pixels: float,
) -> None:
    valid_targets = torch.where(targets["mask"][batch_index])[0]
    expected = targets["endpoints"][batch_index, valid_targets]
    desired = len(valid_targets)
    selected = _topk_valid(
        logits[batch_index], output["proposal_mask"][batch_index], desired
    )
    local_matches = _selected_target_indices(
        output["endpoints"][batch_index],
        selected,
        expected,
        tolerance_pixels=tolerance_pixels,
    )
    matched_targets = {int(valid_targets[index]) for index in local_matches}
    representable_targets = set(
        torch.where(labels["target_representable"][batch_index])[0].tolist()
    )
    totals["samples"] += 1
    totals["gt_bonds"] += len(valid_targets)
    totals["representable_bonds"] += len(representable_targets)
    totals["selected_geometry_bonds"] += len(matched_targets)
    totals["image_geometry_exact"] += int(
        len(matched_targets) == len(valid_targets)
    )

    for indices in _target_cells(targets, batch_index, grid_size).values():
        if len(indices) < 2:
            continue
        totals["collision_cells"] += 1
        totals["collision_bonds"] += len(indices)
        totals["collision_representable"] += sum(
            index in representable_targets for index in indices
        )
        totals["collision_selected"] += sum(
            index in matched_targets for index in indices
        )
        totals["collision_secondary"] += len(indices) - 1
        totals["collision_secondary_representable"] += sum(
            index in representable_targets for index in indices[1:]
        )
        totals["collision_secondary_selected"] += sum(
            index in matched_targets for index in indices[1:]
        )
        totals["collision_candidate_cell_exact"] += int(
            all(index in representable_targets for index in indices)
        )
        totals["collision_selected_cell_exact"] += int(
            all(index in matched_targets for index in indices)
        )


@torch.inference_mode()
def evaluate_records(
    vision,
    dense,
    group,
    selector,
    records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    *,
    batch_size: int,
    queries: int,
    grid_size: int,
    tolerance_pixels: float,
) -> dict:
    selector.eval()
    totals: Counter = Counter()
    maximum_residual = 0.0
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        group_output, proposals = _frozen_group_outputs(
            vision, dense, group, pixels, queries=queries
        )
        with _autocast(device):
            selector_output = selector(group_output, proposals)
        labels = build_geometry_equivalence_labels(
            group_output["endpoints"],
            group_output["proposal_mask"],
            targets,
            tolerance_pixels=tolerance_pixels,
            image_size=IMAGE_SIZE,
        )
        if selector_output["presence_residual"].numel():
            maximum_residual = max(
                maximum_residual,
                float(selector_output["presence_residual"].float().abs().max()),
            )
        logits = selector_output["keep_logits"].float()
        for batch_index in range(len(pixels)):
            _update_representable_selection(
                totals, logits, labels, batch_index
            )
            _update_geometry_selection(
                totals,
                group_output,
                targets,
                labels,
                logits,
                batch_index,
                grid_size=grid_size,
                tolerance_pixels=tolerance_pixels,
            )
    samples = max(totals["samples"], 1)
    bonds = max(totals["gt_bonds"], 1)
    representable = max(totals["representable_bonds"], 1)
    groups = max(totals["representable_groups"], 1)
    collision_bonds = max(totals["collision_bonds"], 1)
    secondary = max(totals["collision_secondary"], 1)
    cells = max(totals["collision_cells"], 1)
    return {
        "schema_version": "bond-group-presence-metrics-0.1",
        "samples": totals["samples"],
        "gt_bonds": totals["gt_bonds"],
        "representable_bonds": totals["representable_bonds"],
        "candidate_ceiling_recall": totals["representable_bonds"] / bonds,
        "known_gt_count_geometry_recall": (
            totals["selected_geometry_bonds"] / bonds
        ),
        "known_gt_count_image_exact": totals["image_geometry_exact"] / samples,
        "representable_group_recall": (
            totals["representable_recalled"] / groups
        ),
        "representable_exactly_one_rate": (
            totals["representable_exactly_one"] / groups
        ),
        "representable_set_exact_rate": (
            totals["representable_set_exact"] / samples
        ),
        "missed_representable_per_image": (
            totals["representable_bonds"] - totals["representable_recalled"]
        )
        / samples,
        "collision_cells": totals["collision_cells"],
        "collision_bonds": totals["collision_bonds"],
        "collision_candidate_recall": (
            totals["collision_representable"] / collision_bonds
        ),
        "collision_known_count_recall": (
            totals["collision_selected"] / collision_bonds
        ),
        "collision_secondary_candidate_recall": (
            totals["collision_secondary_representable"] / secondary
        ),
        "collision_secondary_known_count_recall": (
            totals["collision_secondary_selected"] / secondary
        ),
        "collision_candidate_cell_exact_rate": (
            totals["collision_candidate_cell_exact"] / cells
        ),
        "collision_known_count_cell_exact_rate": (
            totals["collision_selected_cell_exact"] / cells
        ),
        "maximum_presence_residual": maximum_residual,
        "tolerance_pixels": tolerance_pixels,
    }


@torch.inference_mode()
def audit_zero_step_parity(
    vision,
    dense,
    group,
    selector,
    records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    *,
    batch_size: int,
    queries: int,
) -> dict:
    maximum_delta = 0.0
    mismatches = nonfinite = 0
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        group_output, proposals = _frozen_group_outputs(
            vision, dense, group, pixels, queries=queries
        )
        with _autocast(device):
            output = selector(group_output, proposals)
        difference = (
            output["keep_logits"].float() - group_output["presence_logits"].float()
        ).abs()
        nonfinite += int((~torch.isfinite(difference)).sum())
        if difference.numel():
            maximum_delta = max(maximum_delta, float(difference.max()))
        mismatches += int(
            (output["proposal_mask"] != group_output["proposal_mask"]).sum()
        )
    passed = maximum_delta == 0.0 and nonfinite == 0 and mismatches == 0
    return {
        "schema_version": "bond-group-presence-zero-step-parity-0.1",
        "status": "passed" if passed else "failed",
        "samples": len(records),
        "maximum_logit_delta": maximum_delta,
        "nonfinite_values": nonfinite,
        "proposal_mask_mismatches": mismatches,
    }


def metric_score(metrics: dict, mode: str) -> tuple[float, ...]:
    if mode == "capacity100":
        return (
            float(metrics["representable_set_exact_rate"]),
            float(metrics["representable_group_recall"]),
            float(metrics["representable_exactly_one_rate"]),
        )
    holdout = metrics["collision_holdout"]
    ordinary = metrics["ordinary_validation"]
    return (
        float(holdout["collision_known_count_cell_exact_rate"]),
        float(holdout["collision_known_count_recall"]),
        float(holdout["collision_secondary_known_count_recall"]),
        float(ordinary["known_gt_count_geometry_recall"]),
    )


def capacity_checks(metrics: dict, frozen_unchanged: bool) -> dict:
    checks = {
        "representable_group_recall_at_least_0_98": (
            metrics["representable_group_recall"] >= 0.98
        ),
        "representable_exactly_one_at_least_0_97": (
            metrics["representable_exactly_one_rate"] >= 0.97
        ),
        "frozen_pipeline_bitwise_unchanged": frozen_unchanged,
    }
    return {**checks, "go": all(checks.values())}


def generalization_checks(
    initial: dict, best: dict, frozen_unchanged: bool
) -> dict:
    holdout = best["collision_holdout"]
    ordinary = best["ordinary_validation"]
    initial_ordinary = initial["ordinary_validation"]
    checks = {
        "collision_known_count_recall_at_least_0_25": (
            holdout["collision_known_count_recall"] >= 0.25
        ),
        "collision_recovers_at_least_0_55_of_ceiling": (
            holdout["collision_known_count_recall"]
            >= 0.55 * holdout["collision_candidate_recall"]
        ),
        "ordinary_recall_drop_at_most_0_005": (
            ordinary["known_gt_count_geometry_recall"]
            >= initial_ordinary["known_gt_count_geometry_recall"] - 0.005
        ),
        "ordinary_exact_drop_at_most_0_005": (
            ordinary["known_gt_count_image_exact"]
            >= initial_ordinary["known_gt_count_image_exact"] - 0.005
        ),
        "candidate_ceiling_unchanged": (
            holdout["candidate_ceiling_recall"]
            == initial["collision_holdout"]["candidate_ceiling_recall"]
            and ordinary["candidate_ceiling_recall"]
            == initial_ordinary["candidate_ceiling_recall"]
        ),
        "frozen_pipeline_bitwise_unchanged": frozen_unchanged,
    }
    return {**checks, "go": all(checks.values())}


def generalization_checkpoint_eligible(initial: dict, metrics: dict) -> bool:
    ordinary = metrics["ordinary_validation"]
    initial_ordinary = initial["ordinary_validation"]
    return (
        ordinary["known_gt_count_geometry_recall"]
        >= initial_ordinary["known_gt_count_geometry_recall"] - 0.005
        and ordinary["known_gt_count_image_exact"]
        >= initial_ordinary["known_gt_count_image_exact"] - 0.005
        and metrics["collision_holdout"]["candidate_ceiling_recall"]
        == initial["collision_holdout"]["candidate_ceiling_recall"]
        and ordinary["candidate_ceiling_recall"]
        == initial_ordinary["candidate_ceiling_recall"]
    )


def _checkpoint(
    selector: BondGroupPresenceSelector,
    config: BondGroupPresenceConfig,
    args,
    *,
    samples_seen: int,
    metrics: dict,
) -> dict:
    return {
        "schema_version": "bond-group-presence-checkpoint-0.1",
        "bond_group_presence": selector.state_dict(),
        "config": asdict(config),
        "samples_seen": samples_seen,
        "metrics": metrics,
        "training_args": vars(args),
        "bond_group_query_checkpoint": args.group_checkpoint,
        "geometry_frozen": True,
        "complete_gt_count_loss_enabled": False,
        "type_loss_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v0.8.33 local-evidence Bond presence ranking"
    )
    parser.add_argument(
        "--mode", choices=("capacity100", "generalization"), default="capacity100"
    )
    parser.add_argument("--capacity-exposure-control-4k", action="store_true")
    parser.add_argument("--group-checkpoint", default=DEFAULT_GROUP_CHECKPOINT)
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=[
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"
        ],
    )
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--set-layers", type=int, default=0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--exactly-one-weight", type=float, default=1.0)
    parser.add_argument("--representative-weight", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    apply_mode_protocol(args)
    positive = (
        args.batch_size,
        args.learning_rate,
        args.queries,
        args.grid_size,
        args.tolerance_pixels,
        args.log_every,
        args.max_grad_norm,
    )
    if (
        min(positive) <= 0
        or args.weight_decay < 0.0
        or args.set_layers < 0
        or min(
            args.positive_weight,
            args.negative_weight,
            args.exactly_one_weight,
            args.representative_weight,
            args.margin_weight,
            args.margin,
        )
        < 0.0
    ):
        raise ValueError("Bond Group presence training arguments are invalid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    _prepare_output_directory(output_dir)
    collision_train, collision_holdout, ordinary_replay, split = build_collision_split(
        args.train_shards,
        grid_size=args.grid_size,
        collision_train_images=args.collision_train_images,
        collision_holdout_images=args.collision_holdout_images,
        ordinary_replay_images=args.ordinary_replay_images,
        seed=args.seed,
    )
    expected_split = json.loads(Path(args.expected_split).read_text(encoding="utf-8"))
    if split != expected_split:
        raise RuntimeError("Bond Group presence split differs from v0.8.32")
    ordinary_validation = (
        load_records(args.validation_shards, args.validation_samples)
        if args.mode == "generalization"
        else []
    )
    train_records = (
        collision_train + ordinary_replay
        if args.mode == "generalization"
        else collision_train
    )
    exposure_order = (
        mixed_exposure_order(
            len(collision_train),
            len(ordinary_replay),
            args.train_exposures,
            args.batch_size,
            args.seed,
        )
        if args.mode == "generalization"
        else capacity_exposure_order(
            len(train_records), args.train_exposures, args.batch_size, args.seed
        )
    )

    device = torch.device(args.device)
    vision, dense, group, group_payload = _load_frozen_pipeline(
        Path(args.group_checkpoint), device
    )
    group_config = BondGroupQueryConfig(**group_payload["config"])
    patch_size = 2 * group_config.endpoint_patch_radius_cells + 1
    config = BondGroupPresenceConfig(
        endpoint_local_dim=(
            group_config.endpoint_local_channels * patch_size * patch_size
        ),
        endpoint_ink_dim=group_config.endpoint_ink_line_feature_dim,
        object_dim=group_config.d_model,
        set_layers=args.set_layers,
    )
    selector = BondGroupPresenceSelector(config).to(device)
    collator = ChemSceneBondCollator()
    frozen_initial = {
        "vision": _state_dict_sha256(vision),
        "dense": _state_dict_sha256(dense),
        "group": _state_dict_sha256(group),
    }

    def evaluate_current() -> dict:
        if args.mode == "capacity100":
            return evaluate_records(
                vision,
                dense,
                group,
                selector,
                collision_train,
                collator,
                device,
                batch_size=args.batch_size,
                queries=args.queries,
                grid_size=args.grid_size,
                tolerance_pixels=args.tolerance_pixels,
            )
        return {
            "collision_holdout": evaluate_records(
                vision,
                dense,
                group,
                selector,
                collision_holdout,
                collator,
                device,
                batch_size=args.batch_size,
                queries=args.queries,
                grid_size=args.grid_size,
                tolerance_pixels=args.tolerance_pixels,
            ),
            "ordinary_validation": evaluate_records(
                vision,
                dense,
                group,
                selector,
                ordinary_validation,
                collator,
                device,
                batch_size=args.batch_size,
                queries=args.queries,
                grid_size=args.grid_size,
                tolerance_pixels=args.tolerance_pixels,
            ),
        }

    parity = audit_zero_step_parity(
        vision,
        dense,
        group,
        selector,
        collision_train,
        collator,
        device,
        batch_size=args.batch_size,
        queries=args.queries,
    )
    if parity["status"] != "passed":
        raise RuntimeError("Bond Group presence zero-step parity failed")
    initial_metrics = evaluate_current()
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "zero_step_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    initial_checkpoint = _checkpoint(
        selector, config, args, samples_seen=0, metrics=initial_metrics
    )
    torch.save(initial_checkpoint, output_dir / "initial.pt")
    shutil.copy2(output_dir / "initial.pt", output_dir / "best.pt")

    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = [{"samples_seen": 0, "metrics": initial_metrics}]
    best_score = metric_score(initial_metrics, args.mode)
    best_samples_seen = 0
    running_loss = 0.0
    running_batches = 0
    next_validation = args.validate_every
    for start in range(0, len(exposure_order), args.batch_size):
        indices = exposure_order[start : start + args.batch_size]
        batch = collator([train_records[index] for index in indices])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        group_output, proposals = _frozen_group_outputs(
            vision, dense, group, pixels, queries=args.queries
        )
        labels = build_geometry_equivalence_labels(
            group_output["endpoints"],
            group_output["proposal_mask"],
            targets,
            tolerance_pixels=args.tolerance_pixels,
            image_size=IMAGE_SIZE,
        )
        selector.train()
        with _autocast(device):
            output = selector(group_output, proposals)
            losses = bond_group_presence_loss(
                output,
                labels,
                positive_weight=args.positive_weight,
                negative_weight=args.negative_weight,
                exactly_one_weight=args.exactly_one_weight,
                representative_weight=args.representative_weight,
                cardinality_weight=0.0,
                margin_weight=args.margin_weight,
                margin=args.margin,
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        samples_seen = start + len(indices)
        running_loss += float(losses["loss"].detach())
        running_batches += 1
        if samples_seen % args.log_every == 0:
            print(
                f"samples={samples_seen} loss={running_loss / running_batches:.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_batches = 0
        if samples_seen < next_validation and samples_seen != len(exposure_order):
            continue
        metrics = evaluate_current()
        history.append({"samples_seen": samples_seen, "metrics": metrics})
        checkpoint = _checkpoint(
            selector, config, args, samples_seen=samples_seen, metrics=metrics
        )
        torch.save(checkpoint, output_dir / "last.pt")
        torch.save(checkpoint, output_dir / f"checkpoint-{samples_seen:06d}.pt")
        score = metric_score(metrics, args.mode)
        eligible = (
            args.mode == "capacity100"
            or generalization_checkpoint_eligible(initial_metrics, metrics)
        )
        if eligible and score > best_score:
            best_score = score
            best_samples_seen = samples_seen
            shutil.copy2(output_dir / "last.pt", output_dir / "best.pt")
        current = (
            metrics
            if args.mode == "capacity100"
            else metrics["collision_holdout"]
        )
        print(
            f"validation samples={samples_seen} "
            f"group_recall={current['representable_group_recall']:.4f} "
            f"exactly_one={current['representable_exactly_one_rate']:.4f} "
            f"collision={current['collision_known_count_recall']:.4f} "
            f"cell={current['collision_known_count_cell_exact_rate']:.4f}",
            flush=True,
        )
        while next_validation <= samples_seen:
            next_validation += args.validate_every

    final_metrics = history[-1]["metrics"]
    best_metrics = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )["metrics"]
    frozen_final = {
        "vision": _state_dict_sha256(vision),
        "dense": _state_dict_sha256(dense),
        "group": _state_dict_sha256(group),
    }
    frozen_unchanged = frozen_initial == frozen_final
    frozen_audit = {
        "schema_version": "bond-group-presence-frozen-state-audit-0.1",
        "status": "passed" if frozen_unchanged else "failed",
        "initial": frozen_initial,
        "final": frozen_final,
    }
    (output_dir / "frozen_state_audit.json").write_text(
        json.dumps(frozen_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checks = (
        capacity_checks(best_metrics, frozen_unchanged)
        if args.mode == "capacity100"
        else generalization_checks(initial_metrics, best_metrics, frozen_unchanged)
    )
    summary = {
        "schema_version": "bond-group-presence-summary-0.1",
        "mode": args.mode,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_samples_seen": best_samples_seen,
        "history": history,
        "zero_step_parity": parity,
        "frozen_state_audit": frozen_audit,
        "go_checks": checks,
        "protocol": {
            **vars(args),
            "training_records": len(train_records),
            "optimizer_steps": len(exposure_order) // args.batch_size,
            "selector_parameters": sum(p.numel() for p in selector.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in selector.parameters() if p.requires_grad
            ),
            "group_checkpoint_sha256": sha256_file(Path(args.group_checkpoint)),
            "expected_split_sha256": sha256_file(Path(args.expected_split)),
            "actual_split_sha256": sha256_file(output_dir / "split.json"),
            "vision_frozen": True,
            "dense_detector_frozen": True,
            "group_geometry_frozen": True,
            "complete_gt_count_loss_enabled": False,
            "type_loss_enabled": False,
            "uses_mr_tadf_for_tuning": False,
            "uses_uspto_for_tuning": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
