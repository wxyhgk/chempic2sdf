#!/usr/bin/env python3
"""Train full-adjacency node-edge denoising over dual vertex candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.dual_graph_denoiser import (
    DualGraphDenoiserConfig,
    GroundedDualGraphDenoiser,
)
from chemvector.dual_vertex_selector import grouped_candidate_selection_loss
from chemvector.edge_evidence import load_edge_teacher
from chemvector.pretrain_dual_vertex_selector import (
    evaluate_cached,
    load_dual_pipeline,
    prepare_cache,
    selector_forward,
    stack_records,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)
from chemvector.train import expand_shards


def best_deployed_threshold(result: dict) -> dict:
    threshold, metrics = max(
        result["real_thresholds"].items(),
        key=lambda item: (
            item[1]["exact_rate"],
            item[1]["count_exact_rate"],
            item[1]["f1"],
        ),
    )
    return {"threshold": float(threshold), **metrics}


def validation_key(result: dict) -> tuple[float, float, float]:
    deployed = best_deployed_threshold(result)
    return (
        deployed["exact_rate"],
        deployed["count_exact_rate"],
        deployed["f1"],
    )


def joint_edge_selection_loss(
    edge_logits: torch.Tensor,
    edge_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    negative_ratio: int,
) -> torch.Tensor:
    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    valid = candidate_mask[:, begin] & candidate_mask[:, end]
    positives = valid & (edge_targets > 0.5)
    negatives = valid & ~positives
    selected_logits = []
    selected_targets = []
    for current_logits, current_positives, current_negatives in zip(
        edge_logits, positives, negatives
    ):
        positive_indices = torch.where(current_positives)[0]
        negative_indices = torch.where(current_negatives)[0]
        negative_count = min(
            len(negative_indices), max(len(positive_indices) * negative_ratio, 32)
        )
        if negative_count:
            hard_offsets = current_logits[negative_indices].detach().topk(
                negative_count
            ).indices
            negative_indices = negative_indices[hard_offsets]
        chosen = torch.cat((positive_indices, negative_indices))
        if len(chosen):
            selected_logits.append(current_logits[chosen])
            selected_targets.append(current_positives[chosen].float())
    if not selected_logits:
        return edge_logits.sum() * 0.0
    logits = torch.cat(selected_logits)
    targets = torch.cat(selected_targets)
    positive_count = targets.sum().clamp_min(1.0)
    negative_count = (1.0 - targets).sum().clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=(negative_count / positive_count).clamp_max(32.0),
    )


def pass_through_auxiliary_loss(
    logits: torch.Tensor,
    line_interior_mask: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Train PASS_THROUGH only on explicit line interiors and protected REAL groups."""

    if not (
        logits.shape
        == line_interior_mask.shape
        == group_ids.shape
        == candidate_mask.shape
    ):
        raise ValueError("Pass-through supervision tensors must have matching shapes")
    labeled = candidate_mask & (line_interior_mask | (group_ids >= 0))
    if not bool(labeled.any()):
        return logits.sum() * 0.0
    targets = line_interior_mask[labeled].float()
    positive_count = targets.sum().clamp_min(1.0)
    negative_count = (1.0 - targets).sum().clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        logits[labeled],
        targets,
        pos_weight=(negative_count / positive_count).clamp_max(32.0),
    )


def save_checkpoint(path, model, optimizer, config, args, samples_seen, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "dual_graph_denoiser": model.state_dict(),
            "dual_graph_denoiser_config": asdict(config),
            "base_selector_checkpoint": args.base_selector_checkpoint,
            "edge_teacher_checkpoint": args.edge_teacher_checkpoint,
            "optimizer": optimizer.state_dict(),
            "samples_seen": samples_seen,
            "validation": validation,
            "training_kind": "dual_vertex_full_adjacency_graph_denoiser",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selector-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--edge-dim", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=8.0)
    parser.add_argument("--edge-residual-scale", type=float, default=8.0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-negative-ratio", type=int, default=8)
    parser.add_argument("--endpoint-ray-features", action="store_true")
    parser.add_argument("--straight-through-topk", type=int, default=0)
    parser.add_argument(
        "--patch-mode",
        choices=("none", "feature", "pixel", "semantic_pixel"),
        default="none",
    )
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-residual-scale", type=float, default=4.0)
    parser.add_argument("--group-ranking-weight", type=float, default=4.0)
    parser.add_argument("--group-ranking-margin", type=float, default=1.0)
    parser.add_argument("--group-background-weight", type=float, default=1.0)
    parser.add_argument("--absolute-boundary-weight", type=float, default=1.0)
    parser.add_argument("--positive-floor-probability", type=float, default=0.8)
    parser.add_argument("--background-ceiling-probability", type=float, default=0.2)
    parser.add_argument("--hard-background-count", type=int, default=16)
    parser.add_argument("--line-interior-ranking-weight", type=float, default=0.0)
    parser.add_argument("--line-interior-ranking-margin", type=float, default=1.0)
    parser.add_argument("--line-interior-distance-pixels", type=float, default=3.0)
    parser.add_argument(
        "--line-interior-endpoint-distance-pixels", type=float, default=8.0
    )
    parser.add_argument(
        "--line-interior-ambiguous-atom-radius-pixels", type=float, default=10.0
    )
    parser.add_argument("--pass-through-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--edge-teacher-checkpoint")
    parser.add_argument("--edge-teacher-chunk-size", type=int, default=2048)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for graph denoiser training")
    if args.patch_mode == "semantic_pixel" and args.pass_through_auxiliary_weight <= 0.0:
        raise ValueError(
            "Semantic pixel mode requires a positive pass-through auxiliary weight"
        )
    if args.patch_mode != "semantic_pixel" and args.pass_through_auxiliary_weight > 0.0:
        raise ValueError(
            "Pass-through auxiliary loss requires semantic pixel patch mode"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    base_payload = torch.load(
        args.base_selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector_config = StructuredSelectorConfig(
        **base_payload["structured_selector_config"]
    )
    base_selector = GroundedStructuredSelector(selector_config).to(device)
    base_selector.load_state_dict(base_payload["dual_vertex_selector"])
    base_selector.eval().requires_grad_(False)
    config = DualGraphDenoiserConfig(
        d_model=selector_config.d_model,
        edge_dim=args.edge_dim,
        layers=args.message_layers,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        refine_edges=args.edge_loss_weight > 0.0,
        edge_residual_scale=args.edge_residual_scale,
        edge_auxiliary_dim=4 if args.endpoint_ray_features else 0,
        straight_through_topk=args.straight_through_topk,
        patch_mode=args.patch_mode,
        patch_size=args.patch_size,
        patch_residual_scale=args.patch_residual_scale,
    )
    model = GroundedDualGraphDenoiser(base_selector, config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    base, detector, grounded_selector, graph, text_head, implicit_head = load_dual_pipeline(
        base_payload["text_object_checkpoint"],
        base_payload["implicit_vertex_checkpoint"],
        device,
        maximum_candidates=selector_config.maximum_candidates,
    )
    edge_teacher = (
        load_edge_teacher(args.edge_teacher_checkpoint, device)
        if args.edge_teacher_checkpoint
        else None
    )

    def loader(shards):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards), sequence_version="0.4"
            ),
            batch_size=args.cache_batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    cache_kwargs = {
        "text_candidate_threshold": 0.05,
        "text_filter_threshold": 0.30,
        "implicit_candidate_threshold": 0.05,
        "bbox_expansion_pixels": 8.0,
        "matching_radius_pixels": 6.0,
        "candidate_pixel_patch_size": (
            args.patch_size
            if args.patch_mode in {"pixel", "semantic_pixel"}
            else 0
        ),
        "include_line_interior_mask": (
            args.line_interior_ranking_weight > 0.0
            or args.pass_through_auxiliary_weight > 0.0
        ),
        "line_interior_distance_pixels": args.line_interior_distance_pixels,
        "line_interior_endpoint_distance_pixels": (
            args.line_interior_endpoint_distance_pixels
        ),
        "line_interior_ambiguous_atom_radius_pixels": (
            args.line_interior_ambiguous_atom_radius_pixels
        ),
    }
    train_cache, pair_indices = prepare_cache(
        base,
        detector,
        grounded_selector,
        graph,
        text_head,
        implicit_head,
        loader(args.train_shards),
        device,
        samples=args.train_cache_samples,
        cache_name="train",
        edge_teacher=edge_teacher,
        edge_teacher_chunk_size=args.edge_teacher_chunk_size,
        include_endpoint_ray_features=args.endpoint_ray_features,
        **cache_kwargs,
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
    ):
        validation_cache = train_cache
    else:
        validation_cache, validation_pairs = prepare_cache(
            base,
            detector,
            grounded_selector,
            graph,
            text_head,
            implicit_head,
            loader(args.validation_shards),
            device,
            samples=args.validation_samples,
            cache_name="validation",
            edge_teacher=edge_teacher,
            edge_teacher_chunk_size=args.edge_teacher_chunk_size,
            include_endpoint_ray_features=args.endpoint_ray_features,
            **cache_kwargs,
        )
        if not torch.equal(pair_indices, validation_pairs):
            raise ValueError("Train and validation pair ordering differs")
    pair_indices = pair_indices.to(device)
    loss_kwargs = {
        "whole_set_weight": 0.0,
        "whole_set_margin": 1.0,
        "boundary_size": 16,
        "extreme_set_weight": 0.0,
        "extreme_set_margin": 1.0,
        "group_ranking_weight": args.group_ranking_weight,
        "group_ranking_margin": args.group_ranking_margin,
        "group_background_weight": args.group_background_weight,
        "absolute_boundary_weight": args.absolute_boundary_weight,
        "positive_floor_probability": args.positive_floor_probability,
        "background_ceiling_probability": args.background_ceiling_probability,
        "hard_background_count": args.hard_background_count,
        "line_interior_weight": args.line_interior_ranking_weight,
        "line_interior_margin": args.line_interior_ranking_margin,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    initial = evaluate_cached(
        model,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        loss_kwargs=loss_kwargs,
        loss_mode="target_group",
    )
    print("initial=" + json.dumps(initial), flush=True)
    print(
        f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0)
    started = time.perf_counter()
    running_loss = 0.0
    running_samples = 0
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = stack_records(train_cache, indices, device)
            model.train()
            model.base_selector.eval()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                model_output = model.forward_with_edges(
                    batch["candidate_hidden"],
                    batch["base_logits"],
                    batch["points"],
                    batch["candidate_mask"],
                    batch["edge_logits"].detach(),
                    pair_indices,
                    batch["candidate_auxiliary"],
                    batch["candidate_visual"],
                    batch["candidate_roles"],
                    batch.get("edge_auxiliary"),
                    batch.get("candidate_pixel_patches"),
                )
                logits = model_output["node_logits"]
                losses = grouped_candidate_selection_loss(
                    logits,
                    batch["candidate_groups"],
                    batch["candidate_mask"],
                    ranking_weight=args.group_ranking_weight,
                    ranking_margin=args.group_ranking_margin,
                    background_weight=args.group_background_weight,
                    absolute_boundary_weight=args.absolute_boundary_weight,
                    positive_floor_probability=args.positive_floor_probability,
                    background_ceiling_probability=args.background_ceiling_probability,
                    hard_background_count=args.hard_background_count,
                    line_interior_mask=batch.get("line_interior_mask"),
                    line_interior_weight=args.line_interior_ranking_weight,
                    line_interior_margin=args.line_interior_ranking_margin,
                )
                edge_loss = joint_edge_selection_loss(
                    model_output["edge_logits"],
                    batch["edge_targets"],
                    batch["candidate_mask"],
                    pair_indices,
                    negative_ratio=args.edge_negative_ratio,
                )
                losses["edge_loss"] = edge_loss
                losses["loss"] = losses["loss"] + args.edge_loss_weight * edge_loss
                pass_through_loss = logits.sum() * 0.0
                if "pass_through_logits" in model_output:
                    if batch.get("line_interior_mask") is None:
                        raise ValueError(
                            "Semantic pixel mode requires line-interior supervision"
                        )
                    pass_through_loss = pass_through_auxiliary_loss(
                        model_output["pass_through_logits"],
                        batch["line_interior_mask"],
                        batch["candidate_groups"],
                        batch["candidate_mask"],
                    )
                losses["pass_through_auxiliary_loss"] = pass_through_loss
                losses["loss"] = (
                    losses["loss"]
                    + args.pass_through_auxiliary_weight * pass_through_loss
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_loss += float(losses["loss"].detach()) * len(indices)
            running_samples += len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / max(running_samples, 1):.6f} "
                    f"samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                validation = evaluate_cached(
                    model,
                    validation_cache,
                    pair_indices,
                    device,
                    batch_size=args.batch_size,
                    loss_kwargs=loss_kwargs,
                    loss_mode="target_group",
                )
                validation["best_deployed"] = best_deployed_threshold(validation)
                print("validation=" + json.dumps(validation), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    config,
                    args,
                    samples_seen,
                    validation,
                )
                if validation_key(validation) > best_score:
                    best_score = validation_key(validation)
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break
    validation = evaluate_cached(
        model,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        loss_kwargs=loss_kwargs,
        loss_mode="target_group",
    )
    validation["best_deployed"] = best_deployed_threshold(validation)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        config,
        args,
        samples_seen,
        validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
