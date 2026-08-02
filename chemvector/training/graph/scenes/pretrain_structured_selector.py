#!/usr/bin/env python3
"""Train a cached whole-set selector over 256 high-recall grounded candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.grounded_graph import (
    GroundedCandidateGraphRefiner,
    grounded_graph_targets,
    load_grounded_graph_checkpoint,
)
from chemvector.grounded_selector import decode_grounded_queries
from chemvector.pretrain_grounded_graph import (
    frozen_candidates,
    load_frozen_grounded_pipeline,
)
from chemvector.pretrain_vertex import (
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
    structured_selector_loss,
)
from chemvector.train import expand_shards


def load_resized_graph(
    checkpoint: str,
    device: torch.device,
    maximum_candidates: int,
) -> tuple[GroundedCandidateGraphRefiner, dict]:
    source, payload = load_grounded_graph_checkpoint(checkpoint, device)
    config = replace(source.config, maximum_candidates=maximum_candidates)
    graph = GroundedCandidateGraphRefiner(config).to(device)
    graph.load_state_dict(source.state_dict())
    graph.eval().requires_grad_(False)
    return graph, payload


@torch.inference_mode()
def prepare_cache(
    base_model,
    detector,
    grounded_selector,
    graph,
    loader,
    device: torch.device,
    *,
    samples: int,
    dense_threshold: float,
    candidate_nms_kernel_size: int,
    matching_radius_pixels: float,
    cache_name: str,
) -> tuple[list[dict], torch.Tensor]:
    records = []
    pair_indices = None
    for batch in loader:
        if len(records) >= samples:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        grounded = frozen_candidates(
            base_model,
            detector,
            grounded_selector,
            pixels,
            dense_threshold=dense_threshold,
            maximum_candidates=graph.config.maximum_candidates,
            candidate_nms_kernel_size=candidate_nms_kernel_size,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            graph_output = graph(
                grounded["candidate_hidden"],
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                grounded["selector_logits"],
            )
        node_targets, _, _ = grounded_graph_targets(
            grounded["grounded_centers"].float(),
            grounded["candidate_mask"],
            graph_output["pair_indices"],
            batch["anchor_points"].to(device),
            batch["anchor_mask"].to(device),
            batch["adjacency"].to(device),
            matching_radius_pixels=matching_radius_pixels,
            image_size=pixels.size(-1),
        )
        current_pairs = graph_output["pair_indices"].cpu()
        if pair_indices is None:
            pair_indices = current_pairs
        elif not torch.equal(pair_indices, current_pairs):
            raise ValueError("Structured pair ordering changed inside the cache")
        for batch_index in range(pixels.size(0)):
            valid = batch["anchor_mask"][batch_index]
            records.append(
                {
                    "key": batch["keys"][batch_index],
                    "candidate_hidden": grounded["candidate_hidden"][batch_index]
                    .bfloat16()
                    .cpu(),
                    "base_logits": grounded["selector_logits"][batch_index]
                    .half()
                    .cpu(),
                    "points": grounded["grounded_centers"][batch_index].float().cpu(),
                    "candidate_mask": grounded["candidate_mask"][batch_index].cpu(),
                    "edge_logits": graph_output["edge_logits"][batch_index]
                    .half()
                    .cpu(),
                    "aromatic_logits": grounded["aromatic_logits"][batch_index]
                    .half()
                    .cpu(),
                    "candidate_auxiliary": grounded["candidate_auxiliary"][batch_index]
                    .half()
                    .cpu(),
                    "candidate_visual": grounded["candidate_visual"][batch_index]
                    .bfloat16()
                    .cpu(),
                    "targets": node_targets[batch_index].cpu(),
                    "target_count": int(valid.sum()),
                    "anchor_points": batch["anchor_points"][batch_index][valid].cpu(),
                    "aromatic_targets": batch["aromatic_targets"][batch_index][valid]
                    .cpu(),
                }
            )
            if len(records) >= samples:
                break
        if len(records) % 100 == 0:
            print(f"{cache_name}_cached={len(records)}", flush=True)
    if len(records) != samples or pair_indices is None:
        raise ValueError(f"Requested {samples} {cache_name} samples, found {len(records)}")
    return records, pair_indices


def stack_records(records: list[dict], indices: list[int], device: torch.device) -> dict:
    return {
        name: torch.stack([records[index][name] for index in indices]).to(
            device, non_blocking=True
        )
        for name in (
            "candidate_hidden",
            "base_logits",
            "points",
            "candidate_mask",
            "edge_logits",
            "aromatic_logits",
            "targets",
            "candidate_auxiliary",
            "candidate_visual",
        )
    }


@torch.no_grad()
def evaluate_cached(
    selector,
    records: list[dict],
    pair_indices: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    whole_set_weight: float,
    whole_set_margin: float,
    boundary_size: int,
    extreme_set_weight: float,
    extreme_set_margin: float,
) -> dict:
    selector.eval()
    selected_totals = new_detection_totals()
    oracle_totals = new_detection_totals()
    losses = {
        "loss": 0.0,
        "selection_loss": 0.0,
        "whole_set_loss": 0.0,
        "extreme_set_loss": 0.0,
    }
    ranking_exact = 0
    ranking_gaps = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_records(records, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = selector(
                batch["candidate_hidden"],
                batch["base_logits"],
                batch["points"],
                batch["candidate_mask"],
                batch["edge_logits"],
                pair_indices,
                batch["candidate_auxiliary"],
                batch["candidate_visual"],
            )
            batch_losses = structured_selector_loss(
                logits,
                batch["targets"],
                batch["candidate_mask"],
                whole_set_weight=whole_set_weight,
                whole_set_margin=whole_set_margin,
                boundary_size=boundary_size,
                extreme_set_weight=extreme_set_weight,
                extreme_set_margin=extreme_set_margin,
            )
        counts = torch.tensor(
            [records[index]["target_count"] for index in indices],
            device=device,
            dtype=torch.long,
        )
        selected = decode_grounded_queries(
            logits,
            batch["points"],
            batch["aromatic_logits"],
            counts=counts,
            candidate_mask=batch["candidate_mask"],
        )
        oracle = decode_grounded_queries(
            (batch["targets"] * 2.0 - 1.0) * 100.0,
            batch["points"],
            batch["aromatic_logits"],
            counts=counts,
            candidate_mask=batch["candidate_mask"],
        )
        for offset, count in enumerate(counts.tolist()):
            valid = batch["candidate_mask"][offset]
            targets = batch["targets"][offset].bool()
            selected_indices = logits[offset].masked_fill(~valid, -30.0).topk(count).indices
            ranking_exact += int(bool(targets[selected_indices].all()))
            positive_scores = logits[offset][valid & targets]
            negative_scores = logits[offset][valid & ~targets]
            if len(positive_scores) and len(negative_scores):
                ranking_gaps.append(
                    float((positive_scores.min() - negative_scores.max()).detach())
                )
        for name in losses:
            losses[name] += float(batch_losses[name]) * len(indices)
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            anchors = record["anchor_points"].unsqueeze(0)
            mask = torch.ones((1, anchors.size(1)), dtype=torch.bool)
            aromatic = record["aromatic_targets"].unsqueeze(0)
            for totals, decoded in (
                (selected_totals, selected[offset]),
                (oracle_totals, oracle[offset]),
            ):
                update_detection_metrics(
                    totals,
                    [decoded],
                    anchors,
                    mask,
                    aromatic,
                    768,
                    6.0,
                )
    return {
        "samples": len(records),
        "selected": summarize_detection_metrics(selected_totals),
        "candidate_oracle": summarize_detection_metrics(oracle_totals),
        "ranking": {
            "exact_rate": ranking_exact / max(len(records), 1),
            "mean_extreme_gap": sum(ranking_gaps) / max(len(ranking_gaps), 1),
            "positive_gap_rate": sum(gap > 0.0 for gap in ranking_gaps)
            / max(len(ranking_gaps), 1),
        },
        "losses": {
            name: value / max(len(records), 1) for name, value in losses.items()
        },
    }


def validation_key(result: dict) -> tuple[float, ...]:
    selected = result["selected"]
    return (
        selected["vertex_exact_rate"],
        selected["f1"],
        -selected["count_mae"],
    )


def save_checkpoint(path, selector, optimizer, config, args, *, samples_seen, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "structured_selector": selector.state_dict(),
            "structured_selector_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
            "samples_seen": samples_seen,
            "validation": validation,
            "dense_threshold": args.dense_threshold,
            "candidate_nms_kernel_size": args.candidate_nms_kernel_size,
            "matching_radius_pixels": args.matching_radius_pixels,
            "training_kind": "cached_high_recall_whole_set_selector",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--maximum-candidates", type=int, default=256)
    parser.add_argument("--dense-threshold", type=float, default=0.05)
    parser.add_argument("--candidate-nms-kernel-size", type=int, default=1)
    parser.add_argument("--matching-radius-pixels", type=float, default=6.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--incident-topk", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=12.0)
    parser.add_argument("--whole-set-weight", type=float, default=10.0)
    parser.add_argument("--whole-set-margin", type=float, default=1.0)
    parser.add_argument("--boundary-size", type=int, default=16)
    parser.add_argument("--extreme-set-weight", type=float, default=0.0)
    parser.add_argument("--extreme-set-margin", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for structured selector training")
    if min(
        args.train_cache_samples,
        args.validation_samples,
        args.train_samples,
        args.epochs,
        args.batch_size,
    ) < 1:
        raise ValueError("Structured selector training counts must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    graph, graph_payload = load_resized_graph(
        args.grounded_graph_checkpoint, device, args.maximum_candidates
    )
    base_model, detector, grounded_selector, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    config = StructuredSelectorConfig(
        candidate_dim=grounded_selector.config.d_model,
        auxiliary_dim=30,
        visual_dim=base_model.config.vision_dims[0] * 9,
        d_model=args.d_model,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
        maximum_candidates=args.maximum_candidates,
        incident_topk=args.incident_topk,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    )
    selector = GroundedStructuredSelector(config).to(device)
    if args.init_checkpoint:
        initial_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        initial_config = StructuredSelectorConfig(
            **initial_payload["structured_selector_config"]
        )
        if replace(initial_config, visual_dim=config.visual_dim) != config:
            raise ValueError("Initial structured selector config does not match")
        if Path(initial_payload["grounded_graph_checkpoint"]).resolve() != Path(
            args.grounded_graph_checkpoint
        ).resolve():
            raise ValueError("Initial selector and grounded graph checkpoints differ")
        if float(initial_payload["dense_threshold"]) != args.dense_threshold or int(
            initial_payload["candidate_nms_kernel_size"]
        ) != args.candidate_nms_kernel_size:
            raise ValueError("Initial selector candidate decoding config differs")
        initial_state = initial_payload["structured_selector"]
        if initial_config.visual_dim != config.visual_dim:
            initial_state = {
                name: value
                for name, value in initial_state.items()
                if not name.startswith("visual_projection.")
            }
        missing, unexpected = selector.load_state_dict(initial_state, strict=False)
        allowed_missing = {
            "visual_projection.0.weight",
            "visual_projection.0.bias",
            "visual_projection.1.weight",
            "visual_projection.1.bias",
        }
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"Initial selector tensors differ: missing={missing}, unexpected={unexpected}"
            )
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def loader(shards, reject_file):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                reject_keys=load_reject_keys(reject_file),
            ),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    train_cache, pair_indices = prepare_cache(
        base_model,
        detector,
        grounded_selector,
        graph,
        loader(args.train_shards, args.train_reject_keys_file),
        device,
        samples=args.train_cache_samples,
        dense_threshold=args.dense_threshold,
        candidate_nms_kernel_size=args.candidate_nms_kernel_size,
        matching_radius_pixels=args.matching_radius_pixels,
        cache_name="train",
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
        and load_reject_keys(args.train_reject_keys_file)
        == load_reject_keys(args.validation_reject_keys_file)
    ):
        validation_cache = train_cache
    else:
        validation_cache, validation_pairs = prepare_cache(
            base_model,
            detector,
            grounded_selector,
            graph,
            loader(args.validation_shards, args.validation_reject_keys_file),
            device,
            samples=args.validation_samples,
            dense_threshold=args.dense_threshold,
            candidate_nms_kernel_size=args.candidate_nms_kernel_size,
            matching_radius_pixels=args.matching_radius_pixels,
            cache_name="validation",
        )
        if not torch.equal(pair_indices, validation_pairs):
            raise ValueError("Train and validation pair ordering differs")
    pair_indices = pair_indices.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    initial = evaluate_cached(
        selector,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        whole_set_weight=args.whole_set_weight,
        whole_set_margin=args.whole_set_margin,
        boundary_size=args.boundary_size,
        extreme_set_weight=args.extreme_set_weight,
        extreme_set_margin=args.extreme_set_margin,
    )
    print("initial=" + json.dumps(initial), flush=True)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in selector.parameters()):,}")

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, float("-inf"))
    running = {
        "loss": 0.0,
        "selection_loss": 0.0,
        "whole_set_loss": 0.0,
        "extreme_set_loss": 0.0,
    }
    running_samples = 0
    started = time.perf_counter()
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
            selector.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = selector(
                    batch["candidate_hidden"],
                    batch["base_logits"],
                    batch["points"],
                    batch["candidate_mask"],
                    batch["edge_logits"],
                    pair_indices,
                    batch["candidate_auxiliary"],
                    batch["candidate_visual"],
                )
                losses = structured_selector_loss(
                    logits,
                    batch["targets"],
                    batch["candidate_mask"],
                    whole_set_weight=args.whole_set_weight,
                    whole_set_margin=args.whole_set_margin,
                    boundary_size=args.boundary_size,
                    extreme_set_weight=args.extreme_set_weight,
                    extreme_set_margin=args.extreme_set_margin,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_samples += len(indices)
            for name in running:
                running[name] += float(losses[name].detach()) * len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} "
                    + " ".join(
                        f"{name}={value / running_samples:.6f}"
                        for name, value in running.items()
                    )
                    + f" samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                running_samples = 0
            if args.validate_every > 0 and samples_seen >= next_validation:
                validation = evaluate_cached(
                    selector,
                    validation_cache,
                    pair_indices,
                    device,
                    batch_size=args.batch_size,
                    whole_set_weight=args.whole_set_weight,
                    whole_set_margin=args.whole_set_margin,
                    boundary_size=args.boundary_size,
                    extreme_set_weight=args.extreme_set_weight,
                    extreme_set_margin=args.extreme_set_margin,
                )
                print("validation=" + json.dumps(validation), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    selector,
                    optimizer,
                    config,
                    args,
                    samples_seen=samples_seen,
                    validation=validation,
                )
                if validation_key(validation) > best_score:
                    best_score = validation_key(validation)
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    validation = evaluate_cached(
        selector,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        whole_set_weight=args.whole_set_weight,
        whole_set_margin=args.whole_set_margin,
        boundary_size=args.boundary_size,
        extreme_set_weight=args.extreme_set_weight,
        extreme_set_margin=args.extreme_set_margin,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        selector,
        optimizer,
        config,
        args,
        samples_seen=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
