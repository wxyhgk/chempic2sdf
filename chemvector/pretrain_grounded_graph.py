#!/usr/bin/env python3
"""Train visible-line graph feedback on frozen v0.7.26 grounded candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    GroundedGraphConfig,
    grounded_graph_loss,
    grounded_graph_targets,
    load_grounded_graph_checkpoint,
)
from chemvector.grounded_selector import (
    GroundedQuerySelector,
    GroundedSelectorConfig,
    decode_grounded_queries,
    fuse_dense_candidates_with_queries,
    sample_candidate_features,
)
from chemvector.pretrain_grounded_selector import load_frozen_pipeline
from chemvector.pretrain_proposal_graph import (
    edge_totals,
    summarize_edges,
    update_edge_metrics,
)
from chemvector.pretrain_vertex import (
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item) for item in value.split(","))
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be a comma-separated [0, 1] list")
    return thresholds


def load_frozen_grounded_pipeline(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("candidate_centric", False):
        raise ValueError("Grounded graph requires a candidate-centric selector")
    base_model, detector, _ = load_frozen_pipeline(
        Path(payload["query_checkpoint"]), device
    )
    selector_config = GroundedSelectorConfig(**payload["grounded_selector_config"])
    selector = GroundedQuerySelector(selector_config).to(device)
    selector.load_state_dict(payload["grounded_selector"])
    for module in (base_model, detector, selector):
        module.eval()
        module.requires_grad_(False)
    return base_model, detector, selector, payload


@torch.no_grad()
def frozen_candidates(
    base_model,
    detector,
    selector,
    pixels: torch.Tensor,
    *,
    dense_threshold: float,
    maximum_candidates: int = 128,
    candidate_nms_kernel_size: int = 3,
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        stroke_features = base_model.vision_encoder.forward_stroke_resolution(pixels)
        query_output = detector(stroke_features)
        dense_output = base_model.vertex_head(stroke_features)
    dense_candidates = decode_vertices(
        dense_output,
        threshold=dense_threshold,
        maximum_vertices=maximum_candidates,
        nms_kernel_size=candidate_nms_kernel_size,
    )
    grounded = fuse_dense_candidates_with_queries(
        query_output,
        dense_candidates,
        maximum_candidates=maximum_candidates,
        image_size=pixels.size(-1),
    )
    grounded["candidate_visual"] = sample_candidate_features(
        stroke_features,
        grounded["grounded_centers"],
        grounded["candidate_mask"],
    )
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        hidden = selector.encode(
            grounded["query_features"],
            grounded["scalar_features"],
            grounded["grounded_centers"],
            grounded["candidate_mask"],
        )
        logits = selector(
            grounded["query_features"],
            grounded["scalar_features"],
            grounded["grounded_centers"],
            grounded["base_logits"],
            grounded["candidate_mask"],
        )
    grounded["candidate_hidden"] = hidden
    grounded["selector_logits"] = logits
    return grounded


@torch.no_grad()
def evaluate(
    base_model,
    detector,
    selector,
    graph,
    loader,
    device: torch.device,
    *,
    samples: int,
    dense_threshold: float,
    edge_thresholds: tuple[float, ...],
    loss_kwargs: dict,
    candidate_nms_kernel_size: int,
) -> dict:
    graph.eval()
    node_totals = new_detection_totals()
    known_count_totals = new_detection_totals()
    candidate_oracle_totals = new_detection_totals()
    edge_metrics = {threshold: edge_totals() for threshold in edge_thresholds}
    losses = {name: 0.0 for name in ("loss", "node_loss", "edge_loss")}
    oracle_target_matches = 0
    oracle_target_exact = 0
    seen = 0
    for batch in loader:
        if seen >= samples:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        grounded = frozen_candidates(
            base_model,
            detector,
            selector,
            pixels,
            dense_threshold=dense_threshold,
            maximum_candidates=graph.config.maximum_candidates,
            candidate_nms_kernel_size=candidate_nms_kernel_size,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = graph(
                grounded["candidate_hidden"],
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                grounded["selector_logits"],
            )
        batch_losses = grounded_graph_loss(
            output,
            batch["anchor_points"].to(device),
            batch["anchor_mask"].to(device),
            batch["adjacency"].to(device),
            matching_radius_pixels=graph.config.matching_radius_pixels,
            image_size=pixels.size(-1),
            **loss_kwargs,
        )
        count = min(pixels.size(0), samples - seen)
        decoded = decode_grounded_queries(
            output["node_logits"][:count],
            grounded["grounded_centers"][:count],
            grounded["aromatic_logits"][:count],
            candidate_mask=grounded["candidate_mask"][:count],
        )
        expected_counts = batch["anchor_mask"][:count].sum(dim=1).long().to(device)
        known = decode_grounded_queries(
            output["node_logits"][:count],
            grounded["grounded_centers"][:count],
            grounded["aromatic_logits"][:count],
            counts=expected_counts,
            candidate_mask=grounded["candidate_mask"][:count],
        )
        oracle_targets, _, _ = grounded_graph_targets(
            output["points"][:count].detach().float(),
            output["candidate_mask"][:count],
            output["pair_indices"],
            batch["anchor_points"][:count].to(device),
            batch["anchor_mask"][:count].to(device),
            batch["adjacency"][:count].to(device),
            matching_radius_pixels=graph.config.matching_radius_pixels,
            image_size=pixels.size(-1),
        )
        candidate_oracle = decode_grounded_queries(
            (oracle_targets * 2.0 - 1.0) * 100.0,
            grounded["grounded_centers"][:count],
            grounded["aromatic_logits"][:count],
            counts=expected_counts,
            candidate_mask=grounded["candidate_mask"][:count],
        )
        oracle_counts = oracle_targets.sum(dim=1).long()
        truth_counts = batch["anchor_mask"][:count].sum(dim=1).long().to(device)
        oracle_target_matches += int(oracle_counts.sum())
        oracle_target_exact += int((oracle_counts == truth_counts).sum())
        update_detection_metrics(
            node_totals,
            decoded,
            batch["anchor_points"][:count],
            batch["anchor_mask"][:count],
            batch["aromatic_targets"][:count],
            pixels.size(-1),
            6.0,
        )
        update_detection_metrics(
            known_count_totals,
            known,
            batch["anchor_points"][:count],
            batch["anchor_mask"][:count],
            batch["aromatic_targets"][:count],
            pixels.size(-1),
            6.0,
        )
        update_detection_metrics(
            candidate_oracle_totals,
            candidate_oracle,
            batch["anchor_points"][:count],
            batch["anchor_mask"][:count],
            batch["aromatic_targets"][:count],
            pixels.size(-1),
            6.0,
        )
        for threshold in edge_thresholds:
            update_edge_metrics(
                edge_metrics[threshold],
                output["edge_logits"][:count],
                batch_losses["edge_targets"][:count],
                output["pair_mask"][:count],
                threshold,
            )
        for name in losses:
            losses[name] += float(batch_losses[name]) * count
        seen += count
    edge_grid = {
        f"{threshold:.4f}": summarize_edges(edge_metrics[threshold])
        for threshold in edge_thresholds
    }
    best_edge_threshold = max(
        edge_thresholds, key=lambda item: edge_grid[f"{item:.4f}"]["f1"]
    )
    return {
        "samples": seen,
        "set_count": summarize_detection_metrics(node_totals),
        "known_count": summarize_detection_metrics(known_count_totals),
        "candidate_oracle": summarize_detection_metrics(candidate_oracle_totals),
        "candidate_oracle_target_matches": oracle_target_matches,
        "candidate_oracle_target_exact_rate": oracle_target_exact / max(seen, 1),
        "matching_radius_pixels": graph.config.matching_radius_pixels,
        "edge_thresholds": edge_grid,
        "best_edge_threshold": best_edge_threshold,
        "best_edges": edge_grid[f"{best_edge_threshold:.4f}"],
        "losses": {name: value / max(seen, 1) for name, value in losses.items()},
    }


def validation_key(result: dict) -> tuple[float, ...]:
    return (
        result["set_count"]["vertex_exact_rate"],
        result["set_count"]["f1"],
        result["set_count"]["count_exact_rate"],
        result["best_edges"]["f1"],
    )


def save_checkpoint(path, graph, optimizer, args, *, samples_seen, validation) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "grounded_graph": graph.state_dict(),
            "grounded_graph_config": asdict(graph.config),
            "optimizer": optimizer.state_dict(),
            "grounded_selector_checkpoint": args.grounded_selector_checkpoint,
            "samples_seen": samples_seen,
            "validation": validation,
            "training_kind": "grounded_candidate_visible_line_graph",
            "dense_threshold": args.dense_threshold,
            "candidate_nms_kernel_size": args.candidate_nms_kernel_size,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-selector-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--maximum-candidates", type=int, default=128)
    parser.add_argument("--candidate-nms-kernel-size", type=int, default=3)
    parser.add_argument("--matching-radius-pixels", type=float, default=16.0)
    parser.add_argument("--blocker-radius-pixels", type=float, default=8.0)
    parser.add_argument("--node-residual-scale", type=float)
    parser.add_argument("--edge-warmup-samples", type=int, default=0)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--set-count-weight", type=float, default=1.0)
    parser.add_argument("--cardinality-weight", type=float, default=0.1)
    parser.add_argument("--whole-set-weight", type=float, default=0.0)
    parser.add_argument("--whole-set-margin", type=float, default=1.0)
    parser.add_argument("--init-grounded-graph-checkpoint")
    parser.add_argument("--node-only", action="store_true")
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grounded graph training")
    if min(
        args.train_samples,
        args.validation_samples,
        args.epochs,
        args.batch_size,
        args.negative_ratio,
    ) < 1:
        raise ValueError("Grounded graph training counts must be positive")
    if args.edge_warmup_samples < 0 or args.edge_warmup_samples >= args.train_samples:
        if args.edge_warmup_samples != 0:
            raise ValueError("Edge warmup must be nonnegative and shorter than training")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base_model, detector, selector, _ = load_frozen_grounded_pipeline(
        Path(args.grounded_selector_checkpoint), device
    )
    config = GroundedGraphConfig(
        candidate_dim=selector.config.d_model,
        maximum_candidates=args.maximum_candidates,
        blocker_radius_pixels=args.blocker_radius_pixels,
        matching_radius_pixels=args.matching_radius_pixels,
        node_residual_scale=args.node_residual_scale,
    )
    graph = GroundedCandidateGraphRefiner(config).to(device)
    if args.init_grounded_graph_checkpoint:
        initial_graph, initial_payload = load_grounded_graph_checkpoint(
            args.init_grounded_graph_checkpoint, device
        )
        initial_config = asdict(initial_graph.config)
        target_config = asdict(config)
        for name in (
            "node_residual_scale",
            "matching_radius_pixels",
            "maximum_candidates",
        ):
            initial_config.pop(name)
            target_config.pop(name)
        if initial_config != target_config:
            raise ValueError("Initial grounded graph config does not match")
        if Path(initial_payload["grounded_selector_checkpoint"]).resolve() != Path(
            args.grounded_selector_checkpoint
        ).resolve():
            raise ValueError("Initial graph and grounded selector checkpoints differ")
        graph.load_state_dict(initial_graph.state_dict())
    if args.node_only:
        if not args.init_grounded_graph_checkpoint:
            raise ValueError("Node-only training requires an initial grounded graph")
        for parameter in graph.parameters():
            parameter.requires_grad_(False)
        for module in (graph.edge_message, graph.node_residual):
            module.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in graph.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def loader(shards, reject_file, *, shuffle):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                reject_keys=load_reject_keys(reject_file),
                sample_shuffle_buffer=1000 if shuffle else 0,
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    train_loader = loader(args.train_shards, args.train_reject_keys_file, shuffle=True)
    validation_loader = loader(
        args.validation_shards, args.validation_reject_keys_file, shuffle=False
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    loss_kwargs = {
        "edge_loss_weight": args.edge_loss_weight,
        "negative_ratio": args.negative_ratio,
        "ranking_weight": args.ranking_weight,
        "set_count_weight": args.set_count_weight,
        "cardinality_weight": args.cardinality_weight,
        "whole_set_weight": args.whole_set_weight,
        "whole_set_margin": args.whole_set_margin,
    }
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"trainable_parameters="
        f"{sum(p.numel() for p in graph.parameters() if p.requires_grad):,}"
    )
    initial = evaluate(
        base_model,
        detector,
        selector,
        graph,
        validation_loader,
        device,
        samples=args.validation_samples,
        dense_threshold=args.dense_threshold,
        edge_thresholds=args.edge_thresholds,
        loss_kwargs=loss_kwargs,
        candidate_nms_kernel_size=args.candidate_nms_kernel_size,
    )
    print("initial=" + json.dumps(initial), flush=True)

    samples_seen = 0
    best_score = (-1.0, -1.0, -1.0, -1.0)
    started = time.perf_counter()
    running = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0}
    running_samples = 0
    next_validation = args.validate_every
    stop = False
    node_stage_started = args.edge_warmup_samples == 0 or args.node_only
    for _ in range(args.epochs):
        for batch in train_loader:
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            if pixels.size(0) > remaining:
                pixels = pixels[:remaining]
                for name in ("anchor_points", "anchor_mask", "adjacency"):
                    batch[name] = batch[name][:remaining]
            grounded = frozen_candidates(
                base_model,
                detector,
                selector,
                pixels,
                dense_threshold=args.dense_threshold,
                maximum_candidates=config.maximum_candidates,
                candidate_nms_kernel_size=args.candidate_nms_kernel_size,
            )
            graph.train()
            if not node_stage_started and samples_seen >= args.edge_warmup_samples:
                for parameter in graph.parameters():
                    parameter.requires_grad_(False)
                for module in (graph.edge_message, graph.node_residual):
                    module.requires_grad_(True)
                node_stage_started = True
                print(
                    f"node_stage_started_at={samples_seen} "
                    f"trainable_parameters={sum(p.numel() for p in graph.parameters() if p.requires_grad):,}",
                    flush=True,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = graph(
                    grounded["candidate_hidden"],
                    pixels,
                    grounded["grounded_centers"],
                    grounded["candidate_mask"],
                    grounded["selector_logits"],
                )
            losses = grounded_graph_loss(
                output,
                batch["anchor_points"].to(device),
                batch["anchor_mask"].to(device),
                batch["adjacency"].to(device),
                matching_radius_pixels=config.matching_radius_pixels,
                image_size=pixels.size(-1),
                **loss_kwargs,
            )
            training_loss = (
                args.edge_loss_weight * losses["edge_loss"]
                if not node_stage_started
                else losses["node_loss"]
            )
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += pixels.size(0)
            running_samples += pixels.size(0)
            for name in running:
                running[name] += float(losses[name].detach()) * pixels.size(0)
            if samples_seen // args.log_every != (
                samples_seen - pixels.size(0)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} "
                    + " ".join(
                        f"{name}={value / max(running_samples, 1):.6f}"
                        for name, value in running.items()
                    )
                    + f" samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                running_samples = 0
            if args.validate_every > 0 and samples_seen >= next_validation:
                validation = evaluate(
                    base_model,
                    detector,
                    selector,
                    graph,
                    validation_loader,
                    device,
                    samples=args.validation_samples,
                    dense_threshold=args.dense_threshold,
                    edge_thresholds=args.edge_thresholds,
                    loss_kwargs=loss_kwargs,
                    candidate_nms_kernel_size=args.candidate_nms_kernel_size,
                )
                print("validation=" + json.dumps(validation), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    graph,
                    optimizer,
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

    validation = evaluate(
        base_model,
        detector,
        selector,
        graph,
        validation_loader,
        device,
        samples=args.validation_samples,
        dense_threshold=args.dense_threshold,
        edge_thresholds=args.edge_thresholds,
        loss_kwargs=loss_kwargs,
        candidate_nms_kernel_size=args.candidate_nms_kernel_size,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        graph,
        optimizer,
        args,
        samples_seen=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
