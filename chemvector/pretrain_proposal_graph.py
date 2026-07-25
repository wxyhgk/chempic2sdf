#!/usr/bin/env python3
"""Train joint node-edge reasoning over frozen high-recall proposals."""

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
from chemvector.pretrain_proposal_keep import forward_dense, parse_thresholds
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
    vertex_metric_key,
)
from chemvector.proposal_graph import (
    ProposalGraphCollator,
    ProposalGraphConfig,
    ProposalGraphRefiner,
    decode_proposal_graph,
    proposal_graph_loss,
)
from chemvector.proposal_keep import load_proposal_keep_checkpoint
from chemvector.train import expand_shards, iter_mixed_batches


def edge_totals() -> dict[str, int]:
    return {"true_positives": 0, "predictions": 0, "targets": 0}


def update_edge_metrics(
    totals: dict[str, int],
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> None:
    predictions = logits.sigmoid() >= threshold
    labels = targets >= 0.5
    totals["true_positives"] += int((predictions & labels & mask).sum())
    totals["predictions"] += int((predictions & mask).sum())
    totals["targets"] += int((labels & mask).sum())


def summarize_edges(totals: dict[str, int]) -> dict[str, float | int]:
    precision = totals["true_positives"] / max(totals["predictions"], 1)
    recall = totals["true_positives"] / max(totals["targets"], 1)
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def balanced_validation_key(primary: dict, retention: dict | None) -> tuple[float, ...]:
    """Rank checkpoints without allowing a new style domain to hide old-domain loss."""

    domains = [primary] if retention is None else [primary, retention]
    return (
        sum(item["best_6px"]["vertex_exact_rate"] for item in domains) / len(domains),
        min(item["best_6px"]["vertex_exact_rate"] for item in domains),
        sum(item["best_6px"]["count_exact_rate"] for item in domains) / len(domains),
        sum(item["best_6px"]["f1"] for item in domains) / len(domains),
        sum(item["best_edges"]["f1"] for item in domains) / len(domains),
    )


def initialize_graph_head(graph_head, checkpoint_path: str) -> dict:
    """Load shared v0.7.18 weights while allowing new global-only parameters."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_config = ProposalGraphConfig(**payload["proposal_graph_config"])
    target_config = graph_head.config
    compatible_fields = (
        "input_channels",
        "node_dim",
        "edge_dim",
        "maximum_proposals",
        "proposal_threshold",
        "proposal_nms_kernel_size",
        "matching_radius_pixels",
        "line_samples",
        "line_band",
        "line_band_radius_pixels",
        "incident_topk",
    )
    mismatched = [
        name
        for name in compatible_fields
        if getattr(source_config, name) != getattr(target_config, name)
    ]
    if mismatched:
        raise ValueError(f"Initial proposal graph config differs for: {mismatched}")
    incompatible = graph_head.load_state_dict(payload["proposal_graph"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Initial proposal graph has unexpected tensors: "
            f"{sorted(incompatible.unexpected_keys)}"
        )
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("global_encoder.")
    ]
    if invalid_missing:
        raise ValueError(
            f"Initial proposal graph is missing shared tensors: {sorted(invalid_missing)}"
        )
    return payload


@torch.no_grad()
def evaluate(
    dense_model,
    graph_head,
    loader,
    device,
    *,
    node_thresholds: tuple[float, ...],
    edge_thresholds: tuple[float, ...],
    maximum_batches: int,
    keep_head=None,
    edge_loss_weight: float = 1.0,
    negative_ratio: int = 8,
    ablate_line_evidence: bool = False,
) -> dict:
    dense_model.eval()
    graph_head.eval()
    if keep_head is not None:
        keep_head.eval()
    node_totals = {
        (threshold, tolerance): new_detection_totals()
        for threshold in node_thresholds
        for tolerance in (6.0, 16.0)
    }
    edges = {threshold: edge_totals() for threshold in edge_thresholds}
    losses_total = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0}
    proposal_recall = 0.0
    selected_edges = 0.0
    batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        anchors = batch["anchor_points"].to(device, non_blocking=True)
        mask = batch["anchor_mask"].to(device, non_blocking=True)
        adjacency = batch["adjacency"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features, dense_output = forward_dense(dense_model, pixels)
            base_keep = keep_head(features, dense_output) if keep_head is not None else None
            output = graph_head(
                features,
                torch.ones_like(pixels) if ablate_line_evidence else pixels,
                dense_output,
                base_keep_logits=base_keep,
            )
            losses = proposal_graph_loss(
                output,
                anchors,
                mask,
                adjacency,
                matching_radius_pixels=graph_head.config.matching_radius_pixels,
                image_size=pixels.size(-1),
                edge_loss_weight=edge_loss_weight,
                negative_ratio=negative_ratio,
            )
        for threshold in node_thresholds:
            decoded = decode_proposal_graph(output, threshold=threshold)
            for tolerance in (6.0, 16.0):
                update_detection_metrics(
                    node_totals[(threshold, tolerance)],
                    decoded,
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    pixels.size(-1),
                    tolerance,
                )
        for threshold in edge_thresholds:
            update_edge_metrics(
                edges[threshold],
                output["edge_logits"],
                losses["edge_targets"],
                output["pair_mask"],
                threshold,
            )
        for name in losses_total:
            losses_total[name] += float(losses[name])
        proposal_recall += float(losses["proposal_recall"])
        selected_edges += float(losses["selected_edges"])
        batches += 1
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    grid = {
        f"{threshold:.4f}": {
            f"{tolerance:.4f}": summarize_detection_metrics(
                node_totals[(threshold, tolerance)]
            )
            for tolerance in (6.0, 16.0)
        }
        for threshold in node_thresholds
    }
    best_threshold = max(
        node_thresholds,
        key=lambda item: vertex_metric_key(grid[f"{item:.4f}"]["6.0000"]),
    )
    edge_grid = {
        f"{threshold:.4f}": summarize_edges(edges[threshold])
        for threshold in edge_thresholds
    }
    best_edge_threshold = max(
        edge_thresholds,
        key=lambda item: edge_grid[f"{item:.4f}"]["f1"],
    )
    return {
        **{name: total / max(batches, 1) for name, total in losses_total.items()},
        "proposal_recall": proposal_recall / max(batches, 1),
        "selected_edges_per_batch": selected_edges / max(batches, 1),
        "batches": batches,
        "node_thresholds": grid,
        "best_node_threshold": best_threshold,
        "best_6px": grid[f"{best_threshold:.4f}"]["6.0000"],
        "best_16px": grid[f"{best_threshold:.4f}"]["16.0000"],
        "edge_thresholds": edge_grid,
        "best_edge_threshold": best_edge_threshold,
        "best_edges": edge_grid[f"{best_edge_threshold:.4f}"],
        "line_evidence_ablated": ablate_line_evidence,
    }


def save_checkpoint(
    path,
    graph_head,
    optimizer,
    args,
    *,
    step,
    samples,
    validation,
    retention_validation=None,
):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "proposal_graph": graph_head.state_dict(),
            "proposal_graph_config": asdict(graph_head.config),
            "optimizer": optimizer.state_dict(),
            "dense_checkpoint": args.dense_checkpoint,
            "proposal_keep_checkpoint": args.proposal_keep_checkpoint,
            "initial_graph_checkpoint": args.init_graph_checkpoint,
            "global_step": step,
            "samples_seen": samples,
            "validation": validation,
            "retention_validation": retention_validation,
            "training_kind": "joint_node_edge_proposal_graph",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--proposal-keep-checkpoint")
    parser.add_argument("--init-graph-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--train-variants", nargs="+")
    parser.add_argument("--replay-shards", nargs="+")
    parser.add_argument("--replay-probability", type=float, default=0.0)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--retention-validation-shards", nargs="+")
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--replay-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--retention-validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--shard-shuffle", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, default=32)
    parser.add_argument("--maximum-proposals", type=int, default=192)
    parser.add_argument("--proposal-threshold", type=float, default=0.05)
    parser.add_argument("--proposal-nms-kernel-size", type=int, default=3)
    parser.add_argument("--matching-radius-pixels", type=float, default=16.0)
    parser.add_argument("--line-samples", type=int, default=24)
    parser.add_argument("--line-band", type=int, default=3)
    parser.add_argument("--line-band-radius-pixels", type=float, default=3.0)
    parser.add_argument("--incident-topk", type=int, default=4)
    parser.add_argument("--global-layers", type=int, default=0)
    parser.add_argument("--global-heads", type=int, default=4)
    parser.add_argument("--global-ffn-dim", type=int, default=256)
    parser.add_argument("--joint-node-edge-gradients", action="store_true")
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument(
        "--node-thresholds",
        type=parse_thresholds,
        default=parse_thresholds(
            "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.82,0.83,0.84,"
            "0.85,0.86,0.87,0.88,0.89,0.9,0.95"
        ),
    )
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.5,0.6,0.7,0.8,0.85,0.9,0.95"),
    )
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--evaluate-line-ablation", action="store_true")
    parser.add_argument("--validate-before-training", action="store_true")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for proposal graph training")
    if min(
        args.train_samples,
        args.epochs,
        args.batch_size,
        args.negative_ratio,
    ) < 1:
        raise ValueError("Training counts must be positive")
    if args.edge_loss_weight < 0:
        raise ValueError("edge-loss-weight must be nonnegative")
    if not 0.0 <= args.replay_probability < 1.0:
        raise ValueError("replay-probability must be in [0, 1)")
    if bool(args.replay_shards) != bool(args.replay_probability):
        raise ValueError("replay-shards and a positive replay-probability are required together")
    if args.resume_checkpoint and args.init_graph_checkpoint:
        raise ValueError("Resume and initial graph checkpoints are mutually exclusive")
    if bool(args.retention_validation_reject_keys_file) and not bool(
        args.retention_validation_shards
    ):
        raise ValueError("Retention reject keys require retention validation shards")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dense_model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
    dense_model.requires_grad_(False)
    keep_head = None
    if args.proposal_keep_checkpoint:
        keep_head, keep_payload = load_proposal_keep_checkpoint(
            args.proposal_keep_checkpoint,
            device,
        )
        if Path(keep_payload["dense_checkpoint"]).resolve() != Path(
            args.dense_checkpoint
        ).resolve():
            raise ValueError("Proposal keep and dense checkpoints do not match")
        keep_head.requires_grad_(False)
    config = ProposalGraphConfig(
        input_channels=dense_model.config.vision_dims[0],
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        maximum_proposals=args.maximum_proposals,
        proposal_threshold=args.proposal_threshold,
        proposal_nms_kernel_size=args.proposal_nms_kernel_size,
        matching_radius_pixels=args.matching_radius_pixels,
        line_samples=args.line_samples,
        line_band=args.line_band,
        line_band_radius_pixels=args.line_band_radius_pixels,
        incident_topk=args.incident_topk,
        global_layers=args.global_layers,
        global_heads=args.global_heads,
        global_ffn_dim=args.global_ffn_dim,
        joint_node_edge_gradients=args.joint_node_edge_gradients,
    )
    graph_head = ProposalGraphRefiner(config).to(device)
    if args.init_graph_checkpoint:
        initial_payload = initialize_graph_head(graph_head, args.init_graph_checkpoint)
        if Path(initial_payload["dense_checkpoint"]).resolve() != Path(
            args.dense_checkpoint
        ).resolve():
            raise ValueError("Initial proposal graph and dense checkpoints do not match")
        initial_keep = initial_payload.get("proposal_keep_checkpoint")
        requested_keep = args.proposal_keep_checkpoint
        if (Path(initial_keep).resolve() if initial_keep else None) != (
            Path(requested_keep).resolve() if requested_keep else None
        ):
            raise ValueError("Initial proposal graph and proposal keep checkpoints do not match")
    optimizer = torch.optim.AdamW(
        graph_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if ProposalGraphConfig(**resume_payload["proposal_graph_config"]) != config:
            raise ValueError("Resume proposal graph config does not match")
        if Path(resume_payload["dense_checkpoint"]).resolve() != Path(
            args.dense_checkpoint
        ).resolve():
            raise ValueError("Resume and dense checkpoints do not match")
        resumed_keep = resume_payload.get("proposal_keep_checkpoint")
        if (Path(resumed_keep).resolve() if resumed_keep else None) != (
            Path(args.proposal_keep_checkpoint).resolve()
            if args.proposal_keep_checkpoint
            else None
        ):
            raise ValueError("Resume and proposal keep checkpoints do not match")
        graph_head.load_state_dict(resume_payload["proposal_graph"])
        if not args.reset_optimizer:
            optimizer.load_state_dict(resume_payload["optimizer"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
                parameter_group["weight_decay"] = args.weight_decay

    def loader(
        shards,
        reject_file,
        *,
        shuffle,
        shard_shuffle=False,
        allowed_variants=None,
    ):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                shardshuffle=100 if shard_shuffle else False,
                sample_shuffle_buffer=args.sample_shuffle_buffer if shuffle else 0,
                sequence_version="0.4",
                reject_keys=load_reject_keys(reject_file),
                allowed_variants=(set(allowed_variants) if allowed_variants else None),
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    train_loader = loader(
        args.train_shards,
        args.train_reject_keys_file,
        shuffle=True,
        shard_shuffle=args.shard_shuffle,
        allowed_variants=args.train_variants,
    )
    replay_loader = (
        loader(
            args.replay_shards,
            args.replay_reject_keys_file,
            shuffle=True,
            shard_shuffle=args.shard_shuffle,
        )
        if args.replay_shards
        else None
    )
    validation_loader = loader(
        args.validation_shards,
        args.validation_reject_keys_file,
        shuffle=False,
    )
    retention_validation_loader = (
        loader(
            args.retention_validation_shards,
            args.retention_validation_reject_keys_file,
            shuffle=False,
        )
        if args.retention_validation_shards
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in graph_head.parameters()):,}")
    torch.cuda.reset_peak_memory_stats(device)
    samples = int(resume_payload["samples_seen"]) if resume_payload else 0
    step = int(resume_payload["global_step"]) if resume_payload else 0
    initial_samples = samples
    best_score = (-1.0, -1.0, -1.0, -1.0, float("-inf"))
    started = time.perf_counter()
    running = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0, "recall": 0.0}
    replay_samples = 0
    interval = 0
    stop = False
    if args.validate_before_training:
        initial_validation = evaluate(
            dense_model,
            graph_head,
            validation_loader,
            device,
            node_thresholds=args.node_thresholds,
            edge_thresholds=args.edge_thresholds,
            maximum_batches=args.validation_batches,
            keep_head=keep_head,
            edge_loss_weight=args.edge_loss_weight,
            negative_ratio=args.negative_ratio,
        )
        initial_retention = (
            evaluate(
                dense_model,
                graph_head,
                retention_validation_loader,
                device,
                node_thresholds=args.node_thresholds,
                edge_thresholds=args.edge_thresholds,
                maximum_batches=args.validation_batches,
                keep_head=keep_head,
                edge_loss_weight=args.edge_loss_weight,
                negative_ratio=args.negative_ratio,
            )
            if retention_validation_loader is not None
            else None
        )
        initial_payload = {
            "validation": initial_validation,
            "retention_validation": initial_retention,
            "selection_key": balanced_validation_key(
                initial_validation,
                initial_retention,
            ),
        }
        print("initial_validation=" + json.dumps(initial_payload))
        (output_dir / "initial_validation.json").write_text(
            json.dumps(initial_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for epoch in range(args.epochs):
        batches = iter_mixed_batches(
            train_loader,
            replay_loader,
            args.replay_probability,
            rng=random.Random(args.seed + epoch),
        )
        for batch, is_replay in batches:
            step += 1
            interval += 1
            graph_head.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            anchors = batch["anchor_points"].to(device, non_blocking=True)
            mask = batch["anchor_mask"].to(device, non_blocking=True)
            adjacency = batch["adjacency"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features, dense_output = forward_dense(dense_model, pixels)
                base_keep = keep_head(features, dense_output) if keep_head is not None else None
                output = graph_head(
                    features,
                    pixels,
                    dense_output,
                    base_keep_logits=base_keep,
                )
                losses = proposal_graph_loss(
                    output,
                    anchors,
                    mask,
                    adjacency,
                    matching_radius_pixels=config.matching_radius_pixels,
                    image_size=pixels.size(-1),
                    edge_loss_weight=args.edge_loss_weight,
                    negative_ratio=args.negative_ratio,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(graph_head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += int(pixels.size(0))
            if is_replay:
                replay_samples += int(pixels.size(0))
            running["loss"] += float(losses["loss"].detach())
            running["node_loss"] += float(losses["node_loss"].detach())
            running["edge_loss"] += float(losses["edge_loss"].detach())
            running["recall"] += float(losses["proposal_recall"])
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    f"loss={running['loss'] / interval:.6f} "
                    f"node_loss={running['node_loss'] / interval:.6f} "
                    f"edge_loss={running['edge_loss'] / interval:.6f} "
                    f"proposal_recall={running['recall'] / interval:.4f} "
                    f"samples_per_second="
                    f"{(samples - initial_samples) / (time.perf_counter() - started):.2f}",
                    f"replay_fraction="
                    f"{replay_samples / max(samples - initial_samples, 1):.4f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                interval = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                validation = evaluate(
                    dense_model,
                    graph_head,
                    validation_loader,
                    device,
                    node_thresholds=args.node_thresholds,
                    edge_thresholds=args.edge_thresholds,
                    maximum_batches=args.validation_batches,
                    keep_head=keep_head,
                    edge_loss_weight=args.edge_loss_weight,
                    negative_ratio=args.negative_ratio,
                )
                retention_validation = (
                    evaluate(
                        dense_model,
                        graph_head,
                        retention_validation_loader,
                        device,
                        node_thresholds=args.node_thresholds,
                        edge_thresholds=args.edge_thresholds,
                        maximum_batches=args.validation_batches,
                        keep_head=keep_head,
                        edge_loss_weight=args.edge_loss_weight,
                        negative_ratio=args.negative_ratio,
                    )
                    if retention_validation_loader is not None
                    else None
                )
                print(
                    "validation_step="
                    + str(step)
                    + " "
                    + json.dumps(
                        {
                            "validation": validation,
                            "retention_validation": retention_validation,
                        }
                    )
                )
                checkpoint = output_dir / f"checkpoint-{step:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    graph_head,
                    optimizer,
                    args,
                    step=step,
                    samples=samples,
                    validation=validation,
                    retention_validation=retention_validation,
                )
                score = balanced_validation_key(validation, retention_validation)
                if score > best_score:
                    best_score = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
            if samples >= args.train_samples:
                stop = True
                break
        if stop:
            break

    validation = evaluate(
        dense_model,
        graph_head,
        validation_loader,
        device,
        node_thresholds=args.node_thresholds,
        edge_thresholds=args.edge_thresholds,
        maximum_batches=args.validation_batches,
        keep_head=keep_head,
        edge_loss_weight=args.edge_loss_weight,
        negative_ratio=args.negative_ratio,
    )
    retention_validation = (
        evaluate(
            dense_model,
            graph_head,
            retention_validation_loader,
            device,
            node_thresholds=args.node_thresholds,
            edge_thresholds=args.edge_thresholds,
            maximum_batches=args.validation_batches,
            keep_head=keep_head,
            edge_loss_weight=args.edge_loss_weight,
            negative_ratio=args.negative_ratio,
        )
        if retention_validation_loader is not None
        else None
    )
    print("validation=" + json.dumps(validation))
    if retention_validation is not None:
        print("retention_validation=" + json.dumps(retention_validation))
    print(
        f"peak_cuda_memory_gib={torch.cuda.max_memory_allocated(device) / 2**30:.3f}"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if retention_validation is not None:
        (output_dir / "retention_validation.json").write_text(
            json.dumps(retention_validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    save_checkpoint(
        output_dir / "last.pt",
        graph_head,
        optimizer,
        args,
        step=step,
        samples=samples,
        validation=validation,
        retention_validation=retention_validation,
    )
    if args.evaluate_line_ablation:
        ablation = evaluate(
            dense_model,
            graph_head,
            validation_loader,
            device,
            node_thresholds=args.node_thresholds,
            edge_thresholds=args.edge_thresholds,
            maximum_batches=args.validation_batches,
            keep_head=keep_head,
            edge_loss_weight=args.edge_loss_weight,
            negative_ratio=args.negative_ratio,
            ablate_line_evidence=True,
        )
        print("line_ablation=" + json.dumps(ablation))
        (output_dir / "line_ablation.json").write_text(
            json.dumps(ablation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
