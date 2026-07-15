#!/usr/bin/env python3
"""Train factorized edge and bond heads on frozen visual pair features."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.relation import (
    factorized_relation_loss,
    new_relation_totals,
    relation_teacher_consistency_loss,
    summarize_relation_metrics,
    update_relation_metrics,
)
from chemvector.stroke import StrokePairCollator
from chemvector.train import expand_shards, iter_mixed_batches


def initialize_relation_from_stroke(model: ChemVectorModel) -> None:
    """Warm-start the new heads from the old five-class stroke classifier."""

    if model.stroke_head is None or model.relation_head is None:
        raise ValueError("Stroke and relation heads are required")
    old_classifier = model.stroke_head.classifier[3]
    relation = model.relation_head
    with torch.no_grad():
        torch.nn.init.eye_(relation.shared[0].weight)
        relation.shared[0].bias.zero_()
        relation.bond.weight.copy_(old_classifier.weight[1:4])
        relation.bond.bias.copy_(old_classifier.bias[1:4])
        relation.edge.weight.copy_(
            old_classifier.weight[1:].mean(dim=0, keepdim=True)
            - old_classifier.weight[0:1]
        )
        relation.edge.bias.copy_(
            old_classifier.bias[1:].mean(dim=0, keepdim=True)
            - old_classifier.bias[0:1]
        )


def load_relation_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    stroke_band: int,
    stroke_band_radius: float,
    residual_adapter: bool = False,
    adapter_dim: int = 32,
) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    values = {key: value for key, value in checkpoint["config"].items() if key in valid_keys}
    values.update(
        stroke_enabled=True,
        relation_enabled=True,
        stroke_band=stroke_band,
        stroke_band_radius=stroke_band_radius,
    )
    if residual_adapter:
        values["structure_residual_adapter"] = True
        values["structure_adapter_dim"] = adapter_dim
    model = ChemVectorModel(ChemVectorConfig(**values))
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key
        for key in model.state_dict()
        if key.startswith("relation_head.")
        or ".pair_feature_adapter." in key
        or ".output_adapter." in key
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Relation checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    base_relation_missing = {
        key
        for key in missing
        if key.startswith("relation_head.") and ".logit_adapter." not in key
    }
    if base_relation_missing:
        initialize_relation_from_stroke(model)
    return model.to(device)


def set_relation_trainable(
    model: ChemVectorModel,
    *,
    train_pair_encoder: bool = False,
    adapter_only: bool = False,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.relation_head is None:
        raise ValueError("Relation head is disabled")
    if adapter_only:
        modules = (
            model.stroke_head.pair_feature_adapter,
            model.relation_head.logit_adapter,
        )
        if any(module is None for module in modules):
            raise ValueError("Adapter-only relation training requires structure adapters")
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    else:
        for parameter in model.relation_head.parameters():
            parameter.requires_grad_(True)
    if train_pair_encoder and not adapter_only:
        for module in (
            model.stroke_head.line_encoder,
            model.stroke_head.geometry_encoder,
            model.stroke_head.classifier[0],
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def forward_relation_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
    *,
    teacher_stroke_head: torch.nn.Module | None = None,
    teacher_relation_head: torch.nn.Module | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor] | None,
]:
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    pairs = batch["pair_indices"].to(device, non_blocking=True)
    pair_mask = batch["pair_mask"].to(device, non_blocking=True)
    labels = batch["pair_labels"].to(device, non_blocking=True)[pair_mask]
    with torch.no_grad():
        stroke_features = model.vision_encoder.forward_stroke_resolution(pixels)
    pair_encoder_trainable = any(
        parameter.requires_grad for parameter in model.stroke_head.parameters()
    )
    if pair_encoder_trainable:
        pair_features = model.stroke_head.encode_pairs(
            stroke_features,
            anchors,
            pairs,
            pair_mask,
            pixels.shape[-2:],
        )
    else:
        with torch.no_grad():
            pair_features = model.stroke_head.encode_pairs(
                stroke_features,
                anchors,
                pairs,
                pair_mask,
                pixels.shape[-2:],
            )
    teacher_output = None
    if teacher_stroke_head is not None and teacher_relation_head is not None:
        with torch.no_grad():
            teacher_pair_features = teacher_stroke_head.encode_pairs(
                stroke_features,
                anchors,
                pairs,
                pair_mask,
                pixels.shape[-2:],
            )
            teacher_output = teacher_relation_head(teacher_pair_features)
    return model.relation_head(pair_features), labels, teacher_output


def compute_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    args: argparse.Namespace,
    teacher_output: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    losses = factorized_relation_loss(
        output["edge_logits"],
        output["bond_logits"],
        labels,
        edge_pos_weight=args.edge_pos_weight,
        edge_focal_gamma=args.edge_focal_gamma,
        bond_class_weights=tuple(args.bond_class_weights),
        edge_loss_weight=args.edge_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
    )
    if teacher_output is not None:
        teacher_losses = relation_teacher_consistency_loss(output, teacher_output)
        losses.update(teacher_losses)
        losses["loss"] = (
            losses["loss"]
            + args.teacher_consistency_weight * teacher_losses["teacher_loss"]
        )
    return losses


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    model.eval()
    totals = new_relation_totals()
    loss_totals = {name: 0.0 for name in ("loss", "edge_loss", "bond_loss")}
    batches = 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, labels, _ = forward_relation_batch(model, batch, device)
            losses = compute_loss(output, labels, args)
        update_relation_metrics(
            totals,
            output["edge_logits"],
            output["bond_logits"],
            labels,
        )
        for name in loss_totals:
            loss_totals[name] += float(losses[name])
        batches += 1
        if batches >= args.validation_batches:
            break
    result: dict[str, float | int] = summarize_relation_metrics(totals)
    result.update({name: total / max(batches, 1) for name, total in loss_totals.items()})
    result["batches"] = batches
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-checkpoint",
        default="outputs/checkpoints/chemvector-v5-vertex-10k/last.pt",
    )
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/validation-000000.tar"],
    )
    parser.add_argument("--replay-shards", nargs="+")
    parser.add_argument("--replay-probability", type=float, default=0.0)
    parser.add_argument("--output-dir", default="outputs/checkpoints/chemvector-v072-relation-5k")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--anchor-jitter-pixels", type=float, default=12.0)
    parser.add_argument("--stroke-band", type=int, default=9)
    parser.add_argument("--stroke-band-radius", type=float, default=12.0)
    parser.add_argument("--edge-pos-weight", type=float, default=4.0)
    parser.add_argument("--edge-focal-gamma", type=float, default=2.0)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bond-class-weights",
        type=float,
        nargs=3,
        default=(1.0, 2.0, 6.0),
        metavar=("SINGLE", "DOUBLE", "TRIPLE"),
    )
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--train-pair-encoder", action="store_true")
    parser.add_argument("--include-all-train-pairs", action="store_true")
    parser.add_argument("--include-all-validation-pairs", action="store_true")
    parser.add_argument("--residual-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--freeze-base-head", action="store_true")
    parser.add_argument("--teacher-consistency-weight", type=float, default=0.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for relation training")
    if args.train_samples < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Relation training sizes and learning rate must be positive")
    if not 0.0 <= args.replay_probability < 1.0:
        raise ValueError("replay-probability must be in [0, 1)")
    if args.replay_probability and not args.replay_shards:
        raise ValueError("replay-shards are required when replay is enabled")
    if args.adapter_dim < 1:
        raise ValueError("adapter-dim must be positive")
    if args.freeze_base_head and not args.residual_adapter:
        raise ValueError("Frozen base relation training requires a residual adapter")
    if args.freeze_base_head and args.train_pair_encoder:
        raise ValueError("Adapter-only training cannot open the base pair encoder")
    if args.teacher_consistency_weight < 0:
        raise ValueError("teacher-consistency-weight must be nonnegative")
    if args.teacher_consistency_weight and not args.replay_shards:
        raise ValueError("Teacher consistency requires replay shards")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_relation_model(
        checkpoint_path,
        device,
        stroke_band=args.stroke_band,
        stroke_band_radius=args.stroke_band_radius,
        residual_adapter=args.residual_adapter,
        adapter_dim=args.adapter_dim,
    )
    teacher_stroke_head = None
    teacher_relation_head = None
    if args.teacher_consistency_weight:
        teacher_stroke_head = deepcopy(model.stroke_head).eval().requires_grad_(False)
        teacher_relation_head = (
            deepcopy(model.relation_head).eval().requires_grad_(False)
        )
    trainable = set_relation_trainable(
        model,
        train_pair_encoder=args.train_pair_encoder,
        adapter_only=args.freeze_base_head,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.train_shards),
            sample_shuffle_buffer=args.sample_shuffle_buffer,
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=StrokePairCollator(
            negative_ratio=args.negative_ratio,
            anchor_jitter_pixels=args.anchor_jitter_pixels,
            seed=args.seed,
            include_all_pairs=args.include_all_train_pairs,
        ),
    )
    replay_loader = None
    if args.replay_shards:
        replay_loader = DataLoader(
            ChemVectorIterableDataset(
                expand_shards(args.replay_shards),
                sample_shuffle_buffer=args.sample_shuffle_buffer,
                sequence_version="0.4",
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            collate_fn=StrokePairCollator(
                negative_ratio=args.negative_ratio,
                anchor_jitter_pixels=args.anchor_jitter_pixels,
                seed=args.seed + 2,
                include_all_pairs=args.include_all_train_pairs,
            ),
        )
    validation_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.validation_shards),
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=StrokePairCollator(
            negative_ratio=args.negative_ratio,
            anchor_jitter_pixels=0.0,
            seed=args.seed + 1,
            include_all_pairs=args.include_all_validation_pairs,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.is_file():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"trainable_parameters={sum(parameter.numel() for parameter in trainable):,}")
    model.eval()
    if args.freeze_base_head:
        model.stroke_head.pair_feature_adapter.train()
        model.relation_head.logit_adapter.train()
    elif args.train_pair_encoder:
        model.stroke_head.train()
        model.relation_head.train()
    else:
        model.relation_head.train()
    totals = new_relation_totals()
    running_names = ["loss", "edge_loss", "bond_loss"]
    if teacher_relation_head is not None:
        running_names.extend(
            ("teacher_loss", "teacher_edge_loss", "teacher_bond_loss")
        )
    running_losses = {name: 0.0 for name in running_names}
    interval_batches = 0
    replay_batches = 0
    samples = 0
    started = time.perf_counter()
    training_batches = iter_mixed_batches(
        train_loader,
        replay_loader,
        args.replay_probability,
        rng=random.Random(args.seed),
    )
    for step, (batch, is_replay) in enumerate(training_batches, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, labels, teacher_output = forward_relation_batch(
                model,
                batch,
                device,
                teacher_stroke_head=teacher_stroke_head if is_replay else None,
                teacher_relation_head=teacher_relation_head if is_replay else None,
            )
            losses = compute_loss(output, labels, args, teacher_output)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name in running_losses:
            if name in losses:
                running_losses[name] += float(losses[name].detach())
        interval_batches += 1
        replay_batches += int(is_replay)
        samples += batch["pixel_values"].size(0)
        update_relation_metrics(
            totals,
            output["edge_logits"].detach(),
            output["bond_logits"].detach(),
            labels,
        )
        if step % args.log_every == 0:
            metrics = summarize_relation_metrics(totals)
            teacher_text = (
                f"teacher_loss={running_losses['teacher_loss'] / interval_batches:.6f} "
                if "teacher_loss" in running_losses
                else ""
            )
            print(
                f"step={step} samples={samples} "
                f"loss={running_losses['loss'] / interval_batches:.6f} "
                f"edge_loss={running_losses['edge_loss'] / interval_batches:.6f} "
                f"bond_loss={running_losses['bond_loss'] / interval_batches:.6f} "
                f"edge_f1={metrics['edge_f1']:.6f} "
                f"visible_bond_accuracy={metrics['visible_bond_accuracy']:.6f} "
                + teacher_text
                +
                f"replay_batches={replay_batches} "
                f"samples_per_second={samples / (time.perf_counter() - started):.2f}"
            )
            totals = new_relation_totals()
            running_losses = {name: 0.0 for name in running_names}
            interval_batches = 0
        if samples >= args.train_samples:
            break

    validation = validate(model, validation_loader, device, args)
    print("validation=" + json.dumps(validation, ensure_ascii=False))
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": step,
            "training_kind": "factorized_relation_pretraining",
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
