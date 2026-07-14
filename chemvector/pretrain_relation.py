#!/usr/bin/env python3
"""Train factorized edge and bond heads on frozen visual pair features."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.relation import (
    factorized_relation_loss,
    new_relation_totals,
    summarize_relation_metrics,
    update_relation_metrics,
)
from chemvector.stroke import StrokePairCollator
from chemvector.train import expand_shards


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
    model = ChemVectorModel(ChemVectorConfig(**values))
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("relation_head.")
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Relation checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if missing:
        initialize_relation_from_stroke(model)
    return model.to(device)


def set_relation_trainable(
    model: ChemVectorModel,
    *,
    train_pair_encoder: bool = False,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.relation_head is None:
        raise ValueError("Relation head is disabled")
    for parameter in model.relation_head.parameters():
        parameter.requires_grad_(True)
    if train_pair_encoder:
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
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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
    return model.relation_head(pair_features), labels


def compute_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return factorized_relation_loss(
        output["edge_logits"],
        output["bond_logits"],
        labels,
        edge_pos_weight=args.edge_pos_weight,
        edge_focal_gamma=args.edge_focal_gamma,
        bond_class_weights=tuple(args.bond_class_weights),
        edge_loss_weight=args.edge_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
    )


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
            output, labels = forward_relation_batch(model, batch, device)
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
        default="outputs/chemvector-v5-vertex-10k/last.pt",
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
    parser.add_argument("--output-dir", default="outputs/chemvector-v072-relation-5k")
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for relation training")
    if args.train_samples < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Relation training sizes and learning rate must be positive")
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
    )
    trainable = set_relation_trainable(
        model,
        train_pair_encoder=args.train_pair_encoder,
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
    if args.train_pair_encoder:
        model.stroke_head.train()
    model.relation_head.train()
    totals = new_relation_totals()
    running_losses = {name: 0.0 for name in ("loss", "edge_loss", "bond_loss")}
    interval_batches = 0
    samples = 0
    started = time.perf_counter()
    for step, batch in enumerate(train_loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, labels = forward_relation_batch(model, batch, device)
            losses = compute_loss(output, labels, args)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name in running_losses:
            running_losses[name] += float(losses[name].detach())
        interval_batches += 1
        samples += batch["pixel_values"].size(0)
        update_relation_metrics(
            totals,
            output["edge_logits"].detach(),
            output["bond_logits"].detach(),
            labels,
        )
        if step % args.log_every == 0:
            metrics = summarize_relation_metrics(totals)
            print(
                f"step={step} samples={samples} "
                f"loss={running_losses['loss'] / interval_batches:.6f} "
                f"edge_loss={running_losses['edge_loss'] / interval_batches:.6f} "
                f"bond_loss={running_losses['bond_loss'] / interval_batches:.6f} "
                f"edge_f1={metrics['edge_f1']:.6f} "
                f"visible_bond_accuracy={metrics['visible_bond_accuracy']:.6f} "
                f"samples_per_second={samples / (time.perf_counter() - started):.2f}"
            )
            totals = new_relation_totals()
            running_losses = {
                name: 0.0 for name in ("loss", "edge_loss", "bond_loss")
            }
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
