#!/usr/bin/env python3
"""Pretrain the ChemVector OCR branch on generated chemical-label crops."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from functools import partial
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.ocr import OCRTextCodec
from chemvector.synthetic_ocr import (
    SyntheticOCRConfig,
    SyntheticOCRDataset,
    collate_synthetic_ocr,
)
from chemvector.train import ocr_sequence_accuracy, ocr_token_accuracy


def load_ocr_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    config_values = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in valid_keys
    }
    config_values["ocr_enabled"] = True
    config = ChemVectorConfig(**config_values)
    model = ChemVectorModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("ocr_head.")
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "OCR initialization checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device)


def set_ocr_trainable(
    model: ChemVectorModel,
    *,
    tune_vision: bool = False,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.ocr_head]
    if tune_vision:
        modules.extend(
            [
                model.vision_encoder.stem,
                *model.vision_encoder.stages[:3],
            ]
        )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_optimizer(
    model: ChemVectorModel,
    ocr_learning_rate: float,
    vision_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    ocr_parameters = list(model.ocr_head.parameters())
    ocr_ids = {id(parameter) for parameter in ocr_parameters}
    vision_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in ocr_ids
    ]
    parameter_groups = [
        {"params": ocr_parameters, "lr": ocr_learning_rate, "name": "ocr"}
    ]
    if vision_parameters:
        parameter_groups.append(
            {
                "params": vision_parameters,
                "lr": vision_learning_rate,
                "name": "vision",
            }
        )
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def forward_ocr_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    anchor_points = batch["ocr_anchor_points"].to(device, non_blocking=True)
    anchor_mask = batch["ocr_anchor_mask"].to(device, non_blocking=True)
    token_ids = batch["ocr_token_ids"].to(device, non_blocking=True)
    features = model.vision_encoder.forward_high_resolution(pixel_values)
    return model.ocr_head(
        features,
        anchor_points,
        anchor_mask,
        token_ids,
        pixel_values.shape[-2:],
    )


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    codec: OCRTextCodec,
) -> dict[str, float]:
    model.eval()
    loss = 0.0
    token_correct = token_count = 0
    totals = {
        "exact": 0,
        "anchors": 0,
        "visible_exact": 0,
        "visible_anchors": 0,
        "blank_exact": 0,
        "blank_anchors": 0,
    }
    batches = 0
    greedy_exact = greedy_visible_exact = greedy_blank_exact = 0
    greedy_visible = greedy_blank = 0
    category_totals: dict[str, int] = {}
    category_exact: dict[str, int] = {}
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_ocr_batch(model, batch, device)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            anchor_points = batch["ocr_anchor_points"].to(device, non_blocking=True)
            anchor_mask = batch["ocr_anchor_mask"].to(device, non_blocking=True)
            features = model.vision_encoder.forward_high_resolution(pixel_values)
            generated_ids = model.ocr_head.generate(
                features,
                anchor_points,
                anchor_mask,
                pixel_values.shape[-2:],
            )
        loss += float(output["ocr_loss"])
        correct, count = ocr_token_accuracy(output)
        token_correct += correct
        token_count += count
        metrics = ocr_sequence_accuracy(output)
        for key in totals:
            totals[key] += metrics[key]
        generated_texts = [codec.decode(row.tolist()) for row in generated_ids.cpu()]
        for expected, actual, category in zip(
            batch["texts"], generated_texts, batch["categories"]
        ):
            exact = expected == actual
            greedy_exact += int(exact)
            category_totals[category] = category_totals.get(category, 0) + 1
            category_exact[category] = category_exact.get(category, 0) + int(exact)
            if expected:
                greedy_visible += 1
                greedy_visible_exact += int(exact)
            else:
                greedy_blank += 1
                greedy_blank_exact += int(exact)
        batches += 1
    model.train()
    result = {
        "loss": loss / max(batches, 1),
        "token_accuracy": token_correct / max(token_count, 1),
        "exact_accuracy": totals["exact"] / max(totals["anchors"], 1),
        "visible_exact_accuracy": totals["visible_exact"]
        / max(totals["visible_anchors"], 1),
        "blank_exact_accuracy": totals["blank_exact"]
        / max(totals["blank_anchors"], 1),
        "greedy_exact_accuracy": greedy_exact / max(greedy_visible + greedy_blank, 1),
        "greedy_visible_exact_accuracy": greedy_visible_exact / max(greedy_visible, 1),
        "greedy_blank_exact_accuracy": greedy_blank_exact / max(greedy_blank, 1),
    }
    for category in sorted(category_totals):
        result[f"greedy_{category}_exact_accuracy"] = (
            category_exact[category] / category_totals[category]
        )
    return result


def save_checkpoint(
    path: Path,
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": global_step,
            "training_kind": "synthetic_ocr_pretraining",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/chemvector-v5-ocr-pretrain")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ocr-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--vision-learning-rate",
        type=float,
        default=0.0,
        help="Keep at zero during crop-only pretraining to protect geometry features",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCR pretraining")
    if min(args.steps, args.batch_size, args.validation_samples) < 1:
        raise ValueError("Steps, batch size, and validation samples must be positive")
    if args.ocr_learning_rate <= 0 or args.vision_learning_rate < 0:
        raise ValueError("OCR LR must be positive and vision LR must be nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_ocr_model(checkpoint_path, device)
    trainable = set_ocr_trainable(
        model,
        tune_vision=args.vision_learning_rate > 0,
    )
    optimizer = make_optimizer(
        model,
        args.ocr_learning_rate,
        args.vision_learning_rate,
        args.weight_decay,
    )
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    collator = partial(collate_synthetic_ocr, codec=codec)
    render_config = SyntheticOCRConfig(roi_radius=round(model.config.ocr_roi_radius))
    train_dataset = SyntheticOCRDataset(
        args.steps * args.batch_size,
        seed=args.seed,
        config=render_config,
    )
    validation_dataset = SyntheticOCRDataset(
        args.validation_samples,
        seed=args.seed + 1_000_000_007,
        config=render_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collator,
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
    print(f"fonts={len(train_dataset.font_paths)}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_correct = running_tokens = 0
    running_exact = running_anchors = 0
    running_visible_exact = running_visible = 0
    running_blank_exact = running_blank = 0
    interval_started = time.perf_counter()
    for global_step, batch in enumerate(train_loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_ocr_batch(model, batch, device)
        output["ocr_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += float(output["ocr_loss"].detach())
        correct, tokens = ocr_token_accuracy(output)
        running_correct += correct
        running_tokens += tokens
        metrics = ocr_sequence_accuracy(output)
        running_exact += metrics["exact"]
        running_anchors += metrics["anchors"]
        running_visible_exact += metrics["visible_exact"]
        running_visible += metrics["visible_anchors"]
        running_blank_exact += metrics["blank_exact"]
        running_blank += metrics["blank_anchors"]

        if global_step % args.log_every == 0:
            elapsed = time.perf_counter() - interval_started
            print(
                f"step={global_step} "
                f"loss={running_loss / args.log_every:.6f} "
                f"token_accuracy={running_correct / max(running_tokens, 1):.6f} "
                f"exact={running_exact / max(running_anchors, 1):.6f} "
                f"visible_exact={running_visible_exact / max(running_visible, 1):.6f} "
                f"blank_exact={running_blank_exact / max(running_blank, 1):.6f} "
                f"samples_per_second={args.log_every * args.batch_size / elapsed:.2f}"
            )
            running_loss = 0.0
            running_correct = running_tokens = 0
            running_exact = running_anchors = 0
            running_visible_exact = running_visible = 0
            running_blank_exact = running_blank = 0
            interval_started = time.perf_counter()

        if global_step % args.validate_every == 0 or global_step == args.steps:
            validation = validate(model, validation_loader, device, codec)
            print("validation=" + json.dumps(validation, ensure_ascii=False))
            save_checkpoint(output_dir / "last.pt", model, optimizer, global_step)
        if global_step >= args.steps:
            break


if __name__ == "__main__":
    main()
