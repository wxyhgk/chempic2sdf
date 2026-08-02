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
from chemvector.layer_ocr import (
    ChemVisionLayerOCRConfig,
    ChemVisionLayerOCRDataset,
    ROLE_TO_ID,
    assign_predicted_points,
    collate_layer_ocr,
)
from chemvector.ocr import OCRTextCodec
from chemvector.replacement_ocr import (
    MoleculeOCRReplacementDataset,
    ReplacementOCRConfig,
    collate_replacement_ocr,
)
from chemvector.synthetic_ocr import (
    SyntheticOCRConfig,
    SyntheticOCRDataset,
    collate_synthetic_ocr,
)
from chemvector.train import ocr_sequence_accuracy, ocr_token_accuracy
from chemvector.vertex import decode_vertices


def load_ocr_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    pixel_ocr: bool = False,
    pixel_multiscale: bool = False,
    ocr_roi_radius: float | None = None,
    residual_adapter: bool = False,
    adapter_dim: int = 48,
    role_head: bool = False,
    role_loss_weight: float = 1.0,
) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    config_values = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in valid_keys
    }
    config_values["ocr_enabled"] = True
    if pixel_ocr:
        config_values["ocr_pixel_enabled"] = True
    if pixel_multiscale:
        config_values["ocr_pixel_enabled"] = True
        config_values["ocr_pixel_multiscale"] = True
    if ocr_roi_radius is not None:
        config_values["ocr_roi_radius"] = ocr_roi_radius
    if residual_adapter:
        config_values["ocr_residual_adapter"] = True
        config_values["ocr_adapter_dim"] = adapter_dim
    if role_head:
        config_values["ocr_role_head"] = True
        config_values["ocr_role_loss_weight"] = role_loss_weight
    config = ChemVectorConfig(**config_values)
    model = ChemVectorModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("ocr_head.")
    }
    allowed_missing.update(
        {
            "vertex_head.role_heatmap.weight",
            "vertex_head.role_heatmap.bias",
        }
    )
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
    adapter_only: bool = False,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if adapter_only:
        modules = [
            module
            for module in (
                model.ocr_head.memory_adapter,
                model.ocr_head.role_attention,
                model.ocr_head.role_classifier,
            )
            if module is not None
        ]
        if not modules:
            raise ValueError("Adapter-only training requires an OCR adapter or role head")
    else:
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
    ocr_parameters = [
        parameter for parameter in model.ocr_head.parameters() if parameter.requires_grad
    ]
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
    role_targets = batch.get("ocr_role_targets")
    if role_targets is not None and model.ocr_head.role_classifier is not None:
        role_targets = role_targets.to(device, non_blocking=True)
    else:
        role_targets = None
    features = model.ocr_source(pixel_values)
    return model.ocr_head(
        features,
        anchor_points,
        anchor_mask,
        token_ids,
        pixel_values.shape[-2:],
        role_targets=role_targets,
    )


def forward_predicted_anchor_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
    codec: OCRTextCodec,
    *,
    vertex_threshold: float = 0.3,
) -> dict[str, torch.Tensor] | None:
    if model.vertex_head is None:
        raise ValueError("Predicted-anchor OCR training requires a vertex head")
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    with torch.no_grad():
        stroke_features = model.vision_encoder.forward_stroke_resolution(pixel_values)
        detections = decode_vertices(
            model.vertex_head(stroke_features),
            threshold=vertex_threshold,
            maximum_vertices=128,
        )
    point_groups = []
    text_groups = []
    role_groups = []
    for image_detections, atoms, width, height in zip(
        detections,
        batch["layer_atoms"],
        batch["layer_widths"],
        batch["layer_heights"],
    ):
        points = [
            (float(item["x"]) * width, float(item["y"]) * height)
            for item in image_detections
        ]
        point_groups.append(
            [(x / width, y / height) for x, y in points]
        )
        texts, roles = assign_predicted_points(points, atoms)
        text_groups.append(texts)
        role_groups.append(roles)
    maximum_anchors = max((len(group) for group in point_groups), default=0)
    if maximum_anchors == 0:
        return None
    encoded = [[codec.encode(text) for text in group] for group in text_groups]
    maximum_length = max(len(item) for group in encoded for item in group)
    anchors = torch.zeros(
        (len(point_groups), maximum_anchors, 2),
        dtype=torch.float32,
        device=device,
    )
    anchor_mask = torch.zeros(
        (len(point_groups), maximum_anchors),
        dtype=torch.bool,
        device=device,
    )
    token_ids = torch.full(
        (len(point_groups), maximum_anchors, maximum_length),
        codec.PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    role_targets = torch.full(
        (len(point_groups), maximum_anchors),
        -1,
        dtype=torch.long,
        device=device,
    )
    for batch_index, (points, encoded_group, roles) in enumerate(
        zip(point_groups, encoded, role_groups)
    ):
        if not points:
            continue
        anchors[batch_index, : len(points)] = torch.tensor(points, device=device)
        anchor_mask[batch_index, : len(points)] = True
        role_targets[batch_index, : len(points)] = torch.tensor(
            [ROLE_TO_ID[role] for role in roles],
            dtype=torch.long,
            device=device,
        )
        for anchor_index, item in enumerate(encoded_group):
            token_ids[batch_index, anchor_index, : len(item)] = torch.tensor(
                item,
                device=device,
            )
    features = model.ocr_source(pixel_values, stroke_features)
    return model.ocr_head(
        features,
        anchors,
        anchor_mask,
        token_ids,
        pixel_values.shape[-2:],
        role_targets=(
            role_targets if model.ocr_head.role_classifier is not None else None
        ),
    )


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    codec: OCRTextCodec,
    *,
    use_auxiliary_routing: bool | None = None,
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
    tracked_labels = {
        "B",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Si",
        "Cl",
        "Se",
        "Br",
        "I",
        "Ph",
        "R1",
        "NH",
        "NH2",
        "NH3",
        "N+",
        "N-",
        "NH+",
        "NH2+",
        "OH",
        "O-",
        "O+",
        "B-",
        "BH",
        "BH2",
        "SH",
        "S+",
        "S-",
        "P+",
    }
    label_totals: dict[str, int] = {}
    label_exact: dict[str, int] = {}
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_ocr_batch(model, batch, device)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            anchor_points = batch["ocr_anchor_points"].to(device, non_blocking=True)
            anchor_mask = batch["ocr_anchor_mask"].to(device, non_blocking=True)
            features = model.ocr_source(pixel_values)
            generated_ids = model.ocr_head.generate(
                features,
                anchor_points,
                anchor_mask,
                pixel_values.shape[-2:],
                use_auxiliary_routing=use_auxiliary_routing,
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
            if expected in tracked_labels:
                label_totals[expected] = label_totals.get(expected, 0) + 1
                label_exact[expected] = label_exact.get(expected, 0) + int(exact)
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
    for label in sorted(label_totals):
        result[f"greedy_label_{label}_exact_accuracy"] = (
            label_exact[label] / label_totals[label]
        )
        result[f"greedy_label_{label}_samples"] = label_totals[label]
    return result


def save_checkpoint(
    path: Path,
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    training_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": global_step,
            "training_kind": training_kind,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--pixel-ocr", action="store_true")
    parser.add_argument("--pixel-multiscale", action="store_true")
    parser.add_argument("--pixel-hard-negative", action="store_true")
    parser.add_argument("--pixel-original-refinement", action="store_true")
    parser.add_argument("--ocr-roi-radius", type=float)
    parser.add_argument("--residual-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=48)
    parser.add_argument("--role-head", action="store_true")
    parser.add_argument("--role-loss-weight", type=float, default=1.0)
    parser.add_argument("--freeze-base-ocr", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=("synthetic-crop", "molecule-replacement", "chemvision-layers"),
        default="synthetic-crop",
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
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v5-ocr-pretrain",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int)
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
    parser.add_argument("--visible-jitter-copies", type=int, default=0)
    parser.add_argument("--n-jitter-probability", type=float, default=0.05)
    parser.add_argument("--predicted-anchor-weight", type=float, default=0.0)
    parser.add_argument("--modifier-focus", action="store_true")
    parser.add_argument("--color-augmentation-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 64 if args.dataset == "synthetic-crop" else 4

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCR pretraining")
    if min(args.steps, args.batch_size, args.validation_samples) < 1:
        raise ValueError("Steps, batch size, and validation samples must be positive")
    if args.ocr_learning_rate <= 0 or args.vision_learning_rate < 0:
        raise ValueError("OCR LR must be positive and vision LR must be nonnegative")
    if args.ocr_roi_radius is not None and args.ocr_roi_radius <= 0:
        raise ValueError("OCR ROI radius must be positive")
    if args.pixel_original_refinement and not args.pixel_ocr:
        raise ValueError("Pixel original refinement requires --pixel-ocr")
    if args.pixel_hard_negative and not args.pixel_ocr:
        raise ValueError("Pixel hard-negative training requires --pixel-ocr")
    if args.pixel_hard_negative and args.pixel_original_refinement:
        raise ValueError("Pixel OCR training stages are mutually exclusive")
    if args.pixel_multiscale and not args.pixel_ocr:
        raise ValueError("Multi-scale OCR requires --pixel-ocr")
    if args.visible_jitter_copies < 0:
        raise ValueError("Visible jitter copies must be nonnegative")
    if not 0.0 <= args.n_jitter_probability <= 1.0:
        raise ValueError("N jitter probability must be in [0, 1]")
    if args.visible_jitter_copies and args.dataset != "chemvision-layers":
        raise ValueError("Visible jitter copies require chemvision-layers")
    if args.predicted_anchor_weight < 0:
        raise ValueError("Predicted anchor weight must be nonnegative")
    if not 0.0 <= args.color_augmentation_probability <= 1.0:
        raise ValueError("Color augmentation probability must be in [0, 1]")
    if args.predicted_anchor_weight and args.dataset != "chemvision-layers":
        raise ValueError("Predicted anchor training requires chemvision-layers")
    if args.adapter_dim < 1 or args.role_loss_weight < 0:
        raise ValueError("Adapter dimension and role loss weight are invalid")
    if args.freeze_base_ocr and not (args.residual_adapter or args.role_head):
        raise ValueError("Frozen base OCR training requires an adapter or role head")
    if args.role_head and args.dataset != "chemvision-layers":
        raise ValueError("OCR role training requires chemvision-layers")
    if args.role_head and not args.predicted_anchor_weight:
        raise ValueError("OCR role training requires predicted anchors")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_ocr_model(
        checkpoint_path,
        device,
        pixel_ocr=args.pixel_ocr,
        pixel_multiscale=args.pixel_multiscale,
        ocr_roi_radius=args.ocr_roi_radius,
        residual_adapter=args.residual_adapter,
        adapter_dim=args.adapter_dim,
        role_head=args.role_head,
        role_loss_weight=args.role_loss_weight,
    )
    trainable = set_ocr_trainable(
        model,
        tune_vision=args.vision_learning_rate > 0,
        adapter_only=args.freeze_base_ocr,
    )
    optimizer = make_optimizer(
        model,
        args.ocr_learning_rate,
        args.vision_learning_rate,
        args.weight_decay,
    )
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    if args.dataset == "synthetic-crop":
        collator = partial(collate_synthetic_ocr, codec=codec)
        render_config = SyntheticOCRConfig(
            roi_radius=round(model.config.ocr_roi_radius)
        )
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
        training_kind = "synthetic_ocr_pretraining"
    elif args.dataset == "molecule-replacement":
        collator = partial(collate_replacement_ocr, codec=codec)
        if args.pixel_original_refinement:
            replacement_config = ReplacementOCRConfig(
                replacement_image_probability=0.0,
                maximum_original_blank_anchors=4,
                maximum_original_visible_anchors=8,
                priority_original_labels=("B", "Br", "F", "Cl", "I"),
            )
        elif args.pixel_hard_negative:
            replacement_config = ReplacementOCRConfig(
                replacement_image_probability=0.65,
                maximum_original_blank_anchors=4,
                maximum_original_visible_anchors=8,
            )
        else:
            replacement_config = ReplacementOCRConfig()
        train_dataset = MoleculeOCRReplacementDataset(
            args.train_shards,
            seed=args.seed,
            config=replacement_config,
            shardshuffle=100,
            sample_shuffle_buffer=128,
        )
        validation_dataset = MoleculeOCRReplacementDataset(
            args.validation_shards,
            seed=args.seed + 1_000_000_007,
            config=ReplacementOCRConfig(),
            sample_limit=args.validation_samples,
        )
        training_kind = "molecule_replacement_ocr_pretraining"
    else:
        collator = partial(collate_layer_ocr, codec=codec)
        modifier_texts = (
            "N", "NH", "NH2", "NH3", "N+", "N-", "NH+", "NH2+",
            "O", "OH", "O-", "O+",
            "B", "B-", "BH", "BH2",
            "S", "SH", "S+", "S-", "P", "P+",
        )
        train_dataset = ChemVisionLayerOCRDataset(
            args.train_shards,
            seed=args.seed,
            config=ChemVisionLayerOCRConfig(
                visible_jitter_copies=args.visible_jitter_copies,
                n_jitter_probability=args.n_jitter_probability,
                priority_elements=(
                    ()
                    if args.modifier_focus
                    else ChemVisionLayerOCRConfig().priority_elements
                ),
                priority_texts=modifier_texts if args.modifier_focus else (),
                color_probability=args.color_augmentation_probability,
            ),
            shardshuffle=100,
            sample_shuffle_buffer=128,
        )
        validation_dataset = ChemVisionLayerOCRDataset(
            args.validation_shards,
            seed=args.seed + 1_000_000_007,
            config=ChemVisionLayerOCRConfig(
                color_probability=(
                    0.5 if args.color_augmentation_probability > 0.0 else 0.0
                )
            ),
            sample_limit=args.validation_samples,
        )
        training_kind = "chemvision_layers_ocr_pretraining"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collator,
    )
    validation_workers = args.workers if args.dataset == "synthetic-crop" else 0
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
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
    print(f"ocr_dataset={args.dataset}")
    print(f"pixel_ocr={model.config.ocr_pixel_enabled}")
    print(f"pixel_multiscale={model.config.ocr_pixel_multiscale}")
    if args.dataset == "synthetic-crop":
        print(f"fonts={len(train_dataset.font_paths)}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_predicted_loss = 0.0
    running_correct = running_tokens = 0
    running_exact = running_anchors = 0
    running_visible_exact = running_visible = 0
    running_blank_exact = running_blank = 0
    interval_started = time.perf_counter()
    best_visible_accuracy = -1.0
    for global_step, batch in enumerate(train_loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_ocr_batch(model, batch, device)
            predicted_output = (
                forward_predicted_anchor_batch(model, batch, device, codec)
                if args.predicted_anchor_weight
                else None
            )
            predicted_loss = (
                predicted_output["ocr_loss"]
                if predicted_output is not None
                else output["ocr_loss"].new_zeros(())
            )
            training_loss = (
                output["ocr_loss"]
                + args.predicted_anchor_weight * predicted_loss
            )
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += float(output["ocr_loss"].detach())
        running_predicted_loss += float(predicted_loss.detach())
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
                f"predicted_loss={running_predicted_loss / args.log_every:.6f} "
                f"token_accuracy={running_correct / max(running_tokens, 1):.6f} "
                f"exact={running_exact / max(running_anchors, 1):.6f} "
                f"visible_exact={running_visible_exact / max(running_visible, 1):.6f} "
                f"blank_exact={running_blank_exact / max(running_blank, 1):.6f} "
                f"samples_per_second={args.log_every * args.batch_size / elapsed:.2f}"
            )
            running_loss = 0.0
            running_predicted_loss = 0.0
            running_correct = running_tokens = 0
            running_exact = running_anchors = 0
            running_visible_exact = running_visible = 0
            running_blank_exact = running_blank = 0
            interval_started = time.perf_counter()

        if global_step % args.validate_every == 0 or global_step == args.steps:
            validation = validate(model, validation_loader, device, codec)
            print("validation=" + json.dumps(validation, ensure_ascii=False))
            (output_dir / f"validation-{global_step:07d}.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                global_step,
                training_kind,
            )
            save_checkpoint(
                output_dir / f"checkpoint-{global_step:07d}.pt",
                model,
                optimizer,
                global_step,
                training_kind,
            )
            visible_accuracy = validation["greedy_visible_exact_accuracy"]
            if visible_accuracy > best_visible_accuracy:
                best_visible_accuracy = visible_accuracy
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    global_step,
                    training_kind,
                )
                (output_dir / "best_validation.json").write_text(
                    json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        if global_step >= args.steps:
            break


if __name__ == "__main__":
    main()
