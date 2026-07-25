#!/usr/bin/env python3
"""Train ChemVector to emit restricted visual primitives."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import glob
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from chemvector.model import ChemVectorModel
from chemvector.train import build_config
from chemvector.visual_prefix_model import VisualPrefixChemVectorModel
from chemvector.visual_svg_data import VisualSVGCollator, VisualSVGWebDataset
from chemvector.visual_svg_losses import coordinate_expectation_metrics


CROSS_ATTENTION_MODEL_TYPE = "chemvector-cross-attention-v0.1"


def build_visual_svg_model(
    config,
    architecture: str,
    visual_grid_size: int,
    ink_subgrid: int = 2,
):
    if architecture == "visual-prefix":
        return VisualPrefixChemVectorModel(
            config,
            visual_grid_size=visual_grid_size,
            ink_subgrid=ink_subgrid,
        )
    if architecture == "cross-attention":
        return ChemVectorModel(config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            shards.extend(matches)
        elif Path(pattern).is_file():
            shards.append(pattern)
    shards = list(dict.fromkeys(shards))
    if not shards:
        raise ValueError(f"No shards matched: {patterns}")
    return shards


def make_loader(
    shards: list[str],
    tokenizer: VisualSVGTokenizerV01,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    return DataLoader(
        VisualSVGWebDataset(
            shards,
            shardshuffle=shuffle,
            sample_shuffle_buffer=100 if shuffle else 0,
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=VisualSVGCollator(tokenizer, max_length=max_length),
    )


def tokenizer_sha256(tokenizer: VisualSVGTokenizerV01) -> str:
    payload = json.dumps(
        tokenizer.vocabulary,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def corrupt_teacher_forcing_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
    *,
    token_probability: float,
    record_probability: float,
) -> torch.Tensor:
    corrupted = input_ids.clone()
    coordinate_or_width = torch.tensor(
        [
            token_id
            for token, token_id in tokenizer.token_to_id.items()
            if token.startswith(("c:", "w:"))
        ],
        device=input_ids.device,
    )
    if token_probability:
        token_mask = (
            torch.rand(input_ids.shape, device=input_ids.device) < token_probability
        ) & attention_mask.bool()
        token_mask &= torch.isin(input_ids, coordinate_or_width)
        corrupted[token_mask] = tokenizer.unk_token_id
    if record_probability:
        arities = {
            tokenizer.token_to_id[tokenizer.LINE]: 5,
            tokenizer.token_to_id[tokenizer.TRIANGLE]: 7,
            tokenizer.token_to_id[tokenizer.QUAD]: 9,
            tokenizer.token_to_id[tokenizer.TEXT_BOX]: 4,
        }
        document_end_id = tokenizer.token_to_id[tokenizer.DOCUMENT_END]
        for row_index in range(input_ids.size(0)):
            cursor = 2
            valid_length = int(attention_mask[row_index].sum())
            while cursor < valid_length:
                marker = int(input_ids[row_index, cursor])
                if marker == document_end_id:
                    break
                arity = arities.get(marker)
                if arity is None:
                    raise ValueError(
                        f"Unexpected token at Visual SVG record boundary: {marker}"
                    )
                if float(torch.rand((), device=input_ids.device)) < record_probability:
                    corrupted[row_index, cursor : cursor + arity + 1] = tokenizer.unk_token_id
                cursor += arity + 1
    return corrupted


def visual_only_teacher_forcing_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
) -> torch.Tensor:
    visual_only = input_ids.clone()
    replace_mask = attention_mask.bool()
    replace_mask[:, 0] = False
    visual_only[replace_mask] = tokenizer.unk_token_id
    return visual_only


def batch_metrics(
    output: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
) -> dict:
    labels = input_ids[:, 1:]
    predictions = output["logits"].argmax(dim=-1)
    valid = labels != tokenizer.pad_token_id
    exact = ((predictions == labels) | ~valid).all(dim=1)
    marker_ids = torch.tensor(
        [
            tokenizer.token_to_id[token]
            for token in (
                tokenizer.LINE,
                tokenizer.TRIANGLE,
                tokenizer.QUAD,
                tokenizer.TEXT_BOX,
                tokenizer.DOCUMENT_END,
            )
        ],
        device=labels.device,
    )
    marker_mask = torch.isin(labels, marker_ids)
    eos_mask = labels == tokenizer.eos_token_id
    return {
        "correct": int(((predictions == labels) & valid).sum()),
        "tokens": int(valid.sum()),
        "exact": int(exact.sum()),
        "samples": int(exact.numel()),
        "marker_correct": int(((predictions == labels) & marker_mask).sum()),
        "marker_tokens": int(marker_mask.sum()),
        "eos_correct": int(((predictions == labels) & eos_mask).sum()),
        "eos_tokens": int(eos_mask.sum()),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: VisualSVGTokenizerV01,
    device: torch.device,
    max_batches: int,
    primitive_count_loss_weight: float = 0.0,
    visual_only_metrics: bool = False,
) -> dict[str, float]:
    model.eval()
    totals = CounterLike()
    for batch_index, batch in enumerate(loader, 1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values, input_ids, attention_mask)
            visual_only = None
            visual_only_mismatched = None
            if visual_only_metrics:
                visual_only_ids = visual_only_teacher_forcing_ids(
                    input_ids,
                    attention_mask,
                    tokenizer,
                )
                visual_only = model(
                    pixel_values,
                    input_ids,
                    attention_mask,
                    teacher_forcing_ids=visual_only_ids,
                    attention_layout="visual-only",
                )
            count_loss = None
            count_mae = None
            if primitive_count_loss_weight:
                targets = batch["primitive_counts"].to(device, non_blocking=True)
                count_loss = model.primitive_count_loss(
                    output["primitive_count_log"],
                    targets,
                )
                predicted_counts = torch.expm1(output["primitive_count_log"]).clamp_min(0)
                count_mae = (predicted_counts - targets).abs().mean()
            mismatch_loss = None
            mismatch_logits_mae = None
            if pixel_values.size(0) > 1:
                mismatched = model(pixel_values.roll(1, dims=0), input_ids, attention_mask)
                mismatch_loss = mismatched["loss"]
                mismatch_logits_mae = (
                    output["logits"] - mismatched["logits"]
                ).abs().mean()
                if visual_only is not None:
                    visual_only_mismatched = model(
                        pixel_values.roll(1, dims=0),
                        input_ids,
                        attention_mask,
                        teacher_forcing_ids=visual_only_ids,
                        attention_layout="visual-only",
                    )
        coordinate_metrics = coordinate_expectation_metrics(
            output["logits"],
            input_ids[:, 1:],
            tokenizer,
        )
        metrics = batch_metrics(output, input_ids, tokenizer)
        totals.add("loss", float(output["loss"]))
        totals.add("batches", 1)
        if visual_only is not None:
            totals.add("visual_only_loss", float(visual_only["loss"]))
            totals.add("visual_only_batches", 1)
        if visual_only_mismatched is not None:
            totals.add(
                "visual_only_mismatch_loss",
                float(visual_only_mismatched["loss"]),
            )
            totals.add("visual_only_mismatch_batches", 1)
        if count_loss is not None:
            totals.add("primitive_count_loss", float(count_loss))
            totals.add("primitive_count_mae", float(count_mae))
            totals.add("primitive_count_batches", 1)
        coordinate_tokens = int(coordinate_metrics["tokens"])
        totals.add(
            "coordinate_loss_sum",
            float(coordinate_metrics["loss"]) * coordinate_tokens,
        )
        totals.add(
            "coordinate_soft_mae_q_sum",
            float(coordinate_metrics["soft_mae_q"]) * coordinate_tokens,
        )
        totals.add(
            "coordinate_hard_exact_sum",
            float(coordinate_metrics["hard_exact"]) * coordinate_tokens,
        )
        totals.add(
            "coordinate_hard_valid_sum",
            float(coordinate_metrics["hard_valid_rate"]) * coordinate_tokens,
        )
        totals.add("coordinate_tokens", coordinate_tokens)
        visual = output.get("visual_prefix", output.get("memory"))
        if visual is not None and visual.size(0) > 1:
            totals.add("visual_prefix_batch_std", float(visual.std(dim=0).mean()))
            totals.add("visual_prefix_batches", 1)
        for key, value in metrics.items():
            totals.add(key, value)
        if mismatch_loss is not None:
            totals.add("mismatch_loss", float(mismatch_loss))
            totals.add("mismatch_logits_mae", float(mismatch_logits_mae))
            totals.add("mismatch_batches", 1)
        if batch_index >= max_batches:
            break
    model.train()
    loss = totals["loss"] / max(totals["batches"], 1)
    mismatch = totals["mismatch_loss"] / max(totals["mismatch_batches"], 1)
    visual_only_loss = totals["visual_only_loss"] / max(
        totals["visual_only_batches"], 1
    )
    visual_only_mismatch = totals["visual_only_mismatch_loss"] / max(
        totals["visual_only_mismatch_batches"], 1
    )
    return {
        "loss": loss,
        "token_accuracy": totals["correct"] / max(totals["tokens"], 1),
        "teacher_sequence_exact": totals["exact"] / max(totals["samples"], 1),
        "marker_accuracy": totals["marker_correct"] / max(totals["marker_tokens"], 1),
        "eos_accuracy": totals["eos_correct"] / max(totals["eos_tokens"], 1),
        "mismatch_loss": mismatch,
        "mismatch_loss_gap": mismatch - loss,
        "mismatch_loss_ratio": mismatch / max(loss, 1e-8),
        "mismatch_logits_mae": totals["mismatch_logits_mae"]
        / max(totals["mismatch_batches"], 1),
        "visual_prefix_batch_std": totals["visual_prefix_batch_std"]
        / max(totals["visual_prefix_batches"], 1),
        "primitive_count_loss": totals["primitive_count_loss"]
        / max(totals["primitive_count_batches"], 1),
        "primitive_count_mae": totals["primitive_count_mae"]
        / max(totals["primitive_count_batches"], 1),
        "coordinate_loss": totals["coordinate_loss_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_soft_mae_q": totals["coordinate_soft_mae_q_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_hard_exact": totals["coordinate_hard_exact_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_hard_valid_rate": totals["coordinate_hard_valid_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_tokens": totals["coordinate_tokens"],
        "visual_only_loss": visual_only_loss,
        "visual_only_mismatch_loss": visual_only_mismatch,
        "visual_only_mismatch_loss_ratio": visual_only_mismatch
        / max(visual_only_loss, 1e-8),
        "batches": totals["batches"],
    }


class CounterLike(dict):
    def __missing__(self, key):
        return 0

    def add(self, key, value) -> None:
        self[key] = self[key] + value


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "epoch": epoch,
            "global_step": global_step,
            "training_args": vars(args),
            "validation": metrics,
            "sequence_schema_version": "visual-svg-0.1",
            "tokenizer_sha256": tokenizer_sha256(VisualSVGTokenizerV01()),
            "checkpoint_schema_version": "visual-svg-checkpoint-0.2",
            "model_type": getattr(model, "model_type", CROSS_ATTENTION_MODEL_TYPE),
            "visual_grid_size": getattr(model, "visual_grid_size", None),
            "ink_subgrid": getattr(model, "ink_subgrid", None),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/visual_svg_v0_1/capacity_100/shards/*.tar"],
    )
    parser.add_argument("--validation-shards", nargs="+")
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v090-visual-svg-capacity-100",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--token-corruption-probability", type=float, default=0.0)
    parser.add_argument("--record-corruption-probability", type=float, default=0.0)
    parser.add_argument("--image-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--image-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--primitive-count-loss-weight", type=float, default=0.25)
    parser.add_argument("--coordinate-loss-weight", type=float, default=0.0)
    parser.add_argument("--causal-loss-weight", type=float, default=1.0)
    parser.add_argument("--visual-only-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-target-length", type=int, default=2048)
    parser.add_argument(
        "--architecture",
        choices=("visual-prefix", "cross-attention"),
        default="visual-prefix",
    )
    parser.add_argument("--visual-grid-size", type=int, default=12)
    parser.add_argument("--ink-subgrid", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Visual SVG training")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be positive")
    if not 0.0 <= args.token_corruption_probability <= 1.0:
        raise ValueError("--token-corruption-probability must be in [0, 1]")
    if not 0.0 <= args.record_corruption_probability <= 1.0:
        raise ValueError("--record-corruption-probability must be in [0, 1]")
    if args.image_contrastive_weight < 0 or args.image_contrastive_margin < 0:
        raise ValueError("Image contrastive settings must be nonnegative")
    if args.primitive_count_loss_weight < 0:
        raise ValueError("Primitive count loss weight must be nonnegative")
    if args.coordinate_loss_weight < 0:
        raise ValueError("Coordinate loss weight must be nonnegative")
    if args.causal_loss_weight < 0 or args.visual_only_loss_weight < 0:
        raise ValueError("Sequence loss weights must be nonnegative")
    if not args.causal_loss_weight and not args.visual_only_loss_weight:
        raise ValueError("At least one sequence loss must be enabled")
    if args.architecture == "cross-attention" and args.visual_only_loss_weight:
        raise ValueError("Visual-only attention is only available for visual-prefix")
    if args.architecture == "cross-attention" and args.primitive_count_loss_weight:
        raise ValueError("Primitive count loss is only available for visual-prefix")
    if args.visual_grid_size < 1:
        raise ValueError("--visual-grid-size must be positive")
    if args.ink_subgrid < 1:
        raise ValueError("--ink-subgrid must be positive")
    if args.image_contrastive_weight and args.batch_size < 2:
        raise ValueError("Image contrastive training requires --batch-size >= 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if min(args.log_every, args.validate_every, args.save_every) < 1:
        raise ValueError("Logging, validation, and save intervals must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = VisualSVGTokenizerV01()
    config = replace(
        build_config(tokenizer, args.tiny),
        dropout=args.dropout,
        max_target_length=args.max_target_length,
    )
    device = torch.device("cuda")
    model = build_visual_svg_model(
        config,
        args.architecture,
        args.visual_grid_size,
        args.ink_subgrid,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config != asdict(config):
            raise ValueError("Initial checkpoint config does not match this training run")
        if checkpoint.get("sequence_schema_version") != "visual-svg-0.1":
            raise ValueError("Initial checkpoint uses a different sequence schema")
        if checkpoint.get("tokenizer_sha256") != tokenizer_sha256(tokenizer):
            raise ValueError("Initial checkpoint uses a different tokenizer vocabulary")
        expected_model_type = getattr(model, "model_type", CROSS_ATTENTION_MODEL_TYPE)
        checkpoint_model_type = checkpoint.get("model_type", CROSS_ATTENTION_MODEL_TYPE)
        if checkpoint_model_type != expected_model_type:
            raise ValueError("Initial checkpoint uses a different model architecture")
        if checkpoint.get("visual_grid_size") != getattr(model, "visual_grid_size", None):
            raise ValueError("Initial checkpoint uses a different visual grid")
        if checkpoint.get("ink_subgrid", 2) != getattr(model, "ink_subgrid", None):
            raise ValueError("Initial checkpoint uses a different ink subgrid")
        model.load_state_dict(checkpoint["model"])
        print(f"initialized_from={args.init_checkpoint}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards or args.train_shards)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"train_shards={len(train_shards)} validation_shards={len(validation_shards)}")

    validation_loader = make_loader(
        validation_shards,
        tokenizer,
        batch_size=max(2, args.batch_size),
        workers=args.workers,
        shuffle=False,
        max_length=args.max_target_length,
    )
    global_step = 0
    best_loss = float("inf")
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        loader = make_loader(
            train_shards,
            tokenizer,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=not args.no_shuffle,
            max_length=args.max_target_length,
        )
        running_loss = 0.0
        running_correct = 0
        running_tokens = 0
        running_contrastive = 0.0
        running_count_loss = 0.0
        running_coordinate_loss = 0.0
        running_visual_only_loss = 0.0
        running_marker_correct = 0
        running_marker_tokens = 0
        running_eos_correct = 0
        running_eos_tokens = 0
        logged_batches = 0
        accumulated = 0
        started = time.perf_counter()
        model.train()
        for batch_index, batch in enumerate(loader, 1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            teacher_forcing_ids = corrupt_teacher_forcing_ids(
                input_ids,
                attention_mask,
                tokenizer,
                token_probability=args.token_corruption_probability,
                record_probability=args.record_corruption_probability,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.causal_loss_weight or args.coordinate_loss_weight:
                    model_output = model(
                        pixel_values,
                        input_ids,
                        attention_mask,
                        teacher_forcing_ids=teacher_forcing_ids,
                    )
                else:
                    with torch.no_grad():
                        model_output = model(
                            pixel_values,
                            input_ids,
                            attention_mask,
                            teacher_forcing_ids=teacher_forcing_ids,
                        )
                contrastive_loss = model_output["loss"].new_zeros(())
                primitive_count_loss = model_output["loss"].new_zeros(())
                coordinate_loss = model_output["loss"].new_zeros(())
                visual_only_loss = model_output["loss"].new_zeros(())
                visual_only_output = None
                visual_only_ids = None
                if args.visual_only_loss_weight:
                    visual_only_ids = visual_only_teacher_forcing_ids(
                        input_ids,
                        attention_mask,
                        tokenizer,
                    )
                    visual_only_output = model(
                        pixel_values,
                        input_ids,
                        attention_mask,
                        teacher_forcing_ids=visual_only_ids,
                        attention_layout="visual-only",
                    )
                    visual_only_loss = visual_only_output["loss"]
                if args.primitive_count_loss_weight:
                    count_source = visual_only_output or model_output
                    primitive_count_loss = model.primitive_count_loss(
                        count_source["primitive_count_log"],
                        batch["primitive_counts"].to(device, non_blocking=True),
                    )
                if args.coordinate_loss_weight:
                    coordinate_loss = coordinate_expectation_metrics(
                        model_output["logits"],
                        input_ids[:, 1:],
                        tokenizer,
                    )["loss"]
                if args.image_contrastive_weight and pixel_values.size(0) > 1:
                    if visual_only_output is not None:
                        mismatched = model(
                            pixel_values.roll(1, dims=0),
                            input_ids,
                            attention_mask,
                            teacher_forcing_ids=visual_only_ids,
                            attention_layout="visual-only",
                        )
                    else:
                        mismatched = model(
                            pixel_values.roll(1, dims=0),
                            input_ids,
                            attention_mask,
                            teacher_forcing_ids=teacher_forcing_ids,
                        )
                    matched_losses = (
                        visual_only_output["loss_per_sample"]
                        if visual_only_output is not None
                        else model_output["loss_per_sample"]
                    )
                    contrastive_loss = torch.relu(
                        args.image_contrastive_margin
                        + matched_losses
                        - mismatched["loss_per_sample"]
                    ).mean()
                combined_loss = (
                    args.causal_loss_weight * model_output["loss"]
                    + args.visual_only_loss_weight * visual_only_loss
                    + args.image_contrastive_weight * contrastive_loss
                    + args.primitive_count_loss_weight * primitive_count_loss
                    + args.coordinate_loss_weight * coordinate_loss
                )
                loss = combined_loss / args.gradient_accumulation
            loss.backward()
            accumulated += 1
            metrics = batch_metrics(
                model_output,
                input_ids,
                tokenizer,
            )
            running_loss += float(model_output["loss"].detach())
            running_correct += metrics["correct"]
            running_tokens += metrics["tokens"]
            running_contrastive += float(contrastive_loss.detach())
            running_count_loss += float(primitive_count_loss.detach())
            running_coordinate_loss += float(coordinate_loss.detach())
            running_visual_only_loss += float(visual_only_loss.detach())
            running_marker_correct += metrics["marker_correct"]
            running_marker_tokens += metrics["marker_tokens"]
            running_eos_correct += metrics["eos_correct"]
            running_eos_tokens += metrics["eos_tokens"]
            logged_batches += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_step += 1
                if global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={running_loss / max(logged_batches, 1):.5f} "
                        f"contrastive={running_contrastive / max(logged_batches, 1):.5f} "
                        f"count_loss={running_count_loss / max(logged_batches, 1):.5f} "
                        f"coordinate_loss={running_coordinate_loss / max(logged_batches, 1):.5f} "
                        f"visual_only={running_visual_only_loss / max(logged_batches, 1):.5f} "
                        f"token_acc={running_correct / max(running_tokens, 1):.5f} "
                        f"marker_acc={running_marker_correct / max(running_marker_tokens, 1):.5f} "
                        f"eos_acc={running_eos_correct / max(running_eos_tokens, 1):.5f} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_correct = 0
                    running_tokens = 0
                    running_contrastive = 0.0
                    running_count_loss = 0.0
                    running_coordinate_loss = 0.0
                    running_visual_only_loss = 0.0
                    running_marker_correct = 0
                    running_marker_tokens = 0
                    running_eos_correct = 0
                    running_eos_tokens = 0
                    logged_batches = 0
                    started = time.perf_counter()
                if global_step % args.validate_every == 0:
                    validation = validate(
                        model,
                        validation_loader,
                        tokenizer,
                        device,
                        args.validation_batches,
                        args.primitive_count_loss_weight,
                        bool(args.visual_only_loss_weight),
                    )
                    print("validation=" + json.dumps(validation, sort_keys=True), flush=True)
                    if validation["loss"] < best_loss:
                        best_loss = validation["loss"]
                        save_checkpoint(
                            output / "best-teacher.pt",
                            model,
                            optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            args=args,
                            metrics=validation,
                        )
                if global_step % args.save_every == 0:
                    save_checkpoint(
                        output / f"step-{global_step:06d}.pt",
                        model,
                        optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        metrics=None,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
        if accumulated:
            scale = args.gradient_accumulation / accumulated
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if stop:
            break

    final_validation = validate(
        model,
        validation_loader,
        tokenizer,
        device,
        args.validation_batches,
        args.primitive_count_loss_weight,
        bool(args.visual_only_loss_weight),
    )
    save_checkpoint(
        output / "last.pt",
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        args=args,
        metrics=final_validation,
    )
    if not (output / "best-teacher.pt").exists():
        save_checkpoint(
            output / "best-teacher.pt",
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            metrics=final_validation,
        )
    print("final_validation=" + json.dumps(final_validation, sort_keys=True))


SEED = 20260721


if __name__ == "__main__":
    main()
