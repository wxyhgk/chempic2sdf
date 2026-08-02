#!/usr/bin/env python3
"""Single-GPU ChemVector training entry point."""

from __future__ import annotations

import argparse
from functools import partial
import glob
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from chemvector.chemtraverse_decision_loss import (
    ChemTraverseDecisionPlan,
    balanced_control_token_weights,
    build_chemtraverse_decision_plan,
    grammar_aware_decision_balanced_loss,
)
from chemvector.data import ChemVectorCollator, ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.ocr import OCRTextCodec
from chemvector.synthetic_ocr import SyntheticOCRDataset, collate_synthetic_ocr


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            shards.extend(matches)
        elif Path(pattern).is_file():
            shards.append(pattern)
    unique = list(dict.fromkeys(shards))
    if not unique:
        raise ValueError(f"No shards matched: {patterns}")
    return unique


def load_sample_keys(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    keys = {
        json.loads(line)["sample_key"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not keys:
        raise ValueError(f"Sample manifest contains no sample keys: {path}")
    return keys


def file_record(path: str | Path | None) -> dict[str, str | int] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def git_state() -> dict[str, str | bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "tracked_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_config(tokenizer, tiny: bool, *, ocr_enabled: bool = False) -> ChemVectorConfig:
    common = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "ocr_enabled": ocr_enabled,
    }
    if tiny:
        return ChemVectorConfig(
            **common,
            d_model=256,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=1024,
            vision_dims=(48, 96, 192, 256),
            vision_depths=(1, 1, 2, 1),
        )
    return ChemVectorConfig(**common)


def load_vision_encoder_checkpoint(
    model: ChemVectorModel, checkpoint_path: str | Path
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model")
    if not isinstance(source, dict):
        raise ValueError("Vision initialization checkpoint has no model state")
    target = model.state_dict()
    vision_keys = {key for key in target if key.startswith("vision_encoder.")}
    missing = sorted(vision_keys - set(source))
    mismatched = sorted(
        key
        for key in vision_keys & set(source)
        if source[key].shape != target[key].shape
    )
    if missing or mismatched:
        raise ValueError(
            "Vision initialization architecture mismatch: "
            f"missing={missing}, mismatched={mismatched}"
        )
    target.update({key: source[key] for key in vision_keys})
    model.load_state_dict(target)
    return len(vision_keys)


def make_loader(
    shards: list[str],
    tokenizer,
    config: ChemVectorConfig,
    *,
    batch_size: int,
    image_size: int,
    workers: int,
    shuffle: bool,
    reject_keys: set[str] | None = None,
    include_keys: set[str] | None = None,
    loader_seed: int | None = None,
    target_molecule_id: str | None = None,
) -> DataLoader:
    schema_version = tokenizer.sequence_schema_version
    sequence_version = (
        "traverse-0.1"
        if schema_version == "chemtraverse-0.1"
        else schema_version.removeprefix("chemmolseq-").removeprefix("chemsvg-")
    )
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=shuffle,
        sample_shuffle_buffer=1000 if shuffle else 0,
        sample_shuffle_seed=loader_seed,
        sequence_version=sequence_version,
        reject_keys=reject_keys,
        include_keys=include_keys,
        target_molecule_id=target_molecule_id,
    )
    collator = ChemVectorCollator(
        tokenizer,
        image_size=image_size,
        max_length=config.max_target_length,
        ocr_codec=(
            OCRTextCodec(max_length=config.ocr_max_length)
            if config.ocr_enabled
            else None
        ),
    )
    generator = None
    if loader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collator,
        generator=generator,
    )


def iter_mixed_batches(
    primary_loader,
    replay_loader=None,
    replay_probability: float = 0.0,
    *,
    rng=random,
):
    """Yield primary and replay batches without exhausting the replay stream."""
    primary_iterator = iter(primary_loader)
    replay_iterator = iter(replay_loader) if replay_loader is not None else None
    while True:
        use_replay = replay_iterator is not None and rng.random() < replay_probability
        if use_replay:
            try:
                batch = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                try:
                    batch = next(replay_iterator)
                except StopIteration as error:
                    raise ValueError("Replay loader produced no batches") from error
            yield batch, True
            continue
        try:
            yield next(primary_iterator), False
        except StopIteration:
            return


def token_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> tuple[int, int]:
    labels = input_ids[:, 1:]
    predictions = logits.argmax(dim=-1)
    valid = labels != pad_token_id
    return int(((predictions == labels) & valid).sum()), int(valid.sum())


def traversal_control_token_ids(tokenizer: object) -> set[int]:
    if not isinstance(tokenizer, ChemTraverseTokenizerV01):
        return set()
    tokens = (
        tokenizer.COMPONENT,
        tokenizer.DOWN,
        tokenizer.UP,
        tokenizer.LINK,
        tokenizer.COMPONENT_END,
        tokenizer.MOL_END,
        tokenizer.EOS,
    )
    return {tokenizer.token_to_id[token] for token in tokens}


def weighted_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
    weighted_token_ids: set[int],
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids[:, 1:]
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).reshape_as(labels)
    valid = labels != pad_token_id
    weights = valid.to(token_losses.dtype)
    if weight != 1.0:
        emphasized = torch.zeros_like(valid)
        for token_id in weighted_token_ids:
            emphasized |= labels == token_id
        weights = weights * torch.where(
            emphasized,
            weights.new_tensor(weight),
            weights.new_tensor(1.0),
        )
    loss_per_sample = (token_losses * weights).sum(dim=1) / weights.sum(
        dim=1
    ).clamp_min(1)
    loss = (token_losses * weights).sum() / weights.sum().clamp_min(1)
    return loss, loss_per_sample


def add_chemtraverse_decision_loss_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the opt-in ChemTraverse objective without changing defaults."""

    parser.add_argument(
        "--chemtraverse-decision-loss",
        action="store_true",
        help="Use grammar-masked CE only at non-forced ChemTraverse decisions",
    )
    parser.add_argument(
        "--chemtraverse-decision-frequencies",
        help=(
            "Train-only JSON frequencies for clipped inverse-sqrt decision "
            "token weighting"
        ),
    )


def validate_chemtraverse_decision_loss_arguments(args: argparse.Namespace) -> None:
    enabled = bool(args.chemtraverse_decision_loss)
    frequency_path = args.chemtraverse_decision_frequencies
    if enabled and args.sequence_version != "traverse-0.1":
        raise ValueError("ChemTraverse decision loss requires --sequence-version traverse-0.1")
    if enabled and not frequency_path:
        raise ValueError(
            "--chemtraverse-decision-frequencies is required with "
            "--chemtraverse-decision-loss"
        )
    if frequency_path and not enabled:
        raise ValueError(
            "--chemtraverse-decision-frequencies requires "
            "--chemtraverse-decision-loss"
        )
    if enabled and args.traversal_control_loss_weight != 1.0:
        raise ValueError(
            "ChemTraverse decision loss and uniform traversal control weighting "
            "are mutually exclusive"
        )
    if enabled and not args.target_molecule_id:
        raise ValueError(
            "ChemTraverse decision loss requires an explicit non-empty "
            "--target-molecule-id sentinel"
        )


def load_chemtraverse_decision_objective(
    path: str | Path,
    tokenizer: ChemTraverseTokenizerV01,
    *,
    expected_target_molecule_id: str | None = None,
) -> tuple[dict[int, float], dict]:
    """Load, validate, and freeze one train-only decision-frequency artifact."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "chemtraverse-decision-frequencies-0.1":
        raise ValueError("Unsupported ChemTraverse decision frequency schema")
    if payload.get("sequence_schema_version") != tokenizer.sequence_schema_version:
        raise ValueError("Decision frequencies and tokenizer sequence schemas differ")
    target_molecule_id = payload.get("target_molecule_id")
    if not isinstance(target_molecule_id, str) or not target_molecule_id:
        raise ValueError("Decision frequencies must bind a non-empty target_molecule_id")
    if (
        expected_target_molecule_id is not None
        and target_molecule_id != expected_target_molecule_id
    ):
        raise ValueError("Decision frequency artifact and --target-molecule-id differ")
    raw_frequencies = payload.get("decision_frequencies")
    if not isinstance(raw_frequencies, dict) or not raw_frequencies:
        raise ValueError("decision_frequencies must be a non-empty object")
    frequencies: dict[str, int] = {}
    for token, value in raw_frequencies.items():
        if token not in tokenizer.token_to_id:
            raise ValueError(f"Unknown token in decision frequencies: {token!r}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Decision frequencies must be nonnegative integers")
        frequencies[token] = value
    weights = balanced_control_token_weights(tokenizer, frequencies)
    control_frequencies = {
        tokenizer.id_to_token[token_id]: frequencies.get(
            tokenizer.id_to_token[token_id], 0
        )
        for token_id in weights
    }
    weight_names = {
        tokenizer.id_to_token[token_id]: weight for token_id, weight in weights.items()
    }
    objective = {
        "schema_version": "chemtraverse-decision-loss-0.1",
        "name": "grammar-aware-decision-balanced-ce",
        "sequence_schema_version": tokenizer.sequence_schema_version,
        "target_molecule_id": target_molecule_id,
        "forced_position_weight": 0.0,
        "masked_softmax": True,
        "weight_formula": "clip(sqrt(max_frequency/frequency),1,4)",
        "frequency_source": file_record(source),
        "control_decision_frequencies": control_frequencies,
        "control_token_weights": weight_names,
    }
    return weights, objective


def validate_resume_training_objective(
    checkpoint: dict,
    training_objective: dict | None,
) -> None:
    if checkpoint.get("training_objective") != training_objective:
        raise ValueError(
            "Resume checkpoint training objective does not match the current "
            "ChemTraverse decision-loss configuration"
        )


def apply_training_sequence_loss(
    output: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    tokenizer: object,
    *,
    traversal_token_ids: set[int] | None = None,
    traversal_control_loss_weight: float = 1.0,
    decision_token_weights: dict[int, float] | None = None,
    decision_plan: ChemTraverseDecisionPlan | None = None,
):
    """Apply exactly one optional sequence objective and return its diagnostics."""

    if decision_token_weights is not None:
        if traversal_control_loss_weight != 1.0:
            raise ValueError("Decision-balanced and uniform control losses cannot be combined")
        if not isinstance(tokenizer, ChemTraverseTokenizerV01):
            raise ValueError("Decision-balanced loss requires a ChemTraverse tokenizer")
        result = grammar_aware_decision_balanced_loss(
            output["logits"],
            input_ids,
            tokenizer,
            token_weights=decision_token_weights,
            plan=decision_plan,
        )
        output["loss"] = result.loss
        output["loss_per_sample"] = result.loss_per_sample
        return result
    if traversal_control_loss_weight != 1.0:
        output["loss"], output["loss_per_sample"] = weighted_token_loss(
            output["logits"],
            input_ids,
            tokenizer.pad_token_id,
            traversal_token_ids or set(),
            traversal_control_loss_weight,
        )
    return None


def ocr_token_accuracy(
    output: dict[str, torch.Tensor],
    pad_token_id: int = 0,
) -> tuple[int, int]:
    labels = output["ocr_labels"]
    predictions = output["ocr_logits"].argmax(dim=-1)
    valid = labels != pad_token_id
    return int(((predictions == labels) & valid).sum()), int(valid.sum())


def ocr_sequence_accuracy(output: dict[str, torch.Tensor]) -> dict[str, int]:
    labels = output["ocr_labels"]
    predictions = output["ocr_logits"].argmax(dim=-1)
    valid = labels != OCRTextCodec.PAD_TOKEN_ID
    exact = ((predictions == labels) | ~valid).all(dim=1)
    visible = labels[:, 0] != OCRTextCodec.EOS_TOKEN_ID
    return {
        "exact": int(exact.sum()),
        "anchors": int(exact.numel()),
        "visible_exact": int((exact & visible).sum()),
        "visible_anchors": int(visible.sum()),
        "blank_exact": int((exact & ~visible).sum()),
        "blank_anchors": int((~visible).sum()),
    }


def synthetic_ocr_replay_forward(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if model.ocr_head is None:
        raise ValueError("Synthetic OCR replay requires an OCR-enabled model")
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    anchor_points = batch["ocr_anchor_points"].to(device, non_blocking=True)
    anchor_mask = batch["ocr_anchor_mask"].to(device, non_blocking=True)
    token_ids = batch["ocr_token_ids"].to(device, non_blocking=True)
    with torch.no_grad():
        features = model.ocr_source(pixel_values)
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
    max_batches: int,
    *,
    tokenizer: object | None = None,
    decision_token_weights: dict[int, float] | None = None,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    tokens = 0
    ocr_loss_total = 0.0
    ocr_correct = 0
    ocr_tokens = 0
    ocr_exact = 0
    ocr_anchors = 0
    ocr_visible_exact = 0
    ocr_visible_anchors = 0
    ocr_blank_exact = 0
    ocr_blank_anchors = 0
    decision_positions = 0
    forced_positions = 0
    batches = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        ocr_arguments = {}
        if model.config.ocr_enabled:
            ocr_arguments = {
                "ocr_anchor_points": batch["ocr_anchor_points"].to(
                    device, non_blocking=True
                ),
                "ocr_anchor_mask": batch["ocr_anchor_mask"].to(
                    device, non_blocking=True
                ),
                "ocr_token_ids": batch["ocr_token_ids"].to(device, non_blocking=True),
            }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values, input_ids, attention_mask, **ocr_arguments)
            decision_result = None
            if decision_token_weights is not None:
                if tokenizer is None:
                    raise ValueError("Decision loss validation requires a tokenizer")
                decision_result = apply_training_sequence_loss(
                    output,
                    input_ids,
                    tokenizer,
                    decision_token_weights=decision_token_weights,
                )
        loss_total += float(output["loss"])
        batch_correct, batch_tokens = token_accuracy(
            output["logits"], input_ids, model.config.pad_token_id
        )
        correct += batch_correct
        tokens += batch_tokens
        if decision_result is not None:
            decision_positions += decision_result.decision_positions
            forced_positions += decision_result.forced_positions
        if model.config.ocr_enabled:
            ocr_loss_total += float(output["ocr_loss"])
            batch_ocr_correct, batch_ocr_tokens = ocr_token_accuracy(output)
            ocr_correct += batch_ocr_correct
            ocr_tokens += batch_ocr_tokens
            sequence_metrics = ocr_sequence_accuracy(output)
            ocr_exact += sequence_metrics["exact"]
            ocr_anchors += sequence_metrics["anchors"]
            ocr_visible_exact += sequence_metrics["visible_exact"]
            ocr_visible_anchors += sequence_metrics["visible_anchors"]
            ocr_blank_exact += sequence_metrics["blank_exact"]
            ocr_blank_anchors += sequence_metrics["blank_anchors"]
        batches += 1
        if batches >= max_batches:
            break
    model.train()
    metrics = {
        "loss": loss_total / max(batches, 1),
        "token_accuracy": correct / max(tokens, 1),
        "batches": batches,
    }
    if decision_token_weights is not None:
        metrics.update(
            {
                "decision_positions": decision_positions,
                "forced_positions": forced_positions,
                "decision_fraction": decision_positions
                / max(decision_positions + forced_positions, 1),
            }
        )
    if model.config.ocr_enabled:
        metrics.update(
            {
                "ocr_loss": ocr_loss_total / max(batches, 1),
                "ocr_token_accuracy": ocr_correct / max(ocr_tokens, 1),
                "ocr_exact_accuracy": ocr_exact / max(ocr_anchors, 1),
                "ocr_visible_exact_accuracy": ocr_visible_exact
                / max(ocr_visible_anchors, 1),
                "ocr_blank_exact_accuracy": ocr_blank_exact
                / max(ocr_blank_anchors, 1),
                "ocr_tokens": ocr_tokens,
            }
        )
    return metrics


def save_checkpoint(
    path: Path,
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    *,
    epoch_complete: bool,
    samples_in_epoch: int,
    run_metadata: dict | None = None,
    training_objective: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(model.config),
        "epoch": epoch,
        "global_step": global_step,
        "epoch_complete": epoch_complete,
        "samples_in_epoch": samples_in_epoch,
        "rng_state": capture_rng_state(),
        "run_metadata": run_metadata,
    }
    if training_objective is not None:
        payload["training_objective"] = training_objective
    torch.save(payload, path)


def slice_batch(batch: dict, start: int = 0, stop: int | None = None) -> dict:
    """Slice tensors and sample metadata along the batch dimension."""
    trimmed = {}
    for key, value in batch.items():
        trimmed[key] = value[start:stop]
    return trimmed


def optimizer_step(
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    accumulated_batches: int,
    gradient_accumulation: int,
) -> None:
    """Apply accumulated gradients, correcting a short final accumulation."""
    if accumulated_batches < gradient_accumulation:
        scale = gradient_accumulation / accumulated_batches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def build_optimizer(
    model: ChemVectorModel,
    learning_rate: float,
    weight_decay: float,
    ocr_learning_rate_multiplier: float,
) -> torch.optim.Optimizer:
    if model.ocr_head is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    ocr_parameters = list(model.ocr_head.parameters())
    ocr_parameter_ids = {id(parameter) for parameter in ocr_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in ocr_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": learning_rate, "name": "base"},
            {
                "params": ocr_parameters,
                "lr": learning_rate * ocr_learning_rate_multiplier,
                "name": "ocr",
            },
        ],
        weight_decay=weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/train-*.tar"],
    )
    parser.add_argument("--replay-shards", nargs="+")
    parser.add_argument("--replay-probability", type=float, default=0.0)
    parser.add_argument("--validation-shards", nargs="+")
    parser.add_argument("--output-dir", default="outputs/checkpoints/chemvector")
    parser.add_argument(
        "--sequence-version",
        choices=("0.2", "0.3", "0.4", "traverse-0.1"),
        default="0.2",
    )
    parser.add_argument("--reject-keys-file")
    parser.add_argument(
        "--sample-manifest",
        help="JSONL manifest whose sample_key values are the only allowed training samples",
    )
    parser.add_argument(
        "--validation-sample-manifest",
        help="JSONL manifest whose sample_key values are the only allowed validation samples",
    )
    parser.add_argument(
        "--target-molecule-id",
        help="Replace unobservable ChemTraverse molecule_id with a fixed target value",
    )
    parser.add_argument(
        "--data-commitment",
        nargs="*",
        default=[],
        help="Committed dataset metadata files to bind into run_manifest.json",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--token-corruption-probability", type=float, default=0.0)
    parser.add_argument(
        "--traversal-control-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for ChemTraverse branch/link/end control tokens",
    )
    add_chemtraverse_decision_loss_arguments(parser)
    parser.add_argument("--image-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--image-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--visual-aux-weight", type=float, default=0.0)
    parser.add_argument("--visual-aux-positive-weight", type=float, default=20.0)
    parser.add_argument("--ocr-weight", type=float, default=0.0)
    parser.add_argument("--ocr-learning-rate-multiplier", type=float, default=10.0)
    parser.add_argument("--synthetic-ocr-replay-weight", type=float, default=0.0)
    parser.add_argument("--synthetic-ocr-replay-batch-size", type=int, default=64)
    parser.add_argument("--synthetic-ocr-replay-workers", type=int, default=0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CUDA algorithms for reproducible pilot and resume runs",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--init-checkpoint",
        help="Initialize model weights without restoring optimizer or training position",
    )
    parser.add_argument(
        "--init-vision-checkpoint",
        help="Initialize only vision_encoder weights; tokenizer and decoder may differ",
    )
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        help="Override the optimizer learning rate stored in a resumed checkpoint",
    )
    args = parser.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if args.train_samples_per_epoch is not None and args.train_samples_per_epoch < 1:
        raise ValueError("--train-samples-per-epoch must be at least 1")
    if not 0.0 <= args.replay_probability < 1.0:
        raise ValueError("--replay-probability must be in [0, 1)")
    if args.replay_probability and not args.replay_shards:
        raise ValueError("--replay-shards is required when replay is enabled")
    if args.replay_shards and not args.replay_probability:
        raise ValueError("--replay-probability must be positive with replay shards")
    if not 0.0 <= args.token_corruption_probability < 1.0:
        raise ValueError("--token-corruption-probability must be in [0, 1)")
    if args.traversal_control_loss_weight < 1.0:
        raise ValueError("--traversal-control-loss-weight must be at least 1")
    if (
        args.traversal_control_loss_weight != 1.0
        and args.sequence_version != "traverse-0.1"
    ):
        raise ValueError("Traversal control weighting requires ChemTraverse v0.1")
    if args.target_molecule_id is not None and args.sequence_version != "traverse-0.1":
        raise ValueError("Fixed target molecule ID requires ChemTraverse v0.1")
    if args.target_molecule_id == "":
        raise ValueError("--target-molecule-id must be non-empty")
    validate_chemtraverse_decision_loss_arguments(args)
    if args.image_contrastive_weight < 0 or args.image_contrastive_margin < 0:
        raise ValueError("Image contrastive settings must be nonnegative")
    if args.visual_aux_weight < 0 or args.visual_aux_positive_weight <= 0:
        raise ValueError("Visual auxiliary settings must be positive or zero")
    if args.ocr_weight < 0 or args.ocr_learning_rate_multiplier <= 0:
        raise ValueError("OCR weight must be nonnegative and its LR multiplier positive")
    if args.synthetic_ocr_replay_weight < 0:
        raise ValueError("Synthetic OCR replay weight must be nonnegative")
    if args.synthetic_ocr_replay_batch_size < 1:
        raise ValueError("Synthetic OCR replay batch size must be positive")
    if args.synthetic_ocr_replay_workers < 0:
        raise ValueError("Synthetic OCR replay workers must be nonnegative")
    if args.synthetic_ocr_replay_weight and not args.ocr_weight:
        raise ValueError("Synthetic OCR replay requires --ocr-weight")
    initialization_modes = sum(
        value is not None
        for value in (args.resume, args.init_checkpoint, args.init_vision_checkpoint)
    )
    if initialization_modes > 1:
        raise ValueError(
            "--resume, --init-checkpoint and --init-vision-checkpoint are mutually exclusive"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    tokenizers = {
        "0.2": ChemMolTokenizer,
        "0.3": ChemMolTokenizerV3,
        "0.4": ChemSVGTokenizerV4,
        "traverse-0.1": ChemTraverseTokenizerV01,
    }
    tokenizer = tokenizers[args.sequence_version]()
    traversal_token_ids = traversal_control_token_ids(tokenizer)
    decision_token_weights = None
    training_objective = None
    if args.chemtraverse_decision_loss:
        decision_token_weights, training_objective = (
            load_chemtraverse_decision_objective(
                args.chemtraverse_decision_frequencies,
                tokenizer,
                expected_target_molecule_id=args.target_molecule_id,
            )
        )
    config = build_config(tokenizer, args.tiny, ocr_enabled=args.ocr_weight > 0)
    device = torch.device("cuda")
    model = ChemVectorModel(config).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        allowed_missing = {
            key for key in model.state_dict() if key.startswith("ocr_head.")
        }
        if unexpected or missing - allowed_missing:
            raise ValueError(
                "Initial checkpoint architecture mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        print(
            f"initialized_from={args.init_checkpoint} "
            f"new_ocr_parameters={len(missing)}"
        )
    elif args.init_vision_checkpoint:
        loaded = load_vision_encoder_checkpoint(model, args.init_vision_checkpoint)
        print(
            f"vision_initialized_from={args.init_vision_checkpoint} "
            f"vision_tensors={loaded}"
        )
    optimizer = build_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
        args.ocr_learning_rate_multiplier,
    )
    start_epoch = 0
    resume_samples_in_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_training_objective(checkpoint, training_objective)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.resume_learning_rate is not None:
            if args.resume_learning_rate <= 0:
                raise ValueError("--resume-learning-rate must be positive")
            for parameter_group in optimizer.param_groups:
                multiplier = (
                    args.ocr_learning_rate_multiplier
                    if parameter_group.get("name") == "ocr"
                    else 1.0
                )
                parameter_group["lr"] = args.resume_learning_rate * multiplier
        epoch_complete = checkpoint.get("epoch_complete", True)
        start_epoch = checkpoint["epoch"] + int(epoch_complete)
        resume_samples_in_epoch = 0 if epoch_complete else checkpoint.get(
            "samples_in_epoch", 0
        )
        global_step = checkpoint["global_step"]
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

    train_shards = expand_shards(args.train_shards)
    replay_shards = expand_shards(args.replay_shards) if args.replay_shards else []
    validation_shards = expand_shards(args.validation_shards) if args.validation_shards else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_metadata = {
        "schema_version": "chemvector-training-run-0.1",
        "git": git_state(),
        "tokenizer": file_record(output_dir / "tokenizer.json"),
        "sample_manifest": file_record(args.sample_manifest),
        "validation_sample_manifest": file_record(args.validation_sample_manifest),
        "data_commitments": [file_record(path) for path in args.data_commitment],
        "initialization": file_record(
            args.init_vision_checkpoint or args.init_checkpoint or args.resume
        ),
        "train_shards": train_shards,
        "validation_shards": validation_shards,
        "seed": args.seed,
    }
    if training_objective is not None:
        run_metadata["training_objective"] = training_objective
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"train_shards={len(train_shards)}")
    print(f"replay_shards={len(replay_shards)}")
    print(f"replay_probability={args.replay_probability:.4f}")
    print(f"validation_shards={len(validation_shards)}")
    if training_objective is not None:
        print(
            "chemtraverse_decision_loss=enabled "
            f"frequency_sha256={training_objective['frequency_source']['sha256']} "
            "weights="
            + json.dumps(
                training_objective["control_token_weights"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    print(f"reject_keys={len(reject_keys)}")
    include_keys = load_sample_keys(args.sample_manifest)
    validation_include_keys = load_sample_keys(args.validation_sample_manifest)
    print(f"include_keys={len(include_keys) if include_keys is not None else 'all'}")
    print(
        "validation_include_keys="
        f"{len(validation_include_keys) if validation_include_keys is not None else 'all'}"
    )
    if resume_samples_in_epoch:
        if not args.no_shuffle:
            raise ValueError(
                "Resuming a mid-epoch checkpoint requires --no-shuffle so samples can "
                "be skipped deterministically"
            )
        print(
            f"resume_epoch={start_epoch} "
            f"skip_samples={resume_samples_in_epoch}"
        )

    validation_loader = None
    if validation_shards:
        validation_loader = make_loader(
            validation_shards,
            tokenizer,
            config,
            batch_size=args.batch_size,
            image_size=args.image_size,
            workers=args.workers,
            shuffle=False,
            reject_keys=reject_keys,
            include_keys=validation_include_keys,
            loader_seed=args.seed + 10_000,
            target_molecule_id=args.target_molecule_id,
        )

    synthetic_ocr_loader = None
    synthetic_ocr_iterator = None
    if args.synthetic_ocr_replay_weight:
        synthetic_codec = OCRTextCodec(max_length=config.ocr_max_length)
        synthetic_ocr_loader = DataLoader(
            SyntheticOCRDataset(10_000_000, seed=args.seed + 17),
            batch_size=args.synthetic_ocr_replay_batch_size,
            num_workers=args.synthetic_ocr_replay_workers,
            pin_memory=True,
            persistent_workers=args.synthetic_ocr_replay_workers > 0,
            collate_fn=partial(collate_synthetic_ocr, codec=synthetic_codec),
        )
        synthetic_ocr_iterator = iter(synthetic_ocr_loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_loader = make_loader(
            train_shards,
            tokenizer,
            config,
            batch_size=args.batch_size,
            image_size=args.image_size,
            workers=args.workers,
            shuffle=not args.no_shuffle,
            reject_keys=reject_keys,
            include_keys=include_keys,
            loader_seed=args.seed + epoch,
            target_molecule_id=args.target_molecule_id,
        )
        replay_loader = None
        if replay_shards:
            replay_loader = make_loader(
                replay_shards,
                tokenizer,
                config,
                batch_size=args.batch_size,
                image_size=args.image_size,
                workers=args.workers,
                shuffle=not args.no_shuffle,
                reject_keys=reject_keys,
                loader_seed=args.seed + 20_000 + epoch,
                target_molecule_id=args.target_molecule_id,
            )
        epoch_samples = resume_samples_in_epoch
        replay_samples = 0
        running_loss = 0.0
        running_correct = 0
        running_tokens = 0
        running_contrastive = 0.0
        running_visual_aux = 0.0
        running_ocr = 0.0
        running_ocr_correct = 0
        running_ocr_tokens = 0
        running_ocr_exact = 0
        running_ocr_anchors = 0
        running_ocr_visible_exact = 0
        running_ocr_visible_anchors = 0
        running_ocr_blank_exact = 0
        running_ocr_blank_anchors = 0
        running_synthetic_ocr = 0.0
        running_synthetic_visible_exact = 0
        running_synthetic_visible_anchors = 0
        running_decision_positions = 0
        running_forced_positions = 0
        running_decision_weight = 0.0
        logged_steps = 0
        accumulated_batches = 0
        skipped_samples = 0
        previous_memory = None
        epoch_started = time.perf_counter()
        batches = iter_mixed_batches(
            train_loader,
            replay_loader,
            args.replay_probability,
        )
        for batch_index, (batch, is_replay) in enumerate(batches, 1):
            batch_size = batch["input_ids"].size(0)
            if skipped_samples < resume_samples_in_epoch:
                remaining_to_skip = resume_samples_in_epoch - skipped_samples
                if remaining_to_skip >= batch_size:
                    skipped_samples += batch_size
                    continue
                batch = slice_batch(batch, start=remaining_to_skip)
                skipped_samples += remaining_to_skip
                batch_size = batch["input_ids"].size(0)
            if args.train_samples_per_epoch is not None:
                remaining_samples = args.train_samples_per_epoch - epoch_samples
                if remaining_samples <= 0:
                    break
                if batch_size > remaining_samples:
                    batch = slice_batch(batch, stop=remaining_samples)

            decision_plan = (
                build_chemtraverse_decision_plan(batch["input_ids"], tokenizer)
                if decision_token_weights is not None
                else None
            )
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            visual_targets = batch["visual_targets"].to(device, non_blocking=True)
            ocr_arguments = {}
            if config.ocr_enabled:
                ocr_arguments = {
                    "ocr_anchor_points": batch["ocr_anchor_points"].to(
                        device, non_blocking=True
                    ),
                    "ocr_anchor_mask": batch["ocr_anchor_mask"].to(
                        device, non_blocking=True
                    ),
                    "ocr_token_ids": batch["ocr_token_ids"].to(
                        device, non_blocking=True
                    ),
                }
            synthetic_batch = (
                next(synthetic_ocr_iterator)
                if synthetic_ocr_iterator is not None
                else None
            )
            teacher_forcing_ids = input_ids
            if args.token_corruption_probability:
                teacher_forcing_ids = input_ids.clone()
                corruption = (
                    torch.rand(input_ids.shape, device=device)
                    < args.token_corruption_probability
                ) & attention_mask.bool()
                corruption[:, 0] = False
                teacher_forcing_ids[corruption] = tokenizer.unk_token_id
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    pixel_values,
                    input_ids,
                    attention_mask,
                    teacher_forcing_ids=teacher_forcing_ids,
                    **ocr_arguments,
                )
                decision_loss_output = apply_training_sequence_loss(
                    output,
                    input_ids,
                    tokenizer,
                    traversal_token_ids=traversal_token_ids,
                    traversal_control_loss_weight=args.traversal_control_loss_weight,
                    decision_token_weights=decision_token_weights,
                    decision_plan=decision_plan,
                )
                contrastive_loss = output["loss"].new_zeros(())
                if args.image_contrastive_weight and input_ids.size(0) > 1:
                    negative_memory = (
                        previous_memory
                        if previous_memory is not None
                        and previous_memory.shape == output["memory"].shape
                        else output["memory"].roll(1, dims=0).detach()
                    )
                    mismatched = model(
                        pixel_values,
                        input_ids,
                        attention_mask,
                        memory=negative_memory,
                        teacher_forcing_ids=teacher_forcing_ids,
                    )
                    apply_training_sequence_loss(
                        mismatched,
                        input_ids,
                        tokenizer,
                        traversal_token_ids=traversal_token_ids,
                        traversal_control_loss_weight=args.traversal_control_loss_weight,
                        decision_token_weights=decision_token_weights,
                        decision_plan=decision_plan,
                    )
                    contrastive_loss = torch.relu(
                        args.image_contrastive_margin
                        + output["loss_per_sample"]
                        - mismatched["loss_per_sample"]
                    ).mean()
                previous_memory = output["memory"].detach()
                visual_aux_loss = model.visual_auxiliary_loss(
                    output["visual_logits"],
                    visual_targets,
                    args.visual_aux_positive_weight,
                )
                ocr_loss = (
                    output["ocr_loss"]
                    if config.ocr_enabled
                    else output["loss"].new_zeros(())
                )
                synthetic_ocr_output = None
                synthetic_ocr_loss = output["loss"].new_zeros(())
                if synthetic_batch is not None:
                    synthetic_ocr_output = synthetic_ocr_replay_forward(
                        model,
                        synthetic_batch,
                        device,
                    )
                    synthetic_ocr_loss = synthetic_ocr_output["ocr_loss"]
                combined_loss = (
                    output["loss"]
                    + args.image_contrastive_weight * contrastive_loss
                    + args.visual_aux_weight * visual_aux_loss
                    + args.ocr_weight * ocr_loss
                    + args.synthetic_ocr_replay_weight * synthetic_ocr_loss
                )
                loss = combined_loss / args.gradient_accumulation
            loss.backward()
            accumulated_batches += 1
            correct, tokens = token_accuracy(
                output["logits"].detach(), input_ids, config.pad_token_id
            )
            running_loss += float(output["loss"].detach())
            if decision_loss_output is not None:
                running_decision_positions += decision_loss_output.decision_positions
                running_forced_positions += decision_loss_output.forced_positions
                running_decision_weight += float(
                    decision_loss_output.weight_sum.detach()
                )
            running_correct += correct
            running_tokens += tokens
            running_contrastive += float(contrastive_loss.detach())
            running_visual_aux += float(visual_aux_loss.detach())
            running_ocr += float(ocr_loss.detach())
            if config.ocr_enabled:
                ocr_correct, ocr_tokens = ocr_token_accuracy(output)
                running_ocr_correct += ocr_correct
                running_ocr_tokens += ocr_tokens
                sequence_metrics = ocr_sequence_accuracy(output)
                running_ocr_exact += sequence_metrics["exact"]
                running_ocr_anchors += sequence_metrics["anchors"]
                running_ocr_visible_exact += sequence_metrics["visible_exact"]
                running_ocr_visible_anchors += sequence_metrics["visible_anchors"]
                running_ocr_blank_exact += sequence_metrics["blank_exact"]
                running_ocr_blank_anchors += sequence_metrics["blank_anchors"]
            running_synthetic_ocr += float(synthetic_ocr_loss.detach())
            if synthetic_ocr_output is not None:
                synthetic_metrics = ocr_sequence_accuracy(synthetic_ocr_output)
                running_synthetic_visible_exact += synthetic_metrics["visible_exact"]
                running_synthetic_visible_anchors += synthetic_metrics[
                    "visible_anchors"
                ]
            logged_steps += 1
            epoch_samples += input_ids.size(0)
            if is_replay:
                replay_samples += input_ids.size(0)

            if accumulated_batches == args.gradient_accumulation:
                optimizer_step(
                    model,
                    optimizer,
                    accumulated_batches,
                    args.gradient_accumulation,
                )
                accumulated_batches = 0
                global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - epoch_started
                    decision_log = ""
                    if decision_token_weights is not None:
                        decision_log = (
                            f" decision_positions={running_decision_positions}"
                            f" forced_positions={running_forced_positions}"
                            " decision_fraction="
                            f"{running_decision_positions / max(running_decision_positions + running_forced_positions, 1):.6f}"
                            " decision_weight_mean="
                            f"{running_decision_weight / max(running_decision_positions, 1):.6f}"
                        )
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={running_loss / max(logged_steps, 1):.6f} "
                        f"contrastive={running_contrastive / max(logged_steps, 1):.6f} "
                        f"visual_aux={running_visual_aux / max(logged_steps, 1):.6f} "
                        f"ocr={running_ocr / max(logged_steps, 1):.6f} "
                        f"token_accuracy={running_correct / max(running_tokens, 1):.6f} "
                        f"ocr_token_accuracy={running_ocr_correct / max(running_ocr_tokens, 1):.6f} "
                        f"ocr_exact={running_ocr_exact / max(running_ocr_anchors, 1):.6f} "
                        f"ocr_visible_exact={running_ocr_visible_exact / max(running_ocr_visible_anchors, 1):.6f} "
                        f"ocr_blank_exact={running_ocr_blank_exact / max(running_ocr_blank_anchors, 1):.6f} "
                        f"synthetic_ocr={running_synthetic_ocr / max(logged_steps, 1):.6f} "
                        f"synthetic_visible_exact={running_synthetic_visible_exact / max(running_synthetic_visible_anchors, 1):.6f} "
                        f"samples={epoch_samples} samples_per_second={epoch_samples / elapsed:.2f}"
                        f" replay_samples={replay_samples}"
                        f" replay_fraction={replay_samples / max(epoch_samples, 1):.4f}"
                        f" learning_rate={optimizer.param_groups[0]['lr']:.8g}"
                        f"{decision_log}"
                    )
                    running_loss = 0.0
                    running_correct = 0
                    running_tokens = 0
                    running_contrastive = 0.0
                    running_visual_aux = 0.0
                    running_ocr = 0.0
                    running_ocr_correct = 0
                    running_ocr_tokens = 0
                    running_ocr_exact = 0
                    running_ocr_anchors = 0
                    running_ocr_visible_exact = 0
                    running_ocr_visible_anchors = 0
                    running_ocr_blank_exact = 0
                    running_ocr_blank_anchors = 0
                    running_synthetic_ocr = 0.0
                    running_synthetic_visible_exact = 0
                    running_synthetic_visible_anchors = 0
                    running_decision_positions = 0
                    running_forced_positions = 0
                    running_decision_weight = 0.0
                    logged_steps = 0
                if args.save_every and global_step % args.save_every == 0:
                    save_checkpoint(
                        output_dir / f"checkpoint-{global_step:07d}.pt",
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        epoch_complete=False,
                        samples_in_epoch=epoch_samples,
                        run_metadata=run_metadata,
                        training_objective=training_objective,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
            if (
                args.train_samples_per_epoch is not None
                and epoch_samples >= args.train_samples_per_epoch
            ):
                break

        if accumulated_batches and not stop:
            optimizer_step(
                model,
                optimizer,
                accumulated_batches,
                args.gradient_accumulation,
            )
            global_step += 1

        resume_samples_in_epoch = 0

        if validation_loader is not None:
            metrics = validate(
                model,
                validation_loader,
                device,
                args.validation_batches,
                tokenizer=tokenizer,
                decision_token_weights=decision_token_weights,
            )
            validation_message = (
                f"validation epoch={epoch} loss={metrics['loss']:.6f} "
                f"token_accuracy={metrics['token_accuracy']:.6f} "
            )
            if config.ocr_enabled:
                validation_message += (
                    f"ocr_loss={metrics['ocr_loss']:.6f} "
                    f"ocr_token_accuracy={metrics['ocr_token_accuracy']:.6f} "
                    f"ocr_exact={metrics['ocr_exact_accuracy']:.6f} "
                    f"ocr_visible_exact={metrics['ocr_visible_exact_accuracy']:.6f} "
                    f"ocr_blank_exact={metrics['ocr_blank_exact_accuracy']:.6f} "
                )
            if decision_token_weights is not None:
                validation_message += (
                    f"decision_positions={int(metrics['decision_positions'])} "
                    f"forced_positions={int(metrics['forced_positions'])} "
                    f"decision_fraction={metrics['decision_fraction']:.6f} "
                )
            validation_message += f"batches={int(metrics['batches'])}"
            print(validation_message)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            epoch_complete=not stop,
            samples_in_epoch=epoch_samples,
            run_metadata=run_metadata,
            training_objective=training_objective,
        )
        if stop:
            break


if __name__ == "__main__":
    main()
