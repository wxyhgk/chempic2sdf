#!/usr/bin/env python3
"""Single-GPU ChemVector training entry point."""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from chemvector.data import ChemVectorCollator, ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel


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


def build_config(tokenizer, tiny: bool) -> ChemVectorConfig:
    common = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
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
) -> DataLoader:
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=shuffle,
        sample_shuffle_buffer=1000 if shuffle else 0,
        sequence_version=(
            tokenizer.sequence_schema_version.removeprefix("chemmolseq-").removeprefix(
                "chemsvg-"
            )
        ),
        reject_keys=reject_keys,
    )
    collator = ChemVectorCollator(
        tokenizer,
        image_size=image_size,
        max_length=config.max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collator,
    )


def token_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> tuple[int, int]:
    labels = input_ids[:, 1:]
    predictions = logits.argmax(dim=-1)
    valid = labels != pad_token_id
    return int(((predictions == labels) & valid).sum()), int(valid.sum())


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    tokens = 0
    batches = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values, input_ids, attention_mask)
        loss_total += float(output["loss"])
        batch_correct, batch_tokens = token_accuracy(
            output["logits"], input_ids, model.config.pad_token_id
        )
        correct += batch_correct
        tokens += batch_tokens
        batches += 1
        if batches >= max_batches:
            break
    model.train()
    return {
        "loss": loss_total / max(batches, 1),
        "token_accuracy": correct / max(tokens, 1),
        "batches": batches,
    }


def save_checkpoint(
    path: Path,
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    *,
    epoch_complete: bool,
    samples_in_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "epoch": epoch,
            "global_step": global_step,
            "epoch_complete": epoch_complete,
            "samples_in_epoch": samples_in_epoch,
        },
        path,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/train-*.tar"],
    )
    parser.add_argument("--validation-shards", nargs="+")
    parser.add_argument("--output-dir", default="outputs/chemvector")
    parser.add_argument(
        "--sequence-version",
        choices=("0.2", "0.3", "0.4"),
        default="0.2",
    )
    parser.add_argument("--reject-keys-file")
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
    parser.add_argument("--image-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--image-contrastive-margin", type=float, default=0.5)
    parser.add_argument("--visual-aux-weight", type=float, default=0.0)
    parser.add_argument("--visual-aux-positive-weight", type=float, default=20.0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        help="Override the optimizer learning rate stored in a resumed checkpoint",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if args.train_samples_per_epoch is not None and args.train_samples_per_epoch < 1:
        raise ValueError("--train-samples-per-epoch must be at least 1")
    if not 0.0 <= args.token_corruption_probability < 1.0:
        raise ValueError("--token-corruption-probability must be in [0, 1)")
    if args.image_contrastive_weight < 0 or args.image_contrastive_margin < 0:
        raise ValueError("Image contrastive settings must be nonnegative")
    if args.visual_aux_weight < 0 or args.visual_aux_positive_weight <= 0:
        raise ValueError("Visual auxiliary settings must be positive or zero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    tokenizers = {
        "0.2": ChemMolTokenizer,
        "0.3": ChemMolTokenizerV3,
        "0.4": ChemSVGTokenizerV4,
    }
    tokenizer = tokenizers[args.sequence_version]()
    config = build_config(tokenizer, args.tiny)
    device = torch.device("cuda")
    model = ChemVectorModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    resume_samples_in_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.resume_learning_rate is not None:
            if args.resume_learning_rate <= 0:
                raise ValueError("--resume-learning-rate must be positive")
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.resume_learning_rate
        epoch_complete = checkpoint.get("epoch_complete", True)
        start_epoch = checkpoint["epoch"] + int(epoch_complete)
        resume_samples_in_epoch = 0 if epoch_complete else checkpoint.get(
            "samples_in_epoch", 0
        )
        global_step = checkpoint["global_step"]

    train_shards = expand_shards(args.train_shards)
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
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"train_shards={len(train_shards)}")
    print(f"validation_shards={len(validation_shards)}")
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    print(f"reject_keys={len(reject_keys)}")
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
        )

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
        )
        epoch_samples = resume_samples_in_epoch
        running_loss = 0.0
        running_correct = 0
        running_tokens = 0
        running_contrastive = 0.0
        running_visual_aux = 0.0
        logged_steps = 0
        accumulated_batches = 0
        skipped_samples = 0
        previous_memory = None
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, 1):
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

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            visual_targets = batch["visual_targets"].to(device, non_blocking=True)
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
                combined_loss = (
                    output["loss"]
                    + args.image_contrastive_weight * contrastive_loss
                    + args.visual_aux_weight * visual_aux_loss
                )
                loss = combined_loss / args.gradient_accumulation
            loss.backward()
            accumulated_batches += 1
            correct, tokens = token_accuracy(
                output["logits"].detach(), input_ids, config.pad_token_id
            )
            running_loss += float(output["loss"].detach())
            running_correct += correct
            running_tokens += tokens
            running_contrastive += float(contrastive_loss.detach())
            running_visual_aux += float(visual_aux_loss.detach())
            logged_steps += 1
            epoch_samples += input_ids.size(0)

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
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={running_loss / max(logged_steps, 1):.6f} "
                        f"contrastive={running_contrastive / max(logged_steps, 1):.6f} "
                        f"visual_aux={running_visual_aux / max(logged_steps, 1):.6f} "
                        f"token_accuracy={running_correct / max(running_tokens, 1):.6f} "
                        f"samples={epoch_samples} samples_per_second={epoch_samples / elapsed:.2f}"
                        f" learning_rate={optimizer.param_groups[0]['lr']:.8g}"
                    )
                    running_loss = 0.0
                    running_correct = 0
                    running_tokens = 0
                    running_contrastive = 0.0
                    running_visual_aux = 0.0
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
            )
            print(
                f"validation epoch={epoch} loss={metrics['loss']:.6f} "
                f"token_accuracy={metrics['token_accuracy']:.6f} "
                f"batches={int(metrics['batches'])}"
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            epoch_complete=True,
            samples_in_epoch=epoch_samples,
        )
        if stop:
            break


if __name__ == "__main__":
    main()
