#!/usr/bin/env python3
"""Calibrate Open Text OCR with character-presence and paired-style losses."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from chemvector.ctc_ocr import ctc_loss
from chemvector.pretrain_open_text_ctc import (
    DEFAULT_DATA_ROOT,
    OpenTextDataset,
    collate_open_text,
    evaluate_checkpoint,
    evaluate_open_text,
    gate_results,
    load_initial_model,
    make_validation_loader,
    metric_score,
    optimizer_for_arm,
    serializable_args,
    training_weights,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/chemvector-v0811-open-text-full-1500k/full/best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/checkpoints/chemvector-v0812-counterfactual-200k"
)
PRESENCE_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-()[]*'? "
)
PRESENCE_TOKEN_IDS = tuple(ord(character) + 1 for character in PRESENCE_ALPHABET)
ARM_CONFIGS = {
    "ctc": {
        "presence_weight": 0.0,
        "pair_weight": 0.0,
        "pair_fraction": 0.0,
    },
    "presence": {
        "presence_weight": 0.20,
        "pair_weight": 0.0,
        "pair_fraction": 0.0,
    },
    "presence_pair": {
        "presence_weight": 0.20,
        "pair_weight": 0.20,
        "pair_fraction": 0.25,
    },
}


def character_presence_targets(
    texts: list[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    byte_to_index = {
        token_id - 1: index for index, token_id in enumerate(PRESENCE_TOKEN_IDS)
    }
    targets = torch.zeros(
        (len(texts), len(PRESENCE_TOKEN_IDS)),
        dtype=torch.float32,
        device=device,
    )
    for row, text in enumerate(texts):
        for value in set(text.encode("utf-8")):
            index = byte_to_index.get(value)
            if index is not None:
                targets[row, index] = 1.0
    return targets


def character_presence_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(-1)[..., list(PRESENCE_TOKEN_IDS)]
    log_absent = torch.log1p(-probabilities.clamp(max=1.0 - 1e-6)).sum(dim=1)
    return (-torch.expm1(log_absent)).clamp(1e-6, 1.0 - 1e-6)


def character_presence_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if positive_weight <= 0:
        raise ValueError("presence positive weight must be positive")
    return -(
        positive_weight * targets * probabilities.log()
        + (1.0 - targets) * torch.log1p(-probabilities)
    ).mean()


def counterfactual_pair_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    pair_ids: list[str | None],
    *,
    margin: float,
) -> torch.Tensor:
    if margin <= 0:
        raise ValueError("pair margin must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        if pair_id is not None:
            groups[pair_id].append(index)
    losses = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_ids = torch.tensor(members, device=probabilities.device)
        group_probabilities = probabilities[member_ids]
        group_targets = targets[member_ids]
        positive = group_targets.bool()
        negative = ~positive
        comparable = positive.any(dim=0) & negative.any(dim=0)
        for character_index in torch.where(comparable)[0].tolist():
            positive_score = group_probabilities[
                positive[:, character_index], character_index
            ].mean()
            negative_score = group_probabilities[
                negative[:, character_index], character_index
            ].mean()
            losses.append(F.relu(margin - positive_score + negative_score))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


class PairAwareBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: OpenTextDataset,
        weights: torch.Tensor,
        *,
        batch_size: int,
        samples: int,
        pair_fraction: float,
        seed: int,
    ) -> None:
        if not 0.0 <= pair_fraction <= 1.0:
            raise ValueError("pair fraction must be in [0, 1]")
        self.weights = weights
        self.batch_size = batch_size
        self.samples = samples
        self.pair_fraction = pair_fraction
        self.seed = seed
        by_pair: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            if entry.counterfactual_pair_id is not None:
                by_pair[entry.counterfactual_pair_id].append(index)
        self.pairs = [tuple(values) for _, values in sorted(by_pair.items())]
        if pair_fraction and not self.pairs:
            raise ValueError("pair-aware sampler found no counterfactual pairs")

    def __len__(self) -> int:
        return math.ceil(self.samples / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        rng = random.Random(self.seed)
        remaining = self.samples
        while remaining > 0:
            current_size = min(self.batch_size, remaining)
            pair_target = round(current_size * self.pair_fraction)
            batch = []
            attempts = 0
            while self.pairs and len(batch) < pair_target and attempts < len(self.pairs) * 2:
                members = self.pairs[rng.randrange(len(self.pairs))]
                attempts += 1
                if len(batch) + len(members) <= pair_target:
                    batch.extend(members)
            fill = current_size - len(batch)
            if fill:
                sampled = torch.multinomial(
                    self.weights,
                    fill,
                    replacement=True,
                    generator=generator,
                ).tolist()
                batch.extend(sampled)
            rng.shuffle(batch)
            yield batch
            remaining -= current_size


def make_train_loader(
    dataset: OpenTextDataset,
    weights: torch.Tensor,
    *,
    batch_size: int,
    samples: int,
    pair_fraction: float,
    seed: int,
    maximum_bytes: int,
) -> DataLoader:
    collate = lambda values: collate_open_text(values, maximum_bytes)
    if pair_fraction:
        batch_sampler = PairAwareBatchSampler(
            dataset,
            weights,
            batch_size=batch_size,
            samples=samples,
            pair_fraction=pair_fraction,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate,
        )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
    )


def train_arm(
    arm: str,
    args,
    train_dataset: OpenTextDataset,
    validation_loader: DataLoader,
    internal_loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    config = ARM_CONFIGS[arm]
    model, initial_payload = load_initial_model(args.initial_checkpoint, arm="full")
    model.to(device)
    optimizer = optimizer_for_arm(
        model,
        learning_rate=args.learning_rate,
        pixel_learning_rate=args.pixel_learning_rate,
    )
    output = args.output_dir / arm
    output.mkdir(parents=True, exist_ok=True)
    samples_per_epoch = math.ceil(args.train_crops / args.epochs)
    samples_seen = 0
    history = []
    best_score = -math.inf
    for epoch in range(1, args.epochs + 1):
        current_samples = min(
            samples_per_epoch, args.train_crops - samples_seen
        )
        if current_samples <= 0:
            break
        loader = make_train_loader(
            train_dataset,
            weights,
            batch_size=args.batch_size,
            samples=current_samples,
            pair_fraction=config["pair_fraction"],
            seed=args.seed + epoch,
            maximum_bytes=model.config.maximum_bytes,
        )
        model.train()
        totals = defaultdict(float)
        batches = 0
        for batch in loader:
            pixels = batch["pixels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixels)
            targets = character_presence_targets(batch["texts"], device=device)
            presence = character_presence_probabilities(logits)
            losses = {
                "ctc_loss": ctc_loss(
                    logits.float(),
                    batch["targets"].to(device),
                    batch["target_lengths"].to(device),
                ),
                "presence_loss": character_presence_loss(
                    presence,
                    targets,
                    positive_weight=args.presence_positive_weight,
                ),
                "pair_loss": counterfactual_pair_loss(
                    presence,
                    targets,
                    batch["counterfactual_pair_ids"],
                    margin=args.pair_margin,
                ),
            }
            loss = (
                losses["ctc_loss"]
                + config["presence_weight"] * losses["presence_loss"]
                + config["pair_weight"] * losses["pair_loss"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            totals["loss"] += float(loss.detach())
            for name, value in losses.items():
                totals[name] += float(value.detach())
            batches += 1
        samples_seen += current_samples
        metrics = evaluate_open_text(model, validation_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "samples_seen": samples_seen,
                "selection_score": metric_score(metrics),
                **{
                    name: value / max(batches, 1)
                    for name, value in totals.items()
                },
            }
        )
        history.append(metrics)
        custom = metrics["segments"]["custom"]
        print(
            f"arm={arm} epoch={epoch} samples={samples_seen} "
            + json.dumps(
                {
                    "overall": metrics["exact_rate"],
                    "real": custom["real_predicted"]["exact_rate"],
                    "modifier": custom["molecule_modifier"]["exact_rate"],
                    "common_single": custom["molecule_common_single"][
                        "exact_rate"
                    ],
                    "false_modifier": custom["molecule_simple"][
                        "false_modifier_rate"
                    ],
                    "open_random": custom["open_random"]["exact_rate"],
                    "score": metrics["selection_score"],
                    "loss": metrics["loss"],
                    "presence_loss": metrics["presence_loss"],
                    "pair_loss": metrics["pair_loss"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        payload = {
            "pretrained_ctc_ocr": model.state_dict(),
            "config": asdict(model.config),
            "arm": arm,
            "auxiliary_config": config,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "validation": metrics,
            "initial_checkpoint": str(args.initial_checkpoint),
            "initial_validation": initial_payload.get("validation"),
            "training_args": serializable_args(args),
        }
        torch.save(payload, output / "last.pt")
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            torch.save(payload, output / "best.pt")
            write_json(output / "best-validation.json", metrics)
    best_validation = json.loads((output / "best-validation.json").read_text())
    best_checkpoint = output / "best.pt"
    internal = evaluate_checkpoint(best_checkpoint, internal_loader, device)
    write_json(output / "best-internal-test.json", internal)
    write_json(output / "history.json", {"arm": arm, "history": history})
    return {
        "arm": arm,
        "best": best_validation,
        "internal_test": internal,
        "checkpoint": str(best_checkpoint),
        "gates": gate_results(best_validation, internal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arms", nargs="+", choices=tuple(ARM_CONFIGS), default=tuple(ARM_CONFIGS)
    )
    parser.add_argument("--train-crops", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--pixel-learning-rate", type=float, default=1e-5)
    parser.add_argument("--presence-positive-weight", type=float, default=4.0)
    parser.add_argument("--pair-margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if min(
        args.train_crops,
        args.epochs,
        args.batch_size,
        args.evaluation_batch_size,
    ) < 1:
        parser.error("counts must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = OpenTextDataset(args.data_root, split="train")
    validation_dataset = OpenTextDataset(args.data_root, split="validation")
    internal_dataset = OpenTextDataset(args.data_root, split="internal_test")
    weights, sampler_audit = training_weights(train_dataset.entries)
    pair_counts = defaultdict(int)
    pair_members = defaultdict(int)
    for entry in train_dataset.entries:
        if entry.counterfactual_pair_id is not None:
            pair_counts[entry.counterfactual_pair_group] += 1
            pair_members[entry.counterfactual_pair_id] += 1
    initial_model, _ = load_initial_model(args.initial_checkpoint, arm="full")
    maximum_bytes = initial_model.config.maximum_bytes
    validation_loader = make_validation_loader(
        validation_dataset,
        batch_size=args.evaluation_batch_size,
        workers=0,
        maximum_bytes=maximum_bytes,
    )
    internal_loader = make_validation_loader(
        internal_dataset,
        batch_size=args.evaluation_batch_size,
        workers=0,
        maximum_bytes=maximum_bytes,
    )
    baseline_validation = evaluate_open_text(
        initial_model.to(device), validation_loader, device
    )
    baseline_internal = evaluate_open_text(initial_model, internal_loader, device)
    write_json(args.output_dir / "baseline-validation.json", baseline_validation)
    write_json(args.output_dir / "baseline-internal-test.json", baseline_internal)
    write_json(
        args.output_dir / "run.json",
        {
            "protocol": "v0.8.12-counterfactual-presence-ab",
            "args": serializable_args(args),
            "sampler": sampler_audit,
            "train_pair_members_by_group": dict(sorted(pair_counts.items())),
            "train_pairs": len(pair_members),
            "presence_alphabet": PRESENCE_ALPHABET,
            "arms": ARM_CONFIGS,
        },
    )
    results = [
        train_arm(
            arm,
            args,
            train_dataset,
            validation_loader,
            internal_loader,
            weights,
            device,
        )
        for arm in args.arms
    ]
    selected = max(results, key=lambda item: item["best"]["selection_score"])
    summary = {
        "protocol": "v0.8.12-counterfactual-presence-ab",
        "status": selected["gates"]["status"],
        "baseline": {
            "validation": baseline_validation,
            "internal_test": baseline_internal,
        },
        "arms": results,
        "selected_arm": selected["arm"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation": selected["best"],
        "selected_internal_test": selected["internal_test"],
        "gates": selected["gates"],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        "summary="
        + json.dumps(
            {
                "status": summary["status"],
                "selected_arm": summary["selected_arm"],
                "score": summary["selected_validation"]["selection_score"],
                "gates": summary["gates"]["checks"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
