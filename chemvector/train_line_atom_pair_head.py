#!/usr/bin/env python3
"""Train first-class line-only AtomPair presence and type heads."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from chemvector.line_atom_pair_head import (
    LineAtomPairHead,
    LineAtomPairHeadConfig,
    line_atom_pair_loss,
)


DEFAULT_CACHE = "outputs/cache/chemvector-v0985-line-atom-pair-full"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0985-line-atom-pair-head"
DOMAINS = ("chemvision", "full_v0")


def load_cache(root: Path, split: str, domain: str) -> dict:
    payload = torch.load(
        root / f"{split}-{domain}.pt", map_location="cpu", weights_only=False
    )
    if payload.get("schema_version") != "line-atom-pair-cache-0.1":
        raise ValueError("Unsupported line AtomPair cache schema")
    if payload["split"] != split or payload["domain"] != domain:
        raise ValueError("Line AtomPair cache identity differs from its filename")
    return payload


def _sample_exact(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_count: int,
) -> int:
    wrong = torch.zeros(sample_count, dtype=torch.bool)
    if len(predictions):
        for sample_index in torch.where(predictions != targets)[0].tolist():
            wrong[int(sample_indices[sample_index])] = True
    return int((~wrong).sum())


@torch.no_grad()
def evaluate(
    model: LineAtomPairHead,
    validation: dict[str, dict],
    device: torch.device,
) -> dict:
    raw = {}
    threshold_candidates = {-20.0, 20.0, 0.0}
    for domain, cache in validation.items():
        features = cache["features"].float().to(device)
        output = model(features)
        logits = output["presence_logits"].float().cpu()
        threshold_candidates.update(float(value) for value in logits.tolist())
        raw[domain] = {
            "logits": logits,
            "display": output["display_logits"].argmax(-1).cpu(),
            "chemical": output["chemical_logits"].argmax(-1).cpu(),
        }

    best = None
    for threshold in sorted(threshold_candidates):
        domains = {}
        pooled = Counter()
        for domain, cache in validation.items():
            predicted = raw[domain]["logits"] >= threshold
            expected = cache["presence_targets"]
            true_positive = int((predicted & expected).sum())
            false_positive = int((predicted & ~expected).sum())
            false_negative = int((~predicted & expected).sum())
            rescue_expected = cache["rescue_targets"]
            rescue_true = int((predicted & rescue_expected).sum())
            sample_count = len(cache["sample_keys"])
            exact = _sample_exact(
                predicted,
                expected,
                cache["sample_indices"],
                sample_count,
            )
            positives = expected.bool()
            display_correct = int(
                (raw[domain]["display"][positives] == cache["display_targets"][positives]).sum()
            )
            chemical_correct = int(
                (raw[domain]["chemical"][positives] == cache["chemical_targets"][positives]).sum()
            )
            domains[domain] = {
                "samples": sample_count,
                "candidates": len(expected),
                "positives": int(expected.sum()),
                "negatives": int((~expected).sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": true_positive / max(true_positive + false_positive, 1),
                "recall": true_positive / max(int(expected.sum()), 1),
                "sample_exact": exact,
                "sample_exact_rate": exact / max(sample_count, 1),
                "rescue_expected": int(rescue_expected.sum()),
                "rescue_true": rescue_true,
                "rescue_recall": rescue_true / max(int(rescue_expected.sum()), 1),
                "display_accuracy": display_correct / max(int(positives.sum()), 1),
                "chemical_accuracy": chemical_correct / max(int(positives.sum()), 1),
            }
            pooled.update(
                samples=sample_count,
                exact=exact,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                rescue_expected=int(rescue_expected.sum()),
                rescue_true=rescue_true,
                positives=int(expected.sum()),
                display_correct=display_correct,
                chemical_correct=chemical_correct,
            )
        minimum_exact = min(item["sample_exact_rate"] for item in domains.values())
        pooled_exact = pooled["exact"] / max(pooled["samples"], 1)
        precision = pooled["true_positive"] / max(
            pooled["true_positive"] + pooled["false_positive"], 1
        )
        recall = pooled["true_positive"] / max(pooled["positives"], 1)
        rescue_recall = pooled["rescue_true"] / max(pooled["rescue_expected"], 1)
        chemical_accuracy = pooled["chemical_correct"] / max(pooled["positives"], 1)
        selection_key = (
            minimum_exact,
            pooled_exact,
            rescue_recall,
            precision,
            recall,
            chemical_accuracy,
            -abs(threshold),
        )
        row = {
            "threshold": threshold,
            "domains": domains,
            "minimum_domain_sample_exact_rate": minimum_exact,
            "pooled_sample_exact_rate": pooled_exact,
            "pooled_precision": precision,
            "pooled_recall": recall,
            "pooled_rescue_recall": rescue_recall,
            "pooled_false_positive": pooled["false_positive"],
            "pooled_chemical_accuracy": chemical_accuracy,
            "selection_key": selection_key,
        }
        if best is None or selection_key > best[0]:
            best = (selection_key, row)
    if best is None:
        raise RuntimeError("Validation threshold scan produced no result")
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--type-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Line AtomPair head training requires CUDA")
    if min(args.epochs, args.batch_size) < 1:
        raise ValueError("Epochs and batch size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cache_root = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = {domain: load_cache(cache_root, "train", domain) for domain in DOMAINS}
    validation = {
        domain: load_cache(cache_root, "validation", domain) for domain in DOMAINS
    }
    feature_config = LineAtomPairHeadConfig(**train[DOMAINS[0]]["feature_config"])
    for cache in (*train.values(), *validation.values()):
        if cache["feature_config"] != asdict(feature_config):
            raise ValueError("Line AtomPair caches use different feature configs")

    features = torch.cat([train[domain]["features"] for domain in DOMAINS]).float()
    presence = torch.cat([train[domain]["presence_targets"] for domain in DOMAINS])
    display = torch.cat([train[domain]["display_targets"] for domain in DOMAINS])
    chemical = torch.cat([train[domain]["chemical_targets"] for domain in DOMAINS])
    domain_ids = torch.cat(
        [
            torch.full((len(train[domain]["features"]),), index, dtype=torch.long)
            for index, domain in enumerate(DOMAINS)
        ]
    )
    if not len(features):
        raise ValueError("Training cache contains no line-only candidates")
    class_counts = Counter(
        (int(domain), bool(target))
        for domain, target in zip(domain_ids.tolist(), presence.tolist())
    )
    weights = torch.tensor(
        [1.0 / class_counts[(int(domain), bool(target))] for domain, target in zip(domain_ids, presence)],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(class_counts.values()) * len(class_counts),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(features, presence, display, chemical),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
    )
    device = torch.device("cuda")
    model = LineAtomPairHead(feature_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for batch_features, batch_presence, batch_display, batch_chemical in loader:
            batch_features = batch_features.to(device)
            batch_presence = batch_presence.to(device)
            batch_display = batch_display.to(device)
            batch_chemical = batch_chemical.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_features)
            losses = line_atom_pair_loss(
                output,
                batch_presence,
                batch_display,
                batch_chemical,
                type_weight=args.type_weight,
            )
            losses["loss"].backward()
            optimizer.step()
            totals["batches"] += 1
            for name in ("loss", "presence_loss", "display_loss", "chemical_loss"):
                totals[name] += float(losses[name].detach())
        model.eval()
        metrics = evaluate(model, validation, device)
        row = {
            "epoch": epoch,
            "train": {
                name: totals[name] / max(totals["batches"], 1)
                for name in ("loss", "presence_loss", "display_loss", "chemical_loss")
            },
            "validation": metrics,
        }
        history.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)
        key = tuple(metrics["selection_key"])
        if best is None or key > best[0]:
            best = (key, epoch, metrics)
            torch.save(
                {
                    "checkpoint_schema_version": "line-atom-pair-head-0.1",
                    "config": asdict(feature_config),
                    "head": model.state_dict(),
                    "epoch": epoch,
                    "threshold": metrics["threshold"],
                    "validation": metrics,
                    "cache_dir": str(cache_root),
                    "training_args": vars(args),
                    "class_counts": {
                        f"{DOMAINS[domain]}:{'positive' if target else 'negative'}": count
                        for (domain, target), count in class_counts.items()
                    },
                },
                output_dir / "best.pt",
            )
    if best is None:
        raise RuntimeError("Training produced no checkpoint")
    summary = {
        "best_epoch": best[1],
        "best_validation": best[2],
        "class_counts": {
            f"{DOMAINS[domain]}:{'positive' if target else 'negative'}": count
            for (domain, target), count in class_counts.items()
        },
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
