#!/usr/bin/env python3
"""Evaluate Semantic Bond existence on every unordered GT Junction pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.data import ChemVectorIterableDataset
from chemvector.pretrain_semantic_bond_existence import BondExistenceCollator
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelationConfig,
)
from chemvector.train import expand_shards


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_metrics(
    rows: list[tuple[torch.Tensor, torch.Tensor]], threshold: float
) -> dict:
    true_positive = false_positive = false_negative = sample_exact = 0
    for scores, targets in rows:
        predictions = scores >= threshold
        true_positive += int((predictions & targets).sum())
        false_positive += int((predictions & ~targets).sum())
        false_negative += int((~predictions & targets).sum())
        sample_exact += int(torch.equal(predictions, targets))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "sample_set_exact": sample_exact / max(len(rows), 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


@torch.inference_mode()
def collect_dense_rows(
    model: SemanticBondExistence,
    records: list[dict],
    collator: BondExistenceCollator,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rows = []
    for record in records:
        batch = collator([record])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["pixel_values"].to(device),
                batch["points"].to(device),
                batch["pair_indices"].to(device),
                batch["pair_mask"].to(device),
            )
        rows.append(
            (
                output["edge_logits"].float().cpu().sigmoid(),
                batch["edge_targets"].bool(),
            )
        )
    return rows


def default_thresholds() -> list[float]:
    return [
        *[value / 100.0 for value in range(50, 100)],
        0.995,
        0.9975,
        0.999,
        0.9995,
        0.9999,
    ]


def evaluate_style(
    model: SemanticBondExistence,
    records: list[dict],
    device: torch.device,
    *,
    degradation_probability: float,
    seed: int,
    predicted_junctions: dict[str, dict] | None = None,
) -> dict:
    collator = BondExistenceCollator(
        negative_ratio=1e9,
        degradation_probability=degradation_probability,
        endpoint_jitter_sigma_pixels=(0.0 if degradation_probability == 0.0 else 1.5),
        predicted_junctions=predicted_junctions,
        seed=seed,
    )
    rows = collect_dense_rows(model, records, collator, device)
    sweep = [threshold_metrics(rows, value) for value in default_thresholds()]
    best = max(sweep, key=lambda item: (item["sample_set_exact"], item["f1"]))
    return {
        "samples": len(rows),
        "pairs": sum(len(targets) for _, targets in rows),
        "positives": sum(int(targets.sum()) for _, targets in rows),
        "best": best,
        "threshold_0_5": threshold_metrics(rows, 0.5),
        "threshold_sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--predicted-junction-manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SemanticBondExistence(
        SemanticBondRelationConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["semantic_bond_existence"])
    model.eval().requires_grad_(False)
    predicted_junctions = None
    allowed_keys = None
    if args.predicted_junction_manifest:
        manifest = json.loads(
            Path(args.predicted_junction_manifest).read_text(encoding="utf-8")
        )
        predicted_junctions = {
            str(item["key"]): item for item in manifest["samples"]
        }
        allowed_keys = set(predicted_junctions)
    records = []
    for record in ChemVectorIterableDataset(
        expand_shards(args.validation_shards), sequence_version="0.4"
    ):
        if allowed_keys is not None and record["key"] not in allowed_keys:
            continue
        records.append(record)
        if len(records) >= args.samples:
            break
    if len(records) < args.samples:
        raise ValueError(
            f"Only {len(records)} eligible validation records were found"
        )
    result = {
        "schema_version": "semantic-bond-existence-dense-eval-0.1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "protocol": "all unordered GT Junction pairs; no pair deleted by length",
        "clean": evaluate_style(
            model,
            records,
            device,
            degradation_probability=0.0,
            seed=args.seed,
            predicted_junctions=predicted_junctions,
        ),
        "degraded": evaluate_style(
            model,
            records,
            device,
            degradation_probability=1.0,
            seed=args.seed + 1,
            predicted_junctions=predicted_junctions,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["default_thresholds", "threshold_metrics"]
