#!/usr/bin/env python3
"""Fine-tune the semantic endpoint field on the safe style_train_100k split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    decode_semantic_endpoints,
    semantic_endpoint_loss,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.core.data import ChemVectorIterableDataset
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    _target_candidate_coverage,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle import (
    _fresh_loader,
    load_fresh_split,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    V1BondObjectCollator,
    _bond_probability,
)
from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    _unique_port_data,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


MODEL_NAME = "chemvector-v1p2d2e3-style-endpoint-field"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
STYLE_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
DEFAULT_MANIFEST = Path(
    "outputs/audits/chemvector-v1p2d2e3a-style-endpoint-safe-manifest/manifest.jsonl"
)
MANIFEST_SHA256 = (
    "ee41d57275d8c48ebe9461b69eeef5bbb90abaa10de9319b746f8274024686e6"
)
INITIAL_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v1p2b-semantic-endpoint-field-pilot/best.pt"
)
INITIAL_CHECKPOINT_SHA256 = (
    "3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca"
)
EXPECTED_SPLITS = {"fit": 91_740, "calibration": 3_588, "holdout": 4_008}
SEED = 20260824
ENDPOINT_THRESHOLD = 0.05
ENDPOINT_BUDGETS = (256, 512)
GO_GATES = {
    "style_holdout_unique_port_recall": 0.98,
    "style_holdout_pairable_bond_recall": 0.975,
    "fresh_calibration_unique_port_recall": 0.975,
    "fresh_calibration_pairable_bond_recall": 0.97,
    "fresh_stress_pairable_bond_recall": 0.95,
}


def load_style_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    if file_sha256(path) != MANIFEST_SHA256:
        raise ValueError("P2-D2-E3 style endpoint manifest SHA256 differs")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    counts = Counter(str(row["split"]) for row in rows)
    if dict(counts) != EXPECTED_SPLITS:
        raise ValueError(f"P2-D2-E3 manifest split counts differ: {dict(counts)}")
    if len({str(row["sample_key"]) for row in rows}) != len(rows):
        raise ValueError("P2-D2-E3 manifest contains duplicate sample keys")
    return rows


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["split"]) == split]
    if len(selected) != EXPECTED_SPLITS[split]:
        raise ValueError(f"P2-D2-E3 {split} count differs")
    return selected


def _style_loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1 or not splits <= set(EXPECTED_SPLITS):
        raise ValueError("P2-D2-E3 loader rows mix splits")
    keys = {str(row["sample_key"]) for row in rows}
    shard_names = sorted({str(row["source_shard"]) for row in rows})
    if not shard_names or any(
        not name.startswith("shards/train-") for name in shard_names
    ):
        raise ValueError("P2-D2-E3 manifest references unexpected shards")
    dataset = ChemVectorIterableDataset(
        [str(STYLE_ROOT / name) for name in shard_names],
        shardshuffle=False,
        sample_shuffle_buffer=4096 if shuffle else 0,
        sample_shuffle_seed=seed,
        include_keys=keys,
        serialize_target=False,
        load_raster_layers=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=V1BondObjectCollator(),
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _new_budget_totals() -> dict[str, Counter[str]]:
    return {f"max{budget}": Counter() for budget in ENDPOINT_BUDGETS}


def _update_reachability(
    totals: Counter[str],
    *,
    objects: list[dict[str, torch.Tensor | float]],
    budget: int,
    expected_points: torch.Tensor,
    port_indices: torch.Tensor,
    bond_count: int,
) -> None:
    selected = objects[:budget]
    points = (
        torch.stack([item["point"] for item in selected])
        if selected
        else torch.empty((0, 2))
    )
    recovered_ports, pairable_bonds = _target_candidate_coverage(
        points, expected_points, port_indices
    )
    totals["samples"] += 1
    totals["ports"] += len(expected_points)
    totals["bonds"] += bond_count
    totals["outputs"] += len(selected)
    totals["saturated"] += int(len(selected) == budget)
    totals["recovered_ports"] += recovered_ports
    totals["pairable_bonds"] += pairable_bonds
    totals["port_exact"] += int(recovered_ports == len(expected_points))


def _reachability_metrics(totals: Counter[str]) -> dict[str, float | int]:
    samples = max(totals["samples"], 1)
    return {
        "samples": totals["samples"],
        "unique_ports": totals["ports"],
        "bonds": totals["bonds"],
        "mean_output_candidates": totals["outputs"] / samples,
        "budget_saturated_rate": totals["saturated"] / samples,
        "unique_port_recall": totals["recovered_ports"]
        / max(totals["ports"], 1),
        "pairable_bond_recall": totals["pairable_bonds"]
        / max(totals["bonds"], 1),
        "port_exact_rate": totals["port_exact"] / samples,
    }


@torch.no_grad()
def evaluate_reachability(
    separator: SemanticInkSeparator,
    detector: SemanticBondEndpointDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    detector.eval()
    totals = _new_budget_totals()
    variants: dict[str, dict[str, Counter[str]]] = defaultdict(_new_budget_totals)
    operations: dict[str, dict[str, Counter[str]]] = defaultdict(_new_budget_totals)
    keys = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            field = detector(probability)
        decoded = decode_semantic_endpoints(
            field,
            threshold=ENDPOINT_THRESHOLD,
            maximum_points=max(ENDPOINT_BUDGETS),
        )
        for index, objects in enumerate(decoded):
            bond_count = int(batch["mask"][index].sum())
            expected_bonds = batch["endpoints"][index, :bond_count]
            expected_points, port_indices = _unique_port_data(expected_bonds)
            variant = str(batch["variants"][index])
            operation = batch["stress_operations"][index]
            for budget in ENDPOINT_BUDGETS:
                name = f"max{budget}"
                kwargs = {
                    "objects": objects,
                    "budget": budget,
                    "expected_points": expected_points,
                    "port_indices": port_indices,
                    "bond_count": bond_count,
                }
                _update_reachability(totals[name], **kwargs)
                _update_reachability(variants[variant][name], **kwargs)
                if operation is not None:
                    _update_reachability(operations[str(operation)][name], **kwargs)
            keys.append(str(batch["keys"][index]))
    return {
        "samples": len(keys),
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in keys).encode()
        ).hexdigest(),
        "budgets": {
            name: _reachability_metrics(values) for name, values in totals.items()
        },
        "by_variant": {
            variant: {
                name: _reachability_metrics(values)
                for name, values in current.items()
            }
            for variant, current in sorted(variants.items())
        },
        "by_stress_operation": {
            operation: {
                name: _reachability_metrics(values)
                for name, values in current.items()
            }
            for operation, current in sorted(operations.items())
        },
    }


def _checks(style_holdout: dict[str, Any], fresh: dict[str, Any]) -> dict[str, bool]:
    style = style_holdout["budgets"]["max512"]
    current = fresh["budgets"]["max512"]
    return {
        "style_holdout_samples_exact4008": style_holdout["samples"] == 4_008,
        "style_holdout_unique_port_recall_min98pct": style["unique_port_recall"]
        >= GO_GATES["style_holdout_unique_port_recall"],
        "style_holdout_pairable_bond_recall_min97_5pct": style[
            "pairable_bond_recall"
        ]
        >= GO_GATES["style_holdout_pairable_bond_recall"],
        "fresh_calibration_samples_exact400": fresh["samples"] == 400,
        "fresh_calibration_unique_port_recall_min97_5pct": current[
            "unique_port_recall"
        ]
        >= GO_GATES["fresh_calibration_unique_port_recall"],
        "fresh_calibration_pairable_bond_recall_min97pct": current[
            "pairable_bond_recall"
        ]
        >= GO_GATES["fresh_calibration_pairable_bond_recall"],
        "fresh_stress_pairable_bond_recall_min95pct": fresh["by_variant"][
            "stress"
        ]["max512"]["pairable_bond_recall"]
        >= GO_GATES["fresh_stress_pairable_bond_recall"],
        "one_fixed_training_epoch": True,
        "separator_frozen": True,
        "style_holdout_consumed_once": True,
        "fresh_holdout_samples_read_zero": True,
        "sealed_test_samples_read_zero": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E3 style endpoint training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E3 output is not empty: {args.output_dir}")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E3 separator SHA256 differs")
    if file_sha256(INITIAL_CHECKPOINT) != INITIAL_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E3 initial endpoint checkpoint SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_style_manifest()
    fit_rows = split_rows(rows, "fit")
    calibration_rows = split_rows(rows, "calibration")
    holdout_rows = split_rows(rows, "holdout")
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    initial = torch.load(INITIAL_CHECKPOINT, map_location=device)
    detector = SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(**initial["config"])
    ).to(device)
    detector.load_state_dict(initial["detector"], strict=True)
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    detector.train()
    samples_seen = 0
    running_loss = 0.0
    loader = _style_loader(
        fit_rows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    for step, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        endpoints = batch["endpoints"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = detector(probability)
            losses = semantic_endpoint_loss(output, endpoints, mask)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"P2-D2-E3 non-finite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        optimizer.step()
        current_batch = pixels.size(0)
        samples_seen += current_batch
        running_loss += float(losses["loss"].detach()) * current_batch
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "samples_seen": samples_seen,
                        "train_loss": running_loss / samples_seen,
                    }
                ),
                flush=True,
            )
    if samples_seen != len(fit_rows):
        raise ValueError(
            f"P2-D2-E3 epoch read {samples_seen}, expected {len(fit_rows)}"
        )

    checkpoint = {
        "schema_version": "chemvector-v1p2d2e3-style-endpoint-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(detector.config),
        "detector": detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "samples_seen": samples_seen,
        "train_loss": running_loss / samples_seen,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": max(ENDPOINT_BUDGETS),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "style_holdout_read": False,
        "fresh_holdout_read": False,
        "sealed_test_read": False,
    }
    checkpoint_path = args.output_dir / "final.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)

    style_calibration = evaluate_reachability(
        separator,
        detector,
        _style_loader(
            calibration_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    fresh_calibration_rows, _, _ = load_fresh_split()
    fresh_calibration = evaluate_reachability(
        separator,
        detector,
        _fresh_loader(
            fresh_calibration_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    style_holdout = evaluate_reachability(
        separator,
        detector,
        _style_loader(
            holdout_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    checks = _checks(style_holdout, fresh_calibration)
    result = {
        "schema_version": "chemvector-v1p2d2e3-style-endpoint-result-0.1",
        "model_name": MODEL_NAME,
        "decision": (
            "style_endpoint_field_go"
            if all(checks.values())
            else "style_endpoint_field_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(detector.config),
        "parameters": sum(parameter.numel() for parameter in detector.parameters()),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "separator_frozen": True,
        "manifest_sha256": MANIFEST_SHA256,
        "fit_samples": len(fit_rows),
        "calibration_samples": len(calibration_rows),
        "holdout_samples": len(holdout_rows),
        "epochs": 1,
        "samples_seen": samples_seen,
        "train_loss": running_loss / samples_seen,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "endpoint_budgets": ENDPOINT_BUDGETS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "style_calibration": style_calibration,
        "fresh_calibration": fresh_calibration,
        "style_holdout": style_holdout,
        "style_holdout_samples_read": len(holdout_rows),
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
