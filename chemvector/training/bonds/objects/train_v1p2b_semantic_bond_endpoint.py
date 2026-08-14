#!/usr/bin/env python3
"""Train a local visible-port field from the frozen v1 P1-A bond map."""

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
from scipy.optimize import linear_sum_assignment
import torch

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
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    _bond_probability,
    _loader,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    MANIFEST_SHA256,
    load_safe_manifest,
)


MODEL_NAME = "chemvector-v1p2b-semantic-endpoint-field-pilot"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
SPLIT_SEED = 20260817
CALIBRATION_MOLECULES = 100
HOLDOUT_MOLECULES = 100
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0
THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))
GO_GATES = {
    "known_unique_port_recall": 0.95,
    "known_pairable_bond_recall": 0.90,
    "known_unique_port_exact": 0.30,
    "predicted_unique_port_f1": 0.88,
    "predicted_count_accuracy": 0.30,
    "text_adjacent_pairable_bond_recall": 0.85,
    "stress_known_pairable_bond_recall": 0.85,
}


def split_train_endpoint_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_molecule[str(row["molecule_id"])].append(row)
    expected_variants = {"original", "rdkit", "coordgen", "stress"}
    for molecule_id, group in by_molecule.items():
        if len(group) != 4 or {str(row["variant"]) for row in group} != expected_variants:
            raise ValueError(f"P2-B incomplete train molecule group: {molecule_id}")
    ranked = sorted(
        by_molecule,
        key=lambda molecule_id: (
            hashlib.sha256(f"{SPLIT_SEED}:{molecule_id}".encode()).hexdigest(),
            molecule_id,
        ),
    )
    calibration_ids = set(ranked[:CALIBRATION_MOLECULES])
    holdout_ids = set(
        ranked[CALIBRATION_MOLECULES : CALIBRATION_MOLECULES + HOLDOUT_MOLECULES]
    )
    fit = [
        row
        for row in train_rows
        if row["molecule_id"] not in calibration_ids | holdout_ids
    ]
    calibration = [
        row for row in train_rows if row["molecule_id"] in calibration_ids
    ]
    holdout = [row for row in train_rows if row["molecule_id"] in holdout_ids]
    for group in (fit, calibration, holdout):
        group.sort(key=lambda row: str(row["sample_key"]))
    if (len(fit), len(calibration), len(holdout)) != (8_056, 400, 400):
        raise ValueError("P2-B fit/calibration/holdout counts differ")
    molecule_sets = [
        {str(row["molecule_id"]) for row in group}
        for group in (fit, calibration, holdout)
    ]
    if any(
        molecule_sets[first] & molecule_sets[second]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("P2-B molecule split overlap")
    return fit, calibration, holdout


def _unique_port_data(
    endpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if endpoints.ndim != 3 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("P2-B endpoints must have shape [bonds, 2, 2]")
    quantized = (endpoints.reshape(-1, 2) * 1023.0).round().long()
    unique, inverse = torch.unique(
        quantized, dim=0, sorted=True, return_inverse=True
    )
    return unique.float() / 1023.0, inverse.reshape(-1, 2)


@torch.no_grad()
def collect_predictions(
    separator: SemanticInkSeparator,
    detector: SemanticBondEndpointDetector,
    loader: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    detector.eval()
    records = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = detector(probability)
        decoded = decode_semantic_endpoints(
            output, threshold=0.0, maximum_points=256
        )
        for index, objects in enumerate(decoded):
            bond_count = int(batch["mask"][index].sum())
            endpoints = batch["endpoints"][index, :bond_count].clone()
            points, port_indices = _unique_port_data(endpoints)
            if len(points) > 256:
                raise ValueError(
                    f"P2-B target exceeds 256 unique ports: {batch['keys'][index]}"
                )
            records.append(
                {
                    "key": batch["keys"][index],
                    "variant": batch["variants"][index],
                    "stress_operation": batch["stress_operations"][index],
                    "objects": objects,
                    "points": points,
                    "port_indices": port_indices,
                    "text_adjacent": batch["text_adjacent"][index, :bond_count].clone(),
                }
            )
    return records


def _update_label_audit(totals: Counter[str], record: dict[str, Any]) -> None:
    cells = Counter(
        (
            min(191, max(0, int(float(point[0]) * 192))),
            min(191, max(0, int(float(point[1]) * 192))),
        )
        for point in record["points"]
    )
    maximum = max(cells.values(), default=0)
    totals["grid_collision_excess"] += sum(value - 1 for value in cells.values())
    totals["grid_collision_images"] += int(maximum > 1)
    totals["grid_max_occupancy"] = max(totals["grid_max_occupancy"], maximum)


def _accumulate_record(
    totals: Counter[str], record: dict[str, Any], threshold: float
) -> None:
    expected_points = record["points"]
    expected_count = len(expected_points)
    bonds = record["port_indices"]
    text_adjacent = record["text_adjacent"]
    known = record["objects"][:expected_count]
    predicted = [item for item in record["objects"] if item["score"] >= threshold]
    totals["images"] += 1
    totals["expected_ports"] += expected_count
    totals["expected_bonds"] += len(bonds)
    totals["text_bonds"] += int(text_adjacent.sum())
    totals["implicit_bonds"] += int((~text_adjacent).sum())
    _update_label_audit(totals, record)

    for prefix, objects in (("known", known), ("predicted", predicted)):
        totals[f"{prefix}_output"] += len(objects)
        correct_targets = torch.zeros(expected_count, dtype=torch.bool)
        if objects and expected_count:
            output_points = torch.stack([item["point"] for item in objects])
            distances = torch.cdist(output_points.float(), expected_points.float()) * IMAGE_SIZE
            rows, columns = linear_sum_assignment(distances.numpy())
            for row, column in zip(rows, columns):
                if float(distances[row, column]) <= MATCH_RADIUS_PIXELS:
                    correct_targets[int(column)] = True
        correct = int(correct_targets.sum())
        pairable = (
            correct_targets[bonds].all(dim=1)
            if len(bonds)
            else torch.empty(0, dtype=torch.bool)
        )
        totals[f"{prefix}_correct_ports"] += correct
        totals[f"{prefix}_pairable_bonds"] += int(pairable.sum())
        totals[f"{prefix}_text_pairable"] += int(pairable[text_adjacent].sum())
        totals[f"{prefix}_implicit_pairable"] += int(pairable[~text_adjacent].sum())
        totals[f"{prefix}_port_exact"] += int(
            len(objects) == expected_count and correct == expected_count
        )
    totals["predicted_count_exact"] += int(len(predicted) == expected_count)


def _metrics(totals: Counter[str], threshold: float) -> dict[str, float | int]:
    images = max(totals["images"], 1)
    ports = max(totals["expected_ports"], 1)
    bonds = max(totals["expected_bonds"], 1)
    predicted_output = max(totals["predicted_output"], 1)
    precision = totals["predicted_correct_ports"] / predicted_output
    recall = totals["predicted_correct_ports"] / ports
    return {
        "samples": totals["images"],
        "unique_ports": totals["expected_ports"],
        "bonds": totals["expected_bonds"],
        "threshold": threshold,
        "known_unique_port_recall": totals["known_correct_ports"] / ports,
        "known_pairable_bond_recall": totals["known_pairable_bonds"] / bonds,
        "known_unique_port_exact": totals["known_port_exact"] / images,
        "predicted_unique_port_precision": precision,
        "predicted_unique_port_recall": recall,
        "predicted_unique_port_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_pairable_bond_recall": totals["predicted_pairable_bonds"] / bonds,
        "predicted_unique_port_exact": totals["predicted_port_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "text_adjacent_pairable_bond_recall": totals["known_text_pairable"]
        / max(totals["text_bonds"], 1),
        "implicit_pairable_bond_recall": totals["known_implicit_pairable"]
        / max(totals["implicit_bonds"], 1),
        "grid_collision_excess": totals["grid_collision_excess"],
        "grid_collision_images": totals["grid_collision_images"],
        "grid_max_occupancy": totals["grid_max_occupancy"],
    }


def summarize_predictions(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        _accumulate_record(totals, record, threshold)
        _accumulate_record(by_variant[str(record["variant"])], record, threshold)
        if record["stress_operation"] is not None:
            _accumulate_record(
                by_operation[str(record["stress_operation"])], record, threshold
            )
    return {
        **_metrics(totals, threshold),
        "by_variant": {
            name: _metrics(counts, threshold)
            for name, counts in sorted(by_variant.items())
        },
        "by_stress_operation": {
            name: _metrics(counts, threshold)
            for name, counts in sorted(by_operation.items())
        },
    }


def select_threshold(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    sweep = [summarize_predictions(records, threshold) for threshold in THRESHOLDS]
    best = max(
        sweep,
        key=lambda row: (
            row["predicted_unique_port_f1"],
            row["predicted_count_accuracy"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), sweep


def _save_checkpoint(
    path: Path,
    detector: SemanticBondEndpointDetector,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    threshold: float,
    calibration: dict[str, Any],
    samples_seen: int,
) -> None:
    payload = {
        "schema_version": "chemvector-v1p2b-semantic-endpoint-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(detector.config),
        "detector": detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "threshold": threshold,
        "calibration": calibration,
        "samples_seen": samples_seen,
        "separator_sha256": SEPARATOR_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "holdout_read": False,
        "validation_read": False,
        "internal_test_read": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "holdout_samples_exact400": metrics["samples"] == 400,
        "known_unique_port_recall_min95pct": metrics["known_unique_port_recall"]
        >= GO_GATES["known_unique_port_recall"],
        "known_pairable_bond_recall_min90pct": metrics["known_pairable_bond_recall"]
        >= GO_GATES["known_pairable_bond_recall"],
        "known_unique_port_exact_min30pct": metrics["known_unique_port_exact"]
        >= GO_GATES["known_unique_port_exact"],
        "predicted_unique_port_f1_min88pct": metrics["predicted_unique_port_f1"]
        >= GO_GATES["predicted_unique_port_f1"],
        "predicted_count_accuracy_min30pct": metrics["predicted_count_accuracy"]
        >= GO_GATES["predicted_count_accuracy"],
        "text_adjacent_pairable_min85pct": metrics[
            "text_adjacent_pairable_bond_recall"
        ]
        >= GO_GATES["text_adjacent_pairable_bond_recall"],
        "stress_known_pairable_min85pct": metrics["by_variant"]["stress"][
            "known_pairable_bond_recall"
        ]
        >= GO_GATES["stress_known_pairable_bond_recall"],
        "grid_slot_capacity_sufficient": metrics["grid_max_occupancy"] <= 3,
        "separator_frozen": True,
        "holdout_consumed_once": True,
        "validation_unread": True,
        "internal_test_unread": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separator", type=Path, default=DEFAULT_SEPARATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P2-B endpoint training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P2-B output is not empty: {args.output_dir}")
    if file_sha256(args.separator) != SEPARATOR_SHA256:
        raise ValueError("v1 P2-B separator SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    fit_rows, calibration_rows, holdout_rows = split_train_endpoint_rows(
        load_safe_manifest()
    )
    device = torch.device("cuda")
    separator_payload = torch.load(args.separator, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    detector = SemanticBondEndpointDetector().to(device)
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    samples_seen = 0
    history = []
    best_score = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        detector.train()
        running = 0.0
        epoch_seen = 0
        loader = _loader(
            fit_rows,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
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
                raise FloatingPointError(
                    f"v1 P2-B non-finite loss at epoch={epoch}, step={step}"
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            batch_samples = pixels.size(0)
            epoch_seen += batch_samples
            samples_seen += batch_samples
            running += float(losses["loss"].detach()) * batch_samples
            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "samples_seen": samples_seen,
                            "loss": running / epoch_seen,
                        }
                    ),
                    flush=True,
                )
        if epoch_seen != len(fit_rows):
            raise ValueError(f"v1 P2-B epoch read {epoch_seen}, expected {len(fit_rows)}")
        records = collect_predictions(
            separator,
            detector,
            _loader(
                calibration_rows,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                workers=args.workers,
            ),
            device,
        )
        if len(records) != 400:
            raise ValueError("v1 P2-B calibration count differs")
        threshold, sweep = select_threshold(records)
        calibration = summarize_predictions(records, threshold)
        score = (
            calibration["known_pairable_bond_recall"],
            calibration["known_unique_port_recall"],
            calibration["predicted_unique_port_f1"],
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running / epoch_seen,
            "samples_seen": samples_seen,
            "threshold": threshold,
            "calibration": calibration,
            "threshold_sweep": sweep,
        }
        history.append(epoch_record)
        _save_checkpoint(
            args.output_dir / f"epoch-{epoch:02d}.pt",
            detector,
            optimizer,
            epoch=epoch,
            threshold=threshold,
            calibration=calibration,
            samples_seen=samples_seen,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            _save_checkpoint(
                args.output_dir / "best.pt",
                detector,
                optimizer,
                epoch=epoch,
                threshold=threshold,
                calibration=calibration,
                samples_seen=samples_seen,
            )
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    detector.load_state_dict(best["detector"], strict=True)
    threshold = float(best["threshold"])
    holdout_records = collect_predictions(
        separator,
        detector,
        _loader(
            holdout_rows,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    if len(holdout_records) != 400:
        raise ValueError("v1 P2-B holdout count differs")
    holdout = summarize_predictions(holdout_records, threshold)
    checks = _checks(holdout)
    result = {
        "schema_version": "chemvector-v1p2b-semantic-endpoint-result-0.1",
        "model_name": MODEL_NAME,
        "decision": "endpoint_field_go" if all(checks.values()) else "endpoint_field_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(detector.config),
        "parameters": sum(parameter.numel() for parameter in detector.parameters()),
        "separator_sha256": SEPARATOR_SHA256,
        "separator_frozen": True,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "fit_samples": len(fit_rows),
        "fit_molecules": len({row["molecule_id"] for row in fit_rows}),
        "calibration_samples": len(calibration_rows),
        "calibration_molecules": CALIBRATION_MOLECULES,
        "holdout_samples": len(holdout_rows),
        "holdout_molecules": HOLDOUT_MOLECULES,
        "validation_samples_read": 0,
        "internal_test_samples_read": 0,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "checkpoint": str(args.output_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(args.output_dir / "best.pt"),
        "calibration": best["calibration"],
        "holdout": holdout,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
