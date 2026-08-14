#!/usr/bin/env python3
"""Train complete BondObjects from the frozen v1 P1-A bond-ink map."""

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
from torch.utils.data import DataLoader

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.components.bonds.objects.bond_object import (
    BOND_TYPE_TO_ID,
    DIRECTION_TO_ID,
    unordered_endpoint_cost,
)
from chemvector.components.bonds.objects.dense_bond_object import (
    decode_dense_bond_objects,
    dense_bond_object_loss,
)
from chemvector.components.bonds.objects.semantic_bond_object import (
    SemanticBondObjectConfig,
    SemanticBondObjectDetector,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.core.data import ChemVectorIterableDataset
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    DATA_ROOT,
    MANIFEST_SHA256,
    load_safe_manifest,
)


MODEL_NAME = "chemvector-v1p2-semantic-bondobject-pilot"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
DEFAULT_SEPARATOR = Path(
    "outputs/checkpoints/chemvector-v1p1a-binary-ink-finetune/best.pt"
)
SEPARATOR_SHA256 = (
    "490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74"
)
QUANTIZATION_MAX = 1023.0
CALIBRATION_MOLECULES = 100
SPLIT_SEED = 20260816
THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))
GO_GATES = {
    "known_geometry_accuracy": 0.90,
    "known_typed_accuracy": 0.85,
    "known_typed_exact": 0.20,
    "predicted_count_accuracy": 0.50,
    "predicted_typed_exact": 0.15,
    "predicted_geometry_f1": 0.88,
    "text_adjacent_geometry_accuracy": 0.85,
    "stress_known_geometry_accuracy": 0.85,
}


def split_train_calibration_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_molecule[str(row["molecule_id"])].append(row)
    for molecule_id, group in by_molecule.items():
        if len(group) != 4 or {row["variant"] for row in group} != {
            "original",
            "rdkit",
            "coordgen",
            "stress",
        }:
            raise ValueError(f"P2 incomplete train molecule group: {molecule_id}")
    ranked = sorted(
        by_molecule,
        key=lambda molecule_id: (
            hashlib.sha256(
                f"{SPLIT_SEED}:{molecule_id}".encode("utf-8")
            ).hexdigest(),
            molecule_id,
        ),
    )
    calibration_molecules = set(ranked[:CALIBRATION_MOLECULES])
    fit = [row for row in train_rows if row["molecule_id"] not in calibration_molecules]
    calibration = [
        row for row in train_rows if row["molecule_id"] in calibration_molecules
    ]
    fit.sort(key=lambda row: str(row["sample_key"]))
    calibration.sort(key=lambda row: str(row["sample_key"]))
    if len(fit) != 8_456 or len(calibration) != 400:
        raise ValueError("P2 fit/calibration counts differ")
    return fit, calibration


class V1BondObjectCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        scenes = [
            build_chemscene_v0_2(
                sample["chemmol"], sample["render"], sample["layers"]
            )
            for sample in samples
        ]
        maximum = max(len(scene["bond_objects"]) for scene in scenes)
        endpoints = torch.zeros((len(samples), maximum, 2, 2))
        display = torch.zeros((len(samples), maximum), dtype=torch.long)
        chemical = torch.zeros((len(samples), maximum), dtype=torch.long)
        direction = torch.zeros((len(samples), maximum), dtype=torch.long)
        text_adjacent = torch.zeros((len(samples), maximum), dtype=torch.bool)
        mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        pixels = []
        for batch_index, (sample, scene) in enumerate(zip(samples, scenes)):
            array = np.asarray(sample["image"].convert("RGB"), dtype=np.float32) / 255.0
            pixels.append(torch.from_numpy(array).permute(2, 0, 1))
            roles = {int(atom["id"]): str(atom["role"]) for atom in scene["atom_objects"]}
            for bond_index, bond in enumerate(scene["bond_objects"]):
                endpoints[batch_index, bond_index] = (
                    torch.tensor(bond["ports"], dtype=torch.float32)
                    / QUANTIZATION_MAX
                )
                display[batch_index, bond_index] = BOND_TYPE_TO_ID[
                    str(bond["display_type"])
                ]
                chemical[batch_index, bond_index] = BOND_TYPE_TO_ID[
                    str(bond["chemical_type"])
                ]
                direction[batch_index, bond_index] = DIRECTION_TO_ID[
                    str(bond["direction"])
                ]
                text_adjacent[batch_index, bond_index] = any(
                    roles[int(atom_id)] == "TEXT" for atom_id in bond["atom_refs"]
                )
                mask[batch_index, bond_index] = True
        return {
            "pixel_values": torch.stack(pixels),
            "endpoints": endpoints,
            "display": display,
            "chemical": chemical,
            "direction": direction,
            "text_adjacent": text_adjacent,
            "mask": mask,
            "keys": [sample["key"] for sample in samples],
            "variants": [str(sample["meta"]["variant"]) for sample in samples],
            "stress_operations": [
                sample["render"].get("render_parameters", {}).get("stress_operation")
                for sample in samples
            ],
        }


def _loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    keys = {str(row["sample_key"]) for row in rows}
    shards = sorted({str(DATA_ROOT / row["source_shard"]) for row in rows})
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1:
        raise ValueError("P2 loader rows mix source splits")
    split = next(iter(splits))
    marker = "/train-" if split == "train" else "/validation-"
    if split not in {"train", "validation"} or any(marker not in shard for shard in shards):
        raise ValueError(f"P2 loader shard path differs for {split}")
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=False,
        sample_shuffle_buffer=512 if shuffle else 0,
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


def _targets(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "endpoints": batch["endpoints"].to(device, non_blocking=True),
        "bbox": torch.zeros(
            (*batch["endpoints"].shape[:2], 4), device=device
        ),
        "display": batch["display"].to(device, non_blocking=True),
        "chemical": batch["chemical"].to(device, non_blocking=True),
        "direction": batch["direction"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
    }


@torch.no_grad()
def _bond_probability(
    separator: SemanticInkSeparator, pixels: torch.Tensor
) -> torch.Tensor:
    separator.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = separator(pixels)["bond_logits"]
    return logits.float().sigmoid().unsqueeze(1)


def _clean_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "score": float(item["score"]),
            "endpoints": item["endpoints"].float().cpu(),
            "display": int(item["display"]),
            "chemical": int(item["chemical"]),
            "direction": int(item["direction"]),
        }
        for item in objects
    ]


@torch.no_grad()
def collect_predictions(
    separator: SemanticInkSeparator,
    detector: SemanticBondObjectDetector,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    detector.eval()
    records = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = detector(probability)
        decoded = decode_dense_bond_objects(
            output, threshold=0.0, maximum_objects=128
        )
        for index, objects in enumerate(decoded):
            valid = batch["mask"][index]
            count = int(valid.sum())
            if count > 128:
                raise ValueError(f"P2 target exceeds 128 BondObjects: {batch['keys'][index]}")
            records.append(
                {
                    "key": batch["keys"][index],
                    "variant": batch["variants"][index],
                    "stress_operation": batch["stress_operations"][index],
                    "objects": _clean_objects(objects),
                    "endpoints": batch["endpoints"][index, :count].clone(),
                    "display": batch["display"][index, :count].clone(),
                    "chemical": batch["chemical"][index, :count].clone(),
                    "direction": batch["direction"][index, :count].clone(),
                    "text_adjacent": batch["text_adjacent"][index, :count].clone(),
                }
            )
    return records


def _oriented_errors(predicted: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    forward = torch.linalg.vector_norm((predicted - expected) * 768.0, dim=-1)
    reverse = torch.linalg.vector_norm(
        (predicted - expected.flip(0)) * 768.0, dim=-1
    )
    return forward if float(forward.sum()) <= float(reverse.sum()) else reverse


def _accumulate_record(
    totals: Counter[str], record: dict[str, Any], threshold: float
) -> None:
    expected_count = len(record["endpoints"])
    known = record["objects"][:expected_count]
    predicted = [item for item in record["objects"] if item["score"] >= threshold]
    totals["images"] += 1
    totals["expected"] += expected_count
    totals["text_expected"] += int(record["text_adjacent"].sum())
    totals["implicit_expected"] += int((~record["text_adjacent"]).sum())

    for prefix, objects in (("known", known), ("predicted", predicted)):
        totals[f"{prefix}_output"] += len(objects)
        if not objects or not expected_count:
            continue
        predicted_endpoints = torch.stack([item["endpoints"] for item in objects])
        cost = unordered_endpoint_cost(
            predicted_endpoints, record["endpoints"]
        ).numpy()
        rows, columns = linear_sum_assignment(cost)
        geometry_correct = 0
        typed_correct = 0
        display_correct = 0
        chemical_correct = 0
        direction_correct = 0
        for row, column in zip(rows, columns):
            item = objects[int(row)]
            errors = _oriented_errors(
                item["endpoints"], record["endpoints"][int(column)]
            )
            geometry = bool((errors <= 6.0).all())
            display = item["display"] == int(record["display"][int(column)])
            chemical = item["chemical"] == int(record["chemical"][int(column)])
            direction = item["direction"] == int(record["direction"][int(column)])
            typed = geometry and display and chemical
            geometry_correct += int(geometry)
            typed_correct += int(typed)
            display_correct += int(geometry and display)
            chemical_correct += int(geometry and chemical)
            direction_correct += int(geometry and direction)
            role = "text" if bool(record["text_adjacent"][int(column)]) else "implicit"
            totals[f"{prefix}_{role}_geometry"] += int(geometry)
            totals[f"{prefix}_{role}_typed"] += int(typed)
        totals[f"{prefix}_geometry"] += geometry_correct
        totals[f"{prefix}_typed"] += typed_correct
        totals[f"{prefix}_display"] += display_correct
        totals[f"{prefix}_chemical"] += chemical_correct
        totals[f"{prefix}_direction"] += direction_correct
        exact_geometry = len(objects) == expected_count and geometry_correct == expected_count
        exact_typed = len(objects) == expected_count and typed_correct == expected_count
        totals[f"{prefix}_geometry_exact"] += int(exact_geometry)
        totals[f"{prefix}_typed_exact"] += int(exact_typed)
    totals["predicted_count_exact"] += int(len(predicted) == expected_count)


def _metrics(totals: Counter[str], threshold: float) -> dict[str, float | int]:
    images = max(totals["images"], 1)
    expected = max(totals["expected"], 1)
    predicted = max(totals["predicted_output"], 1)
    precision = totals["predicted_geometry"] / predicted
    recall = totals["predicted_geometry"] / expected
    return {
        "samples": totals["images"],
        "bonds": totals["expected"],
        "threshold": threshold,
        "known_geometry_accuracy": totals["known_geometry"] / expected,
        "known_typed_accuracy": totals["known_typed"] / expected,
        "known_display_accuracy": totals["known_display"] / expected,
        "known_chemical_accuracy": totals["known_chemical"] / expected,
        "known_direction_accuracy": totals["known_direction"] / expected,
        "known_geometry_exact": totals["known_geometry_exact"] / images,
        "known_typed_exact": totals["known_typed_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_geometry_precision": precision,
        "predicted_geometry_recall": recall,
        "predicted_geometry_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_geometry_exact": totals["predicted_geometry_exact"] / images,
        "predicted_typed_exact": totals["predicted_typed_exact"] / images,
        "text_adjacent_geometry_accuracy": totals["known_text_geometry"]
        / max(totals["text_expected"], 1),
        "text_adjacent_typed_accuracy": totals["known_text_typed"]
        / max(totals["text_expected"], 1),
        "implicit_geometry_accuracy": totals["known_implicit_geometry"]
        / max(totals["implicit_expected"], 1),
        "implicit_typed_accuracy": totals["known_implicit_typed"]
        / max(totals["implicit_expected"], 1),
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
            name: _metrics(values, threshold)
            for name, values in sorted(by_variant.items())
        },
        "by_stress_operation": {
            name: _metrics(values, threshold)
            for name, values in sorted(by_operation.items())
        },
    }


def select_threshold(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    sweep = [summarize_predictions(records, threshold) for threshold in THRESHOLDS]
    best = max(
        sweep,
        key=lambda row: (
            row["predicted_typed_exact"],
            row["predicted_geometry_f1"],
            row["predicted_count_accuracy"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), sweep


def _save_checkpoint(
    path: Path,
    detector: SemanticBondObjectDetector,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    threshold: float,
    calibration: dict[str, Any],
    samples_seen: int,
) -> None:
    payload = {
        "schema_version": "chemvector-v1p2-semantic-bondobject-checkpoint-0.1",
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
        "validation_read": False,
        "internal_test_read": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "validation_samples_exact496": metrics["samples"] == 496,
        "known_geometry_accuracy_min90pct": metrics["known_geometry_accuracy"]
        >= GO_GATES["known_geometry_accuracy"],
        "known_typed_accuracy_min85pct": metrics["known_typed_accuracy"]
        >= GO_GATES["known_typed_accuracy"],
        "known_typed_exact_min20pct": metrics["known_typed_exact"]
        >= GO_GATES["known_typed_exact"],
        "predicted_count_accuracy_min50pct": metrics["predicted_count_accuracy"]
        >= GO_GATES["predicted_count_accuracy"],
        "predicted_typed_exact_min15pct": metrics["predicted_typed_exact"]
        >= GO_GATES["predicted_typed_exact"],
        "predicted_geometry_f1_min88pct": metrics["predicted_geometry_f1"]
        >= GO_GATES["predicted_geometry_f1"],
        "text_adjacent_geometry_min85pct": metrics["text_adjacent_geometry_accuracy"]
        >= GO_GATES["text_adjacent_geometry_accuracy"],
        "stress_known_geometry_min85pct": metrics["by_variant"]["stress"]["known_geometry_accuracy"]
        >= GO_GATES["stress_known_geometry_accuracy"],
        "separator_frozen": True,
        "validation_consumed_once": True,
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
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P2 training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P2 output is not empty: {args.output_dir}")
    if file_sha256(args.separator) != SEPARATOR_SHA256:
        raise ValueError("v1 P2 separator SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_safe_manifest()
    fit_rows, calibration_rows = split_train_calibration_rows(rows)
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if len(validation_rows) != 496:
        raise ValueError("v1 P2 validation count differs")
    device = torch.device("cuda")
    separator_payload = torch.load(args.separator, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    detector = SemanticBondObjectDetector().to(device)
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
            targets = _targets(batch, device)
            probability = _bond_probability(separator, pixels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = detector(probability)
                losses = dense_bond_object_loss(
                    output,
                    targets,
                    endpoint_barrier_weight=0.02,
                    endpoint_barrier_margin_pixels=4.0,
                    heatmap_set_weight=0.1,
                    heatmap_set_margin=1.0,
                    slot_matching="hungarian",
                    cell_count_weight=0.1,
                    duplicate_weight=0.05,
                    duplicate_sigma_pixels=8.0,
                    image_size=768.0,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"v1 P2 non-finite loss at epoch={epoch}, step={step}")
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
            raise ValueError(f"v1 P2 epoch read {epoch_seen}, expected {len(fit_rows)}")
        calibration_records = collect_predictions(
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
        if len(calibration_records) != 400:
            raise ValueError("v1 P2 calibration count differs")
        threshold, sweep = select_threshold(calibration_records)
        calibration = summarize_predictions(calibration_records, threshold)
        score = (
            calibration["predicted_typed_exact"],
            calibration["known_typed_exact"],
            calibration["predicted_geometry_f1"],
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
    validation_records = collect_predictions(
        separator,
        detector,
        _loader(
            validation_rows,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    if len(validation_records) != 496:
        raise ValueError("v1 P2 validation read count differs")
    validation = summarize_predictions(validation_records, threshold)
    checks = _checks(validation)
    result = {
        "schema_version": "chemvector-v1p2-semantic-bondobject-result-0.1",
        "model_name": MODEL_NAME,
        "decision": "semantic_bondobject_go" if all(checks.values()) else "semantic_bondobject_no_go",
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
        "validation_samples": len(validation_rows),
        "internal_test_samples_read": 0,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "checkpoint": str(args.output_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(args.output_dir / "best.pt"),
        "calibration": best["calibration"],
        "validation": validation,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
