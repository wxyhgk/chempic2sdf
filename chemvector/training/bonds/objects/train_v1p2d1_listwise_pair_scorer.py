#!/usr/bin/env python3
"""Train existence ranking over frozen top-8 endpoint partner candidates."""

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

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    decode_semantic_endpoints,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
    score_endpoint_pairs,
    select_local_partner_pairs,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    SemanticEndpointPairScorer,
    SemanticEndpointPairScorerConfig,
    deduplicate_scored_pair_segments,
    extract_endpoint_pair_features,
    semantic_endpoint_pair_scorer_loss,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    DEFAULT_ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_THRESHOLD,
    MAXIMUM_ENDPOINTS,
    _pair_costs,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    _bond_probability,
    _loader,
)
from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    split_train_endpoint_rows,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    MANIFEST_SHA256,
    load_safe_manifest,
)


MODEL_NAME = "chemvector-v1p2d1-listwise-pair-scorer-pilot"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
SPLIT_SEED = 20260820
CALIBRATION_MOLECULES = 100
PARTNERS_PER_ENDPOINT = 8
PAIR_MATCH_RADIUS_PIXELS = 6.0
PAIR_NMS_RADIUS_PIXELS = 4.0
THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))
GO_GATES = {
    "known_pair_recall": 0.90,
    "known_image_exact": 0.20,
    "predicted_pair_f1": 0.80,
    "predicted_count_accuracy": 0.20,
    "predicted_image_exact": 0.10,
    "stress_known_pair_recall": 0.85,
    "crossed_known_pair_recall": 0.75,
}


def split_pair_scorer_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    p2b_fit, _, p2b_holdout = split_train_endpoint_rows(rows)
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in p2b_fit:
        by_molecule[str(row["molecule_id"])].append(row)
    ranked = sorted(
        by_molecule,
        key=lambda molecule_id: (
            hashlib.sha256(f"{SPLIT_SEED}:{molecule_id}".encode()).hexdigest(),
            molecule_id,
        ),
    )
    calibration_ids = set(ranked[:CALIBRATION_MOLECULES])
    fit = [row for row in p2b_fit if row["molecule_id"] not in calibration_ids]
    calibration = [
        row for row in p2b_fit if row["molecule_id"] in calibration_ids
    ]
    for group in (fit, calibration, p2b_holdout):
        group.sort(key=lambda row: str(row["sample_key"]))
    if (len(fit), len(calibration), len(p2b_holdout)) != (7_656, 400, 400):
        raise ValueError("P2-D1 fit/calibration/holdout counts differ")
    molecule_sets = [
        {str(row["molecule_id"]) for row in group}
        for group in (fit, calibration, p2b_holdout)
    ]
    if any(
        molecule_sets[first] & molecule_sets[second]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("P2-D1 molecule split overlap")
    return fit, calibration, p2b_holdout


@torch.no_grad()
def build_pair_sample(
    separator: SemanticInkSeparator,
    detector: SemanticBondEndpointDetector,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    probability = _bond_probability(separator, pixels)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        field = detector(probability)
    decoded = decode_semantic_endpoints(
        field, threshold=ENDPOINT_THRESHOLD, maximum_points=MAXIMUM_ENDPOINTS
    )[0]
    points = (
        torch.stack([item["point"] for item in decoded]).to(device)
        if decoded
        else probability.new_empty((0, 2))
    )
    point_scores = torch.tensor(
        [float(item["score"]) for item in decoded], device=device
    )
    pair_config = SemanticEndpointPairConfig()
    evidence = score_endpoint_pairs(
        probability, points, point_scores, pair_config
    )
    selected = select_local_partner_pairs(
        evidence["pairs"],
        evidence["line_score"],
        point_count=len(points),
        partners_per_point=PARTNERS_PER_ENDPOINT,
    )
    pairs = evidence["pairs"][selected]
    features = extract_endpoint_pair_features(
        probability, points, point_scores, pairs, pair_config
    )
    bond_count = int(batch["mask"][0].sum())
    expected = batch["endpoints"][0, :bond_count].to(device)
    costs = _pair_costs(points, pairs, expected)
    if len(pairs) and bond_count:
        minimum, target_ids = costs.min(dim=1)
        target_ids = target_ids.masked_fill(
            minimum > PAIR_MATCH_RADIUS_PIXELS, -1
        )
    else:
        target_ids = torch.full(
            (len(pairs),), -1, device=device, dtype=torch.long
        )
    return {
        "features": features,
        "target_bond_ids": target_ids,
        "points": points,
        "pairs": pairs,
        "bond_count": bond_count,
        "key": batch["keys"][0],
        "variant": batch["variants"][0],
        "stress_operation": batch["stress_operations"][0],
    }


@torch.no_grad()
def collect_records(
    separator: SemanticInkSeparator,
    detector: SemanticBondEndpointDetector,
    scorer: SemanticEndpointPairScorer,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    scorer.eval()
    records = []
    loader = _loader(rows, batch_size=1, shuffle=False, seed=SPLIT_SEED, workers=workers)
    for batch in loader:
        sample = build_pair_sample(separator, detector, batch, device)
        logits = scorer(sample["features"]).float()
        order = deduplicate_scored_pair_segments(
            sample["points"],
            sample["pairs"],
            logits,
            radius_pixels=PAIR_NMS_RADIUS_PIXELS,
        )
        records.append(
            {
                "key": sample["key"],
                "variant": sample["variant"],
                "stress_operation": sample["stress_operation"],
                "scores": logits[order].sigmoid().cpu(),
                "target_bond_ids": sample["target_bond_ids"][order].cpu(),
                "bond_count": sample["bond_count"],
                "candidate_count": len(sample["pairs"]),
                "deduplicated_count": len(order),
            }
        )
    return records


def _accumulate_record(
    totals: Counter[str], record: dict[str, Any], threshold: float
) -> None:
    bond_count = int(record["bond_count"])
    scores = record["scores"]
    target_ids = record["target_bond_ids"]
    known_count = min(bond_count, len(scores))
    selections = {
        "known": torch.arange(known_count),
        "predicted": torch.where(scores >= threshold)[0],
    }
    totals["images"] += 1
    totals["bonds"] += bond_count
    totals["candidates"] += int(record["candidate_count"])
    totals["deduplicated"] += int(record["deduplicated_count"])
    for prefix, selected in selections.items():
        selected_targets = target_ids[selected]
        recovered = len(set(int(value) for value in selected_targets.tolist() if value >= 0))
        totals[f"{prefix}_selected"] += len(selected)
        totals[f"{prefix}_recovered"] += recovered
        totals[f"{prefix}_exact"] += int(
            len(selected) == bond_count and recovered == bond_count
        )
    totals["predicted_count_exact"] += int(len(selections["predicted"]) == bond_count)


def _metrics(totals: Counter[str], threshold: float) -> dict[str, float | int]:
    images = max(totals["images"], 1)
    bonds = max(totals["bonds"], 1)
    predicted = max(totals["predicted_selected"], 1)
    precision = totals["predicted_recovered"] / predicted
    recall = totals["predicted_recovered"] / bonds
    return {
        "samples": totals["images"],
        "bonds": totals["bonds"],
        "threshold": threshold,
        "mean_candidates": totals["candidates"] / images,
        "mean_deduplicated_candidates": totals["deduplicated"] / images,
        "known_pair_recall": totals["known_recovered"] / bonds,
        "known_image_exact": totals["known_exact"] / images,
        "predicted_pair_precision": precision,
        "predicted_pair_recall": recall,
        "predicted_pair_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_image_exact": totals["predicted_exact"] / images,
    }


def summarize_records(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    operations: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        _accumulate_record(totals, record, threshold)
        _accumulate_record(variants[str(record["variant"])], record, threshold)
        if record["stress_operation"] is not None:
            _accumulate_record(
                operations[str(record["stress_operation"])], record, threshold
            )
    return {
        **_metrics(totals, threshold),
        "by_variant": {
            name: _metrics(values, threshold)
            for name, values in sorted(variants.items())
        },
        "by_stress_operation": {
            name: _metrics(values, threshold)
            for name, values in sorted(operations.items())
        },
    }


def select_threshold(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    sweep = [summarize_records(records, threshold) for threshold in THRESHOLDS]
    best = max(
        sweep,
        key=lambda row: (
            row["predicted_image_exact"],
            row["predicted_pair_f1"],
            row["predicted_count_accuracy"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), sweep


def _save_checkpoint(
    path: Path,
    scorer: SemanticEndpointPairScorer,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    threshold: float,
    calibration: dict[str, Any],
    samples_seen: int,
) -> None:
    payload = {
        "schema_version": "chemvector-v1p2d1-pair-scorer-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(scorer.config),
        "scorer": scorer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "threshold": threshold,
        "calibration": calibration,
        "samples_seen": samples_seen,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
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
        "known_pair_recall_min90pct": metrics["known_pair_recall"]
        >= GO_GATES["known_pair_recall"],
        "known_image_exact_min20pct": metrics["known_image_exact"]
        >= GO_GATES["known_image_exact"],
        "predicted_pair_f1_min80pct": metrics["predicted_pair_f1"]
        >= GO_GATES["predicted_pair_f1"],
        "predicted_count_accuracy_min20pct": metrics["predicted_count_accuracy"]
        >= GO_GATES["predicted_count_accuracy"],
        "predicted_image_exact_min10pct": metrics["predicted_image_exact"]
        >= GO_GATES["predicted_image_exact"],
        "stress_known_pair_recall_min85pct": metrics["by_variant"]["stress"][
            "known_pair_recall"
        ]
        >= GO_GATES["stress_known_pair_recall"],
        "crossed_known_pair_recall_min75pct": metrics["by_stress_operation"][
            "crossed_bonds"
        ]["known_pair_recall"]
        >= GO_GATES["crossed_known_pair_recall"],
        "separator_endpoint_and_candidates_frozen": True,
        "holdout_consumed_once": True,
        "validation_unread": True,
        "internal_test_unread": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-checkpoint", type=Path, default=DEFAULT_ENDPOINT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P2-D1 pair scorer training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P2-D1 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("v1 P2-D1 endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("v1 P2-D1 separator SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    fit_rows, calibration_rows, holdout_rows = split_pair_scorer_rows(
        load_safe_manifest()
    )
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    endpoint_payload = torch.load(args.endpoint_checkpoint, map_location=device)
    detector = SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(**endpoint_payload["config"])
    ).to(device)
    detector.load_state_dict(endpoint_payload["detector"], strict=True)
    detector.eval().requires_grad_(False)
    scorer = SemanticEndpointPairScorer().to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    samples_seen = 0
    history = []
    best_score = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        running = 0.0
        epoch_seen = 0
        loader = _loader(
            fit_rows,
            batch_size=1,
            shuffle=True,
            seed=args.seed + epoch,
            workers=args.workers,
        )
        for step, batch in enumerate(loader, 1):
            sample = build_pair_sample(separator, detector, batch, device)
            if not len(sample["features"]):
                raise ValueError(f"P2-D1 empty pair sample: {sample['key']}")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = scorer(sample["features"])
                losses = semantic_endpoint_pair_scorer_loss(
                    logits,
                    sample["target_bond_ids"],
                    bond_count=sample["bond_count"],
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"P2-D1 non-finite loss at epoch={epoch}, step={step}"
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            epoch_seen += 1
            samples_seen += 1
            running += float(losses["loss"].detach())
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
            raise ValueError(f"P2-D1 epoch read {epoch_seen}, expected {len(fit_rows)}")
        records = collect_records(
            separator,
            detector,
            scorer,
            calibration_rows,
            device,
            workers=args.workers,
        )
        threshold, sweep = select_threshold(records)
        calibration = summarize_records(records, threshold)
        score = (
            calibration["known_image_exact"],
            calibration["known_pair_recall"],
            calibration["predicted_pair_f1"],
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
            scorer,
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
                scorer,
                optimizer,
                epoch=epoch,
                threshold=threshold,
                calibration=calibration,
                samples_seen=samples_seen,
            )
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    scorer.load_state_dict(best["scorer"], strict=True)
    threshold = float(best["threshold"])
    holdout_records = collect_records(
        separator,
        detector,
        scorer,
        holdout_rows,
        device,
        workers=args.workers,
    )
    holdout = summarize_records(holdout_records, threshold)
    checks = _checks(holdout)
    result = {
        "schema_version": "chemvector-v1p2d1-pair-scorer-result-0.1",
        "model_name": MODEL_NAME,
        "decision": "listwise_pair_scorer_go"
        if all(checks.values())
        else "listwise_pair_scorer_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(scorer.config),
        "parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "partners_per_endpoint": PARTNERS_PER_ENDPOINT,
        "pair_nms_radius_pixels": PAIR_NMS_RADIUS_PIXELS,
        "fit_samples": len(fit_rows),
        "fit_molecules": len({row["molecule_id"] for row in fit_rows}),
        "calibration_samples": len(calibration_rows),
        "calibration_molecules": CALIBRATION_MOLECULES,
        "holdout_samples": len(holdout_rows),
        "holdout_molecules": len({row["molecule_id"] for row in holdout_rows}),
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
