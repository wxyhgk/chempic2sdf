#!/usr/bin/env python3
"""Audit endpoint threshold and candidate-budget ceilings on fresh calibration."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    decode_semantic_endpoints,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    DEFAULT_ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    _target_candidate_coverage,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle import (
    SOURCE_COMMITMENT,
    _fresh_loader,
    load_fresh_split,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    _bond_probability,
)
from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    _unique_port_data,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p2d2e2-endpoint-budget-ladder"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P2D2E1_AUDIT = Path(
    "outputs/audits/chemvector-v1p2d2e1-fresh-endpoint-reachability/audit.json"
)
P2D2E1_AUDIT_SHA256 = (
    "baab7dccc977500fccee4c9db7dfaf3afc0d5afe36cf3e663a7aeade3caeb0ad"
)
THRESHOLDS = (0.05, 0.02, 0.01)
MAXIMUM_POINTS = (256, 384, 512)
TARGET_RECALL = 0.97


def config_name(threshold: float, maximum_points: int) -> str:
    return f"threshold_{threshold:.2f}_max_{maximum_points}"


def route_decision(configs: dict[str, dict[str, float]]) -> str:
    def passes(threshold: float, maximum_points: int) -> bool:
        metrics = configs[config_name(threshold, maximum_points)]
        return (
            metrics["unique_port_recall"] >= TARGET_RECALL
            and metrics["pairable_bond_recall"] >= TARGET_RECALL
        )

    if any(passes(0.05, budget) for budget in (384, 512)):
        return "maximum_points_ceiling"
    if any(
        passes(threshold, budget)
        for threshold in (0.02, 0.01)
        for budget in MAXIMUM_POINTS
    ):
        return "endpoint_threshold_ceiling"
    return "endpoint_model_retrain"


def _summary(totals: Counter[str]) -> dict[str, float | int]:
    images = max(totals["images"], 1)
    return {
        "samples": totals["images"],
        "unique_ports": totals["ports"],
        "bonds": totals["bonds"],
        "output_candidates": totals["outputs"],
        "mean_output_candidates": totals["outputs"] / images,
        "budget_saturated_images": totals["saturated"],
        "budget_saturated_rate": totals["saturated"] / images,
        "unique_port_recall": totals["recovered_ports"]
        / max(totals["ports"], 1),
        "pairable_bond_recall": totals["pairable_bonds"]
        / max(totals["bonds"], 1),
        "port_exact_images": totals["port_exact"],
        "port_exact_rate": totals["port_exact"] / images,
    }


def _update(
    totals: Counter[str],
    *,
    outputs: int,
    maximum_points: int,
    ports: int,
    recovered_ports: int,
    bonds: int,
    pairable_bonds: int,
) -> None:
    totals["images"] += 1
    totals["ports"] += ports
    totals["bonds"] += bonds
    totals["outputs"] += outputs
    totals["saturated"] += int(outputs == maximum_points)
    totals["recovered_ports"] += recovered_ports
    totals["pairable_bonds"] += pairable_bonds
    totals["port_exact"] += int(recovered_ports == ports)


@torch.no_grad()
def run_audit(
    separator: SemanticInkSeparator,
    detector: SemanticBondEndpointDetector,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    workers: int,
    log_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = [
        config_name(threshold, maximum_points)
        for threshold in THRESHOLDS
        for maximum_points in MAXIMUM_POINTS
    ]
    totals = {name: Counter() for name in names}
    variants: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {name: Counter() for name in names}
    )
    operations: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {name: Counter() for name in names}
    )
    output_rows = []
    loader = _fresh_loader(
        rows, batch_size=1, shuffle=False, seed=20260822, workers=workers
    )
    for sample_index, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            field = detector(probability)
        decoded = decode_semantic_endpoints(
            field,
            threshold=min(THRESHOLDS),
            maximum_points=max(MAXIMUM_POINTS),
        )[0]
        bond_count = int(batch["mask"][0].sum())
        expected_bonds = batch["endpoints"][0, :bond_count]
        expected_points, port_indices = _unique_port_data(expected_bonds)
        variant = str(batch["variants"][0])
        operation = batch["stress_operations"][0]
        row_results = {}
        for threshold in THRESHOLDS:
            thresholded = [
                item for item in decoded if float(item["score"]) >= threshold
            ]
            for maximum_points in MAXIMUM_POINTS:
                selected = thresholded[:maximum_points]
                points = (
                    torch.stack([item["point"] for item in selected])
                    if selected
                    else torch.empty((0, 2))
                )
                recovered_ports, pairable_bonds = _target_candidate_coverage(
                    points, expected_points, port_indices
                )
                name = config_name(threshold, maximum_points)
                values = {
                    "outputs": len(selected),
                    "recovered_ports": recovered_ports,
                    "pairable_bonds": pairable_bonds,
                }
                row_results[name] = values
                kwargs = {
                    **values,
                    "maximum_points": maximum_points,
                    "ports": len(expected_points),
                    "bonds": bond_count,
                }
                _update(totals[name], **kwargs)
                _update(variants[variant][name], **kwargs)
                if operation is not None:
                    _update(operations[str(operation)][name], **kwargs)
        output_rows.append(
            {
                "key": batch["keys"][0],
                "variant": variant,
                "stress_operation": operation,
                "unique_ports": len(expected_points),
                "bonds": bond_count,
                "decoded_at_threshold_0_01_max512": len(decoded),
                "configs": row_results,
            }
        )
        if sample_index % log_every == 0:
            current = _summary(totals[config_name(0.05, 512)])
            print(
                json.dumps(
                    {
                        "samples": sample_index,
                        "fixed_threshold_max512_port_recall": current[
                            "unique_port_recall"
                        ],
                        "fixed_threshold_max512_pairable_recall": current[
                            "pairable_bond_recall"
                        ],
                    }
                ),
                flush=True,
            )

    return (
        {
            "samples": len(output_rows),
            "configs": {name: _summary(values) for name, values in totals.items()},
            "by_variant": {
                variant: {
                    name: _summary(values) for name, values in current.items()
                }
                for variant, current in sorted(variants.items())
            },
            "by_stress_operation": {
                operation: {
                    name: _summary(values) for name, values in current.items()
                }
                for operation, current in sorted(operations.items())
            },
        },
        output_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-checkpoint", type=Path, default=DEFAULT_ENDPOINT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E2 endpoint budget ladder requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E2 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E2 endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E2 separator SHA256 differs")
    if file_sha256(P2D2E1_AUDIT) != P2D2E1_AUDIT_SHA256:
        raise ValueError("P2-D2-E2 prerequisite audit SHA256 differs")

    calibration, _, commitment = load_fresh_split()
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

    started = time.perf_counter()
    summary, rows = run_audit(
        separator,
        detector,
        calibration,
        device,
        workers=args.workers,
        log_every=args.log_every,
    )
    baseline = summary["configs"][config_name(0.05, 256)]
    checks = {
        "calibration_samples_exact400": summary["samples"] == 400,
        "baseline_port_recall_matches_p2d2e1": abs(
            baseline["unique_port_recall"] - 0.9588126763620577
        )
        < 1e-12,
        "baseline_pairable_recall_matches_p2d2e1": abs(
            baseline["pairable_bond_recall"] - 0.9516090297790586
        )
        < 1e-12,
        "source_commitment_matches": commitment["commitment_sha256"]
        == SOURCE_COMMITMENT,
        "separator_endpoint_frozen": True,
        "fresh_holdout_samples_read_zero": True,
        "sealed_test_samples_read_zero": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "chemvector-v1p2d2e2-endpoint-budget-ladder-0.1",
        "audit_name": AUDIT_NAME,
        "decision": route_decision(summary["configs"]),
        "checks": checks,
        "target_recall": TARGET_RECALL,
        "thresholds": THRESHOLDS,
        "maximum_points": MAXIMUM_POINTS,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "source_commitment": SOURCE_COMMITMENT,
        "p2d2e1_audit_sha256": P2D2E1_AUDIT_SHA256,
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
