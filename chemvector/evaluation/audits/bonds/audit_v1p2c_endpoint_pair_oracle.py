#!/usr/bin/env python3
"""Audit sparse Bond pair candidates from wide semantic endpoint proposals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    decode_semantic_endpoints,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
    score_endpoint_pairs,
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
from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    _unique_port_data,
    split_train_endpoint_rows,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    MANIFEST_SHA256,
    load_safe_manifest,
)


AUDIT_NAME = "chemvector-v1p2c-endpoint-pair-oracle"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DEFAULT_ENDPOINT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v1p2b-semantic-endpoint-field-pilot/best.pt"
)
ENDPOINT_CHECKPOINT_SHA256 = (
    "3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca"
)
ENDPOINT_THRESHOLD = 0.05
MAXIMUM_ENDPOINTS = 256
MATCH_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0
ROUTES = (
    "endpoint_score",
    "line_score",
    "line_blocker_score",
    "line_gated_blocker_score",
)
BUDGET_MULTIPLIERS = (1, 2, 4, 8)
GO_GATES = {
    "candidate_unique_port_recall": 0.97,
    "exhaustive_pair_recall": 0.97,
    "winner_4x_pair_recall": 0.95,
    "winner_4x_image_exact": 0.30,
    "winner_4x_stress_pair_recall": 0.90,
    "winner_4x_crossed_pair_recall": 0.85,
}


def _target_candidate_coverage(
    points: torch.Tensor,
    expected_points: torch.Tensor,
    bond_port_indices: torch.Tensor,
) -> tuple[int, int]:
    if not len(points) or not len(expected_points):
        return 0, 0
    distances = torch.cdist(points.float(), expected_points.float()) * IMAGE_SIZE
    rows, columns = linear_sum_assignment(distances.cpu().numpy())
    recovered = torch.zeros(len(expected_points), dtype=torch.bool)
    for row, column in zip(rows, columns):
        if float(distances[row, column]) <= MATCH_RADIUS_PIXELS:
            recovered[int(column)] = True
    pairable = recovered[bond_port_indices.cpu()].all(dim=1)
    return int(recovered.sum()), int(pairable.sum())


def _pair_costs(
    points: torch.Tensor, pairs: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    if not len(pairs) or not len(expected):
        return points.new_empty((len(pairs), len(expected)))
    segments = points[pairs]
    forward = torch.linalg.vector_norm(
        (segments[:, None] - expected[None]) * IMAGE_SIZE, dim=-1
    ).amax(dim=-1)
    reverse = torch.linalg.vector_norm(
        (segments[:, None] - expected.flip(1)[None]) * IMAGE_SIZE, dim=-1
    ).amax(dim=-1)
    return torch.minimum(forward, reverse)


def _route_result(
    costs: torch.Tensor,
    scores: torch.Tensor,
    expected_bonds: int,
    multiplier: int | None,
) -> dict[str, int | bool]:
    valid = torch.where(scores > -1e5)[0]
    if multiplier is None:
        selected = valid
    else:
        budget = min(len(valid), max(1, multiplier * expected_bonds))
        selected = valid[scores[valid].topk(budget).indices]
    if expected_bonds:
        recovered = (
            (costs[selected].amin(dim=0) <= MATCH_RADIUS_PIXELS)
            if len(selected)
            else torch.zeros(expected_bonds, dtype=torch.bool, device=costs.device)
        )
    else:
        recovered = torch.empty(0, dtype=torch.bool, device=costs.device)
    count = int(recovered.sum())
    return {
        "selected": len(selected),
        "recovered": count,
        "exact": count == expected_bonds,
    }


def _new_route_totals() -> dict[str, Counter[str]]:
    return {
        **{f"{multiplier}x": Counter() for multiplier in BUDGET_MULTIPLIERS},
        "full": Counter(),
    }


def _update_route_totals(
    totals: dict[str, Counter[str]], results: dict[str, dict[str, Any]], bonds: int
) -> None:
    for budget, values in results.items():
        totals[budget]["images"] += 1
        totals[budget]["bonds"] += bonds
        totals[budget]["selected"] += int(values["selected"])
        totals[budget]["recovered"] += int(values["recovered"])
        totals[budget]["exact"] += int(values["exact"])


def _summarize_route(values: Counter[str]) -> dict[str, float | int]:
    return {
        "samples": values["images"],
        "bonds": values["bonds"],
        "selected_pairs": values["selected"],
        "mean_selected_pairs": values["selected"] / max(values["images"], 1),
        "pair_recall": values["recovered"] / max(values["bonds"], 1),
        "image_exact": values["exact"] / max(values["images"], 1),
    }


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
    pair_config = SemanticEndpointPairConfig()
    route_totals = {route: _new_route_totals() for route in ROUTES}
    variant_totals: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: {route: _new_route_totals() for route in ROUTES}
    )
    operation_totals: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: {route: _new_route_totals() for route in ROUTES}
    )
    coverage = Counter()
    candidate_counts = []
    pair_counts = []
    output_rows = []
    detector.eval()
    loader = _loader(
        rows, batch_size=1, shuffle=False, seed=20260818, workers=workers
    )
    for sample_index, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            field = detector(probability)
        decoded = decode_semantic_endpoints(
            field,
            threshold=ENDPOINT_THRESHOLD,
            maximum_points=MAXIMUM_ENDPOINTS,
        )[0]
        candidate_points = (
            torch.stack([item["point"] for item in decoded]).to(device)
            if decoded
            else probability.new_empty((0, 2))
        )
        candidate_scores = torch.tensor(
            [float(item["score"]) for item in decoded], device=device
        )
        bond_count = int(batch["mask"][0].sum())
        expected_bonds = batch["endpoints"][0, :bond_count].to(device)
        expected_points, port_indices = _unique_port_data(expected_bonds.cpu())
        endpoint_recovered, pairable = _target_candidate_coverage(
            candidate_points.cpu(), expected_points, port_indices
        )
        coverage["images"] += 1
        coverage["ports"] += len(expected_points)
        coverage["recovered_ports"] += endpoint_recovered
        coverage["bonds"] += bond_count
        coverage["pairable_bonds"] += pairable

        evidence = score_endpoint_pairs(
            probability, candidate_points, candidate_scores, pair_config
        )
        costs = _pair_costs(candidate_points, evidence["pairs"], expected_bonds)
        variant = str(batch["variants"][0])
        operation = batch["stress_operations"][0]
        row_results = {}
        for route in ROUTES:
            route_results = {
                f"{multiplier}x": _route_result(
                    costs, evidence[route], bond_count, multiplier
                )
                for multiplier in BUDGET_MULTIPLIERS
            }
            route_results["full"] = _route_result(
                costs, evidence[route], bond_count, None
            )
            _update_route_totals(route_totals[route], route_results, bond_count)
            _update_route_totals(
                variant_totals[variant][route], route_results, bond_count
            )
            if operation is not None:
                _update_route_totals(
                    operation_totals[str(operation)][route],
                    route_results,
                    bond_count,
                )
            row_results[route] = route_results
        candidate_counts.append(len(candidate_points))
        pair_counts.append(len(evidence["pairs"]))
        output_rows.append(
            {
                "key": batch["keys"][0],
                "variant": variant,
                "stress_operation": operation,
                "bonds": bond_count,
                "unique_ports": len(expected_points),
                "endpoint_candidates": len(candidate_points),
                "candidate_pairs": len(evidence["pairs"]),
                "recovered_ports": endpoint_recovered,
                "pairable_bonds": pairable,
                "routes": row_results,
            }
        )
        if sample_index % log_every == 0:
            print(
                json.dumps(
                    {
                        "samples": sample_index,
                        "endpoint_candidates_mean": sum(candidate_counts)
                        / len(candidate_counts),
                        "pairable_recall": coverage["pairable_bonds"]
                        / max(coverage["bonds"], 1),
                    }
                ),
                flush=True,
            )

    def summarize_groups(
        groups: dict[str, dict[str, dict[str, Counter[str]]]],
    ) -> dict[str, Any]:
        return {
            name: {
                route: {
                    budget: _summarize_route(values)
                    for budget, values in budgets.items()
                }
                for route, budgets in routes.items()
            }
            for name, routes in sorted(groups.items())
        }

    candidate_counts.sort()
    pair_counts.sort()
    summary = {
        "samples": coverage["images"],
        "unique_ports": coverage["ports"],
        "bonds": coverage["bonds"],
        "candidate_unique_port_recall": coverage["recovered_ports"]
        / max(coverage["ports"], 1),
        "candidate_pairable_bond_recall": coverage["pairable_bonds"]
        / max(coverage["bonds"], 1),
        "endpoint_candidates": {
            "mean": sum(candidate_counts) / max(len(candidate_counts), 1),
            "p95": candidate_counts[math.ceil(len(candidate_counts) * 0.95) - 1],
            "maximum": max(candidate_counts),
        },
        "candidate_pairs": {
            "mean": sum(pair_counts) / max(len(pair_counts), 1),
            "p95": pair_counts[math.ceil(len(pair_counts) * 0.95) - 1],
            "maximum": max(pair_counts),
        },
        "routes": {
            route: {
                budget: _summarize_route(values)
                for budget, values in budgets.items()
            }
            for route, budgets in route_totals.items()
        },
        "by_variant": summarize_groups(variant_totals),
        "by_stress_operation": summarize_groups(operation_totals),
    }
    return summary, output_rows


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
        raise RuntimeError("v1 P2-C pair Oracle requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P2-C output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("v1 P2-C endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("v1 P2-C separator SHA256 differs")

    _, calibration_rows, _ = split_train_endpoint_rows(load_safe_manifest())
    if len(calibration_rows) != 400:
        raise ValueError("v1 P2-C calibration count differs")
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
        calibration_rows,
        device,
        workers=args.workers,
        log_every=args.log_every,
    )
    winner = max(
        ROUTES,
        key=lambda route: (
            summary["routes"][route]["4x"]["pair_recall"],
            summary["routes"][route]["4x"]["image_exact"],
        ),
    )
    checks = {
        "samples_exact400": summary["samples"] == 400,
        "candidate_unique_port_recall_min97pct": summary[
            "candidate_unique_port_recall"
        ]
        >= GO_GATES["candidate_unique_port_recall"],
        "exhaustive_pair_recall_min97pct": summary["routes"][winner]["full"][
            "pair_recall"
        ]
        >= GO_GATES["exhaustive_pair_recall"],
        "winner_4x_pair_recall_min95pct": summary["routes"][winner]["4x"][
            "pair_recall"
        ]
        >= GO_GATES["winner_4x_pair_recall"],
        "winner_4x_image_exact_min30pct": summary["routes"][winner]["4x"][
            "image_exact"
        ]
        >= GO_GATES["winner_4x_image_exact"],
        "winner_4x_stress_pair_recall_min90pct": summary["by_variant"]["stress"][
            winner
        ]["4x"]["pair_recall"]
        >= GO_GATES["winner_4x_stress_pair_recall"],
        "winner_4x_crossed_pair_recall_min85pct": summary[
            "by_stress_operation"
        ]["crossed_bonds"][winner]["4x"]["pair_recall"]
        >= GO_GATES["winner_4x_crossed_pair_recall"],
        "no_maximum_bond_length_cutoff": True,
        "endpoint_and_separator_frozen": True,
        "p2b_holdout_unread": True,
        "validation_unread": True,
        "internal_test_unread": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "chemvector-v1p2c-endpoint-pair-oracle-0.1",
        "audit_name": AUDIT_NAME,
        "decision": "pair_candidate_route_select_go"
        if all(checks.values())
        else "pair_candidate_route_select_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "winner": winner,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": MAXIMUM_ENDPOINTS,
        "pair_config": asdict(SemanticEndpointPairConfig()),
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "p2b_holdout_samples_read": 0,
        "validation_samples_read": 0,
        "internal_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
