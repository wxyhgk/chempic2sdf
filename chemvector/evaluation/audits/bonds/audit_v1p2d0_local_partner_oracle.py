#!/usr/bin/env python3
"""Audit per-endpoint top-K partner unions before training a pair scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

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
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    DEFAULT_ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_THRESHOLD,
    IMAGE_SIZE,
    MATCH_RADIUS_PIXELS,
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


AUDIT_NAME = "chemvector-v1p2d0-local-partner-oracle"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
ROUTES = ("line_score", "line_gated_blocker_score")
PARTNER_COUNTS = (2, 4, 8, 16)
GO_GATES = {
    "winner_top8_pair_recall": 0.95,
    "winner_top8_image_exact": 0.30,
    "winner_top8_stress_pair_recall": 0.90,
    "winner_top8_crossed_pair_recall": 0.85,
}


def _selected_result(
    costs: torch.Tensor, selected: torch.Tensor, expected_bonds: int
) -> dict[str, int | bool]:
    if expected_bonds:
        recovered = (
            costs[selected].amin(dim=0) <= MATCH_RADIUS_PIXELS
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


def _new_totals() -> dict[str, Counter[str]]:
    return {f"top{count}": Counter() for count in PARTNER_COUNTS}


def _update(
    totals: dict[str, Counter[str]], results: dict[str, dict[str, Any]], bonds: int
) -> None:
    for name, values in results.items():
        totals[name]["images"] += 1
        totals[name]["bonds"] += bonds
        totals[name]["selected"] += int(values["selected"])
        totals[name]["recovered"] += int(values["recovered"])
        totals[name]["exact"] += int(values["exact"])


def _summary(values: Counter[str]) -> dict[str, float | int]:
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
    loader_factory: Callable[..., Any] = _loader,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = {route: _new_totals() for route in ROUTES}
    variants: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: {route: _new_totals() for route in ROUTES}
    )
    operations: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: {route: _new_totals() for route in ROUTES}
    )
    output_rows = []
    pair_config = SemanticEndpointPairConfig()
    loader = loader_factory(
        rows, batch_size=1, shuffle=False, seed=20260819, workers=workers
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
        points = (
            torch.stack([item["point"] for item in decoded]).to(device)
            if decoded
            else probability.new_empty((0, 2))
        )
        point_scores = torch.tensor(
            [float(item["score"]) for item in decoded], device=device
        )
        bond_count = int(batch["mask"][0].sum())
        expected = batch["endpoints"][0, :bond_count].to(device)
        evidence = score_endpoint_pairs(
            probability, points, point_scores, pair_config
        )
        costs = _pair_costs(points, evidence["pairs"], expected)
        variant = str(batch["variants"][0])
        operation = batch["stress_operations"][0]
        row_routes = {}
        for route in ROUTES:
            results = {}
            for partner_count in PARTNER_COUNTS:
                selected = select_local_partner_pairs(
                    evidence["pairs"],
                    evidence[route],
                    point_count=len(points),
                    partners_per_point=partner_count,
                )
                results[f"top{partner_count}"] = _selected_result(
                    costs, selected, bond_count
                )
            _update(totals[route], results, bond_count)
            _update(variants[variant][route], results, bond_count)
            if operation is not None:
                _update(operations[str(operation)][route], results, bond_count)
            row_routes[route] = results
        output_rows.append(
            {
                "key": batch["keys"][0],
                "variant": variant,
                "stress_operation": operation,
                "endpoint_candidates": len(points),
                "candidate_pairs": len(evidence["pairs"]),
                "bonds": bond_count,
                "routes": row_routes,
            }
        )
        if sample_index % log_every == 0:
            current = totals["line_score"]["top8"]
            print(
                json.dumps(
                    {
                        "samples": sample_index,
                        "line_top8_recall": current["recovered"]
                        / max(current["bonds"], 1),
                        "line_top8_exact": current["exact"]
                        / max(current["images"], 1),
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
                    count: _summary(values) for count, values in counts.items()
                }
                for route, counts in routes.items()
            }
            for name, routes in sorted(groups.items())
        }

    return (
        {
            "samples": len(output_rows),
            "routes": {
                route: {
                    count: _summary(values) for count, values in counts.items()
                }
                for route, counts in totals.items()
            },
            "by_variant": summarize_groups(variants),
            "by_stress_operation": summarize_groups(operations),
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
        raise RuntimeError("v1 P2-D0 local partner Oracle requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P2-D0 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("v1 P2-D0 endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("v1 P2-D0 separator SHA256 differs")

    _, rows, _ = split_train_endpoint_rows(load_safe_manifest())
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
    summary, output_rows = run_audit(
        separator,
        detector,
        rows,
        device,
        workers=args.workers,
        log_every=args.log_every,
    )
    winner = max(
        ROUTES,
        key=lambda route: (
            summary["routes"][route]["top8"]["pair_recall"],
            summary["routes"][route]["top8"]["image_exact"],
        ),
    )
    winner_top8 = summary["routes"][winner]["top8"]
    checks = {
        "samples_exact400": summary["samples"] == 400,
        "winner_top8_pair_recall_min95pct": winner_top8["pair_recall"]
        >= GO_GATES["winner_top8_pair_recall"],
        "winner_top8_image_exact_min30pct": winner_top8["image_exact"]
        >= GO_GATES["winner_top8_image_exact"],
        "winner_top8_stress_pair_recall_min90pct": summary["by_variant"][
            "stress"
        ][winner]["top8"]["pair_recall"]
        >= GO_GATES["winner_top8_stress_pair_recall"],
        "winner_top8_crossed_pair_recall_min85pct": summary[
            "by_stress_operation"
        ]["crossed_bonds"][winner]["top8"]["pair_recall"]
        >= GO_GATES["winner_top8_crossed_pair_recall"],
        "no_maximum_bond_length_cutoff": True,
        "p2b_holdout_unread": True,
        "validation_unread": True,
        "internal_test_unread": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "chemvector-v1p2d0-local-partner-oracle-0.1",
        "audit_name": AUDIT_NAME,
        "decision": "local_partner_oracle_go"
        if all(checks.values())
        else "local_partner_oracle_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "winner": winner,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "partner_counts": PARTNER_COUNTS,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "p2b_holdout_samples_read": 0,
        "validation_samples_read": 0,
        "internal_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "image_size": IMAGE_SIZE,
    }
    _atomic_json(args.output_dir / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
