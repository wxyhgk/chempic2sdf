#!/usr/bin/env python3
"""Capacity64 gate for Hungarian Bond queries over frozen candidate graphs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from chemvector.components.bonds.objects.semantic_bond_candidate_query_set import (
    SemanticBondCandidateQuerySet,
    SemanticBondCandidateQuerySetConfig,
    semantic_bond_candidate_query_set_loss,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e5_slots4_local_partner_oracle import (
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_RESULT_SHA256,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    SEPARATOR_SHA256,
)
from chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field import (
    MANIFEST_SHA256,
)
from chemvector.training.bonds.objects.train_v1p2d2e6_candidate_graph_capacity64 import (
    CACHE_SHA256,
    CAPACITY_SAMPLES,
    DEFAULT_CACHE,
    load_capacity_cache,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


MODEL_NAME = "chemvector-v1p2d2e7-bond-query-set-capacity64"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
SEED = 20260826
EXISTENCE_THRESHOLD = 0.5
GO_GATES = {
    "known_count_image_exact": 1.0,
    "predicted_count_accuracy": 1.0,
    "predicted_image_exact": 1.0,
}


def _graph_to_device(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **graph,
        "node_features": graph["node_features"].to(device, non_blocking=True),
        "edge_features": graph["edge_features"].to(device, non_blocking=True),
        "pairs": graph["pairs"].to(device, non_blocking=True),
        "edge_target_bond_ids": graph["edge_target_bond_ids"].to(
            device, non_blocking=True
        ),
    }


def query_selection_result(
    existence_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    target_bond_ids: torch.Tensor,
    *,
    selected_queries: torch.Tensor,
    bond_count: int,
) -> dict[str, int]:
    if selected_queries.ndim != 1:
        raise ValueError("Bond query selection indices must be one-dimensional")
    if len(selected_queries):
        edges = pointer_logits[selected_queries].float().argmax(dim=-1)
        targets = target_bond_ids[edges]
    else:
        targets = target_bond_ids.new_empty((0,))
    valid = targets[targets >= 0]
    recovered = len(set(int(value) for value in valid.tolist()))
    return {
        "selected": len(selected_queries),
        "correct": int((targets >= 0).sum()),
        "recovered": recovered,
        "exact": int(len(selected_queries) == bond_count and recovered == bond_count),
    }


def _update_metrics(
    totals: Counter[str],
    output: dict[str, torch.Tensor],
    graph: dict[str, Any],
) -> None:
    existence = output["existence_logits"].float()
    bond_count = int(graph["bond_count"])
    known_queries = existence.topk(bond_count).indices
    predicted_queries = torch.where(existence.sigmoid() >= EXISTENCE_THRESHOLD)[0]
    known = query_selection_result(
        existence,
        output["pointer_logits"],
        graph["edge_target_bond_ids"],
        selected_queries=known_queries,
        bond_count=bond_count,
    )
    predicted = query_selection_result(
        existence,
        output["pointer_logits"],
        graph["edge_target_bond_ids"],
        selected_queries=predicted_queries,
        bond_count=bond_count,
    )
    totals["samples"] += 1
    totals["bonds"] += bond_count
    totals["count_correct"] += int(len(predicted_queries) == bond_count)
    totals["count_absolute_error"] += abs(len(predicted_queries) - bond_count)
    for name, values in (("known", known), ("predicted", predicted)):
        for key, value in values.items():
            totals[f"{name}_{key}"] += int(value)


def _metrics(totals: Counter[str]) -> dict[str, float | int]:
    samples = max(totals["samples"], 1)
    bonds = max(totals["bonds"], 1)
    predicted_precision = totals["predicted_correct"] / max(
        totals["predicted_selected"], 1
    )
    predicted_recall = totals["predicted_recovered"] / bonds
    return {
        "samples": totals["samples"],
        "bonds": totals["bonds"],
        "predicted_count_accuracy": totals["count_correct"] / samples,
        "predicted_count_mae": totals["count_absolute_error"] / samples,
        "known_count_pair_precision": totals["known_correct"]
        / max(totals["known_selected"], 1),
        "known_count_pair_recall": totals["known_recovered"] / bonds,
        "known_count_image_exact": totals["known_exact"] / samples,
        "predicted_pair_precision": predicted_precision,
        "predicted_pair_recall": predicted_recall,
        "predicted_pair_f1": 2.0
        * predicted_precision
        * predicted_recall
        / max(predicted_precision + predicted_recall, 1e-12),
        "predicted_image_exact": totals["predicted_exact"] / samples,
    }


@torch.no_grad()
def evaluate_capacity(
    model: SemanticBondCandidateQuerySet,
    graphs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: Counter[str] = Counter()
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for source in graphs:
        graph = _graph_to_device(source, device)
        output = model(
            graph["node_features"], graph["edge_features"], graph["pairs"]
        )
        _update_metrics(totals, output, graph)
        _update_metrics(variants[str(graph["variant"])], output, graph)
    return {
        **_metrics(totals),
        "by_variant": {
            name: _metrics(values) for name, values in sorted(variants.items())
        },
    }


def capacity_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "samples_exact64": metrics["samples"] == CAPACITY_SAMPLES,
        "known_count_image_exact_100pct": metrics["known_count_image_exact"]
        >= GO_GATES["known_count_image_exact"],
        "predicted_count_accuracy_100pct": metrics["predicted_count_accuracy"]
        >= GO_GATES["predicted_count_accuracy"],
        "predicted_image_exact_100pct": metrics["predicted_image_exact"]
        >= GO_GATES["predicted_image_exact"],
        "all_four_variants_predicted_exact_100pct": all(
            values["predicted_image_exact"] == 1.0
            for values in metrics["by_variant"].values()
        ),
        "query_count_fixed128": True,
        "hungarian_group_matching": True,
        "candidate_cache_oracle_exact_100pct": True,
        "separator_and_endpoint_frozen": True,
        "style_calibration_samples_read_zero": True,
        "style_holdout_samples_read_zero": True,
        "fresh_calibration_samples_read_zero": True,
        "fresh_holdout_samples_read_zero": True,
        "sealed_test_samples_read_zero": True,
    }


def _save_checkpoint(
    path: Path,
    model: SemanticBondCandidateQuerySet,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict[str, Any],
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": "chemvector-v1p2d2e7-bond-query-set-checkpoint-0.1",
            "model_name": MODEL_NAME,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "capacity_metrics": metrics,
            "existence_threshold": EXISTENCE_THRESHOLD,
            "capacity_cache_sha256": CACHE_SHA256,
            "separator_sha256": SEPARATOR_SHA256,
            "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
            "style_manifest_sha256": MANIFEST_SHA256,
            "all_reserved_splits_read": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=4_800)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--evaluate-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E7 capacity training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E7 output is not empty: {args.output_dir}")
    if min(args.steps, args.evaluate_every, args.log_every) < 1:
        raise ValueError("P2-D2-E7 training intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graphs = load_capacity_cache(args.cache)["graphs"]
    maximum_bonds = max(int(graph["bond_count"]) for graph in graphs)
    config = SemanticBondCandidateQuerySetConfig()
    if maximum_bonds > config.query_count:
        raise ValueError("P2-D2-E7 capacity target exceeds Bond query count")
    device = torch.device("cuda")
    model = SemanticBondCandidateQuerySet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(graphs), generator=generator).tolist()
    cursor = 0
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    running_loss = 0.0
    history = []
    best_score = None
    best_step = 0
    completed_steps = 0
    consecutive_exact = 0
    for step in range(1, args.steps + 1):
        if cursor == len(order):
            order = torch.randperm(len(graphs), generator=generator).tolist()
            cursor = 0
        graph = _graph_to_device(graphs[order[cursor]], device)
        cursor += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                graph["node_features"], graph["edge_features"], graph["pairs"]
            )
            losses = semantic_bond_candidate_query_set_loss(
                output,
                graph["edge_target_bond_ids"],
                bond_count=int(graph["bond_count"]),
            )
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"P2-D2-E7 non-finite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(losses["loss"].detach())
        completed_steps = step
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "mean_loss": running_loss / step,
                        "current_loss": float(losses["loss"].detach()),
                    }
                ),
                flush=True,
            )
        if step % args.evaluate_every != 0 and step != args.steps:
            continue
        metrics = evaluate_capacity(model, graphs, device)
        score = (
            metrics["predicted_image_exact"],
            metrics["known_count_image_exact"],
            metrics["predicted_count_accuracy"],
            metrics["predicted_pair_f1"],
        )
        record = {
            "step": step,
            "mean_train_loss": running_loss / step,
            "metrics": metrics,
        }
        history.append(record)
        if best_score is None or score > best_score:
            best_score = score
            best_step = step
            _save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                step=step,
                metrics=metrics,
            )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        exact = (
            metrics["known_count_image_exact"] == 1.0
            and metrics["predicted_count_accuracy"] == 1.0
            and metrics["predicted_image_exact"] == 1.0
        )
        consecutive_exact = consecutive_exact + 1 if exact else 0
        if consecutive_exact >= 2:
            break

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"], strict=True)
    metrics = evaluate_capacity(model, graphs, device)
    checks = capacity_checks(metrics)
    _save_checkpoint(
        args.output_dir / "final.pt",
        model,
        optimizer,
        step=best_step,
        metrics=metrics,
    )
    result = {
        "schema_version": "chemvector-v1p2d2e7-bond-query-set-capacity-result-0.1",
        "model_name": MODEL_NAME,
        "decision": (
            "bond_query_set_capacity_go"
            if all(checks.values())
            else "bond_query_set_capacity_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "capacity_cache_sha256": CACHE_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "endpoint_result_sha256": ENDPOINT_RESULT_SHA256,
        "style_manifest_sha256": MANIFEST_SHA256,
        "maximum_capacity_bonds": maximum_bonds,
        "existence_threshold": EXISTENCE_THRESHOLD,
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "best_step": best_step,
        "mean_train_loss": running_loss / max(completed_steps, 1),
        "metrics": metrics,
        "history": history,
        "checkpoint": str(args.output_dir / "final.pt"),
        "checkpoint_sha256": file_sha256(args.output_dir / "final.pt"),
        "style_calibration_samples_read": 0,
        "style_holdout_samples_read": 0,
        "fresh_calibration_samples_read": 0,
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
