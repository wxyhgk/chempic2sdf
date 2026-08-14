#!/usr/bin/env python3
"""Capacity64 gate for joint scoring over the frozen endpoint candidate graph."""

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

from chemvector.components.bonds.objects.semantic_endpoint_candidate_graph import (
    SemanticEndpointCandidateGraph,
    SemanticEndpointCandidateGraphConfig,
    semantic_endpoint_candidate_graph_loss,
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
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


MODEL_NAME = "chemvector-v1p2d2e6-candidate-graph-capacity64"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
DEFAULT_CACHE = Path(
    "outputs/audits/chemvector-v1p2d2e6-candidate-graph-capacity64-cache/capacity64.pt"
)
CACHE_SHA256 = "6d80911067a60dc91ad4cf5f7716e7549122bdef8d2f46015a562ae896b2bcd8"
CAPACITY_SAMPLES = 64
SEED = 20260825
GO_GATES = {
    "known_count_image_exact": 1.0,
    "predicted_count_accuracy": 1.0,
    "predicted_image_exact": 1.0,
}


def load_capacity_cache(path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if file_sha256(path) != CACHE_SHA256:
        raise ValueError("P2-D2-E6 capacity cache SHA256 differs")
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != "chemvector-v1p2d2e6-candidate-graph-cache-0.1":
        raise ValueError("P2-D2-E6 capacity cache schema differs")
    if len(payload.get("graphs", ())) != CAPACITY_SAMPLES:
        raise ValueError("P2-D2-E6 capacity cache sample count differs")
    frozen = {
        "separator_sha256": SEPARATOR_SHA256,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "endpoint_result_sha256": ENDPOINT_RESULT_SHA256,
        "style_manifest_sha256": MANIFEST_SHA256,
        "endpoint_threshold": 0.05,
        "maximum_endpoints": 256,
        "partners_per_endpoint": 8,
    }
    if any(payload.get(key) != value for key, value in frozen.items()):
        raise ValueError("P2-D2-E6 capacity cache frozen protocol differs")
    for graph in payload["graphs"]:
        bond_count = int(graph["bond_count"])
        targets = graph["edge_target_bond_ids"]
        if int(graph["reachable_bond_count"]) != bond_count:
            raise ValueError("P2-D2-E6 capacity graph is not Oracle-exact")
        if set(targets[targets >= 0].tolist()) != set(range(bond_count)):
            raise ValueError("P2-D2-E6 capacity graph target IDs are incomplete")
    return payload


def _graph_to_device(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **graph,
        "node_features": graph["node_features"].to(device, non_blocking=True),
        "edge_features": graph["edge_features"].to(device, non_blocking=True),
        "pairs": graph["pairs"].to(device, non_blocking=True),
        "node_target_degrees": graph["node_target_degrees"].to(
            device, non_blocking=True
        ),
        "edge_target_bond_ids": graph["edge_target_bond_ids"].to(
            device, non_blocking=True
        ),
    }


def _selection_result(
    logits: torch.Tensor,
    target_bond_ids: torch.Tensor,
    *,
    selected_count: int,
    bond_count: int,
) -> dict[str, int]:
    selected_count = max(0, min(int(selected_count), len(logits)))
    selected = (
        logits.float().topk(selected_count).indices
        if selected_count
        else torch.empty(0, dtype=torch.long, device=logits.device)
    )
    selected_targets = target_bond_ids[selected]
    valid = selected_targets[selected_targets >= 0]
    recovered = len(set(int(value) for value in valid.tolist()))
    correct = int((selected_targets >= 0).sum())
    return {
        "selected": selected_count,
        "correct": correct,
        "recovered": recovered,
        "exact": int(selected_count == bond_count and recovered == bond_count),
    }


def _update_metrics(
    totals: Counter[str],
    output: dict[str, torch.Tensor],
    graph: dict[str, Any],
) -> None:
    bond_count = int(graph["bond_count"])
    predicted_count = max(
        0,
        min(
            len(output["edge_logits"]),
            int(torch.round(output["graph_bond_count"].float()).item()),
        ),
    )
    known = _selection_result(
        output["edge_logits"],
        graph["edge_target_bond_ids"],
        selected_count=bond_count,
        bond_count=bond_count,
    )
    predicted = _selection_result(
        output["edge_logits"],
        graph["edge_target_bond_ids"],
        selected_count=predicted_count,
        bond_count=bond_count,
    )
    node_keep_target = graph["node_target_degrees"] > 0
    node_keep_prediction = output["node_keep_logits"] >= 0.0
    degree_prediction = output["node_degree_logits"].argmax(dim=-1)
    totals["samples"] += 1
    totals["bonds"] += bond_count
    totals["nodes"] += len(node_keep_target)
    totals["count_correct"] += int(predicted_count == bond_count)
    totals["count_absolute_error"] += abs(predicted_count - bond_count)
    totals["node_keep_correct"] += int(
        (node_keep_prediction == node_keep_target).sum()
    )
    totals["node_keep_exact"] += int(
        bool((node_keep_prediction == node_keep_target).all())
    )
    totals["node_degree_correct"] += int(
        (degree_prediction == graph["node_target_degrees"]).sum()
    )
    for name, values in (("known", known), ("predicted", predicted)):
        for key, value in values.items():
            totals[f"{name}_{key}"] += int(value)


def _metrics(totals: Counter[str]) -> dict[str, float | int]:
    samples = max(totals["samples"], 1)
    bonds = max(totals["bonds"], 1)
    nodes = max(totals["nodes"], 1)
    precision = totals["predicted_correct"] / max(totals["predicted_selected"], 1)
    recall = totals["predicted_recovered"] / bonds
    return {
        "samples": totals["samples"],
        "bonds": totals["bonds"],
        "predicted_count_accuracy": totals["count_correct"] / samples,
        "predicted_count_mae": totals["count_absolute_error"] / samples,
        "known_count_pair_precision": totals["known_correct"]
        / max(totals["known_selected"], 1),
        "known_count_pair_recall": totals["known_recovered"] / bonds,
        "known_count_image_exact": totals["known_exact"] / samples,
        "predicted_pair_precision": precision,
        "predicted_pair_recall": recall,
        "predicted_pair_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_image_exact": totals["predicted_exact"] / samples,
        "node_keep_accuracy": totals["node_keep_correct"] / nodes,
        "node_keep_image_exact": totals["node_keep_exact"] / samples,
        "node_degree_accuracy": totals["node_degree_correct"] / nodes,
    }


@torch.no_grad()
def evaluate_capacity(
    model: SemanticEndpointCandidateGraph,
    graphs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: Counter[str] = Counter()
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for source in graphs:
        graph = _graph_to_device(source, device)
        output = model(
            graph["node_features"], graph["edge_features"], graph["pairs"]
        )
        _update_metrics(totals, output, graph)
        _update_metrics(by_variant[str(graph["variant"])], output, graph)
    return {
        **_metrics(totals),
        "by_variant": {
            name: _metrics(values) for name, values in sorted(by_variant.items())
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
    model: SemanticEndpointCandidateGraph,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict[str, Any],
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": "chemvector-v1p2d2e6-candidate-graph-checkpoint-0.1",
            "model_name": MODEL_NAME,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "capacity_metrics": metrics,
            "capacity_cache_sha256": CACHE_SHA256,
            "separator_sha256": SEPARATOR_SHA256,
            "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
            "style_manifest_sha256": MANIFEST_SHA256,
            "style_calibration_read": False,
            "style_holdout_read": False,
            "fresh_calibration_read": False,
            "fresh_holdout_read": False,
            "sealed_test_read": False,
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
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E6 capacity training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E6 output is not empty: {args.output_dir}")
    if min(args.steps, args.evaluate_every, args.log_every) < 1:
        raise ValueError("P2-D2-E6 training intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cache = load_capacity_cache(args.cache)
    graphs = cache["graphs"]
    device = torch.device("cuda")
    model = SemanticEndpointCandidateGraph().to(device)
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
    consecutive_exact = 0
    completed_steps = 0
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
            losses = semantic_endpoint_candidate_graph_loss(
                output,
                graph["edge_target_bond_ids"],
                graph["node_target_degrees"],
                bond_count=int(graph["bond_count"]),
                max_degree=model.config.max_degree,
            )
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"P2-D2-E6 non-finite loss at step {step}")
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
            metrics["predicted_count_accuracy"],
            metrics["known_count_image_exact"],
            metrics["predicted_pair_f1"],
            metrics["node_keep_accuracy"],
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
        "schema_version": "chemvector-v1p2d2e6-candidate-graph-capacity-result-0.1",
        "model_name": MODEL_NAME,
        "decision": (
            "candidate_graph_capacity_go"
            if all(checks.values())
            else "candidate_graph_capacity_no_go"
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
