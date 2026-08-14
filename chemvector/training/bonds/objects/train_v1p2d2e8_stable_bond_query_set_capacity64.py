#!/usr/bin/env python3
"""Stable multi-graph capacity gate for latent-max Hungarian Bond queries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

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
    DEFAULT_CACHE,
    load_capacity_cache,
)
from chemvector.training.bonds.objects.train_v1p2d2e7_bond_query_set_capacity64 import (
    EXISTENCE_THRESHOLD,
    GO_GATES,
    _graph_to_device,
    capacity_checks,
    evaluate_capacity,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


MODEL_NAME = "chemvector-v1p2d2e8-stable-bond-query-set-capacity64"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
SEED = 20260828
DEFAULT_ACCUMULATION_STEPS = 8


def _save_checkpoint(
    path: Path,
    model: SemanticBondCandidateQuerySet,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    microsteps: int,
    metrics: dict,
    accumulation_steps: int,
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": "chemvector-v1p2d2e8-stable-bond-query-set-checkpoint-0.1",
            "model_name": MODEL_NAME,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_update": update,
            "microsteps": microsteps,
            "accumulation_steps": accumulation_steps,
            "capacity_metrics": metrics,
            "pointer_group_objective": "latent_max_log_probability",
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
    parser.add_argument("--optimizer-updates", type=int, default=2_400)
    parser.add_argument(
        "--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E8 stable query-set training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E8 output is not empty: {args.output_dir}")
    if min(
        args.optimizer_updates,
        args.accumulation_steps,
        args.evaluate_every,
        args.log_every,
    ) < 1:
        raise ValueError("P2-D2-E8 training intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graphs = load_capacity_cache(args.cache)["graphs"]
    config = SemanticBondCandidateQuerySetConfig()
    maximum_bonds = max(int(graph["bond_count"]) for graph in graphs)
    if maximum_bonds > config.query_count:
        raise ValueError("P2-D2-E8 target exceeds Bond query count")
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
    total_loss = 0.0
    microsteps = 0
    history = []
    best_score = None
    best_update = 0
    consecutive_exact = 0
    completed_updates = 0
    for update in range(1, args.optimizer_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for _ in range(args.accumulation_steps):
            if cursor == len(order):
                order = torch.randperm(len(graphs), generator=generator).tolist()
                cursor = 0
            graph = _graph_to_device(graphs[order[cursor]], device)
            cursor += 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    graph["node_features"], graph["edge_features"], graph["pairs"]
                )
                losses = semantic_bond_candidate_query_set_loss(
                    output,
                    graph["edge_target_bond_ids"],
                    bond_count=int(graph["bond_count"]),
                )
                normalized_loss = losses["loss"] / args.accumulation_steps
            if not torch.isfinite(normalized_loss):
                raise FloatingPointError(
                    f"P2-D2-E8 non-finite loss at update {update}"
                )
            normalized_loss.backward()
            value = float(losses["loss"].detach())
            update_loss += value
            total_loss += value
            microsteps += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_updates = update
        if update % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "optimizer_update": update,
                        "microsteps": microsteps,
                        "mean_loss": total_loss / microsteps,
                        "update_loss": update_loss / args.accumulation_steps,
                    }
                ),
                flush=True,
            )
        if update % args.evaluate_every != 0 and update != args.optimizer_updates:
            continue
        metrics = evaluate_capacity(model, graphs, device)
        score = (
            metrics["predicted_image_exact"],
            metrics["known_count_image_exact"],
            metrics["predicted_count_accuracy"],
            metrics["predicted_pair_f1"],
        )
        record = {
            "optimizer_update": update,
            "microsteps": microsteps,
            "mean_train_loss": total_loss / microsteps,
            "metrics": metrics,
        }
        history.append(record)
        if best_score is None or score > best_score:
            best_score = score
            best_update = update
            _save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                update=update,
                microsteps=microsteps,
                metrics=metrics,
                accumulation_steps=args.accumulation_steps,
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
        update=best_update,
        microsteps=int(best["microsteps"]),
        metrics=metrics,
        accumulation_steps=args.accumulation_steps,
    )
    result = {
        "schema_version": "chemvector-v1p2d2e8-stable-bond-query-set-result-0.1",
        "model_name": MODEL_NAME,
        "decision": (
            "stable_bond_query_set_capacity_go"
            if all(checks.values())
            else "stable_bond_query_set_capacity_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pointer_group_objective": "latent_max_log_probability",
        "capacity_cache_sha256": CACHE_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "endpoint_result_sha256": ENDPOINT_RESULT_SHA256,
        "style_manifest_sha256": MANIFEST_SHA256,
        "maximum_capacity_bonds": maximum_bonds,
        "existence_threshold": EXISTENCE_THRESHOLD,
        "accumulation_steps": args.accumulation_steps,
        "requested_optimizer_updates": args.optimizer_updates,
        "completed_optimizer_updates": completed_updates,
        "microsteps": microsteps,
        "best_optimizer_update": best_update,
        "mean_train_loss": total_loss / max(microsteps, 1),
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
