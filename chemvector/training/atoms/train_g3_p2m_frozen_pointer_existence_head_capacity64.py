#!/usr/bin/env python3
"""Calibrate only the 65-parameter existence head on frozen P2-J queries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    g3_atom_existence_calibration_loss,
)
from chemvector.training.atoms import train_g3_p2e_stride1_top128_selector_train5k as p2e
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p2j_pointer_matched_query_set_capacity64 as p2j
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p2m-frozen-pointer-existence-head-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
P2J_CHECKPOINT = p2j.OUTPUT / "step-004800.pt"
P2J_CHECKPOINT_SHA256 = (
    "a4e5031ae4994b73c62a9ca97625c720aa06d2f816cf322b109fed20f5f2ad0a"
)
STEPS = 1_600
EXPECTED_EXPOSURE = 6_400
SAVE_EVERY = 400
TRAINABLE_PARAMETERS = 65
P2M_PROTOCOL = {
    "experiment": "P2-M",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2m-frozen-pointer-existence-head-0.1",
    "audit_schema_version": "g3-p2m-capacity64-audit-0.1",
    "scope": "frozen P2-J representation, existence-head capacity only",
    "existence_objective": "weighted-bce-plus-fixed-label-extreme-margin1",
    "boundary_weight": 1.0,
    "decision_prefix": "frozen_pointer_existence_head",
}


def build_parser(*, output: Path = OUTPUT, description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(args, *, output: Path = OUTPUT, experiment: str = "P2-M") -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{experiment} requires fixed {name}={expected}")
    if args.output_dir.resolve() != output.resolve():
        raise ValueError(f"{experiment} requires output directory {output}")
    if args.log_every < 1 or not torch.cuda.is_available():
        raise RuntimeError(f"{experiment} requires positive logging interval and CUDA BF16")


def validate_trainable_scope(model, optimizer) -> None:
    expected = {id(value) for value in model.existence_head.parameters()}
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if expected != actual:
        raise ValueError("P2-M optimizer must contain only the existence head")
    if sum(value.numel() for value in model.existence_head.parameters()) != TRAINABLE_PARAMETERS:
        raise ValueError("P2-M existence parameter count differs")
    for name, value in model.named_parameters():
        if value.requires_grad != name.startswith("existence_head."):
            raise ValueError(f"P2-M trainability differs: {name}")


def training_step(
    model, optimizer, batch, *, max_grad_norm: float, boundary_weight: float = 1.0
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = g3_atom_existence_calibration_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            boundary_weight=boundary_weight,
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.existence_head.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-M gradient norm is not finite")
    optimizer.step()
    if any(
        value.grad is not None
        for name, value in model.named_parameters()
        if not name.startswith("existence_head.")
    ):
        raise RuntimeError("P2-M produced gradients outside existence head")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def checkpoint_payload(
    model, optimizer, args, lineage, *, step, exposure, losses, metrics,
    protocol=P2M_PROTOCOL,
):
    return {
        "checkpoint_schema_version": protocol["checkpoint_schema_version"],
        "model_type": protocol["model_type"],
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "calibration_steps": step,
        "calibration_samples_seen": exposure,
        "training_args": {**vars(args), "output_dir": str(args.output_dir)},
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "frozen_parameter_count": model.parameter_count() - TRAINABLE_PARAMETERS,
        "query_matching": "frozen-pointer-mass-with-zero-existence-cost",
        "existence_objective": protocol["existence_objective"],
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def run_experiment(args, *, protocol) -> None:
    experiment = protocol["experiment"]
    validate_args(args, output=protocol["output"], experiment=experiment)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"{experiment} output is not empty: {args.output_dir}")
    if file_sha256(P2J_CHECKPOINT) != P2J_CHECKPOINT_SHA256:
        raise ValueError(f"{experiment} P2-J checkpoint SHA256 differs")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    source = torch.load(P2J_CHECKPOINT, map_location=device, weights_only=False)
    model = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    source_state_sha = module_state_sha256(model)
    model.requires_grad_(False)
    model.existence_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.existence_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validate_trainable_scope(model, optimizer)
    cache, cache_audit = p2e.load_cache()
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    lineage = {
        "p2j_checkpoint": str(P2J_CHECKPOINT.resolve()),
        "p2j_checkpoint_sha256": P2J_CHECKPOINT_SHA256,
        "p2j_model_state_sha256": source_state_sha,
        "p2j_metrics": source["metrics"],
        "candidate_cache_sha256": p2e.CACHE_SHA256,
        "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
            "candidate_cache_sha256"
        ],
        "selection_audit": selection_audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", {
        "schema_version": protocol["audit_schema_version"],
        "scope": protocol["scope"],
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "pointer_query_count_chemistry_frozen": True,
        "existence_removed_from_matching_cost": True,
        "lineage": lineage,
    })
    _atomic_json(
        args.output_dir / "training_args.json",
        {**vars(args), "output_dir": str(args.output_dir)},
    )
    step0 = {
        domain: p2h.evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    _atomic_json(args.output_dir / "step0_metrics.json", step0)

    step = exposure = epoch = 0
    losses = {}
    running = defaultdict(float)
    started = time.perf_counter()
    log_path = args.output_dir / "train.jsonl"
    while step < STEPS:
        epoch += 1
        for domain, records in p2h.epoch_batches(domains):
            model.eval()
            losses = training_step(
                model,
                optimizer,
                p2h.collate(records, device),
                max_grad_norm=args.max_grad_norm,
                boundary_weight=protocol["boundary_weight"],
            )
            step += 1
            exposure += len(records)
            for name, value in losses.items():
                running[name] += value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "exposure": exposure,
                    "domain": domain,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{name: value / args.log_every for name, value in running.items()},
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    model, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=losses, metrics=None,
                    protocol=protocol,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)

    metrics = {
        domain: p2h.evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    repeated = {
        domain: p2h.evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    decision, checks = p2h.gate(metrics)
    decision = decision.replace("atom_query_set", protocol["decision_prefix"])
    protocol_checks = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 100,
        "optimizer_existence_only": validate_trainable_scope(model, optimizer) is None,
        "repeat_metrics": repeated == metrics,
    }
    if not all(protocol_checks.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_metrics": step0,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        model, optimizer, args, lineage,
        step=step, exposure=exposure, losses=losses, metrics=final,
        protocol=protocol,
    )
    checkpoint = args.output_dir / "step-001600.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(args.output_dir / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(args.output_dir / "final_metrics.json", final)
    _atomic_json(args.output_dir / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({
        "decision": decision,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final["checkpoint_sha256"],
    }, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args, protocol=P2M_PROTOCOL)


if __name__ == "__main__":
    main()
