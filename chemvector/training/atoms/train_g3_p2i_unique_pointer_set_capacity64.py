#!/usr/bin/env python3
"""Capacity-test Atom queries with train-time globally unique peak pointers."""

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
    g3_atom_query_set_loss,
)
from chemvector.training.atoms import train_g3_p2e_stride1_top128_selector_train5k as p2e
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p2i-unique-pointer-set-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
P2H_CHECKPOINT = p2h.OUTPUT / "step-004800.pt"
P2H_CHECKPOINT_SHA256 = (
    "168c43a8dadf5f21a8c1f9eae91ae52e3e6f3a6d892eb467014a2e14d542efe5"
)
EXPECTED_INITIAL_SHA256 = (
    "6566dd14c8d4054c1e919281955ab938c26c5e09ef96433da6a60946ffa85d79"
)
P2I_PROTOCOL = {
    "experiment": "P2-I",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2i-unique-pointer-set-capacity64-0.1",
    "audit_schema_version": "g3-p2i-capacity64-audit-0.1",
    "scope": "same-image capacity, P2-H controlled pointer-target ablation",
    "changed_variable": "global unique compatible pointer target",
    "matching": "hungarian-query-to-gt-then-global-unique-compatible-pointer",
    "pointer_target": "one-distinct-compatible-peak-per-matched-query",
    "decision_prefix": "unique_pointer_set",
}


def build_parser(*, output: Path = OUTPUT, description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(
    args: argparse.Namespace, *, output: Path = OUTPUT, experiment: str = "P2-I"
) -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{experiment} requires fixed {name}={expected}")
    if args.output_dir.resolve() != output.resolve():
        raise ValueError(f"{experiment} requires output directory {output}")
    if args.log_every < 1:
        raise ValueError(f"{experiment} log interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{experiment} requires CUDA BF16")


def training_step(model, optimizer, batch, *, max_grad_norm: float) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = g3_atom_query_set_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            unique_pointers=True,
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-I gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def checkpoint_payload(
    model, optimizer, args, lineage, *, step, exposure, losses, metrics,
    protocol=P2I_PROTOCOL,
):
    return {
        "checkpoint_schema_version": protocol["checkpoint_schema_version"],
        "model_type": protocol["model_type"],
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": {**vars(args), "output_dir": str(args.output_dir)},
        "trainable_parameter_count": model.parameter_count(),
        "matching": protocol["matching"],
        "pointer_target": protocol["pointer_target"],
        "existence_objective": protocol.get(
            "existence_objective", "weighted-object-bce"
        ),
        "candidate_preclustering": False,
        "known_count_is_model_input": False,
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def run_experiment(args, *, protocol, training_step_fn) -> None:
    output = protocol["output"]
    experiment = protocol["experiment"]
    validate_args(args, output=output, experiment=experiment)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"{experiment} output is not empty: {args.output_dir}")
    if file_sha256(P2H_CHECKPOINT) != P2H_CHECKPOINT_SHA256:
        raise ValueError(f"{experiment} P2-H reference checkpoint SHA256 differs")
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

    cache, cache_audit = p2e.load_cache()
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    model = G3AtomQuerySet().to(device)
    initial_sha = module_state_sha256(model)
    if initial_sha != EXPECTED_INITIAL_SHA256:
        raise RuntimeError(f"{experiment} no longer shares P2-H initialization")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    start_exposure = 0
    resume = None
    resume_checkpoint = protocol.get("resume_checkpoint")
    if resume_checkpoint is not None:
        if file_sha256(resume_checkpoint) != protocol["resume_checkpoint_sha256"]:
            raise ValueError(f"{experiment} resume checkpoint SHA256 differs")
        resume = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if resume.get("model_type") != protocol["resume_model_type"]:
            raise ValueError(f"{experiment} resume model type differs")
        if resume.get("global_step") != protocol["resume_step"]:
            raise ValueError(f"{experiment} resume step differs")
        model.load_state_dict(resume["atom_query_set"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["global_step"])
        start_exposure = int(resume["samples_seen"])
    lineage = {
        "p2h_checkpoint": str(P2H_CHECKPOINT.resolve()),
        "p2h_checkpoint_sha256": P2H_CHECKPOINT_SHA256,
        "same_initialization_as_p2h": True,
        "candidate_cache": str(p2e.CACHE.resolve()),
        "candidate_cache_sha256": p2e.CACHE_SHA256,
        "candidate_audit": str(p2e.AUDIT.resolve()),
        "candidate_audit_sha256": p2e.AUDIT_SHA256,
        "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
            "candidate_cache_sha256"
        ],
        "selection_audit": selection_audit,
        "initial_model_state_sha256": initial_sha,
        "resume": (
            {
                "checkpoint": str(resume_checkpoint.resolve()),
                "checkpoint_sha256": protocol["resume_checkpoint_sha256"],
                "model_type": protocol["resume_model_type"],
                "step": start_step,
                "samples_seen": start_exposure,
                "optimizer_restored": True,
            }
            if resume_checkpoint is not None else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        args.output_dir / "audit.json",
        {
            "schema_version": protocol["audit_schema_version"],
            "scope": protocol["scope"],
            "changed_variable": protocol["changed_variable"],
            "same_architecture_data_initialization_budget_as_p2h": protocol.get(
                "same_architecture_data_initialization_budget_as_p2h", True
            ),
            "fixed_pixel_clustering": False,
            "selection": selection_audit,
            "lineage": lineage,
        },
    )
    _atomic_json(
        args.output_dir / "training_args.json",
        {**vars(args), "output_dir": str(args.output_dir)},
    )
    step0 = {
        domain: p2h.evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    _atomic_json(args.output_dir / "step0_metrics.json", step0)

    target_steps = int(protocol.get("target_steps", p2h.STEPS))
    expected_exposure = target_steps * args.batch_size
    step = start_step
    exposure = start_exposure
    epoch = step // 16
    losses = resume.get("last_losses", {}) if resume is not None else {}
    running = defaultdict(float)
    started = time.perf_counter()
    log_path = args.output_dir / "train.jsonl"
    while step < target_steps:
        epoch += 1
        for domain, records in p2h.epoch_batches(domains):
            model.train()
            losses = training_step_fn(
                model,
                optimizer,
                p2h.collate(records, device),
                max_grad_norm=args.max_grad_norm,
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
            if step % p2h.SAVE_EVERY == 0 and step < target_steps:
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
        "fixed_steps": step == target_steps,
        "fixed_exposure": exposure == expected_exposure,
        "fixed_epochs": epoch == target_steps // 16,
        "same_initialization_as_p2h": initial_sha == EXPECTED_INITIAL_SHA256,
        "resume_optimizer_restored": resume_checkpoint is None or resume is not None,
        "model_changed": module_state_sha256(model) != initial_sha,
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
    checkpoint = args.output_dir / f"step-{target_steps:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(args.output_dir / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(args.output_dir / "final_metrics.json", final)
    _atomic_json(
        args.output_dir / "status.json",
        {
            "status": "completed",
            "decision": decision,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": final["checkpoint_sha256"],
        },
    )
    print(json.dumps({
        "decision": decision,
        "parameter_count": model.parameter_count(),
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final["checkpoint_sha256"],
    }, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args, protocol=P2I_PROTOCOL, training_step_fn=training_step)


if __name__ == "__main__":
    main()
