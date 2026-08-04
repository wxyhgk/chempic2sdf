#!/usr/bin/env python3
"""Capacity-test permutation-invariant Atom queries on 32 easy + 32 hard images."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    decode_g3_atom_query_set,
    g3_atom_query_set_loss,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p2e_stride1_top128_selector_train5k as p2e
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p2h-atom-query-set-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
STEPS = 4_800
SAVE_EVERY = 800
EXPECTED_EXPOSURE = 19_200
EXPECTED_VARIANTS = ("coordgen", "original", "rdkit", "stress")
EXPECTED_BASELINES = {
    "easy32": {"matched_atoms": 967, "target_atoms": 967, "exact_images": 32},
    "hard32": {"matched_atoms": 695, "target_atoms": 807, "exact_images": 0},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P2-H requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P2-H requires output directory {OUTPUT}")
    if args.log_every < 1:
        raise ValueError("P2-H log interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-H requires CUDA BF16")


def _variant(sample_key: str) -> str:
    variant = sample_key.rsplit("_", 1)[-1]
    if variant not in EXPECTED_VARIANTS:
        raise ValueError(f"P2-H sample has unknown variant: {sample_key}")
    return variant


def _molecule_key(sample_key: str) -> str:
    return sample_key.rsplit("_", 1)[0]


def baseline_statistics(record: dict) -> dict:
    full_mask = record["full_atom_mask"]
    if not torch.equal(record["atom_mask"], full_mask):
        raise ValueError("P2-H baseline record is not fully representable")
    expected = record["atom_anchors"][full_mask]
    count = len(expected)
    matched = p1a.maximum_cardinality_radius_match_count(
        record["candidate_anchors"][:count],
        expected,
        radius_pixels=6.0,
        image_size=768.0,
    )
    return {
        "matched_atoms": matched,
        "target_atoms": count,
        "exact": matched == count,
        "recall": matched / count,
    }


def select_capacity_domains(records: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    candidates = []
    for record in records:
        if not torch.equal(record["atom_mask"], record["full_atom_mask"]):
            continue
        stats = baseline_statistics(record)
        candidates.append((stats, record))
    domains = {}
    used_molecules = set()
    for name, exact in (("easy32", True), ("hard32", False)):
        selected = []
        variant_counts = defaultdict(int)
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item[0]["recall"] if exact else item[0]["recall"],
                item[1]["sample_key"],
            ),
        )
        for stats, record in ordered:
            key = str(record["sample_key"])
            variant = _variant(key)
            molecule = _molecule_key(key)
            if (
                stats["exact"] != exact
                or variant_counts[variant] >= 8
                or molecule in used_molecules
            ):
                continue
            selected.append(record)
            variant_counts[variant] += 1
            used_molecules.add(molecule)
            if len(selected) == 32:
                break
        if len(selected) != 32 or any(variant_counts[value] != 8 for value in EXPECTED_VARIANTS):
            raise RuntimeError(f"P2-H could not select balanced {name}")
        domains[name] = selected

    audit_domains = {}
    digest = hashlib.sha256()
    for name, selected in domains.items():
        totals = {"matched_atoms": 0, "target_atoms": 0, "exact_images": 0}
        keys = []
        for record in selected:
            stats = baseline_statistics(record)
            totals["matched_atoms"] += stats["matched_atoms"]
            totals["target_atoms"] += stats["target_atoms"]
            totals["exact_images"] += int(stats["exact"])
            key = str(record["sample_key"])
            keys.append(key)
            digest.update(f"{name}:{key}\n".encode())
        if totals != EXPECTED_BASELINES[name]:
            raise RuntimeError(f"P2-H {name} baseline changed: {totals}")
        audit_domains[name] = {
            **totals,
            "sample_keys": keys,
            "variant_counts": {
                variant: sum(_variant(key) == variant for key in keys)
                for variant in EXPECTED_VARIANTS
            },
        }
    return domains, {
        "selection": "32 exact easy + 32 lowest-recall hard, 8 per variant",
        "molecule_overlap": len(
            {_molecule_key(item["sample_key"]) for item in domains["easy32"]}
            & {_molecule_key(item["sample_key"]) for item in domains["hard32"]}
        ),
        "domains": audit_domains,
        "sample_keys_sha256": digest.hexdigest(),
    }


def collate(records: list[dict], device: torch.device) -> dict:
    batch = p2e.collate(records, device)
    batch["full_atom_mask"] = torch.stack(
        [record["full_atom_mask"] for record in records]
    ).to(device)
    return batch


def model_candidates(batch: dict) -> dict[str, torch.Tensor]:
    return {
        name: batch[name]
        for name in (
            "candidate_features",
            "candidate_anchors",
            "raw_scores",
            "normalized_scores",
            "scale_ranks",
        )
    }


def epoch_batches(domains: dict[str, list[dict]]):
    for start in range(0, 32, 4):
        yield "easy32", domains["easy32"][start : start + 4]
        yield "hard32", domains["hard32"][start : start + 4]


def training_step(model, optimizer, batch, *, max_grad_norm: float) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(model_candidates(batch))
        losses = g3_atom_query_set_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-H gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


@torch.no_grad()
def evaluate(model, records: list[dict], *, device: torch.device) -> dict:
    model.eval()
    known_totals = p1a._empty_atom_totals()
    predicted_totals = p1a._empty_atom_totals()
    count_predictions = hashlib.sha256()
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(model_candidates(batch))
        known_counts = batch["full_atom_mask"].sum(dim=1).long()
        known = decode_g3_atom_query_set(
            output, batch["candidate_anchors"], counts=known_counts
        )
        predicted = decode_g3_atom_query_set(output, batch["candidate_anchors"])
        predicted_counts = output["count_logits"].argmax(dim=-1)
        count_predictions.update(predicted_counts.cpu().numpy().tobytes())
        for index in range(len(current)):
            expected = batch["atom_anchors"][
                index, batch["full_atom_mask"][index]
            ].float().cpu()
            p1a._merge_atom_totals(
                known_totals,
                p1a.atom_sample_statistics(known[index]["anchors"].float().cpu(), expected),
            )
            p1a._merge_atom_totals(
                predicted_totals,
                p1a.atom_sample_statistics(
                    predicted[index]["anchors"].float().cpu(), expected
                ),
            )
    return {
        "known_count": p1a._finalize_atom_totals(known_totals),
        "predicted_count": p1a._finalize_atom_totals(predicted_totals),
        "count_predictions_sha256": count_predictions.hexdigest(),
    }


def gate(metrics: dict[str, dict]) -> tuple[str, dict[str, bool]]:
    checks = {}
    for domain in ("easy32", "hard32"):
        known = metrics[domain]["known_count"]
        predicted = metrics[domain]["predicted_count"]
        checks[f"{domain}_known_all_atoms"] = (
            known["matched_atoms"] == known["target_atoms"]
        )
        checks[f"{domain}_known_exact32"] = known["whole_set_exact_images"] == 32
        checks[f"{domain}_count_exact32"] = predicted["count_exact_images"] == 32
        checks[f"{domain}_predicted_exact32"] = (
            predicted["whole_set_exact_images"] == 32
        )
    return (
        "atom_query_set_capacity_go"
        if all(checks.values()) else "atom_query_set_capacity_no_go",
        checks,
    )


def checkpoint_payload(model, optimizer, args, lineage, *, step, exposure, losses, metrics):
    return {
        "checkpoint_schema_version": "g3-p2h-atom-query-set-capacity64-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": {**vars(args), "output_dir": str(args.output_dir)},
        "trainable_parameter_count": model.parameter_count(),
        "matching": "hungarian-query-to-gt-coordinate",
        "pointer_target": "set-valued-all-compatible-peaks-within-evaluation-radius",
        "candidate_preclustering": False,
        "known_count_is_model_input": False,
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-H output is not empty: {args.output_dir}")
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
    domains, selection_audit = select_capacity_domains(cache["train5k"])
    if selection_audit["molecule_overlap"]:
        raise RuntimeError("P2-H easy/hard domains overlap by molecule")
    model = G3AtomQuerySet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_sha = module_state_sha256(model)
    lineage = {
        "candidate_cache": str(p2e.CACHE.resolve()),
        "candidate_cache_sha256": p2e.CACHE_SHA256,
        "candidate_audit": str(p2e.AUDIT.resolve()),
        "candidate_audit_sha256": p2e.AUDIT_SHA256,
        "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
            "candidate_cache_sha256"
        ],
        "selection_audit": selection_audit,
        "initial_model_state_sha256": initial_sha,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        args.output_dir / "audit.json",
        {
            "schema_version": "g3-p2h-capacity64-audit-0.1",
            "scope": "same-image capacity, not generalization",
            "candidate_protocol": "frozen-p2a-stride1-top128",
            "fixed_pixel_clustering": False,
            "fixed_candidate_positive_identity": False,
            "selection": selection_audit,
            "lineage": lineage,
        },
    )
    _atomic_json(
        args.output_dir / "training_args.json",
        {**vars(args), "output_dir": str(args.output_dir)},
    )
    step0 = {
        domain: evaluate(model, records, device=device)
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
        for domain, records in epoch_batches(domains):
            model.train()
            losses = training_step(
                model,
                optimizer,
                collate(records, device),
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
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    model, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=losses, metrics=None,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)

    metrics = {
        domain: evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    repeated = {
        domain: evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    decision, checks = gate(metrics)
    protocol_checks = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 300,
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
    )
    checkpoint = args.output_dir / "step-004800.pt"
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


if __name__ == "__main__":
    main()
