#!/usr/bin/env python3
"""Train the P2-E set selector on the frozen stride-1 top128 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.g3_multiscale_peak_set_ranker import (
    G3MultiscalePeakSetRanker,
    select_known_count_multiscale,
)
from chemvector.evaluation.audits.atoms import audit_g3_p2d_stride1_top128_cache as p2d
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1x_multiscale_peak_selector_capacity as p1x
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p2e-stride1-top128-selector-train5k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = p2d.DEFAULT_OUTPUT / "candidate-cache.pt"
CACHE_SHA256 = "ba0495dec9f197927f79ca254723cfe7a519cce672725702b37aa330e96fbc54"
AUDIT = p2d.DEFAULT_OUTPUT / "audit.json"
AUDIT_SHA256 = "8dd7ab70eecd83ff4972bd0ff2f3ed6c77ebb43b18d9a934aa689afbf9b5e2dd"
STEPS = 6_250
SAVE_EVERY = 1_250
EXPECTED_EXPOSURE = 25_000
STEP0_ROUTE_MATCHED = 3_022
STEP0_ROUTE_EXACT = 63


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
            raise ValueError(f"P2-E requires fixed {name}={expected}")
    if args.log_every < 1:
        raise ValueError("P2-E log interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-E requires CUDA BF16")


def load_cache() -> tuple[dict[str, list[dict]], dict]:
    if file_sha256(CACHE) != CACHE_SHA256 or file_sha256(AUDIT) != AUDIT_SHA256:
        raise ValueError("P2-E cache or audit SHA256 differs")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("decision") != "stride1_top128_cache_passed":
        raise ValueError("P2-E cache audit did not pass")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if set(cache) != {"train5k", "route100"}:
        raise ValueError("P2-E cache domains differ")
    if len(cache["train5k"]) != 5_000 or len(cache["route100"]) != 100:
        raise ValueError("P2-E cache sizes differ")
    return cache, audit


def cached_batches(records: list[dict], batch_size: int = 4):
    if len(records) % batch_size:
        raise ValueError("P2-E cache contains a residual batch")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def collate(records: list[dict], device: torch.device) -> dict:
    batch = p1x.collate_cached(records, device)
    batch["full_atom_mask"] = torch.stack(
        [record["full_atom_mask"] for record in records]
    ).to(device)
    return batch


@torch.no_grad()
def evaluate(ranker, records: list[dict], *, device: torch.device) -> dict:
    ranker.eval()
    full_totals = p1a._empty_atom_totals()
    representable_totals = p1a._empty_atom_totals()
    selection_digest = hashlib.sha256()
    for records_batch in cached_batches(records):
        batch = collate(records_batch, device)
        candidates = {
            name: batch[name]
            for name in (
                "candidate_ids",
                "candidate_features",
                "candidate_anchors",
                "raw_scores",
                "normalized_scores",
                "scale_ids",
                "scale_ranks",
            )
        }
        with _autocast_context(device, "bf16"):
            ranked = ranker(candidates)
        for mode, mask, totals in (
            ("full", batch["full_atom_mask"], full_totals),
            ("representable", batch["atom_mask"], representable_totals),
        ):
            counts = mask.sum(dim=1).long()
            selected = select_known_count_multiscale(ranked, counts)["selected_mask"]
            for index in range(len(records_batch)):
                predicted = batch["candidate_anchors"][index, selected[index]].float().cpu()
                expected = batch["atom_anchors"][index, mask[index]].float().cpu()
                p1a._merge_atom_totals(
                    totals, p1a.atom_sample_statistics(predicted, expected)
                )
                if mode == "full":
                    selection_digest.update(
                        batch["candidate_ids"][index, selected[index]].cpu().numpy().tobytes()
                    )

    def finalize(totals):
        metrics = p1a._finalize_atom_totals(totals)
        metrics["exact_rate_6px"] = metrics["whole_set_exact_images"] / len(records)
        return metrics

    return {
        "full_known_count": finalize(full_totals),
        "representable_known_count": finalize(representable_totals),
        "full_selection_ids_sha256": selection_digest.hexdigest(),
    }


def gate(step0: dict, final: dict) -> tuple[str, dict[str, bool]]:
    old = step0["full_known_count"]
    new = final["full_known_count"]
    checks = {
        "step0_matched_matches_frozen_field": old["matched_atoms"] == STEP0_ROUTE_MATCHED,
        "step0_exact_matches_frozen_field": old["whole_set_exact_images"] == STEP0_ROUTE_EXACT,
        "route_recall_gain_0_5pp": new["recall_6px"] >= old["recall_6px"] + 0.005,
        "route_exact_gain_10_images": new["whole_set_exact_images"]
        >= old["whole_set_exact_images"] + 10,
        "capability_recall_99": new["recall_6px"] >= 0.99,
        "capability_exact_85": new["whole_set_exact_images"] >= 85,
    }
    causal = all(checks[name] for name in (
        "step0_matched_matches_frozen_field",
        "step0_exact_matches_frozen_field",
        "route_recall_gain_0_5pp",
        "route_exact_gain_10_images",
    ))
    capability = checks["capability_recall_99"] and checks["capability_exact_85"]
    return (
        "route_causal_go_capability_go"
        if causal and capability
        else "route_causal_go_capability_no_go"
        if causal
        else "route_causal_no_go",
        checks,
    )


def checkpoint_payload(ranker, optimizer, args, lineage, *, step, exposure, losses, metrics):
    return {
        "checkpoint_schema_version": "g3-p2e-stride1-top128-selector-0.1",
        "model_type": MODEL_TYPE,
        "set_ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": vars(args),
        "trainable_parameter_count": ranker.parameter_count(),
        "candidate_protocol": "train5k-stride1-regional-top128",
        "matching": "maximum-cardinality-score-ranked-one-to-one-on-representable-gt",
        "full_gt_retained_for_evaluation": True,
        "known_count_is_model_input": False,
        "lineage": lineage,
        "last_losses": losses,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P2-E output is not empty: {OUTPUT}")
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
    cache, cache_audit = load_cache()
    ranker = G3MultiscalePeakSetRanker().to(device)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    p1x.validate_optimizer(ranker, optimizer)
    initial_ranker_sha = module_state_sha256(ranker)
    lineage = {
        "candidate_cache": str(CACHE.resolve()),
        "candidate_cache_sha256": CACHE_SHA256,
        "candidate_audit": str(AUDIT.resolve()),
        "candidate_audit_sha256": AUDIT_SHA256,
        "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"]["candidate_cache_sha256"],
        "stride1_checkpoint": cache_audit["lineage"]["checkpoint"],
        "stride1_checkpoint_sha256": cache_audit["lineage"]["checkpoint_sha256"],
        "initial_ranker_state_sha256": initial_ranker_sha,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "training_args.json", vars(args))
    step0 = evaluate(ranker, cache["route100"], device=device)
    _atomic_json(OUTPUT / "step0_route_metrics.json", step0)

    step = exposure = 0
    losses = {}
    started = time.perf_counter()
    log_path = OUTPUT / "train.jsonl"
    epoch_key_sha = None
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        for records_batch in cached_batches(cache["train5k"]):
            for record in records_batch:
                key_digest.update(f"{record['sample_key']}\n".encode())
            batch = collate(records_batch, device)
            ranker.train()
            losses = p1x.training_step(
                ranker, optimizer, batch, max_grad_norm=args.max_grad_norm
            )
            step += 1
            exposure += len(records_batch)
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "exposure": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **losses,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    ranker, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=losses, metrics=None,
                )
                _atomic_torch_save(OUTPUT / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(OUTPUT / "last.pt", payload)
        current_sha = key_digest.hexdigest()
        if epoch_key_sha is None:
            epoch_key_sha = current_sha
        elif current_sha != epoch_key_sha:
            raise RuntimeError("P2-E train cache order changed")

    final_train = evaluate(ranker, cache["train5k"], device=device)
    final_route = evaluate(ranker, cache["route100"], device=device)
    repeated_route = evaluate(ranker, cache["route100"], device=device)
    decision, checks = gate(step0, final_route)
    protocol_checks = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 5,
        "fixed_cache_order": epoch_key_sha is not None,
        "ranker_changed": module_state_sha256(ranker) != initial_ranker_sha,
        "optimizer_ranker_only": p1x.validate_optimizer(ranker, optimizer) is None,
        "repeat_route_metrics": repeated_route == final_route,
    }
    if not all(protocol_checks.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_route_metrics": step0,
        "final_train_metrics": final_train,
        "final_route_metrics": final_route,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        ranker, optimizer, args, lineage,
        step=step, exposure=exposure, losses=losses, metrics=final,
    )
    checkpoint = OUTPUT / "step-006250.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(OUTPUT / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(OUTPUT / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({
        "decision": decision,
        "step0_full": step0["full_known_count"],
        "final_full": final_route["full_known_count"],
        "final_representable": final_route["representable_known_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
