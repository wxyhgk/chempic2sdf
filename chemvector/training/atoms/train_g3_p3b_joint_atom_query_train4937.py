#!/usr/bin/env python3
"""Jointly adapt the complete Atom-query module on 4,937 reachable train5k images."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
    g3_atom_query_set_loss,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3a_count_head_train5k as p3a
from chemvector.training.atoms import (
    audit_g3_p2x_joint_cardinality_decode_capacity64 as p2x,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3b-joint-atom-query-train4937"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p3a.OUTPUT / "step-001570.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "2eccba37a58c0b2cadf92be12153fa4697e5fbaa912778129a8571fd6ebed6d6"
)
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 301_206
EXPECTED_ELIGIBLE = 4_937
EXPECTED_STEPS = math.ceil(EXPECTED_ELIGIBLE / BATCH_SIZE) * EPOCHS
EXPECTED_EXPOSURE = EXPECTED_ELIGIBLE * EPOCHS


def training_step(model, optimizer, records, *, device: torch.device) -> dict:
    batch = p2h.collate(records, device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with _autocast_context(device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = g3_atom_query_set_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            unique_pointers=True,
            query_matching="pointer",
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-B gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-B output is not empty: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-B P3-A checkpoint SHA256 differs")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    source = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    base = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model = G3PointerUtilityExistenceModel(base).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    source_state_sha256 = module_state_sha256(model)
    model.requires_grad_(True)
    if model.parameter_count() != TRAINABLE_PARAMETERS:
        raise RuntimeError("P3-B model parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    cache, cache_audit = p2e.load_cache()
    train5k = cache["train5k"]
    route = cache["route100"]
    eligible = [
        record for record in train5k
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"P3-B eligible train count changed: {len(eligible)}")
    route_keys = {str(record["sample_key"]) for record in route}
    train_keys = {str(record["sample_key"]) for record in train5k}
    if route_keys & train_keys:
        raise RuntimeError("P3-B route100 overlaps train5k")
    domains, selection_audit = p2h.select_capacity_domains(train5k)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    step0_route = p2x.evaluate_joint(model, route, device=device)
    step0_capacity = {
        domain: p2x.evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    _atomic_json(OUTPUT / "step0_metrics.json", {
        "route100": step0_route,
        "capacity64": step0_capacity,
    })
    if step0_route["predicted_count"]["matched_atoms"] != 2_674:
        raise RuntimeError("P3-B P3-A route baseline changed")
    _atomic_json(OUTPUT / "audit.json", {
        "schema_version": "g3-p3b-joint-atom-query-train4937-audit-0.1",
        "scope": "joint Atom-query train5k generalization",
        "train5k_images": len(train5k),
        "eligible_training_images": len(eligible),
        "ineligible_images_retained_outside_pointer_training": len(train5k) - len(eligible),
        "route100_used_for_training_or_selection": False,
        "matching": "pointer-compatible-hungarian-with-unique-candidate-targets",
        "decoder": "fixed-cardinality-joint-map",
        "fixed_gate": {
            "route100_count_exact_min": 70,
            "route100_known_matched_atoms_min": 3000,
            "route100_known_whole_exact_min": 50,
            "route100_predicted_whole_exact_min": 40,
            "capacity64_predicted_exact": "32/32 in both domains",
        },
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
        },
    })

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step = exposure = 0
    started = time.perf_counter()
    log_path = OUTPUT / "train.jsonl"
    last = {}
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(eligible), generator=generator).tolist()
        sums = {}
        batches = 0
        for start in range(0, len(order), BATCH_SIZE):
            current = [eligible[index] for index in order[start : start + BATCH_SIZE]]
            last = training_step(model, optimizer, current, device=device)
            step += 1
            exposure += len(current)
            batches += 1
            for name, value in last.items():
                sums[name] = sums.get(name, 0.0) + value
        record = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "elapsed_seconds": time.perf_counter() - started,
            **{name: value / batches for name, value in sums.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)

    route_metrics = p2x.evaluate_joint(model, route, device=device)
    capacity_metrics = {
        domain: p2x.evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    repeated_route = p2x.evaluate_joint(model, route, device=device)
    repeated_capacity = {
        domain: p2x.evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    predicted = route_metrics["predicted_count"]
    known = route_metrics["known_count"]
    checks = {
        "route100_count_exact_min70": predicted["count_exact_images"] >= 70,
        "route100_known_matched_atoms_min3000": known["matched_atoms"] >= 3_000,
        "route100_known_whole_exact_min50": known["whole_set_exact_images"] >= 50,
        "route100_predicted_whole_exact_min40": (
            predicted["whole_set_exact_images"] >= 40
        ),
        "capacity_easy_exact32": (
            capacity_metrics["easy32"]["predicted_count"]["whole_set_exact_images"] == 32
        ),
        "capacity_hard_exact32": (
            capacity_metrics["hard32"]["predicted_count"]["whole_set_exact_images"] == 32
        ),
    }
    protocol_checks = {
        "fixed_steps": step == EXPECTED_STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "all_parameters_trainable": all(value.requires_grad for value in model.parameters()),
        "repeat_route_metrics": repeated_route == route_metrics,
        "repeat_capacity_metrics": repeated_capacity == capacity_metrics,
    }
    decision = (
        "joint_atom_query_train4937_generalization_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "joint_atom_query_train4937_generalization_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_route100": step0_route,
        "route100": route_metrics,
        "capacity64": capacity_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3b-joint-atom-query-train4937-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": len(eligible),
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "matching": "pointer-compatible-hungarian-unique-pointer",
        "decoder": "fixed-cardinality-joint-map",
        "last_losses": last,
        "metrics": final,
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
        },
    }
    checkpoint = OUTPUT / f"step-{step:06d}.pt"
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
        "route100": route_metrics,
        "capacity64": capacity_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final["checkpoint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
