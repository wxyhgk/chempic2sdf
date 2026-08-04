#!/usr/bin/env python3
"""Train Atom queries from a fixed assignment cache with capacity64 replay."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
    hungarian_pointer_query_targets,
    unique_pointer_targets,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2r_p2j_identity_existence_capacity64 as p2r,
)
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
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


MODEL_TYPE = "chemvector-g3-p3c-fixed-assignment-replay-train4937"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9"
)
P2R_CACHE = p2r.OUTPUT / "p2j_identity_action_cache.json"
P2R_CACHE_SHA256 = "d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca"
EPOCHS = 3
GENERAL_PER_BATCH = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 301_206
EXPECTED_ELIGIBLE = 4_937
EXPECTED_CAPACITY = 64
EXPECTED_GENERAL = EXPECTED_ELIGIBLE - EXPECTED_CAPACITY
STEPS_PER_EPOCH = math.ceil(EXPECTED_GENERAL / GENERAL_PER_BATCH)
EXPECTED_STEPS = STEPS_PER_EPOCH * EPOCHS
EXPECTED_EXPOSURE = (EXPECTED_GENERAL + STEPS_PER_EPOCH) * EPOCHS


def assignment_entries_digest(entries: list[dict]) -> str:
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_assignment_cache(payload: dict) -> dict[str, dict[str, torch.Tensor]]:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P3-C assignment cache has no entries")
    if payload.get("entries_sha256") != assignment_entries_digest(entries):
        raise ValueError("P3-C assignment cache digest differs")
    lookup = {}
    total_assignments = 0
    for entry in entries:
        key = entry.get("sample_key")
        queries = entry.get("query_indices")
        targets = entry.get("target_indices")
        candidates = entry.get("candidate_indices")
        if not isinstance(key, str) or not key or key in lookup:
            raise ValueError("P3-C assignment cache sample key is invalid")
        if not all(isinstance(value, list) for value in (queries, targets, candidates)):
            raise ValueError("P3-C assignment cache indices are invalid")
        if not len(queries) == len(targets) == len(candidates) == entry.get("atoms"):
            raise ValueError("P3-C assignment cache cardinality differs")
        if len(set(queries)) != len(queries) or len(set(candidates)) != len(candidates):
            raise ValueError("P3-C assignment cache uniqueness differs")
        if any(not 0 <= value < 128 for value in queries + candidates):
            raise ValueError("P3-C query/candidate index is outside capacity")
        lookup[key] = {
            "query_indices": torch.tensor(queries, dtype=torch.long),
            "target_indices": torch.tensor(targets, dtype=torch.long),
            "candidate_indices": torch.tensor(candidates, dtype=torch.long),
            "capacity_replay": bool(entry.get("capacity_replay")),
        }
        total_assignments += len(queries)
    if payload.get("assignments") != total_assignments:
        raise ValueError("P3-C total assignment count differs")
    return lookup


@torch.no_grad()
def generate_assignment_entries(model, records, capacity_masks, *, device) -> list[dict]:
    model.eval()
    entries = []
    radius = 6.0 / 768.0
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
        for index, record in enumerate(current):
            key = str(record["sample_key"])
            single_output = {
                name: value[index : index + 1]
                for name, value in output.items()
                if isinstance(value, torch.Tensor)
            }
            capacity_replay = key in capacity_masks
            if capacity_replay:
                fixed_queries = torch.where(capacity_masks[key].to(device))[0]
                restricted = {
                    **single_output,
                    "pointer_logits": single_output["pointer_logits"][:, fixed_queries],
                    "existence_logits": single_output["existence_logits"][:, fixed_queries],
                }
                local_queries, target_indices = hungarian_pointer_query_targets(
                    restricted,
                    batch["candidate_anchors"][index : index + 1],
                    batch["atom_anchors"][index : index + 1],
                    batch["full_atom_mask"][index : index + 1],
                    existence_cost_weight=0.0,
                )[0]
                query_indices = fixed_queries[local_queries]
            else:
                query_indices, target_indices = hungarian_pointer_query_targets(
                    single_output,
                    batch["candidate_anchors"][index : index + 1],
                    batch["atom_anchors"][index : index + 1],
                    batch["full_atom_mask"][index : index + 1],
                    existence_cost_weight=1.0,
                )[0]
            candidates = unique_pointer_targets(
                single_output["pointer_logits"],
                batch["candidate_anchors"][index : index + 1],
                batch["atom_anchors"][index : index + 1],
                [(query_indices, target_indices)],
            )[0]
            distances = torch.cdist(
                batch["atom_anchors"][index, target_indices].float(),
                batch["candidate_anchors"][index].float(),
            )[torch.arange(len(candidates), device=device), candidates]
            if bool((distances > radius).any()):
                raise RuntimeError(
                    "P3-C fixed candidate is incompatible with target: "
                    f"{key}, max={float(distances.max())}, "
                    f"queries={query_indices.cpu().tolist()}, "
                    f"targets={target_indices.cpu().tolist()}, "
                    f"candidates={candidates.cpu().tolist()}"
                )
            entries.append({
                "sample_key": key,
                "atoms": len(query_indices),
                "capacity_replay": capacity_replay,
                "query_indices": query_indices.cpu().tolist(),
                "target_indices": target_indices.cpu().tolist(),
                "candidate_indices": candidates.cpu().tolist(),
            })
    return entries


def fixed_assignment_loss(model, records, assignments, *, device) -> dict:
    batch = p2h.collate(records, device)
    output = model(p2h.model_candidates(batch))
    existence = output["existence_logits"]
    object_targets = torch.zeros_like(existence)
    coordinate_losses = []
    pointer_losses = []
    matched = 0
    for index, record in enumerate(records):
        assignment = assignments[str(record["sample_key"])]
        queries = assignment["query_indices"].to(device)
        targets = assignment["target_indices"].to(device)
        candidates = assignment["candidate_indices"].to(device)
        object_targets[index, queries] = 1.0
        coordinate_losses.append(F.l1_loss(
            output["coordinates"][index, queries], batch["atom_anchors"][index, targets]
        ))
        pointer_losses.append(F.cross_entropy(
            output["pointer_logits"][index, queries], candidates
        ))
        matched += len(queries)
    weights = torch.where(
        object_targets.bool(), torch.ones_like(object_targets),
        torch.full_like(object_targets, 0.1),
    )
    object_loss = F.binary_cross_entropy_with_logits(
        existence, object_targets, weight=weights, reduction="sum"
    ) / weights.sum().clamp_min(1.0)
    coordinate_loss = torch.stack(coordinate_losses).mean()
    pointer_loss = torch.stack(pointer_losses).mean()
    count_loss = F.cross_entropy(
        output["count_logits"], batch["full_atom_mask"].sum(dim=1).long()
    )
    loss = object_loss + 5.0 * coordinate_loss + pointer_loss + count_loss
    return {
        "loss": loss,
        "object_loss": object_loss,
        "coordinate_loss": coordinate_loss,
        "pointer_loss": pointer_loss,
        "count_loss": count_loss,
        "matched_queries": loss.new_tensor(float(matched)),
    }


def training_step(model, optimizer, records, assignments, *, device) -> dict:
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with _autocast_context(device, "bf16"):
        losses = fixed_assignment_loss(model, records, assignments, device=device)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-C gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-C output is not empty: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-C P2-W checkpoint SHA256 differs")
    if file_sha256(P2R_CACHE) != P2R_CACHE_SHA256:
        raise ValueError("P3-C P2-R Action Cache SHA256 differs")
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
        raise RuntimeError("P3-C model parameter count changed")
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
        raise RuntimeError("P3-C eligible train count changed")
    p2r_payload = json.loads(P2R_CACHE.read_text(encoding="utf-8"))
    capacity_masks = p2r.validate_action_cache(p2r_payload)
    if len(capacity_masks) != EXPECTED_CAPACITY:
        raise RuntimeError("P3-C capacity replay size changed")
    eligible_keys = {str(record["sample_key"]) for record in eligible}
    if not set(capacity_masks) <= eligible_keys:
        raise RuntimeError("P3-C capacity replay is outside eligible train")
    capacity_records = [
        record for record in eligible if str(record["sample_key"]) in capacity_masks
    ]
    general_records = [
        record for record in eligible if str(record["sample_key"]) not in capacity_masks
    ]
    if len(general_records) != EXPECTED_GENERAL:
        raise RuntimeError("P3-C general train size changed")
    domains, selection_audit = p2h.select_capacity_domains(train5k)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    first_entries = generate_assignment_entries(
        model, eligible, capacity_masks, device=device
    )
    repeated_entries = generate_assignment_entries(
        model, eligible, capacity_masks, device=device
    )
    if repeated_entries != first_entries:
        raise RuntimeError("P3-C assignment cache generation is not deterministic")
    cache_payload = {
        "schema_version": "g3-p3c-fixed-assignment-cache-0.1",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "p2r_capacity_cache_sha256": P2R_CACHE_SHA256,
        "samples": len(first_entries),
        "capacity_replay_samples": sum(item["capacity_replay"] for item in first_entries),
        "assignments": sum(item["atoms"] for item in first_entries),
        "entries_sha256": assignment_entries_digest(first_entries),
        "repeat_generation_equal": True,
        "entries": first_entries,
    }
    assignments = validate_assignment_cache(cache_payload)
    cache_path = OUTPUT / "fixed_assignment_cache.json"
    _atomic_json(cache_path, cache_payload)

    step0_route = p2x.evaluate_joint(model, route, device=device)
    step0_capacity = {
        domain: p2x.evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    _atomic_json(OUTPUT / "step0_metrics.json", {
        "route100": step0_route,
        "capacity64": step0_capacity,
    })
    _atomic_json(OUTPUT / "audit.json", {
        "schema_version": "g3-p3c-fixed-assignment-replay-train4937-audit-0.1",
        "scope": "fixed assignment general training with capacity64 replay",
        "general_per_batch": GENERAL_PER_BATCH,
        "capacity_replay_per_batch": 1,
        "route100_used_for_training_or_selection": False,
        "fixed_gate": {
            "route100_count_exact_min": 27,
            "route100_known_whole_exact_min": 30,
            "route100_predicted_whole_exact_min": 11,
            "capacity64_predicted_exact": "32/32 in both domains",
        },
        "assignment_cache": {
            "path": str(cache_path.resolve()),
            "entries_sha256": cache_payload["entries_sha256"],
            "samples": cache_payload["samples"],
            "assignments": cache_payload["assignments"],
        },
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "p2r_capacity_cache_sha256": P2R_CACHE_SHA256,
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
        general_order = torch.randperm(
            len(general_records), generator=generator
        ).tolist()
        capacity_order = []
        while len(capacity_order) < STEPS_PER_EPOCH:
            capacity_order.extend(torch.randperm(
                len(capacity_records), generator=generator
            ).tolist())
        sums = {}
        for batch_index, start in enumerate(
            range(0, len(general_order), GENERAL_PER_BATCH)
        ):
            records = [
                general_records[index]
                for index in general_order[start : start + GENERAL_PER_BATCH]
            ]
            records.append(capacity_records[capacity_order[batch_index]])
            last = training_step(model, optimizer, records, assignments, device=device)
            step += 1
            exposure += len(records)
            for name, value in last.items():
                sums[name] = sums.get(name, 0.0) + value
        record = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "elapsed_seconds": time.perf_counter() - started,
            **{name: value / STEPS_PER_EPOCH for name, value in sums.items()},
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
        "route100_count_exact_min27": predicted["count_exact_images"] >= 27,
        "route100_known_whole_exact_min30": known["whole_set_exact_images"] >= 30,
        "route100_predicted_whole_exact_min11": predicted["whole_set_exact_images"] >= 11,
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
        "repeat_route_metrics": repeated_route == route_metrics,
        "repeat_capacity_metrics": repeated_capacity == capacity_metrics,
        "fixed_assignment_cache": len(assignments) == EXPECTED_ELIGIBLE,
    }
    decision = (
        "fixed_assignment_replay_train4937_generalization_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "fixed_assignment_replay_train4937_generalization_no_go"
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
        "checkpoint_schema_version": "g3-p3c-fixed-assignment-replay-train4937-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": len(eligible),
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "assignment_cache_sha256": cache_payload["entries_sha256"],
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
