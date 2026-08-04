#!/usr/bin/env python3
"""Train joint Atom decoding with loss-augmented fixed-cardinality matching."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import (
    train_g3_p3c_fixed_assignment_replay_train4937 as p3c,
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


MODEL_TYPE = "chemvector-g3-p3d-structured-matching-replay-train4937"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9"
)
ASSIGNMENT_CACHE = p3c.OUTPUT / "fixed_assignment_cache.json"
ASSIGNMENT_CACHE_FILE_SHA256 = (
    "b62323e567c4336b9084b0e2a807b3be5a15ffc4a5e49b0ce2fe0959bfe74b5e"
)
ASSIGNMENT_ENTRIES_SHA256 = (
    "cfcc96b1fa8a5c0e5d69e668ec7a13300d091ef94b9a22525f5e315e9ad5ad86"
)
EPOCHS = 3
GENERAL_PER_BATCH = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
STRUCTURED_MARGIN = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 301_206
EXPECTED_ELIGIBLE = 4_937
EXPECTED_CAPACITY = 64
EXPECTED_GENERAL = EXPECTED_ELIGIBLE - EXPECTED_CAPACITY
STEPS_PER_EPOCH = math.ceil(EXPECTED_GENERAL / GENERAL_PER_BATCH)
EXPECTED_STEPS = STEPS_PER_EPOCH * EPOCHS
EXPECTED_EXPOSURE = (EXPECTED_GENERAL + STEPS_PER_EPOCH) * EPOCHS


def loss_augmented_fixed_cardinality_assignment(
    edge_scores: torch.Tensor,
    count: int,
    target_queries: torch.Tensor,
    target_candidates: torch.Tensor,
    *,
    margin: float = STRUCTURED_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the loss-augmented MAP matching and normalized Hamming loss."""

    if edge_scores.ndim != 2 or not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("P3-D edge scores must be one finite matrix")
    queries, candidates = edge_scores.shape
    if not 0 < count <= min(queries, candidates):
        raise ValueError("P3-D matching count is outside capacity")
    if (
        target_queries.shape != (count,)
        or target_candidates.shape != (count,)
        or len(target_queries.unique()) != count
        or len(target_candidates.unique()) != count
    ):
        raise ValueError("P3-D target matching is not fixed-cardinality unique")
    if margin < 0.0:
        raise ValueError("P3-D structured margin must be nonnegative")

    augmentation = torch.full_like(edge_scores.detach(), margin / count)
    augmentation[target_queries, target_candidates] = 0.0
    augmented_scores = edge_scores.detach() + augmentation
    dummy_queries = candidates - count
    dummy_candidates = queries - count
    size = queries + dummy_queries
    cost = torch.zeros((size, size), dtype=torch.float64)
    cost[:queries, :candidates] = -augmented_scores.double().cpu()
    if dummy_queries and dummy_candidates:
        cost[queries:, candidates:] = 1_000_000_000.0
    rows, columns = linear_sum_assignment(cost.numpy())
    rows = torch.as_tensor(rows, dtype=torch.long, device=edge_scores.device)
    columns = torch.as_tensor(columns, dtype=torch.long, device=edge_scores.device)
    real = (rows < queries) & (columns < candidates)
    selected_queries = rows[real]
    selected_candidates = columns[real]
    if len(selected_queries) != count:
        raise RuntimeError("P3-D loss-augmented matching cardinality differs")
    target_edges = set(zip(target_queries.tolist(), target_candidates.tolist()))
    selected_edges = set(zip(selected_queries.tolist(), selected_candidates.tolist()))
    hamming = margin * len(selected_edges - target_edges) / count
    return selected_queries, selected_candidates, hamming


def structured_matching_loss(
    existence_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    fixed_assignments: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if pointer_logits.shape[:2] != existence_logits.shape:
        raise ValueError("P3-D existence/pointer batch shape differs")
    edge_scores = (
        F.logsigmoid(existence_logits).unsqueeze(-1)
        + pointer_logits.log_softmax(dim=-1)
    )
    losses = []
    exact = 0
    hamming_total = 0.0
    for index, assignment in enumerate(fixed_assignments):
        target_queries = assignment["query_indices"].to(edge_scores.device)
        target_candidates = assignment["candidate_indices"].to(edge_scores.device)
        count = len(target_queries)
        selected_queries, selected_candidates, hamming = (
            loss_augmented_fixed_cardinality_assignment(
                edge_scores[index], count, target_queries, target_candidates
            )
        )
        target_score = edge_scores[index, target_queries, target_candidates].sum()
        selected_score = edge_scores[
            index, selected_queries, selected_candidates
        ].sum()
        losses.append((selected_score + hamming - target_score).clamp_min(0.0))
        exact += int(
            set(zip(selected_queries.tolist(), selected_candidates.tolist()))
            == set(zip(target_queries.tolist(), target_candidates.tolist()))
        )
        hamming_total += hamming
    loss = torch.stack(losses).mean()
    return {
        "structured_matching_loss": loss,
        "loss_augmented_exact_images": loss.new_tensor(float(exact)),
        "loss_augmented_hamming": loss.new_tensor(hamming_total),
    }


def training_step(model, optimizer, records, assignments, *, device) -> dict:
    batch = p2h.collate(records, device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    fixed = [assignments[str(record["sample_key"])] for record in records]
    with _autocast_context(device, "bf16"):
        output = model(p2h.model_candidates(batch))
        structured = structured_matching_loss(
            output["existence_logits"], output["pointer_logits"], fixed
        )
        count_loss = F.cross_entropy(
            output["count_logits"], batch["full_atom_mask"].sum(dim=1).long()
        )
        loss = structured["structured_matching_loss"] + count_loss
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-D gradient norm is not finite")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "count_loss": float(count_loss.detach()),
        **{name: float(value.detach()) for name, value in structured.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-D output is not empty: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-D P2-W checkpoint SHA256 differs")
    if file_sha256(ASSIGNMENT_CACHE) != ASSIGNMENT_CACHE_FILE_SHA256:
        raise ValueError("P3-D fixed assignment cache file SHA256 differs")
    cache_payload = json.loads(ASSIGNMENT_CACHE.read_text(encoding="utf-8"))
    if cache_payload.get("entries_sha256") != ASSIGNMENT_ENTRIES_SHA256:
        raise ValueError("P3-D fixed assignment entries SHA256 differs")
    assignments = p3c.validate_assignment_cache(cache_payload)
    if len(assignments) != EXPECTED_ELIGIBLE:
        raise RuntimeError("P3-D fixed assignment cache size changed")

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
        raise RuntimeError("P3-D model parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    candidate_cache, candidate_audit = p2e.load_cache()
    train5k = candidate_cache["train5k"]
    route = candidate_cache["route100"]
    eligible = [
        record for record in train5k
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    eligible_keys = {str(record["sample_key"]) for record in eligible}
    if eligible_keys != set(assignments):
        raise RuntimeError("P3-D eligible samples differ from fixed assignment cache")
    capacity_records = [
        record for record in eligible
        if assignments[str(record["sample_key"])]["capacity_replay"]
    ]
    general_records = [
        record for record in eligible
        if not assignments[str(record["sample_key"])]["capacity_replay"]
    ]
    if len(capacity_records) != EXPECTED_CAPACITY or len(general_records) != EXPECTED_GENERAL:
        raise RuntimeError("P3-D replay strata size changed")
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
    _atomic_json(OUTPUT / "audit.json", {
        "schema_version": "g3-p3d-structured-matching-replay-train4937-audit-0.1",
        "scope": "loss-augmented fixed-cardinality matching with capacity replay",
        "edge_score": "logsigmoid(existence)+logsoftmax(pointer)",
        "structured_margin": STRUCTURED_MARGIN,
        "general_per_batch": GENERAL_PER_BATCH,
        "capacity_replay_per_batch": 1,
        "route100_used_for_training_or_selection": False,
        "fixed_gate": {
            "route100_count_exact_min": 27,
            "route100_known_whole_exact_min": 30,
            "route100_predicted_whole_exact_min": 11,
            "capacity64_predicted_exact": "32/32 in both domains",
        },
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "assignment_cache": str(ASSIGNMENT_CACHE.resolve()),
            "assignment_cache_file_sha256": ASSIGNMENT_CACHE_FILE_SHA256,
            "assignment_entries_sha256": ASSIGNMENT_ENTRIES_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": candidate_audit["candidate_protocol"][
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
        "structured_matching_replay_train4937_generalization_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "structured_matching_replay_train4937_generalization_no_go"
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
        "checkpoint_schema_version": "g3-p3d-structured-matching-replay-train4937-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": len(eligible),
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "structured_margin": STRUCTURED_MARGIN,
        "assignment_cache_sha256": ASSIGNMENT_ENTRIES_SHA256,
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
