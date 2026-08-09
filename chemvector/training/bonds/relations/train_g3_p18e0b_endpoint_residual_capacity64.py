#!/usr/bin/env python3
"""Capacity-test a proposal-level endpoint residual on frozen Bond top192."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.bonds.objects.proposal_endpoint_residual import (
    ProposalEndpointResidualConfig,
    ProposalEndpointResidualHead,
    proposal_endpoint_residual_loss,
)
from chemvector.components.graph.joint.predicted_incidence import (
    decode_predicted_bond_proposals,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as bond_source,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0a_stress_bond_ceiling as p18e0a,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p4e3_global_peak_rank_pilot20k as p4e3
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p18e0b-bond-endpoint-residual-capacity64"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = OUTPUT / "capacity-cache.pt"
PILOT_MANIFEST = p4e3.P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = p4e3.PILOT_MANIFEST_SHA256
E0A_AUDIT = p18e0a.OUTPUT / "audit.json"
E0A_AUDIT_SHA256 = "748bac73d3323a2f100a2c9428561c86dd9d5546b7242dff7622a4a41dd19446"
CAPACITY_IMAGES = 64
EASY_IMAGES = 32
HARD_IMAGES = 32
MIN_CROSSED_HARD = 8
STEPS = 3_000
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
SEED = 20260810
CONFIG = ProposalEndpointResidualConfig()


def _target_bonds(record: dict) -> torch.Tensor:
    atoms = record["target_atom_anchors"][record["target_atom_mask"]].float()
    refs = record["target_bond_atom_refs"][record["target_bond_mask"]].long()
    return atoms[refs]


def build_residual_targets(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    target_bonds = _target_bonds(record)
    valid = record["bond_candidate_mask"]
    predicted = record["bond_candidate_endpoints"][valid].float()
    assignment, reversed_assignment = p18e0._maximum_bond_assignment(
        predicted, target_bonds, 24.0
    )
    if int((assignment >= 0).sum()) != len(target_bonds):
        raise ValueError("P18-E0B capacity record is not fully representable at 24px")
    valid_indices = torch.where(valid)[0]
    targets = record["bond_candidate_endpoints"].float().clone()
    supervision = torch.zeros_like(valid)
    for target_index, local_prediction in enumerate(assignment.tolist()):
        proposal_index = int(valid_indices[local_prediction])
        expected = target_bonds[target_index]
        if bool(reversed_assignment[target_index]):
            expected = expected.flip(0)
        targets[proposal_index] = expected
        supervision[proposal_index] = True
    return targets, supervision


def _record_metrics(record: dict, endpoints: torch.Tensor) -> dict:
    target_bonds = _target_bonds(record)
    valid = record["bond_candidate_mask"]
    assignment, _ = p18e0._maximum_bond_assignment(
        endpoints[valid].float(), target_bonds, 12.0
    )
    matched = int((assignment >= 0).sum())
    return {
        "targets": len(target_bonds),
        "matched": matched,
        "exact": matched == len(target_bonds),
    }


def _selection_ready(easy: list[dict], hard: list[dict]) -> bool:
    operations = Counter(record["stress_operation"] for record in hard)
    return (
        len(easy) >= EASY_IMAGES
        and len(hard) >= HARD_IMAGES
        and operations["crossed_bonds"] >= MIN_CROSSED_HARD
    )


@torch.inference_mode()
def build_capacity_cache(device: torch.device) -> tuple[list[dict], dict]:
    rows = [
        row
        for row in p18e0._jsonl(PILOT_MANIFEST)
        if str(row["variant"]) == "stress"
    ]
    manifest = p18e0._manifest_records(rows)
    source = p1a.iter_manifest_samples(manifest, accepted_split="train")
    vision, bond_head, proposal = bond_source._load_bond_source(device)
    collator = G3StructuralCollator(
        maximum_atoms=p18e0.ATOM_CAPACITY,
        maximum_bonds=p18e0.BOND_CAPACITY,
        image_size=768,
    )
    state_before = {
        "vision": module_state_sha256(vision),
        "bond": module_state_sha256(bond_head),
    }
    easy: list[dict] = []
    hard: list[dict] = []
    scanned = 0
    pending = []

    def process(samples: list[dict]) -> None:
        nonlocal scanned
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stroke = vision.forward_stroke_resolution(pixels)
            dense = bond_source.forward_bond_source(vision, bond_head, stroke)
            proposals = decode_predicted_bond_proposals(dense, proposal)
        for index, sample in enumerate(samples):
            scanned += 1
            parameters = sample["render"].get("render_parameters", {})
            record = {
                "sample_key": str(sample["key"]),
                "molecule_id": str(sample["chemmol"]["molecule_id"]),
                "variant": "stress",
                "stress_operation": str(parameters.get("stress_operation", "unknown")),
                "target_atom_anchors": batch["atom_anchors"][index].clone(),
                "target_atom_mask": batch["atom_mask"][index].clone(),
                "target_bond_atom_refs": batch["bond_atom_refs"][index].clone(),
                "target_bond_types": batch["bond_types"][index].clone(),
                "target_bond_mask": batch["bond_mask"][index].clone(),
                "bond_candidate_ids": proposals["bond_indices"][index].long().cpu(),
                "bond_candidate_features": proposals["bond_features"][index].half().cpu(),
                "bond_candidate_endpoints": proposals["bond_endpoints"][index].float().cpu(),
                "bond_candidate_types": proposals["bond_types"][index].half().cpu(),
                "bond_candidate_scores": proposals["bond_scores"][index].float().cpu(),
                "bond_candidate_mask": proposals["bond_mask"][index].cpu(),
            }
            exact12 = _record_metrics(record, record["bond_candidate_endpoints"])["exact"]
            metric24 = p18e0a._bond_metrics(record, budget=192, radius_px=24.0)
            if exact12 and len(easy) < EASY_IMAGES:
                record["difficulty"] = "easy"
                easy.append(record)
            elif not exact12 and metric24["bond_exact"]:
                record["difficulty"] = "hard"
                hard.append(record)

    for sample in source:
        pending.append(sample)
        if len(pending) == BATCH_SIZE:
            process(pending)
            pending = []
            if _selection_ready(easy, hard):
                break
    if pending and not _selection_ready(easy, hard):
        process(pending)
    if not _selection_ready(easy, hard):
        raise RuntimeError(
            f"P18-E0B could not build capacity64: easy={len(easy)} hard={len(hard)} "
            f"operations={Counter(record['stress_operation'] for record in hard)}"
        )

    crossed = [record for record in hard if record["stress_operation"] == "crossed_bonds"]
    other = [record for record in hard if record["stress_operation"] != "crossed_bonds"]
    selected_hard = crossed[:MIN_CROSSED_HARD] + other[: HARD_IMAGES - MIN_CROSSED_HARD]
    if len(selected_hard) < HARD_IMAGES:
        used = {record["sample_key"] for record in selected_hard}
        selected_hard.extend(
            record for record in hard if record["sample_key"] not in used
        )
        selected_hard = selected_hard[:HARD_IMAGES]
    records = easy[:EASY_IMAGES] + selected_hard
    for record in records:
        targets, mask = build_residual_targets(record)
        record["residual_target_endpoints"] = targets
        record["residual_supervision_mask"] = mask
    state_after = {
        "vision": module_state_sha256(vision),
        "bond": module_state_sha256(bond_head),
    }
    del vision, bond_head
    gc.collect()
    torch.cuda.empty_cache()
    selection_digest = hashlib.sha256(
        "".join(f"{record['sample_key']}\n" for record in records).encode()
    ).hexdigest()
    return records, {
        "scanned_stress_images": scanned,
        "selected_images": len(records),
        "difficulty": dict(Counter(record["difficulty"] for record in records)),
        "operations": dict(Counter(record["stress_operation"] for record in records)),
        "hard_operations": dict(
            Counter(
                record["stress_operation"]
                for record in records
                if record["difficulty"] == "hard"
            )
        ),
        "sample_keys_sha256": selection_digest,
        "source_state_unchanged": state_before == state_after,
        "proposal_config": asdict(proposal),
    }


def _stack(records: list[dict], device: torch.device) -> dict:
    names = (
        "bond_candidate_features",
        "bond_candidate_endpoints",
        "bond_candidate_types",
        "bond_candidate_scores",
        "bond_candidate_mask",
        "residual_target_endpoints",
        "residual_supervision_mask",
    )
    return {
        name: torch.stack([record[name] for record in records]).to(device)
        for name in names
    }


def _forward(head: ProposalEndpointResidualHead, batch: dict) -> dict:
    return head(
        batch["bond_candidate_features"],
        batch["bond_candidate_endpoints"],
        batch["bond_candidate_types"],
        batch["bond_candidate_scores"],
        batch["bond_candidate_mask"],
    )


@torch.inference_mode()
def evaluate(head, records: list[dict], device: torch.device) -> dict:
    head.eval()
    rows = []
    for start in range(0, len(records), BATCH_SIZE):
        current = records[start : start + BATCH_SIZE]
        batch = _stack(current, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(head, batch)
        for index, record in enumerate(current):
            rows.append(
                {
                    "difficulty": record["difficulty"],
                    "stress_operation": record["stress_operation"],
                    **_record_metrics(record, output["endpoints"][index].float().cpu()),
                }
            )

    def summary(current: list[dict]) -> dict:
        targets = sum(row["targets"] for row in current)
        matched = sum(row["matched"] for row in current)
        exact = sum(int(row["exact"]) for row in current)
        return {
            "images": len(current),
            "targets": targets,
            "matched": matched,
            "recall": matched / max(targets, 1),
            "exact": exact,
            "exact_rate": exact / max(len(current), 1),
        }

    return {
        "overall": summary(rows),
        "by_difficulty": {
            name: summary([row for row in rows if row["difficulty"] == name])
            for name in ("easy", "hard")
        },
        "hard_by_operation": {
            operation: summary(
                [
                    row
                    for row in rows
                    if row["difficulty"] == "hard"
                    and row["stress_operation"] == operation
                ]
            )
            for operation in sorted(
                {
                    row["stress_operation"]
                    for row in rows
                    if row["difficulty"] == "hard"
                }
            )
        },
    }


def _checkpoint(head, optimizer, step: int, metrics: dict, cache_audit: dict) -> dict:
    return {
        "checkpoint_schema_version": "g3-p18e0b-endpoint-residual-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(head.config),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "metrics": metrics,
        "cache_audit": cache_audit,
        "source": {
            "bond_checkpoint": str(bond_source.BOND_SOURCE.resolve()),
            "bond_checkpoint_sha256": bond_source.BOND_SOURCE_SHA256,
            "pilot_manifest": str(PILOT_MANIFEST.resolve()),
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "p18e0a_audit": str(E0A_AUDIT.resolve()),
            "p18e0a_audit_sha256": E0A_AUDIT_SHA256,
        },
        "training_args": {
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P18-E0B output is not empty: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (E0A_AUDIT, E0A_AUDIT_SHA256),
        (bond_source.BOND_SOURCE, bond_source.BOND_SOURCE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P18-E0B input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P18-E0B requires CUDA BF16")
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
    started = time.perf_counter()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    records, cache_audit = build_capacity_cache(device)
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p18e0b-capacity-cache-0.1",
            "records": records,
            "audit": cache_audit,
        },
    )
    cache_audit = {
        **cache_audit,
        "path": str(CACHE.resolve()),
        "sha256": file_sha256(CACHE),
    }
    _atomic_json(OUTPUT / "cache-audit.json", cache_audit)

    head = ProposalEndpointResidualHead(CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial = evaluate(head, records, device)
    _atomic_json(OUTPUT / "initial-metrics.json", initial)
    best_metrics = initial
    best_key = (
        initial["by_difficulty"]["easy"]["exact"],
        initial["by_difficulty"]["hard"]["exact"],
        initial["overall"]["matched"],
    )
    history = []
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(records), generator=generator).tolist()
    offset = 0
    running = Counter()
    for step in range(1, STEPS + 1):
        if offset + BATCH_SIZE > len(order):
            order = torch.randperm(len(records), generator=generator).tolist()
            offset = 0
        indices = order[offset : offset + BATCH_SIZE]
        offset += BATCH_SIZE
        batch = _stack([records[index] for index in indices], device)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(head, batch)
            losses = proposal_endpoint_residual_loss(
                output,
                batch["bond_candidate_endpoints"],
                batch["residual_target_endpoints"],
                batch["residual_supervision_mask"],
                batch["bond_candidate_mask"],
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P18-E0B gradient is not finite")
        optimizer.step()
        for name, value in losses.items():
            running[name] += float(value.detach())
        if step % 500 == 0 or step == STEPS:
            metrics = evaluate(head, records, device)
            key = (
                metrics["by_difficulty"]["easy"]["exact"],
                metrics["by_difficulty"]["hard"]["exact"],
                metrics["overall"]["matched"],
            )
            selected = key > best_key
            if selected:
                best_key = key
                best_metrics = metrics
                _atomic_torch_save(
                    OUTPUT / "best.pt",
                    _checkpoint(head, optimizer, step, metrics, cache_audit),
                )
            row = {
                "step": step,
                "selected_best": selected,
                "metrics": metrics,
                "mean_losses": {
                    name: value / 500.0 for name, value in running.items()
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            running.clear()
            print(json.dumps(row, separators=(",", ":")), flush=True)

    final = evaluate(head, records, device)
    checks = {
        "capacity64_is_24px_representable": all(
            p18e0a._bond_metrics(record, budget=192, radius_px=24.0)["bond_exact"]
            for record in records
        ),
        "easy_zero_break": best_metrics["by_difficulty"]["easy"]["exact"] == EASY_IMAGES,
        "hard_rescue_min24_of32": best_metrics["by_difficulty"]["hard"]["exact"] >= 24,
        "predicted_exact_min56_of64": best_metrics["overall"]["exact"] >= 56,
        "crossed_hard_present_min8": cache_audit["hard_operations"].get("crossed_bonds", 0) >= 8,
        "source_state_unchanged": cache_audit["source_state_unchanged"],
    }
    decision = "endpoint_residual_capacity64_go" if all(checks.values()) else "endpoint_residual_capacity64_no_go"
    if not (OUTPUT / "best.pt").exists():
        _atomic_torch_save(
            OUTPUT / "best.pt",
            _checkpoint(head, optimizer, STEPS, final, cache_audit),
        )
    _atomic_torch_save(
        OUTPUT / "last.pt",
        _checkpoint(head, optimizer, STEPS, final, cache_audit),
    )
    result = {
        "schema_version": "g3-p18e0b-endpoint-residual-capacity64-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "initial": initial,
        "best": best_metrics,
        "final": final,
        "history": history,
        "trainable_parameters": sum(value.numel() for value in head.parameters()),
        "cache": cache_audit,
        "best_checkpoint": str((OUTPUT / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(OUTPUT / "best.pt"),
        "sealed_test_read": False,
        "confirm_read": False,
        "fresh_dev_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
