#!/usr/bin/env python3
"""Evaluate the fixed P18-E0B endpoint residual on unseen fresh640 proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from chemvector.components.bonds.objects.proposal_endpoint_residual import (
    ProposalEndpointResidualConfig,
    ProposalEndpointResidualHead,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0a_stress_bond_ceiling as p18e0a,
)
from chemvector.training.bonds.relations import (
    train_g3_p18e0b_endpoint_residual_capacity64 as p18e0b,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p18e0c-endpoint-residual-fresh640"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = p18e0b.OUTPUT / "best.pt"
CHECKPOINT_SHA256 = "a2e2164e1f7de83dc891a8369b92bcd43657be73b995b54f85b79824459f0357"
E0_CACHE = p18e0.CACHE
E0_CACHE_SHA256 = "009d9dcf77c0d65a17530c2afff6566f7f2265adc9383f750f9b43a7f7cd88cc"
E0A_CACHE = p18e0a.CACHE
E0A_CACHE_SHA256 = "37e84038112a7fdd7a04f1c6a20671f70bafeb3c65e969ab10c96d98898928e8"
BATCH_SIZE = 32
GATES = {
    "overall_exact_rate_min": 0.80,
    "stress_exact_rate_min": 0.50,
    "nonstress_regression_max": 0.02,
    "crossed_exact_min": 8,
}


def _target_bonds(record: dict) -> torch.Tensor:
    atoms = record["target_atom_anchors"][record["target_atom_mask"]].float()
    refs = record["target_bond_atom_refs"][record["target_bond_mask"]].long()
    return atoms[refs]


def _metrics(record: dict, endpoints: torch.Tensor) -> dict:
    targets = _target_bonds(record)
    mask = record["bond_candidate_mask"]
    assignment, _ = p18e0._maximum_bond_assignment(
        endpoints[mask].float(), targets, p18e0.BOND_RADIUS_PX
    )
    matched = int((assignment >= 0).sum())
    return {"targets": len(targets), "matched": matched, "exact": matched == len(targets)}


def _summarize(rows: list[dict]) -> dict:
    targets = sum(row["targets"] for row in rows)
    matched = sum(row["matched"] for row in rows)
    exact = sum(int(row["exact"]) for row in rows)
    return {
        "images": len(rows),
        "targets": targets,
        "matched": matched,
        "recall": matched / max(targets, 1),
        "exact": exact,
        "exact_rate": exact / max(len(rows), 1),
    }


def _endpoint_digest(values: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for key, endpoints in values:
        digest.update(f"{key}\n".encode())
        digest.update(endpoints.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_once(head, records: list[dict], operations: dict[str, str], device) -> tuple[dict, str]:
    rows = []
    corrected_values = []
    for start in range(0, len(records), BATCH_SIZE):
        current = records[start : start + BATCH_SIZE]
        features = torch.stack([record["bond_candidate_features"] for record in current]).to(device)
        endpoints = torch.stack([record["bond_candidate_endpoints"] for record in current]).to(device)
        types = torch.stack([record["bond_candidate_types"] for record in current]).to(device)
        scores = torch.stack([record["bond_candidate_scores"] for record in current]).to(device)
        masks = torch.stack([record["bond_candidate_mask"] for record in current]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = head(features, endpoints, types, scores, masks)
        for index, record in enumerate(current):
            key = str(record["sample_key"])
            base = _metrics(record, record["bond_candidate_endpoints"])
            corrected_endpoints = output["endpoints"][index].float().cpu()
            corrected = _metrics(record, corrected_endpoints)
            corrected_values.append((key, corrected_endpoints))
            rows.append(
                {
                    "sample_key": key,
                    "variant": str(record["variant"]),
                    "stress_operation": operations.get(key),
                    "base": base,
                    "corrected": corrected,
                }
            )

    def report(current: list[dict]) -> dict:
        base = _summarize([row["base"] for row in current])
        corrected = _summarize([row["corrected"] for row in current])
        return {
            "base": base,
            "corrected": corrected,
            "exact_delta": corrected["exact"] - base["exact"],
            "exact_rate_delta": corrected["exact_rate"] - base["exact_rate"],
            "rescued": sum(not row["base"]["exact"] and row["corrected"]["exact"] for row in current),
            "broken": sum(row["base"]["exact"] and not row["corrected"]["exact"] for row in current),
        }

    result = {
        "overall": report(rows),
        "nonstress": report([row for row in rows if row["variant"] != "stress"]),
        "by_variant": {
            variant: report([row for row in rows if row["variant"] == variant])
            for variant in p18e0.VARIANTS
        },
        "stress_by_operation": {
            operation: report(
                [row for row in rows if row["stress_operation"] == operation]
            )
            for operation in sorted({value for value in operations.values()})
        },
        "rows": rows,
    }
    return result, _endpoint_digest(corrected_values)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P18-E0C output is not empty: {OUTPUT}")
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256),
        (E0_CACHE, E0_CACHE_SHA256),
        (E0A_CACHE, E0A_CACHE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P18-E0C input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P18-E0C requires CUDA BF16")
    device = torch.device("cuda")
    started = time.perf_counter()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != p18e0b.MODEL_TYPE or checkpoint.get("global_step") != 2500:
        raise ValueError("P18-E0C checkpoint identity differs")
    head = ProposalEndpointResidualHead(
        ProposalEndpointResidualConfig(**checkpoint["config"])
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval().requires_grad_(False)
    e0 = torch.load(E0_CACHE, map_location="cpu", weights_only=False)
    records = e0["records"]
    stress = torch.load(E0A_CACHE, map_location="cpu", weights_only=False)
    operations = {
        str(record["sample_key"]): str(record["stress_operation"])
        for record in stress["records"]
    }
    if len(records) != p18e0.IMAGES or len(operations) != p18e0.MOLECULES:
        raise RuntimeError("P18-E0C cache counts differ")

    metrics, first_digest = evaluate_once(head, records, operations, device)
    repeated, second_digest = evaluate_once(head, records, operations, device)
    if metrics != repeated or first_digest != second_digest:
        raise RuntimeError("P18-E0C repeated evaluation differs")
    overall = metrics["overall"]
    stress_metrics = metrics["by_variant"]["stress"]
    nonstress = metrics["nonstress"]
    crossed = metrics["stress_by_operation"]["crossed_bonds"]
    checks = {
        "overall_exact_rate_min80": overall["corrected"]["exact_rate"] >= GATES["overall_exact_rate_min"],
        "stress_exact_rate_min50": stress_metrics["corrected"]["exact_rate"] >= GATES["stress_exact_rate_min"],
        "nonstress_regression_max2pp": (
            nonstress["corrected"]["exact_rate"]
            >= nonstress["base"]["exact_rate"] - GATES["nonstress_regression_max"]
        ),
        "crossed_exact_min8": crossed["corrected"]["exact"] >= GATES["crossed_exact_min"],
        "repeat_digest_exact": first_digest == second_digest,
        "candidate_identity_fields_immutable_by_interface": True,
    }
    decision = "endpoint_residual_fresh640_go" if all(checks.values()) else "endpoint_residual_fresh640_no_go"
    rows = metrics.pop("rows")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "rows.json", {"rows": rows})
    report = {
        "schema_version": "g3-p18e0c-endpoint-residual-fresh640-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "metrics": metrics,
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "global_step": 2500,
            "config": asdict(head.config),
        },
        "inputs": {
            "fresh640_cache_sha256": E0_CACHE_SHA256,
            "stress_operation_cache_sha256": E0A_CACHE_SHA256,
        },
        "repeat_endpoint_digest": first_digest,
        "training_or_calibration": False,
        "sealed_test_read": False,
        "confirm_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
