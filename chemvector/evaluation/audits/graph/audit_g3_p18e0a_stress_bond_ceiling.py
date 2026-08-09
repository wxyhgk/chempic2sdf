#!/usr/bin/env python3
"""Separate top-k truncation from dense geometry failure on P18 stress Bonds."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.joint.predicted_incidence import (
    decode_predicted_bond_proposals,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as bond_source,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p18e0a-stress-bond-ceiling"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CACHE = OUTPUT / "stress-top512-cache.pt"
E0_AUDIT = p18e0.OUTPUT / "audit.json"
E0_AUDIT_SHA256 = "83b033c4a0e3ed5227804335a6f5eecef49c9dbac3f89a2afa21b0ec2132978f"
E0_CACHE = p18e0.CACHE
E0_CACHE_SHA256 = "009d9dcf77c0d65a17530c2afff6566f7f2265adc9383f750f9b43a7f7cd88cc"
E0_MANIFEST = p18e0.SELECTION_MANIFEST
E0_MANIFEST_SHA256 = "7aeb5df329338faab36b0e5a3baa5bc0e078273034c6cd6718e16f66864bb9bc"
BUDGETS = (128, 192, 256, 384, 512)
RADII_PX = (12.0, 16.0, 24.0)
MAXIMUM_BONDS = max(BUDGETS)
STRESS_IMAGES = p18e0.MOLECULES
BATCH_SIZE = p18e0.BATCH_SIZE
SEED = 20260810


def classify_bottleneck(
    top192_exact_rate: float,
    top512_exact_rate: float,
    radius24_exact_rate: float,
) -> str:
    if top512_exact_rate >= 0.60 and top512_exact_rate - top192_exact_rate >= 0.20:
        return "top192_ranking_truncation"
    if radius24_exact_rate >= 0.60 and radius24_exact_rate - top512_exact_rate >= 0.20:
        return "endpoint_localization_precision"
    return "dense_bond_generation"


def _target_bonds(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    atoms = record["target_atom_anchors"][record["target_atom_mask"]].float()
    refs = record["target_bond_atom_refs"][record["target_bond_mask"]].long()
    return atoms[refs], refs


def _bond_metrics(record: dict, *, budget: int, radius_px: float) -> dict:
    target_bonds, _ = _target_bonds(record)
    mask = record["bond_candidate_mask"][:budget]
    predicted = record["bond_candidate_endpoints"][:budget][mask].float()
    assignment, _ = p18e0._maximum_bond_assignment(
        predicted, target_bonds, radius_px
    )
    matched = int((assignment >= 0).sum())
    return {
        "bond_targets": len(target_bonds),
        "bond_matched": matched,
        "bond_exact": matched == len(target_bonds),
    }


def _summarize(records: list[dict], *, budget: int, radius_px: float) -> dict:
    rows = [_bond_metrics(record, budget=budget, radius_px=radius_px) for record in records]
    targets = sum(row["bond_targets"] for row in rows)
    matched = sum(row["bond_matched"] for row in rows)
    exact = sum(int(row["bond_exact"]) for row in rows)
    return {
        "images": len(rows),
        "bond_targets": targets,
        "bond_matched": matched,
        "bond_recall": matched / max(targets, 1),
        "bond_exact": exact,
        "bond_exact_rate": exact / max(len(rows), 1),
    }


def _first_compatible_ranks(record: dict, radius_px: float = 12.0) -> list[int | None]:
    target_bonds, _ = _target_bonds(record)
    mask = record["bond_candidate_mask"]
    predicted = record["bond_candidate_endpoints"][mask].float()
    if not len(predicted):
        return [None] * len(target_bonds)
    forward = torch.linalg.vector_norm(
        (predicted[:, None] - target_bonds[None]) * p18e0.IMAGE_SIZE, dim=-1
    ).amax(-1)
    reverse = torch.linalg.vector_norm(
        (predicted[:, None] - target_bonds[None].flip(2)) * p18e0.IMAGE_SIZE,
        dim=-1,
    ).amax(-1)
    compatible = torch.minimum(forward, reverse) <= radius_px
    output = []
    for target in range(len(target_bonds)):
        indices = torch.where(compatible[:, target])[0]
        output.append(int(indices[0]) + 1 if len(indices) else None)
    return output


def _length_bucket(length_px: float) -> str:
    if length_px < 24.0:
        return "lt24"
    if length_px < 48.0:
        return "24to48"
    if length_px < 96.0:
        return "48to96"
    return "ge96"


def _diagnostics(records: list[dict]) -> dict:
    rank_buckets = Counter()
    missing_lengths = Counter()
    missing_by_operation = Counter()
    for record in records:
        target_bonds, _ = _target_bonds(record)
        ranks = _first_compatible_ranks(record)
        for target, rank in enumerate(ranks):
            if rank is None:
                rank_buckets["missing_top512"] += 1
            elif rank <= 128:
                rank_buckets["rank_1_128"] += 1
            elif rank <= 192:
                rank_buckets["rank_129_192"] += 1
            elif rank <= 256:
                rank_buckets["rank_193_256"] += 1
            elif rank <= 384:
                rank_buckets["rank_257_384"] += 1
            else:
                rank_buckets["rank_385_512"] += 1
            if rank is None or rank > 192:
                length = float(
                    torch.linalg.vector_norm(
                        (target_bonds[target, 1] - target_bonds[target, 0])
                        * p18e0.IMAGE_SIZE
                    )
                )
                missing_lengths[_length_bucket(length)] += 1
                missing_by_operation[str(record["stress_operation"])] += 1
    return {
        "first_compatible_rank_buckets": dict(sorted(rank_buckets.items())),
        "top192_missing_target_length_buckets": dict(sorted(missing_lengths.items())),
        "top192_missing_targets_by_operation": dict(sorted(missing_by_operation.items())),
    }


def _source_digest(records: list[dict], budget: int) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['sample_key']}\n".encode())
        for name in (
            "bond_candidate_ids",
            "bond_candidate_endpoints",
            "bond_candidate_scores",
            "bond_candidate_mask",
        ):
            value = record[name][:budget].contiguous()
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def collect_top512(samples: list[dict], e0_by_key: dict[str, dict], device) -> tuple[list[dict], dict]:
    vision, bond_head, base_proposal = bond_source._load_bond_source(device)
    top192_config = replace(base_proposal, maximum_bonds=192)
    top512_config = replace(base_proposal, maximum_bonds=MAXIMUM_BONDS)
    collator = G3StructuralCollator(
        maximum_atoms=p18e0.ATOM_CAPACITY,
        maximum_bonds=p18e0.BOND_CAPACITY,
        image_size=768,
    )
    state_before = {
        "vision": module_state_sha256(vision),
        "bond": module_state_sha256(bond_head),
    }
    records = []
    replay_checks = Counter()
    for start in range(0, len(samples), BATCH_SIZE):
        current = samples[start : start + BATCH_SIZE]
        batch = collator(current)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stroke = vision.forward_stroke_resolution(pixels)
            dense = bond_source.forward_bond_source(vision, bond_head, stroke)
            top192 = decode_predicted_bond_proposals(dense, top192_config)
            top512 = decode_predicted_bond_proposals(dense, top512_config)
        for index, sample in enumerate(current):
            if str(sample["render"]["variant"]) != "stress":
                continue
            key = str(sample["key"])
            old = e0_by_key[key]
            replay_checks["images"] += 1
            replay_checks["id_exact"] += int(
                torch.equal(top192["bond_indices"][index].cpu(), old["bond_candidate_ids"])
            )
            replay_checks["mask_exact"] += int(
                torch.equal(top192["bond_mask"][index].cpu(), old["bond_candidate_mask"])
            )
            replay_checks["endpoint_exact"] += int(
                torch.equal(
                    top192["bond_endpoints"][index].float().cpu(),
                    old["bond_candidate_endpoints"],
                )
            )
            parameters = sample["render"].get("render_parameters", {})
            records.append(
                {
                    **{
                        name: old[name]
                        for name in (
                            "sample_key",
                            "molecule_id",
                            "variant",
                            "target_atom_anchors",
                            "target_atom_mask",
                            "target_bond_atom_refs",
                            "target_bond_types",
                            "target_bond_mask",
                        )
                    },
                    "stress_operation": str(parameters.get("stress_operation", "unknown")),
                    "stress_details": parameters.get("stress_details", {}),
                    "bond_candidate_ids": top512["bond_indices"][index].long().cpu(),
                    "bond_candidate_features": top512["bond_features"][index].half().cpu(),
                    "bond_candidate_endpoints": top512["bond_endpoints"][index].float().cpu(),
                    "bond_candidate_types": top512["bond_types"][index].half().cpu(),
                    "bond_candidate_scores": top512["bond_scores"][index].float().cpu(),
                    "bond_candidate_mask": top512["bond_mask"][index].cpu(),
                }
            )
    state_after = {
        "vision": module_state_sha256(vision),
        "bond": module_state_sha256(bond_head),
    }
    del vision, bond_head
    gc.collect()
    torch.cuda.empty_cache()
    return records, {
        "replay_checks": dict(replay_checks),
        "state_unchanged": state_before == state_after,
        "top192_config": asdict(top192_config),
        "top512_config": asdict(top512_config),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P18-E0A output is not empty: {OUTPUT}")
    for path, expected in (
        (E0_AUDIT, E0_AUDIT_SHA256),
        (E0_CACHE, E0_CACHE_SHA256),
        (E0_MANIFEST, E0_MANIFEST_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P18-E0A input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P18-E0A requires CUDA BF16")
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

    e0_payload = torch.load(E0_CACHE, map_location="cpu", weights_only=False)
    e0_by_key = {record["sample_key"]: record for record in e0_payload["records"]}
    rows = p18e0._jsonl(E0_MANIFEST)
    manifest = p18e0._manifest_records(rows)
    samples = list(
        p1a.iter_manifest_samples(
            manifest, accepted_split=p18e0.freshdev.manifest_protocol.EXPECTED_SPLIT
        )
    )
    records, source_audit = collect_top512(samples, e0_by_key, device)
    if len(records) != STRESS_IMAGES:
        raise RuntimeError("P18-E0A did not collect all stress images")
    replay = source_audit["replay_checks"]
    if any(replay.get(name) != STRESS_IMAGES for name in ("id_exact", "mask_exact", "endpoint_exact")):
        raise RuntimeError(f"P18-E0A top192 replay differs from E0: {replay}")

    budget_curve = {
        str(budget): _summarize(records, budget=budget, radius_px=12.0)
        for budget in BUDGETS
    }
    radius_curve = {
        f"{radius:g}px": _summarize(records, budget=MAXIMUM_BONDS, radius_px=radius)
        for radius in RADII_PX
    }
    by_operation = {}
    for operation in sorted({record["stress_operation"] for record in records}):
        current = [record for record in records if record["stress_operation"] == operation]
        by_operation[operation] = {
            "top192_12px": _summarize(current, budget=192, radius_px=12.0),
            "top512_12px": _summarize(current, budget=512, radius_px=12.0),
            "top512_24px": _summarize(current, budget=512, radius_px=24.0),
        }
    top192_rate = budget_curve["192"]["bond_exact_rate"]
    top512_rate = budget_curve["512"]["bond_exact_rate"]
    radius24_rate = radius_curve["24px"]["bond_exact_rate"]
    decision = classify_bottleneck(top192_rate, top512_rate, radius24_rate)
    cache_digest = _source_digest(records, MAXIMUM_BONDS)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p18e0a-stress-top512-cache-0.1",
            "records": records,
            "candidate_tensors_sha256": cache_digest,
            "source": source_audit,
        },
    )
    report = {
        "schema_version": "g3-p18e0a-stress-bond-ceiling-0.1",
        "decision": decision,
        "budget_curve_12px": budget_curve,
        "radius_curve_top512": radius_curve,
        "by_stress_operation": by_operation,
        "diagnostics": _diagnostics(records),
        "source": {
            **source_audit,
            "e0_audit": str(E0_AUDIT.resolve()),
            "e0_audit_sha256": E0_AUDIT_SHA256,
            "e0_cache_sha256": E0_CACHE_SHA256,
            "e0_manifest_sha256": E0_MANIFEST_SHA256,
            "bond_checkpoint": str(bond_source.BOND_SOURCE.resolve()),
            "bond_checkpoint_sha256": bond_source.BOND_SOURCE_SHA256,
        },
        "cache": {
            "path": str(CACHE.resolve()),
            "sha256": file_sha256(CACHE),
            "candidate_tensors_sha256": cache_digest,
        },
        "data_access": {
            "fresh640_read": True,
            "sealed_test_read": False,
            "confirm_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(
        json.dumps(
            {
                "decision": decision,
                "budget_curve_12px": budget_curve,
                "radius_curve_top512": radius_curve,
                "by_stress_operation": by_operation,
                "diagnostics": report["diagnostics"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
