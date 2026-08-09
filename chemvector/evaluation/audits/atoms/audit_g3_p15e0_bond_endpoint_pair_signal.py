#!/usr/bin/env python3
"""Audit frozen Bond endpoint evidence for real-source KEEP/MOVE pairs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.graph.joint.predicted_incidence import (
    decode_predicted_bond_proposals,
)
from chemvector.evaluation.audits.atoms import audit_g3_p3e29_bond_incident_signal as p3e29
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import audit_g3_p4e9_bond_evidence_probe as bond_probe
from chemvector.evaluation.audits.atoms import build_g3_p11e0_predicted_source_pilot as p11e0
from chemvector.evaluation.audits.atoms import build_g3_p14e0_top128_source_context as p14e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as base
from chemvector.training.atoms.train_g3_p7_pairwise_counterfactual import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _autocast_context


AUDIT_NAME = "chemvector-g3-p15e0-bond-endpoint-pair-signal-pilot"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
CACHE = OUTPUT / "endpoint-features.pt"
SOURCE_CACHE_SHA256 = base.CACHE_SHA256
EXPECTED_ROWS = p14e0.EXPECTED_ROWS
EXPECTED_IMAGES = p14e0.EXPECTED_IMAGES
POSITIONS = 9
INCIDENT_DIM = len(p3e29.incident_feature_names())
FOLDS = 3
SEED = 20260901
BATCH_SIZE = 2
MINIMUM_AUC = 0.70
MINIMUM_AUC_LIFT = 0.03
MINIMUM_RESCUE_ENDPOINT_COVERAGE = 0.50


def scaffold_fold(scaffold: str, folds: int = FOLDS) -> int:
    if folds < 2:
        raise ValueError("P15-E0 requires at least two folds")
    digest = hashlib.sha256(f"p15:{SEED}:{scaffold}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def pair_incident_features(
    endpoint_features: torch.Tensor,
    source_rows: torch.Tensor,
    alternative_indices: torch.Tensor,
) -> torch.Tensor:
    if endpoint_features.ndim != 3 or endpoint_features.shape[1:] != (
        POSITIONS,
        INCIDENT_DIM,
    ):
        raise ValueError("P15-E0 endpoint features have invalid shape")
    if source_rows.shape != alternative_indices.shape or source_rows.ndim != 1:
        raise ValueError("P15-E0 pair indices must be equal vectors")
    selected = endpoint_features[source_rows].float()
    local = torch.arange(len(source_rows))
    source = selected[:, 0]
    alternative = selected[local, alternative_indices.long() + 1]
    return torch.cat((source, alternative, alternative - source), dim=-1)


def pair_baseline_features(
    source_cache: dict,
    source_rows: torch.Tensor,
    alternative_indices: torch.Tensor,
) -> torch.Tensor:
    offsets = source_cache["candidate_offsets"][source_rows].float()
    local = torch.arange(len(source_rows))
    alternative = offsets[local, alternative_indices]
    geometry = alternative * (768.0 / 96.0)
    distance = torch.linalg.vector_norm(geometry, dim=-1, keepdim=True)
    peak_logit = source_cache["candidate_peak_logits"][source_rows][
        local, alternative_indices
    ].float()[:, None]
    score = source_cache["score_crops"][source_rows].float()
    source_grid = torch.zeros(len(source_rows), 1, 1, 2)
    alternative_grid = alternative[:, None, None] * 16.0
    source_score = F.grid_sample(
        score,
        source_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).flatten()[:, None]
    alternative_score = F.grid_sample(
        score,
        alternative_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).flatten()[:, None]
    return torch.cat(
        (
            geometry,
            distance,
            peak_logit,
            source_score,
            alternative_score,
            alternative_score - source_score,
        ),
        dim=-1,
    )


def oof_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
) -> dict:
    if set(folds.tolist()) != set(range(FOLDS)):
        raise ValueError("P15-E0 OOF lacks a fold")
    scores = torch.empty(len(features), dtype=torch.float32)
    per_fold = {}
    for fold in range(FOLDS):
        test = folds == fold
        train = ~test
        if min(int(labels[test].sum()), int((~labels[test]).sum())) == 0:
            raise ValueError(f"P15-E0 fold {fold} lacks a class")
        scores[test] = bond_probe.fit_logistic_predict(
            features[train], labels[train], features[test]
        )
        per_fold[str(fold)] = {
            "pairs": int(test.sum()),
            "rescues": int(labels[test].sum()),
            "auc": p3e29.binary_auc(scores[test], labels[test]),
        }
    return {
        "auc": p3e29.binary_auc(scores, labels),
        "macro_auc": sum(row["auc"] for row in per_fold.values()) / FOLDS,
        "per_fold": per_fold,
    }


def gate(augmented_auc: float, baseline_auc: float, coverage: float) -> tuple[str, dict]:
    checks = {
        "augmented_auc_min070": augmented_auc >= MINIMUM_AUC,
        "auc_lift_min003": augmented_auc - baseline_auc >= MINIMUM_AUC_LIFT,
        "rescue_endpoint_coverage_min050": coverage
        >= MINIMUM_RESCUE_ENDPOINT_COVERAGE,
    }
    return (
        "bond_endpoint_pair_signal_go"
        if all(checks.values())
        else "bond_endpoint_pair_signal_no_go",
        checks,
    )


@torch.inference_mode()
def build_endpoint_cache(source_cache: dict, manifest: list, device: torch.device) -> tuple[dict, dict]:
    rows_by_key: dict[str, list[int]] = defaultdict(list)
    for row, key in enumerate(source_cache["sample_key"]):
        rows_by_key[str(key)].append(row)
    if len(rows_by_key) != EXPECTED_IMAGES:
        raise RuntimeError("P15-E0 source image count differs")

    field, atom_head, atom_vision, frozen_block, atom_vision_sha = p4e6._load_model(device)
    bond_vision, bond, proposal = p3e29._load_bond_source(device)
    modules = (field, atom_head, atom_vision, frozen_block, bond_vision, bond)
    if any(any(parameter.requires_grad for parameter in module.parameters()) for module in modules):
        raise RuntimeError("P15-E0 source model is not frozen")

    features = torch.empty(EXPECTED_ROWS, POSITIONS, INCIDENT_DIM, dtype=torch.float16)
    position_mask = torch.zeros(EXPECTED_ROWS, POSITIONS, dtype=torch.bool)
    seen = torch.zeros(EXPECTED_ROWS, dtype=torch.bool)
    collator = bond_probe._PixelOnlyCollator()
    started = time.perf_counter()
    for batch in p1a.iter_manifest_batches(
        manifest, collator, batch_size=BATCH_SIZE, accepted_split="train"
    ):
        pixels = batch["pixel_values"].to(device)
        objects, stroke = p1t.model_inputs(
            atom_head,
            atom_vision,
            frozen_block,
            pixels,
            device=device,
            training=False,
        )
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
            bond_stroke = bond_vision.forward_stroke_resolution(pixels)
            bond_output = p3e29.forward_bond_source(bond_vision, bond, bond_stroke)
            decoded_bonds = decode_predicted_bond_proposals(bond_output, proposal)
        decoded_atoms = decode_atom_center_field(field_output)
        for batch_index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            dense_anchors = (
                decoded_atoms["anchors"][batch_index]
                .permute(1, 2, 0)
                .reshape(-1, 2)
                .float()
            )
            for row in rows_by_key[key]:
                source_id = int(source_cache["base_source_id"][row])
                if not 0 <= source_id < len(dense_anchors):
                    raise RuntimeError("P15-E0 source ID is outside dense field")
                source = dense_anchors[source_id]
                candidate_offsets = source_cache["candidate_offsets"][row].to(device).float()
                positions = torch.cat((source[None], source[None] + candidate_offsets), dim=0)
                current = p3e29.continuous_incident_features(
                    positions,
                    decoded_bonds["bond_scores"][batch_index],
                    decoded_bonds["bond_endpoints"][batch_index],
                    decoded_bonds["bond_types"][batch_index],
                    decoded_bonds["bond_mask"][batch_index],
                )
                features[row] = current.cpu().half()
                position_mask[row, 0] = True
                position_mask[row, 1:] = source_cache["candidate_mask"][row]
                seen[row] = True
    checks = {
        "rows_2152": int(seen.sum()) == EXPECTED_ROWS,
        "all_rows_seen_once": bool(seen.all()),
        "source_positions_valid": bool(position_mask[:, 0].all()),
        "candidate_mask_matches_source_cache": bool(
            torch.equal(position_mask[:, 1:], source_cache["candidate_mask"])
        ),
        "features_finite": bool(torch.isfinite(features).all()),
    }
    return {
        "schema_version": "g3-p15e0-bond-endpoint-features-0.1",
        "source_cache_sha256": SOURCE_CACHE_SHA256,
        "bond_checkpoint_sha256": p3e29.BOND_SOURCE_SHA256,
        "bond_vision_sha256": p3e29.VISION_SOURCE_SHA256,
        "atom_vision_sha256": atom_vision_sha,
        "sample_key": list(source_cache["sample_key"]),
        "endpoint_features": features,
        "position_mask": position_mask,
    }, {
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P15-E0 output already exists: {OUTPUT}")
    if file_sha256(base.CACHE) != SOURCE_CACHE_SHA256:
        raise ValueError("P15-E0 source cache SHA256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P15-E0 requires CUDA")

    source_cache = torch.load(base.CACHE, map_location="cpu", weights_only=False)
    full_manifest, manifest_audit = load_g3_sample_manifest(
        p11e0.PILOT_MANIFEST,
        expected_sha256=p11e0.PILOT_MANIFEST_SHA256,
        expected_samples=20_000,
        expected_molecules=5_000,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    keys = set(source_cache["sample_key"])
    manifest = [record for record in full_manifest if record.sample_key in keys]
    if len(manifest) != EXPECTED_IMAGES:
        raise RuntimeError("P15-E0 manifest is incomplete")
    payload, cache_audit = build_endpoint_cache(source_cache, manifest, torch.device("cuda"))

    table = base.build_pair_table(source_cache, "train")
    pools, pool_audit = base.select_stratified_pools(table)
    pair_rows = torch.cat((pools["rescue"], pools["harmful"]))
    labels = table["label"][pair_rows] == 1
    source_rows = table["source_row"][pair_rows]
    alternatives = table["alternative_index"][pair_rows]
    folds = torch.tensor(
        [scaffold_fold(source_cache["scaffold"][row]) for row in source_rows],
        dtype=torch.long,
    )
    baseline_features = pair_baseline_features(source_cache, source_rows, alternatives)
    incident_features = pair_incident_features(
        payload["endpoint_features"], source_rows, alternatives
    )
    baseline = oof_logistic(baseline_features, labels, folds)
    augmented = oof_logistic(
        torch.cat((baseline_features, incident_features), dim=-1), labels, folds
    )
    rescue_alt = payload["endpoint_features"][source_rows[labels], alternatives[labels] + 1]
    rescue_coverage = float((rescue_alt[:, 0] * 768.0 <= 6.0).float().mean())
    decision, checks = gate(augmented["auc"], baseline["auc"], rescue_coverage)

    OUTPUT.mkdir(parents=True)
    _atomic_torch_save(CACHE, payload)
    audit = {
        "schema_version": "g3-p15e0-bond-endpoint-pair-signal-audit-0.1",
        "decision": decision,
        "checks": {**cache_audit["checks"], **checks},
        "cache": str(CACHE.resolve()),
        "cache_sha256": file_sha256(CACHE),
        "rows": EXPECTED_ROWS,
        "images": EXPECTED_IMAGES,
        "train_pairs": len(pair_rows),
        "train_rescues": int(labels.sum()),
        "train_harmful": int((~labels).sum()),
        "baseline_oof": baseline,
        "augmented_oof": augmented,
        "auc_lift": augmented["auc"] - baseline["auc"],
        "rescue_alternative_endpoint_coverage_6px": rescue_coverage,
        "pool": pool_audit,
        "cache_build": cache_audit,
        "manifest": manifest_audit,
        "protocol": {
            "features": "frozen BondObject endpoint evidence at source and P6 top8",
            "selection": "P11-E1 stratified 281 rescue + 281 harmful",
            "signal_gate": "train-only three-fold scaffold OOF",
            "calibration_features_cached_without_labels": True,
            "calibration_labels_read": False,
            "holdout_read": False,
            "confirm_external_sealed_read": False,
        },
        "holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
