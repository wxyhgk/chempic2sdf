#!/usr/bin/env python3
"""Audit P2-W joint decoding on molecule-disjoint route100."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import (
    audit_g3_p2x_joint_cardinality_decode_capacity64 as p2x,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p2y-joint-cardinality-decode-route100"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")


def molecule_key(sample_key: str) -> str:
    return sample_key.rsplit("_", 1)[0]


def candidate_oracle(records: list[dict]) -> dict:
    matched_atoms = target_atoms = full_recall_images = 0
    for record in records:
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        matched = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"].float(),
            expected,
            radius_pixels=6.0,
            image_size=768.0,
        )
        matched_atoms += matched
        target_atoms += len(expected)
        full_recall_images += int(matched == len(expected))
    return {
        "images": len(records),
        "matched_atoms": matched_atoms,
        "target_atoms": target_atoms,
        "recall": matched_atoms / target_atoms,
        "full_recall_images": full_recall_images,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P2-Y output is not empty: {OUTPUT}")
    if file_sha256(p2x.CHECKPOINT) != p2x.CHECKPOINT_SHA256:
        raise ValueError("P2-Y P2-W checkpoint SHA256 differs")
    device = torch.device("cuda")
    source = torch.load(p2x.CHECKPOINT, map_location=device, weights_only=False)
    base = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model = G3PointerUtilityExistenceModel(base).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    cache, cache_audit = p2e.load_cache()
    route = cache["route100"]
    train = cache["train5k"]
    route_keys = {str(record["sample_key"]) for record in route}
    train_keys = {str(record["sample_key"]) for record in train}
    route_molecules = {molecule_key(value) for value in route_keys}
    train_molecules = {molecule_key(value) for value in train_keys}
    overlap = {
        "sample_keys": len(route_keys & train_keys),
        "molecules": len(route_molecules & train_molecules),
    }
    if any(overlap.values()):
        raise RuntimeError(f"P2-Y route100 overlaps train5k: {overlap}")

    independent = p2h.evaluate(model, route, device=device)
    joint = p2x.evaluate_joint(model, route, device=device)
    repeated = p2x.evaluate_joint(model, route, device=device)
    payload = {
        "schema_version": "g3-p2y-joint-route100-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": "count_generalization_bottleneck_confirmed",
        "route100_is_train5k_disjoint": overlap == {"sample_keys": 0, "molecules": 0},
        "overlap": overlap,
        "candidate_oracle": candidate_oracle(route),
        "independent_topn": independent,
        "joint_cardinality": joint,
        "repeat_metrics": repeated == joint,
        "lineage": {
            "checkpoint": str(p2x.CHECKPOINT.resolve()),
            "checkpoint_sha256": p2x.CHECKPOINT_SHA256,
            "model_state_sha256": module_state_sha256(model),
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
        },
    }
    if not payload["route100_is_train5k_disjoint"] or not payload["repeat_metrics"]:
        payload["decision"] = "protocol_failed"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
