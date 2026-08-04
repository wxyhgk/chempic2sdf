#!/usr/bin/env python3
"""Audit fixed-cardinality joint query-pointer decoding on P2-W capacity64."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
    decode_g3_atom_query_set_joint,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p2x-joint-cardinality-decode-capacity64"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
CHECKPOINT_SHA256 = (
    "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9"
)


@torch.no_grad()
def evaluate_joint(model, records, *, device: torch.device) -> dict:
    model.eval()
    known_totals = p1a._empty_atom_totals()
    predicted_totals = p1a._empty_atom_totals()
    count_predictions = hashlib.sha256()
    failures = []
    for start in range(0, len(records), 4):
        current = records[start : start + 4]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
        known_counts = batch["full_atom_mask"].sum(dim=1).long()
        predicted_counts = output["count_logits"].argmax(dim=-1)
        known = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=known_counts
        )
        predicted = decode_g3_atom_query_set_joint(
            output, batch["candidate_anchors"], counts=predicted_counts
        )
        count_predictions.update(predicted_counts.cpu().numpy().tobytes())
        for index, record in enumerate(current):
            expected = batch["atom_anchors"][
                index, batch["full_atom_mask"][index]
            ].float().cpu()
            known_stats = p1a.atom_sample_statistics(
                known[index]["anchors"].float().cpu(), expected
            )
            predicted_stats = p1a.atom_sample_statistics(
                predicted[index]["anchors"].float().cpu(), expected
            )
            p1a._merge_atom_totals(known_totals, known_stats)
            p1a._merge_atom_totals(predicted_totals, predicted_stats)
            if not predicted_stats["whole_set_exact_images"]:
                failures.append(str(record["sample_key"]))
    return {
        "known_count": p1a._finalize_atom_totals(known_totals),
        "predicted_count": p1a._finalize_atom_totals(predicted_totals),
        "count_predictions_sha256": count_predictions.hexdigest(),
        "failure_sample_keys": failures,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P2-X output is not empty: {OUTPUT}")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("P2-X P2-W checkpoint SHA256 differs")
    device = torch.device("cuda")
    source = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    base = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model = G3PointerUtilityExistenceModel(base).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    model_state_sha256 = module_state_sha256(model)
    cache, cache_audit = p2e.load_cache()
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])

    independent = {
        domain: p2h.evaluate(model, records, device=device)
        for domain, records in domains.items()
    }
    joint = {
        domain: evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    repeated = {
        domain: evaluate_joint(model, records, device=device)
        for domain, records in domains.items()
    }
    decision, checks = p2h.gate(joint)
    decision = decision.replace("atom_query_set", "joint_cardinality_decode")
    protocol_checks = {
        "checkpoint_sha256": file_sha256(CHECKPOINT) == CHECKPOINT_SHA256,
        "repeat_metrics": repeated == joint,
        "inference_action_cache_used": False,
        "decoder_uses_gt_count": False,
        "decoder_score_has_tuned_weight": False,
        "no_pixel_or_distance_threshold": True,
    }
    if not all(
        value for name, value in protocol_checks.items()
        if name not in {"inference_action_cache_used", "decoder_uses_gt_count", "decoder_score_has_tuned_weight"}
    ) or any(
        protocol_checks[name]
        for name in {"inference_action_cache_used", "decoder_uses_gt_count", "decoder_score_has_tuned_weight"}
    ):
        decision = "protocol_failed"
    payload = {
        "schema_version": "g3-p2x-joint-cardinality-decode-capacity64-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "decoder": {
            "objective": "fixed-cardinality maximum-weight bipartite matching",
            "edge_score": "logsigmoid(existence_logit)+logsoftmax(pointer_logit)",
            "cardinality": "argmax(count_logits)",
            "query_and_candidate_uniqueness": True,
        },
        "independent_topn": independent,
        "joint_cardinality": joint,
        "lineage": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_state_sha256": model_state_sha256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
