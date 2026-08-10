#!/usr/bin/env python3
"""Calibrate old/new Direct AtomPair logit fusion and replay fixed eval640."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.joint.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e0_direct_atom_pair_oracle_ladder as p19e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e2_direct_atom_pair_fresh640 as p19e2,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e4_presence_threshold_calibration as p19e4,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e5-old-new-logit-fusion"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
NEW_CHECKPOINT = p19e4.CHECKPOINT
NEW_CHECKPOINT_SHA256 = p19e4.CHECKPOINT_SHA256
OLD_CHECKPOINT = p19e4.OLD_DIRECT_CHECKPOINT
OLD_CHECKPOINT_SHA256 = p19e4.OLD_DIRECT_CHECKPOINT_SHA256
P19E4_AUDIT = p19e4.OUTPUT / "audit.json"
P19E4_AUDIT_SHA256 = "5a8ee3ff2d65f5419bb0d91d29ac38d9272961a1818f8f4828b73802729493bc"
CALIBRATION_MANIFEST = p19e4.CALIBRATION_MANIFEST
CALIBRATION_MANIFEST_SHA256 = (
    "05a7b51067c0453086ef0d69e725f080cc92ce6f5d80a198d7cb2d7f61cbc9da"
)
EVAL_MANIFEST = p19e4.EVAL_MANIFEST
EVAL_MANIFEST_SHA256 = (
    "93fa7bfa43091125860570250cd73f2b38a39b1f12d0ed28a826da70fe3546da"
)
NEW_REFERENCE_THRESHOLD = 0.99609375
OLD_REFERENCE_THRESHOLD = 0.85
ALPHAS = tuple(index / 20.0 for index in range(21))
SEED = 20260810
GATES = {
    "overall_min": 538,
    "nonstress_min": 456,
    "stress_min": 82,
    "crossed_min": 7,
    "near_nonbonded_min": 9,
}


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.float().clamp(1e-6, 1.0 - 1e-6))


def fuse_probability_records(
    new_records: list[dict],
    old_records: list[dict],
    *,
    alpha: float,
) -> list[dict]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("P19-E5 alpha must be in [0, 1]")
    new_by_key = {str(record["sample_key"]): record for record in new_records}
    old_by_key = {str(record["sample_key"]): record for record in old_records}
    if len(new_by_key) != len(new_records) or len(old_by_key) != len(old_records):
        raise ValueError("P19-E5 probability records contain duplicate keys")
    if set(new_by_key) != set(old_by_key):
        raise ValueError("P19-E5 old/new sample keys differ")
    new_reference = math.log(
        NEW_REFERENCE_THRESHOLD / (1.0 - NEW_REFERENCE_THRESHOLD)
    )
    old_reference = math.log(
        OLD_REFERENCE_THRESHOLD / (1.0 - OLD_REFERENCE_THRESHOLD)
    )
    fused = []
    for key in sorted(new_by_key):
        new = new_by_key[key]
        old = old_by_key[key]
        if not torch.equal(new["presence"], old["presence"]):
            raise ValueError("P19-E5 old/new targets differ")
        if (
            new["variant"] != old["variant"]
            or new["stress_operation"] != old["stress_operation"]
        ):
            raise ValueError("P19-E5 old/new metadata differs")
        new_margin = _logit(new["probabilities"]) - new_reference
        old_margin = _logit(old["probabilities"]) - old_reference
        scores = alpha * new_margin + (1.0 - alpha) * old_margin
        fused.append(
            {
                "sample_key": key,
                "variant": new["variant"],
                "stress_operation": new["stress_operation"],
                # p19e4.metrics_at_threshold only requires a comparable score.
                "probabilities": scores,
                "presence": new["presence"],
            }
        )
    return fused


def select_score_threshold(records: list[dict]) -> tuple[float, dict]:
    intervals = []
    candidates = {0.0}
    for record in records:
        positives = record["probabilities"][record["presence"]]
        negatives = record["probabilities"][~record["presence"]]
        if not len(positives) or not len(negatives):
            raise ValueError("P19-E5 threshold selection needs both pair classes")
        lower = float(negatives.max())
        upper = float(positives.min())
        intervals.append((lower, upper))
        candidates.add(float(np.nextafter(lower, math.inf)))
        candidates.add(upper)
    exact_by_threshold = {
        threshold: sum(int(lower < threshold <= upper) for lower, upper in intervals)
        for threshold in candidates
    }
    maximum_exact = max(exact_by_threshold.values())
    best_threshold = None
    best_metrics = None
    best_key = None
    for threshold in sorted(
        value
        for value, exact in exact_by_threshold.items()
        if exact == maximum_exact
    ):
        metrics = p19e4.metrics_at_threshold(records, threshold)
        overall = metrics["overall"]
        key = (
            int(overall["skeleton_exact"]),
            int(metrics["by_variant"]["stress"]["skeleton_exact"]),
            float(overall["edge_f1"]),
            float(overall["edge_precision"]),
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("P19-E5 threshold selection failed")
    return best_threshold, {
        "candidate_thresholds": len(candidates),
        "images_with_nonempty_exact_interval": sum(
            int(lower < upper) for lower, upper in intervals
        ),
        "metrics": {
            key: value for key, value in best_metrics.items() if key != "rows"
        },
    }


def select_fusion(
    new_records: list[dict], old_records: list[dict]
) -> tuple[dict, list[dict]]:
    arms = []
    best = None
    best_records = None
    best_key = None
    for alpha in ALPHAS:
        records = fuse_probability_records(new_records, old_records, alpha=alpha)
        threshold, selection = select_score_threshold(records)
        metrics = selection["metrics"]
        overall = metrics["overall"]
        key = (
            int(overall["skeleton_exact"]),
            int(metrics["by_variant"]["stress"]["skeleton_exact"]),
            float(overall["edge_f1"]),
            -abs(alpha - 0.5),
        )
        row = {
            "alpha_new": alpha,
            "alpha_old": 1.0 - alpha,
            "score_threshold": threshold,
            **selection,
        }
        arms.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best = row
            best_records = records
    if best is None or best_records is None:
        raise RuntimeError("P19-E5 fusion selection failed")
    return {"winner": best, "arms": arms}, best_records


def _load_model(path: Path, device: torch.device) -> DirectAtomPairBond:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DirectAtomPairBond(DirectAtomPairBondConfig(**payload["config"])).to(
        device
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)


def _nonstress_exact(metrics: dict) -> int:
    return sum(
        int(metrics["by_variant"][variant]["skeleton_exact"])
        for variant in ("coordgen", "original", "rdkit")
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E5 output is not empty: {OUTPUT}")
    for path, expected in (
        (NEW_CHECKPOINT, NEW_CHECKPOINT_SHA256),
        (OLD_CHECKPOINT, OLD_CHECKPOINT_SHA256),
        (P19E4_AUDIT, P19E4_AUDIT_SHA256),
        (CALIBRATION_MANIFEST, CALIBRATION_MANIFEST_SHA256),
        (EVAL_MANIFEST, EVAL_MANIFEST_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E5 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E5 requires CUDA BF16")
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

    calibration_rows = p18e0._jsonl(CALIBRATION_MANIFEST)
    eval_rows = p18e0._jsonl(EVAL_MANIFEST)
    calibration_samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(calibration_rows), accepted_split="train"
        )
    )
    eval_samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(eval_rows), accepted_split="train"
        )
    )
    vision, atom, bond, set_head, _ = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    new_model = _load_model(NEW_CHECKPOINT, device)
    old_model = _load_model(OLD_CHECKPOINT, device)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }

    new_calibration, new_calibration_digest = p19e4.collect_probabilities(
        new_model, vision, atom, calibration_samples, device
    )
    old_calibration, old_calibration_digest = p19e4.collect_probabilities(
        old_model, vision, atom, calibration_samples, device
    )
    selection, _ = select_fusion(new_calibration, old_calibration)
    alpha = float(selection["winner"]["alpha_new"])
    score_threshold = float(selection["winner"]["score_threshold"])

    new_eval, new_eval_digest = p19e4.collect_probabilities(
        new_model, vision, atom, eval_samples, device
    )
    old_eval, old_eval_digest = p19e4.collect_probabilities(
        old_model, vision, atom, eval_samples, device
    )
    fused_eval_records = fuse_probability_records(new_eval, old_eval, alpha=alpha)
    fused_eval = p19e4.metrics_at_threshold(fused_eval_records, score_threshold)
    new_reference = p19e4.metrics_at_threshold(
        new_eval, NEW_REFERENCE_THRESHOLD
    )
    old_reference = p19e4.metrics_at_threshold(
        old_eval, OLD_REFERENCE_THRESHOLD
    )
    paired_old = p19e2.paired_summary(old_reference["rows"], fused_eval["rows"])
    paired_new = p19e2.paired_summary(new_reference["rows"], fused_eval["rows"])
    paired_old_rows = paired_old.pop("rows")
    paired_new_rows = paired_new.pop("rows")
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }
    checks = {
        "alpha_selected_on_calibration": alpha in ALPHAS,
        "eval_overall_preserves_best538": (
            fused_eval["overall"]["skeleton_exact"] >= GATES["overall_min"]
        ),
        "eval_nonstress_preserves_best456": (
            _nonstress_exact(fused_eval) >= GATES["nonstress_min"]
        ),
        "eval_stress_preserves_best82": (
            fused_eval["by_variant"]["stress"]["skeleton_exact"]
            >= GATES["stress_min"]
        ),
        "eval_crossed_preserves_new7": (
            fused_eval["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            >= GATES["crossed_min"]
        ),
        "eval_near_preserves_old9": (
            fused_eval["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            >= GATES["near_nonbonded_min"]
        ),
        "paired_vs_old_rescue_exceeds_break": (
            paired_old["overall"]["rescue"] > paired_old["overall"]["break"]
        ),
        "paired_vs_new_no_net_regression": (
            paired_new["overall"]["rescue"] >= paired_new["overall"]["break"]
        ),
        "all_frozen_states_unchanged": states_before == states_after,
    }
    decision = (
        "old_new_logit_fusion_go"
        if all(checks.values())
        else "old_new_logit_fusion_no_go"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "fused-eval-rows.json", fused_eval.pop("rows"))
    _atomic_json(OUTPUT / "paired-vs-old-rows.json", paired_old_rows)
    _atomic_json(OUTPUT / "paired-vs-new-rows.json", paired_new_rows)
    new_reference.pop("rows")
    old_reference.pop("rows")
    report = {
        "schema_version": "g3-p19e5-old-new-logit-fusion-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "selection": selection,
        "eval_fusion": fused_eval,
        "eval_new_reference": new_reference,
        "eval_old_reference": old_reference,
        "paired_vs_old": paired_old,
        "paired_vs_new": paired_new,
        "digests": {
            "new_calibration": new_calibration_digest,
            "old_calibration": old_calibration_digest,
            "new_eval": new_eval_digest,
            "old_eval": old_eval_digest,
            "fusion": hashlib.sha256(
                f"{alpha:.8f}\0{score_threshold:.12g}".encode()
            ).hexdigest(),
        },
        "training": False,
        "alpha_and_threshold_selected_on_calibration_only": True,
        "eval_labels_read_before_lock": False,
        "fresh640_read": False,
        "confirm_read": False,
        "sealed_test_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
