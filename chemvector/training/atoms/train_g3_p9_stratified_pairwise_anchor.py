#!/usr/bin/env python3
"""Train P9 with strict role/style/degree/distance pair balancing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
)
from chemvector.evaluation.audits.atoms import build_g3_p7e0_pairwise_counterfactual_data as p7e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p8_pairwise_consensus as p8
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save


MODEL_TYPE = "chemvector-g3-p9e1-stratified-pairwise-anchor-train9920"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-008000.pt"
MODEL_SEED = 2026089301
SCHEDULE_SEED = 2026089311
STEPS = 8_000
MINIMUM_PAIR_DISTANCE_PIXELS = 8.0
EXPECTED_PER_CLASS = 4_960
EQUIVALENT_POOL = 2_048
MINIMUM_FULL_RESCUES = 80
MINIMUM_CROSS_CERTIFICATE_RESCUES = 60


def _distance_band(pair_distance_bucket: int) -> int | None:
    if pair_distance_bucket < 2:
        return None
    return 0 if pair_distance_bucket < 4 else 1


def _stratum(pairs: dict, index: int) -> tuple[str, str, str, int | None]:
    return (
        str(pairs["target_role"][index]),
        str(pairs["variant"][index]),
        str(pairs["degree_bucket"][index]),
        _distance_band(int(pairs["pair_distance_bucket"][index])),
    )


def _rank(index: int, pairs: dict, label: str) -> str:
    return hashlib.sha256(
        f"p9:{label}:{pairs['scaffold'][index]}:{int(pairs['source_row'][index])}:{int(pairs['alternative_index'][index])}".encode()
    ).hexdigest()


def select_stratified_pairs(pairs: dict, sources: tuple[dict, dict]) -> tuple[dict[str, torch.Tensor], dict]:
    rescue_by_stratum: dict[tuple, list[int]] = defaultdict(list)
    harmful_by_stratum: dict[tuple, list[int]] = defaultdict(list)
    for index in torch.where(pairs["label"] == PAIR_RESCUE)[0].tolist():
        stratum = _stratum(pairs, index)
        if stratum[-1] is None:
            continue
        kind = int(pairs["source_kind"][index])
        margin = p7._base_margin_for_pair(
            sources[kind],
            int(pairs["source_row"][index]),
            int(pairs["prior_index"][index]),
            int(pairs["alternative_index"][index]),
        )
        if margin >= -1.95:
            rescue_by_stratum[stratum].append(index)
    for index in torch.where(pairs["label"] == PAIR_HARMFUL)[0].tolist():
        stratum = _stratum(pairs, index)
        if stratum[-1] is not None:
            harmful_by_stratum[stratum].append(index)
    rescue = []
    harmful = []
    strata = {}
    for stratum in sorted(rescue_by_stratum):
        positive = sorted(rescue_by_stratum[stratum], key=lambda index: _rank(index, pairs, "rescue"))
        negative = sorted(harmful_by_stratum[stratum], key=lambda index: _rank(index, pairs, "harmful"))
        count = min(len(positive), len(negative))
        rescue.extend(positive[:count])
        harmful.extend(negative[:count])
        strata[str(stratum)] = {
            "rescue_available": len(positive),
            "harmful_available": len(negative),
            "selected_each": count,
        }
    equivalent = torch.where(pairs["label"] == PAIR_EQUIVALENT)[0].tolist()
    equivalent.sort(key=lambda index: _rank(index, pairs, "equivalent"))
    pools = {
        "rescue": torch.tensor(rescue, dtype=torch.long),
        "harmful": torch.tensor(harmful, dtype=torch.long),
        "equivalent": torch.tensor(equivalent[:EQUIVALENT_POOL], dtype=torch.long),
    }
    failure_stratum = str(("TEXT", "stress", "degree3plus", 0))
    audit = {
        "rescue": len(rescue),
        "harmful": len(harmful),
        "equivalent": len(pools["equivalent"]),
        "strata": len(strata),
        "failure_stratum": strata[failure_stratum],
        "minimum_pair_distance_pixels": MINIMUM_PAIR_DISTANCE_PIXELS,
    }
    return pools, audit


def _calibration_records(model, sources: tuple[dict, dict], device: torch.device) -> list[dict]:
    records = p7.evaluate_source_cache(model, sources[0], "hard", device)
    records += p7.evaluate_source_cache(model, sources[1], "easy", device)
    for row in records:
        cache = sources[0] if row["source"] == "hard" else sources[1]
        row["scaffold"] = str(cache["scaffold"][row["row"]])
    return records


def cross_certificate(records: list[dict]) -> dict:
    folds = []
    totals = {"executed": 0, "rescues": 0, "harmful": 0, "useless": 0}
    for fold in range(p8.FOLDS):
        selection = [row for row in records if p8.certificate_fold(row["scaffold"]) != fold]
        certificate = [row for row in records if p8.certificate_fold(row["scaffold"]) == fold]
        threshold = p7.calibrate_zero_harmful(selection)
        metrics = p8.fixed_threshold_metrics(certificate, threshold["threshold"])
        folds.append({"fold": fold, "selection": threshold, "certificate": metrics, "rows": len(certificate)})
        for name in totals:
            totals[name] += int(metrics[name])
    return {"folds": folds, "totals": totals}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P9 output already exists: {OUTPUT}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P9 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P9 requires CUDA")
    device = torch.device("cuda")
    pairs = torch.load(p7e0.TRAIN_PAIRS, map_location="cpu", weights_only=False)
    sources = p7._load_sources()
    pools, pool_audit = select_stratified_pairs(pairs, sources["train"])
    if len(pools["rescue"]) != EXPECTED_PER_CLASS or len(pools["harmful"]) != EXPECTED_PER_CLASS:
        raise RuntimeError(f"P9 stratified pool count differs: {pool_audit}")
    if pool_audit["failure_stratum"]["selected_each"] != 83:
        raise RuntimeError("P9 text-anchor failure stratum is not balanced 83:83")
    model, training = p7._train_model(
        "patch-only",
        pairs,
        sources["train"],
        pools,
        steps=STEPS,
        seed=SCHEDULE_SEED,
        model_seed=MODEL_SEED,
        device=device,
    )
    OUTPUT.mkdir(parents=True)
    records = _calibration_records(model, sources["calibration"], device)
    full = p7.calibrate_zero_harmful(records)
    cross = cross_certificate(records)
    _atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p9-stratified-pairwise-anchor-0.1",
            "model_type": MODEL_TYPE,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "global_step": STEPS,
            "model_seed": MODEL_SEED,
            "schedule_seed": SCHEDULE_SEED,
            "calibration_threshold": full["threshold"],
        },
    )
    checks = {
        "stratified_4960_each": len(pools["rescue"]) == len(pools["harmful"]) == EXPECTED_PER_CLASS,
        "failure_stratum_83_each": pool_audit["failure_stratum"]["selected_each"] == 83,
        "full_zero_harmful": full["harmful"] == 0,
        "full_rescues_min80": full["rescues"] >= MINIMUM_FULL_RESCUES,
        "cross_certificate_zero_harmful": cross["totals"]["harmful"] == 0,
        "cross_certificate_rescues_min60": cross["totals"]["rescues"] >= MINIMUM_CROSS_CERTIFICATE_RESCUES,
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    audit = {
        "schema_version": "g3-p9-stratified-pairwise-anchor-audit-0.1",
        "decision": "stratified_pairwise_anchor_certificate_go" if all(checks.values()) else "stratified_pairwise_anchor_certificate_no_go",
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "pool": pool_audit,
        "training": training,
        "full_calibration": full,
        "cross_certificate": cross,
        "protocol": {
            "stratum": ["target_role", "variant", "degree_bucket", "distance_8_16_or_16plus"],
            "unique_pairs": True,
            "historical_p7_holdout_read": False,
            "dev_confirm_external_sealed_read": False,
        },
        "historical_p7_holdout_read": False,
        "dev_confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
