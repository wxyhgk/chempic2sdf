#!/usr/bin/env python3
"""Run the frozen P19-E9 pairwise-risk router on a fresh confirm20k."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.correction.output_route_risk_router import (
    ALTERNATIVE_ROUTES,
    OutputRouteRiskRouter,
    OutputRouteRiskRouterConfig,
    ROUTE_OUTCOMES,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e0_direct_atom_pair_oracle_ladder as p19e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e5_old_new_logit_fusion as p19e5,
)
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.correction import (
    train_g3_p19e9_pairwise_route_risk12k as p19e9,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e10-pairwise-route-fixed-confirm20k"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CONFIRM_MANIFEST = OUTPUT / "confirm20000.jsonl"
CONFIRM_CACHE = OUTPUT / "confirm-cache.pt"
SOURCE_CHECKPOINT = p19e9.OUTPUT / "best.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "e48cd0622e82ad38939632011f35cb82190f14f43f5aac2fcf433a8e92b14278"
)
SOURCE_RESULT = p19e9.OUTPUT / "result.json"
SOURCE_RESULT_SHA256 = (
    "6e532bf10b68695c1072949586d5d95297e7bdc0a4afccd252724251941acdb9"
)
P19E9_MANIFESTS = (
    (p19e9.TRAIN_MANIFEST, "6e782d38aebd1b5514fc83b5269a33caeb57c285e4105a189e8ee3fb07cd104b"),
    (p19e9.CALIBRATION_MANIFEST, "5f5cea72912685c0200ef295cebaa65a34d5403f578e587bd92253acd1351de5"),
    (p19e9.HOLDOUT_MANIFEST, "1a764867f950a771785b7f2f12adc02852a6881523188a8b44c32dbdd35e3ab0"),
)
CONFIRM_MOLECULES = 5_000
CONFIRM_IMAGES = CONFIRM_MOLECULES * 4
VARIANTS = p19e9.VARIANTS
EXPECTED_FEATURES = 78
EXPECTED_STEP = 250
EXPECTED_RESCUE_THRESHOLD = 0.5
EXPECTED_HARMFUL_THRESHOLD = 0.02
MIN_ACTIONS = 100
MIN_RESCUES = 50
MIN_RESCUE_PRECISION = 0.35
MAX_HARMFUL_UPPER_95 = 0.02
SEED = 20260810


def select_confirm_rows(
    full_rows: list[dict], excluded_rows: list[dict]
) -> tuple[list[dict], dict]:
    excluded_molecules = {str(row["molecule_id"]) for row in excluded_rows}
    excluded_scaffolds = {
        str(row.get("scaffold", ""))
        for row in excluded_rows
        if row.get("scaffold")
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in full_rows:
        grouped[str(row["molecule_id"])].append(row)
    candidates = []
    for molecule_id, rows in grouped.items():
        scaffolds = {str(row.get("scaffold", "")) for row in rows}
        variants = {str(row["variant"]) for row in rows}
        if len(rows) != len(VARIANTS) or variants != set(VARIANTS):
            raise ValueError(f"P19-E10 incomplete molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P19-E10 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        if molecule_id in excluded_molecules or not scaffold or scaffold in excluded_scaffolds:
            continue
        digest = hashlib.sha256(
            f"{SEED}\0p19e10-confirm\0{molecule_id}\0{scaffold}".encode()
        ).hexdigest()
        candidates.append((digest, molecule_id, scaffold))
    candidates.sort()
    selected = []
    used_scaffolds = set()
    for candidate in candidates:
        if candidate[2] in used_scaffolds:
            continue
        selected.append(candidate)
        used_scaffolds.add(candidate[2])
        if len(selected) == CONFIRM_MOLECULES:
            break
    if len(selected) != CONFIRM_MOLECULES:
        raise ValueError("P19-E10 has too few unique-scaffold molecules")
    selected_molecules = {item[1] for item in selected}
    selected_scaffolds = {item[2] for item in selected}
    rows = [
        row for row in full_rows if str(row["molecule_id"]) in selected_molecules
    ]
    if len(rows) != CONFIRM_IMAGES:
        raise RuntimeError("P19-E10 confirm image count differs")
    audit = {
        "eligible_molecules": len(candidates),
        "eligible_unique_scaffolds": len({item[2] for item in candidates}),
        "images": len(rows),
        "molecules": len(selected_molecules),
        "scaffolds": len(selected_scaffolds),
        "excluded_molecule_overlap": len(selected_molecules & excluded_molecules),
        "excluded_scaffold_overlap": len(selected_scaffolds & excluded_scaffolds),
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{row['sample_key']}\n" for row in rows).encode()
        ).hexdigest(),
    }
    return rows, audit


def clopper_pearson_upper(
    successes: int, trials: int, *, alpha: float = 0.05
) -> float:
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("Invalid binomial counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Alpha must be in (0, 1)")
    if trials == 0 or successes == trials:
        return 1.0

    def binomial_cdf(probability: float) -> float:
        if probability <= 0.0:
            return 1.0
        if probability >= 1.0:
            return 0.0
        term = (1.0 - probability) ** trials
        total = term
        ratio = probability / (1.0 - probability)
        for value in range(1, successes + 1):
            term *= (trials - value + 1) / value * ratio
            total += term
        return total

    lower = 0.0
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if binomial_cdf(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def selected_outcome_counts(actions: torch.Tensor, cache: dict) -> dict:
    counts = {
        route: {outcome: 0 for outcome in ROUTE_OUTCOMES}
        for route in ALTERNATIVE_ROUTES
    }
    by_variant = Counter()
    by_operation = Counter()
    for index, action in enumerate(actions.cpu().tolist()):
        if action == 0:
            continue
        route_index = action - 1
        route = ALTERNATIVE_ROUTES[route_index]
        outcome = ROUTE_OUTCOMES[int(cache["outcomes"][index, route_index])]
        counts[route][outcome] += 1
        by_variant[(cache["variants"][index], route, outcome)] += 1
        operation = cache["stress_operations"][index]
        if operation is not None:
            by_operation[(operation, route, outcome)] += 1
    return {
        "by_route": counts,
        "by_variant": {
            "|".join(map(str, key)): value for key, value in sorted(by_variant.items())
        },
        "by_stress_operation": {
            "|".join(map(str, key)): value
            for key, value in sorted(by_operation.items())
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E10 output is not empty: {OUTPUT}")
    inputs = [
        (p19e9.PILOT_MANIFEST, p19e9.PILOT_MANIFEST_SHA256),
        (p19e9.FULL_MANIFEST, p19e9.FULL_MANIFEST_SHA256),
        (p19e9.OLD_DEV_MANIFEST, p19e9.OLD_DEV_MANIFEST_SHA256),
        (p19e9.USED_CALIBRATION_MANIFEST, p19e9.USED_CALIBRATION_MANIFEST_SHA256),
        (p19e9.USED_EVAL_MANIFEST, p19e9.USED_EVAL_MANIFEST_SHA256),
        *p19e9.P19E8_MANIFESTS,
        *P19E9_MANIFESTS,
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256),
        (SOURCE_RESULT, SOURCE_RESULT_SHA256),
        (p19e9.NEW_CHECKPOINT, p19e9.NEW_CHECKPOINT_SHA256),
        (p19e9.OLD_CHECKPOINT, p19e9.OLD_CHECKPOINT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ]
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E10 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E10 requires CUDA")
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

    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["global_step"] != EXPECTED_STEP:
        raise ValueError("P19-E10 source step differs")
    if checkpoint["risk_rule"] != {
        "rescue_threshold": EXPECTED_RESCUE_THRESHOLD,
        "harmful_threshold": EXPECTED_HARMFUL_THRESHOLD,
    }:
        raise ValueError("P19-E10 source risk rule differs")
    if len(checkpoint["feature_names"]) != EXPECTED_FEATURES:
        raise ValueError("P19-E10 source feature schema differs")
    router = OutputRouteRiskRouter(
        OutputRouteRiskRouterConfig(**checkpoint["config"])
    ).to(device)
    router.load_state_dict(checkpoint["router"], strict=True)
    router.eval().requires_grad_(False)
    router_state_before = module_state_sha256(router)

    full_rows = p18e0._jsonl(p19e9.FULL_MANIFEST)
    excluded_rows = []
    for path in (
        p19e9.PILOT_MANIFEST,
        p19e9.OLD_DEV_MANIFEST,
        p19e9.USED_CALIBRATION_MANIFEST,
        p19e9.USED_EVAL_MANIFEST,
        *(item[0] for item in p19e9.P19E8_MANIFESTS),
        *(item[0] for item in P19E9_MANIFESTS),
    ):
        excluded_rows.extend(p18e0._jsonl(path))
    confirm_rows, split_audit = select_confirm_rows(full_rows, excluded_rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    CONFIRM_MANIFEST.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in confirm_rows
        ),
        encoding="utf-8",
    )
    split_audit["manifest_sha256"] = file_sha256(CONFIRM_MANIFEST)
    _atomic_json(OUTPUT / "split-audit.json", split_audit)

    vision, atom, bond, set_head, _ = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    new_model = p19e5._load_model(p19e9.NEW_CHECKPOINT, device)
    old_model = p19e5._load_model(p19e9.OLD_CHECKPOINT, device)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
        "router": router_state_before,
    }
    confirm, feature_names = p19e9.build_split_cache(
        confirm_rows, new_model, old_model, vision, atom, device
    )
    if feature_names != checkpoint["feature_names"]:
        raise RuntimeError("P19-E10 confirm feature schema differs")
    _atomic_torch_save(
        CONFIRM_CACHE,
        {
            "schema_version": "g3-p19e10-pairwise-route-fixed-confirm-cache-0.1",
            "feature_names": feature_names,
            "split": confirm,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "fixed_risk_rule": checkpoint["risk_rule"],
        },
    )
    confirm_cache_sha256 = file_sha256(CONFIRM_CACHE)
    normalized = (
        (confirm["features"] - checkpoint["feature_mean"])
        / checkpoint["feature_std"]
    ).to(device)
    with torch.inference_mode():
        logits = router(normalized)
        actions = p19e9.actions_from_risk_logits(
            logits,
            EXPECTED_RESCUE_THRESHOLD,
            EXPECTED_HARMFUL_THRESHOLD,
        )
    records = p19e9.tensor_records(confirm)
    metrics = p19e9.evaluate_risk_actions(actions, records)
    outcomes = selected_outcome_counts(actions, confirm)
    overall = metrics["overall"]
    action_count = CONFIRM_IMAGES - int(overall["actions"].get("KEEP_FUSION", 0))
    rescue_count = int(overall["rescue"])
    harmful_count = int(overall["harmful"])
    rescue_precision = rescue_count / action_count if action_count else 0.0
    harmful_upper_95 = clopper_pearson_upper(harmful_count, action_count)
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
        "router": module_state_sha256(router),
    }
    checks = {
        "confirm_image_count_exact": len(confirm_rows) == CONFIRM_IMAGES,
        "confirm_unique_scaffold_count_exact": split_audit["scaffolds"] == CONFIRM_MOLECULES,
        "all_prior_overlap_zero": (
            split_audit["excluded_molecule_overlap"] == 0
            and split_audit["excluded_scaffold_overlap"] == 0
        ),
        "source_step_and_rule_locked": (
            checkpoint["global_step"] == EXPECTED_STEP
            and checkpoint["risk_rule"]
            == {
                "rescue_threshold": EXPECTED_RESCUE_THRESHOLD,
                "harmful_threshold": EXPECTED_HARMFUL_THRESHOLD,
            }
        ),
        "action_count_min100": action_count >= MIN_ACTIONS,
        "rescue_min50": rescue_count >= MIN_RESCUES,
        "rescue_precision_min35pct": rescue_precision >= MIN_RESCUE_PRECISION,
        "zero_harmful": harmful_count == 0,
        "harmful_upper_95_max2pct": harmful_upper_95 <= MAX_HARMFUL_UPPER_95,
        "selected_exact_improves_fusion": overall["selected_exact"] > overall["fusion_exact"],
        "all_variants_zero_harmful": all(
            values["harmful"] == 0 for values in metrics["by_variant"].values()
        ),
        "all_stress_operations_zero_harmful": all(
            values["harmful"] == 0
            for values in metrics["stress_by_operation"].values()
        ),
        "frozen_sources_unchanged": states_before == states_after,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    decision = (
        "pairwise_route_fixed_confirm20k_go"
        if all(checks.values())
        else "pairwise_route_fixed_confirm20k_no_go"
    )
    result = {
        "schema_version": "g3-p19e10-pairwise-route-fixed-confirm20k-result-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "fixed_step": EXPECTED_STEP,
        "fixed_risk_rule": checkpoint["risk_rule"],
        "metrics": metrics,
        "selected_outcomes": outcomes,
        "action_count": action_count,
        "rescue_count": rescue_count,
        "harmful_count": harmful_count,
        "rescue_precision": rescue_precision,
        "harmful_clopper_pearson_upper_95": harmful_upper_95,
        "confirm_cache": str(CONFIRM_CACHE.resolve()),
        "confirm_cache_sha256": confirm_cache_sha256,
        "split": split_audit,
        "model_or_rule_selection_on_confirm": False,
        "fresh640_read": False,
        "sealed_test_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
