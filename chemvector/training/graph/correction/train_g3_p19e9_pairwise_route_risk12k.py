#!/usr/bin/env python3
"""Train P19-E9 pairwise route outcome risks on isolated 8k/2k/2k images."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

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
    audit_g3_p19e4_presence_threshold_calibration as p19e4,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e5_old_new_logit_fusion as p19e5,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p4e3_global_peak_rank_pilot20k as p4e3
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.correction import (
    train_g3_p19e7_output_route_capacity640 as p19e7,
)
from chemvector.training.graph.correction import (
    train_g3_p19e8_output_route_train4k as p19e8,
)
from chemvector.training.graph.relations import (
    train_g3_p19e3_direct_atom_pair_presence_pilot20k as p19e3,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p19e9-pairwise-route-risk12k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = OUTPUT / "risk-cache.pt"
HOLDOUT_CACHE = OUTPUT / "holdout-cache.pt"
TRAIN_MANIFEST = OUTPUT / "train8000.jsonl"
CALIBRATION_MANIFEST = OUTPUT / "calibration2000.jsonl"
HOLDOUT_MANIFEST = OUTPUT / "holdout2000.jsonl"
PILOT_MANIFEST = p4e3.P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = p4e3.PILOT_MANIFEST_SHA256
FULL_MANIFEST = p4e3.P4E0_ROOT / "full_train.jsonl"
FULL_MANIFEST_SHA256 = p19e3.FULL_MANIFEST_SHA256
OLD_DEV_MANIFEST = p19e3.DEV_MANIFEST
OLD_DEV_MANIFEST_SHA256 = p19e4.OLD_DEV_MANIFEST_SHA256
USED_CALIBRATION_MANIFEST = p19e4.CALIBRATION_MANIFEST
USED_CALIBRATION_MANIFEST_SHA256 = p19e5.CALIBRATION_MANIFEST_SHA256
USED_EVAL_MANIFEST = p19e4.EVAL_MANIFEST
USED_EVAL_MANIFEST_SHA256 = p19e5.EVAL_MANIFEST_SHA256
P19E8_MANIFESTS = (
    (p19e8.TRAIN_MANIFEST, "4566341a4b3c64f69f0252076277b4687f5f2b2f7aa1236c5648bcb8ef2206fd"),
    (p19e8.CALIBRATION_MANIFEST, "18749ac149257e5d46d2856279a376619fcddd7048a2489e5083b5eef1342f7f"),
    (p19e8.HOLDOUT_MANIFEST, "dac61ca15e1b02fbdc709a116698e0d0355894913d5381185cff8281bc91fd6d"),
)
NEW_CHECKPOINT = p19e5.NEW_CHECKPOINT
NEW_CHECKPOINT_SHA256 = p19e5.NEW_CHECKPOINT_SHA256
OLD_CHECKPOINT = p19e5.OLD_CHECKPOINT
OLD_CHECKPOINT_SHA256 = p19e5.OLD_CHECKPOINT_SHA256
TRAIN_MOLECULES = 2_000
CALIBRATION_MOLECULES = 500
HOLDOUT_MOLECULES = 500
TRAIN_IMAGES = TRAIN_MOLECULES * 4
CALIBRATION_IMAGES = CALIBRATION_MOLECULES * 4
HOLDOUT_IMAGES = HOLDOUT_MOLECULES * 4
VARIANTS = ("coordgen", "original", "rdkit", "stress")
STEPS = 3_000
MINIBATCH_SIZE = 256
EVALUATE_EVERY = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 20260810
RESCUE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
HARMFUL_THRESHOLDS = (0.20, 0.10, 0.05, 0.02, 0.01)


def select_risk_splits(
    full_rows: list[dict], excluded_rows: list[dict]
) -> tuple[dict[str, list[dict]], dict]:
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
            raise ValueError(f"P19-E9 incomplete molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P19-E9 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        if molecule_id in excluded_molecules or not scaffold or scaffold in excluded_scaffolds:
            continue
        digest = hashlib.sha256(
            f"{SEED}\0p19e9-risk\0{molecule_id}\0{scaffold}".encode()
        ).hexdigest()
        candidates.append((digest, molecule_id, scaffold))
    candidates.sort()
    required = TRAIN_MOLECULES + CALIBRATION_MOLECULES + HOLDOUT_MOLECULES
    selected = []
    used_scaffolds = set()
    for candidate in candidates:
        if candidate[2] in used_scaffolds:
            continue
        selected.append(candidate)
        used_scaffolds.add(candidate[2])
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError("P19-E9 has too few unique-scaffold molecules")
    boundaries = {
        "train": (0, TRAIN_MOLECULES),
        "calibration": (TRAIN_MOLECULES, TRAIN_MOLECULES + CALIBRATION_MOLECULES),
        "holdout": (TRAIN_MOLECULES + CALIBRATION_MOLECULES, required),
    }
    split_molecules = {
        name: {item[1] for item in selected[start:end]}
        for name, (start, end) in boundaries.items()
    }
    split_scaffolds = {
        name: {item[2] for item in selected[start:end]}
        for name, (start, end) in boundaries.items()
    }
    splits = {
        name: [row for row in full_rows if str(row["molecule_id"]) in molecules]
        for name, molecules in split_molecules.items()
    }
    expected = {
        "train": TRAIN_IMAGES,
        "calibration": CALIBRATION_IMAGES,
        "holdout": HOLDOUT_IMAGES,
    }
    if any(len(splits[name]) != expected[name] for name in splits):
        raise RuntimeError("P19-E9 split image count differs")
    audit = {
        "eligible_molecules": len(candidates),
        "images": {name: len(rows) for name, rows in splits.items()},
        "molecules": {name: len(values) for name, values in split_molecules.items()},
        "scaffolds": {name: len(values) for name, values in split_scaffolds.items()},
        "excluded_molecule_overlap": {
            name: len(values & excluded_molecules)
            for name, values in split_molecules.items()
        },
        "excluded_scaffold_overlap": {
            name: len(values & excluded_scaffolds)
            for name, values in split_scaffolds.items()
        },
        "cross_split_molecule_overlap": {
            "train_calibration": len(split_molecules["train"] & split_molecules["calibration"]),
            "train_holdout": len(split_molecules["train"] & split_molecules["holdout"]),
            "calibration_holdout": len(split_molecules["calibration"] & split_molecules["holdout"]),
        },
        "cross_split_scaffold_overlap": {
            "train_calibration": len(split_scaffolds["train"] & split_scaffolds["calibration"]),
            "train_holdout": len(split_scaffolds["train"] & split_scaffolds["holdout"]),
            "calibration_holdout": len(split_scaffolds["calibration"] & split_scaffolds["holdout"]),
        },
        "sample_keys_sha256": {
            name: hashlib.sha256(
                "".join(f"{row['sample_key']}\n" for row in rows).encode()
            ).hexdigest()
            for name, rows in splits.items()
        },
    }
    return splits, audit


def route_outcome_labels(route_exact: torch.Tensor) -> torch.Tensor:
    if route_exact.ndim != 2 or route_exact.size(1) != 3:
        raise ValueError("P19-E9 route exact shape differs")
    fusion = route_exact[:, 0].bool()
    outcomes = []
    for route_index in (1, 2):
        alternative = route_exact[:, route_index].bool()
        labels = torch.zeros(
            len(route_exact), dtype=torch.long, device=route_exact.device
        )
        labels[fusion & alternative] = 1
        labels[~fusion & alternative] = 2
        labels[fusion & ~alternative] = 3
        outcomes.append(labels)
    return torch.stack(outcomes, dim=1)


def records_to_tensors(records: list[dict]) -> dict:
    route_exact = torch.stack([record["route_exact"] for record in records])
    return {
        "features": torch.stack([record["features"] for record in records]),
        "route_exact": route_exact,
        "outcomes": route_outcome_labels(route_exact),
        "observable_disagreement": torch.tensor(
            [record["routes_observably_disagree"] for record in records]
        ),
        "sample_keys": [record["sample_key"] for record in records],
        "variants": [record["variant"] for record in records],
        "stress_operations": [record["stress_operation"] for record in records],
    }


def tensor_records(cache: dict) -> list[dict]:
    records = []
    for index in range(len(cache["route_exact"])):
        route_exact = cache["route_exact"][index]
        records.append(
            {
                "sample_key": cache["sample_keys"][index],
                "variant": cache["variants"][index],
                "stress_operation": cache["stress_operations"][index],
                "route_exact": route_exact,
                "target_action": p19e7._route_label(route_exact),
                "routes_observably_disagree": bool(
                    cache["observable_disagreement"][index]
                ),
            }
        )
    return records


def actions_from_risk_logits(
    logits: torch.Tensor, rescue_threshold: float, harmful_threshold: float
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (2, 4):
        raise ValueError("P19-E9 risk logits shape differs")
    probabilities = logits.softmax(-1)
    rescue = probabilities[:, :, 2]
    harmful = probabilities[:, :, 3]
    eligible = (rescue >= rescue_threshold) & (harmful <= harmful_threshold)
    utility = (rescue - harmful).masked_fill(~eligible, float("-inf"))
    best_utility, best_route = utility.max(dim=1)
    return torch.where(
        torch.isfinite(best_utility), best_route + 1, torch.zeros_like(best_route)
    )


def evaluate_risk_actions(actions: torch.Tensor, records: list[dict]) -> dict:
    return p19e8.evaluate_actions(actions.cpu(), records)


def rule_key(metrics: dict, rescue_threshold: float, harmful_threshold: float) -> tuple:
    overall = metrics["overall"]
    non_keep = len(metrics.get("action_rows", [])) or (
        overall["images"] - int(overall["actions"].get("KEEP_FUSION", 0))
    )
    return (
        -int(overall["harmful"]),
        int(overall["selected_exact"]),
        int(overall["rescue"]),
        -int(non_keep),
        float(rescue_threshold),
        -float(harmful_threshold),
    )


@torch.inference_mode()
def calibrate_risk_rule(
    router: OutputRouteRiskRouter,
    normalized_features: torch.Tensor,
    records: list[dict],
) -> tuple[dict, dict]:
    router.eval()
    logits = router(normalized_features)
    best_rule = None
    best_metrics = None
    best_key = None
    for rescue_threshold in RESCUE_THRESHOLDS:
        for harmful_threshold in HARMFUL_THRESHOLDS:
            actions = actions_from_risk_logits(
                logits, rescue_threshold, harmful_threshold
            )
            metrics = evaluate_risk_actions(actions, records)
            key = rule_key(metrics, rescue_threshold, harmful_threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_rule = {
                    "rescue_threshold": rescue_threshold,
                    "harmful_threshold": harmful_threshold,
                }
                best_metrics = metrics
    if best_rule is None or best_metrics is None:
        raise RuntimeError("P19-E9 risk calibration failed")
    return best_rule, best_metrics


def build_split_cache(
    rows: list[dict], new_model, old_model, vision, atom, device: torch.device
) -> tuple[dict, list[str]]:
    samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(rows), accepted_split="train"
        )
    )
    new_records, _ = p19e4.collect_probabilities(
        new_model, vision, atom, samples, device
    )
    old_records, _ = p19e4.collect_probabilities(
        old_model, vision, atom, samples, device
    )
    records, feature_names = p19e7.build_capacity_records(
        new_records, old_records
    )
    return records_to_tensors(records), feature_names


def checkpoint_payload(
    router: OutputRouteRiskRouter,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    rule: dict,
    calibration_metrics: dict,
    feature_names: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    cache_sha256: str,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p19e9-pairwise-route-risk-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(router.config),
        "router": router.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "global_step": step,
        "risk_rule": rule,
        "calibration_metrics": calibration_metrics,
        "cache_sha256": cache_sha256,
        "training_args": {
            "steps": STEPS,
            "minibatch_size": MINIBATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "rescue_thresholds": RESCUE_THRESHOLDS,
            "harmful_thresholds": HARMFUL_THRESHOLDS,
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E9 output is not empty: {OUTPUT}")
    inputs = [
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256),
        (OLD_DEV_MANIFEST, OLD_DEV_MANIFEST_SHA256),
        (USED_CALIBRATION_MANIFEST, USED_CALIBRATION_MANIFEST_SHA256),
        (USED_EVAL_MANIFEST, USED_EVAL_MANIFEST_SHA256),
        *P19E8_MANIFESTS,
        (NEW_CHECKPOINT, NEW_CHECKPOINT_SHA256),
        (OLD_CHECKPOINT, OLD_CHECKPOINT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ]
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E9 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E9 requires CUDA")
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

    full_rows = p18e0._jsonl(FULL_MANIFEST)
    excluded_rows = []
    for path in (
        PILOT_MANIFEST,
        OLD_DEV_MANIFEST,
        USED_CALIBRATION_MANIFEST,
        USED_EVAL_MANIFEST,
        *(item[0] for item in P19E8_MANIFESTS),
    ):
        excluded_rows.extend(p18e0._jsonl(path))
    splits, split_audit = select_risk_splits(full_rows, excluded_rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, path in (
        ("train", TRAIN_MANIFEST),
        ("calibration", CALIBRATION_MANIFEST),
        ("holdout", HOLDOUT_MANIFEST),
    ):
        path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in splits[name]
            ),
            encoding="utf-8",
        )
    split_audit["manifest_sha256"] = {
        "train": file_sha256(TRAIN_MANIFEST),
        "calibration": file_sha256(CALIBRATION_MANIFEST),
        "holdout": file_sha256(HOLDOUT_MANIFEST),
    }
    _atomic_json(OUTPUT / "split-audit.json", split_audit)

    vision, atom, bond, set_head, _ = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    new_model = p19e5._load_model(NEW_CHECKPOINT, device)
    old_model = p19e5._load_model(OLD_CHECKPOINT, device)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }
    caches = {}
    feature_names = None
    for split_name in ("train", "calibration"):
        cache, names = build_split_cache(
            splits[split_name], new_model, old_model, vision, atom, device
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("P19-E9 feature schema differs across splits")
        caches[split_name] = cache
    if feature_names is None or len(feature_names) != 78:
        raise RuntimeError("P19-E9 expected the frozen 78-feature schema")
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p19e9-pairwise-route-risk-cache-0.1",
            "feature_names": feature_names,
            "splits": caches,
            "split_audit": split_audit,
        },
    )
    cache_sha256 = file_sha256(CACHE)
    states_after_cache = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }

    train = caches["train"]
    calibration = caches["calibration"]
    train_records = tensor_records(train)
    calibration_records = tensor_records(calibration)
    mean = train["features"].mean(0)
    std = train["features"].std(0, unbiased=False).clamp_min(1e-4)
    normalized_train = ((train["features"] - mean) / std).to(device)
    normalized_calibration = ((calibration["features"] - mean) / std).to(device)
    train_outcomes = train["outcomes"].to(device)
    class_counts = torch.stack(
        [
            torch.bincount(train["outcomes"][:, route], minlength=4)
            for route in range(2)
        ]
    ).float()
    class_weights = (class_counts.sum(1, keepdim=True) / class_counts.clamp_min(1.0)).sqrt()
    class_weights = (class_weights / class_weights.mean(1, keepdim=True)).to(device)
    router = OutputRouteRiskRouter(
        OutputRouteRiskRouterConfig(input_dim=len(feature_names), hidden_dim=128)
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial_rule, initial_metrics = calibrate_risk_rule(
        router, normalized_calibration, calibration_records
    )
    best_key = rule_key(
        initial_metrics,
        initial_rule["rescue_threshold"],
        initial_rule["harmful_threshold"],
    )
    best_metrics = initial_metrics
    _atomic_torch_save(
        OUTPUT / "best.pt",
        checkpoint_payload(
            router,
            optimizer,
            step=0,
            rule=initial_rule,
            calibration_metrics=initial_metrics,
            feature_names=feature_names,
            mean=mean,
            std=std,
            cache_sha256=cache_sha256,
        ),
    )
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(train_outcomes), generator=generator).tolist()
    offset = 0
    running = 0.0
    history = []
    for step in range(1, STEPS + 1):
        if offset + MINIBATCH_SIZE > len(order):
            order = torch.randperm(len(train_outcomes), generator=generator).tolist()
            offset = 0
        indices = torch.tensor(order[offset : offset + MINIBATCH_SIZE], device=device)
        offset += MINIBATCH_SIZE
        router.train()
        optimizer.zero_grad(set_to_none=True)
        logits = router(normalized_train[indices])
        losses = [
            F.cross_entropy(
                logits[:, route].float(),
                train_outcomes[indices, route],
                weight=class_weights[route],
            )
            for route in range(2)
        ]
        loss = sum(losses) / len(losses)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P19-E9 gradient is not finite")
        optimizer.step()
        running += float(loss.detach())
        if step % EVALUATE_EVERY == 0:
            rule, metrics = calibrate_risk_rule(
                router, normalized_calibration, calibration_records
            )
            key = rule_key(
                metrics, rule["rescue_threshold"], rule["harmful_threshold"]
            )
            selected = key > best_key
            if selected:
                best_key = key
                best_metrics = metrics
                _atomic_torch_save(
                    OUTPUT / "best.pt",
                    checkpoint_payload(
                        router,
                        optimizer,
                        step=step,
                        rule=rule,
                        calibration_metrics=metrics,
                        feature_names=feature_names,
                        mean=mean,
                        std=std,
                        cache_sha256=cache_sha256,
                    ),
                )
            history.append(
                {
                    "step": step,
                    "selected_best": selected,
                    "loss": running / EVALUATE_EVERY,
                    "risk_rule": rule,
                    "calibration": metrics,
                }
            )
            running = 0.0

    best_payload = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    best_router = OutputRouteRiskRouter(
        OutputRouteRiskRouterConfig(**best_payload["config"])
    ).to(device)
    best_router.load_state_dict(best_payload["router"], strict=True)
    best_router.eval()

    holdout, holdout_feature_names = build_split_cache(
        splits["holdout"], new_model, old_model, vision, atom, device
    )
    if holdout_feature_names != feature_names:
        raise RuntimeError("P19-E9 holdout feature schema differs")
    _atomic_torch_save(
        HOLDOUT_CACHE,
        {
            "schema_version": "g3-p19e9-pairwise-route-risk-holdout-cache-0.1",
            "feature_names": feature_names,
            "split": holdout,
            "checkpoint_sha256_at_first_read": file_sha256(OUTPUT / "best.pt"),
            "risk_rule_at_first_read": best_payload["risk_rule"],
        },
    )
    holdout_cache_sha256 = file_sha256(HOLDOUT_CACHE)
    holdout_records = tensor_records(holdout)
    normalized_holdout = ((holdout["features"] - mean) / std).to(device)
    with torch.inference_mode():
        holdout_logits = best_router(normalized_holdout)
        holdout_actions = actions_from_risk_logits(
            holdout_logits,
            best_payload["risk_rule"]["rescue_threshold"],
            best_payload["risk_rule"]["harmful_threshold"],
        )
    holdout_metrics = evaluate_risk_actions(holdout_actions, holdout_records)
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }
    counts_json = {
        ALTERNATIVE_ROUTES[route]: {
            ROUTE_OUTCOMES[outcome]: int(class_counts[route, outcome])
            for outcome in range(4)
        }
        for route in range(2)
    }
    checks = {
        "split_image_counts_exact": split_audit["images"]
        == {
            "train": TRAIN_IMAGES,
            "calibration": CALIBRATION_IMAGES,
            "holdout": HOLDOUT_IMAGES,
        },
        "all_split_overlaps_zero": not any(
            value
            for group in (
                split_audit["excluded_molecule_overlap"],
                split_audit["excluded_scaffold_overlap"],
                split_audit["cross_split_molecule_overlap"],
                split_audit["cross_split_scaffold_overlap"],
            )
            for value in group.values()
        ),
        "each_route_rescue_and_harmful_min20": all(
            class_counts[route, outcome] >= 20
            for route in range(2)
            for outcome in (2, 3)
        ),
        "calibration_zero_harmful": best_metrics["overall"]["harmful"] == 0,
        "calibration_rescue_min5": best_metrics["overall"]["rescue"] >= 5,
        "holdout_zero_harmful": holdout_metrics["overall"]["harmful"] == 0,
        "holdout_rescue_min5": holdout_metrics["overall"]["rescue"] >= 5,
        "holdout_exact_improves_fusion": (
            holdout_metrics["overall"]["selected_exact"]
            > holdout_metrics["overall"]["fusion_exact"]
        ),
        "frozen_sources_unchanged": states_before == states_after_cache == states_after,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    decision = (
        "pairwise_route_risk12k_go"
        if all(checks.values())
        else "pairwise_route_risk12k_no_go"
    )
    result = {
        "schema_version": "g3-p19e9-pairwise-route-risk12k-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "train_outcome_counts": counts_json,
        "best_step": int(best_payload["global_step"]),
        "risk_rule": best_payload["risk_rule"],
        "calibration": best_metrics,
        "holdout": holdout_metrics,
        "history": history,
        "feature_dim": len(feature_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in best_router.parameters()
        ),
        "best_checkpoint": str((OUTPUT / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(OUTPUT / "best.pt"),
        "cache": str(CACHE.resolve()),
        "cache_sha256": cache_sha256,
        "holdout_cache": str(HOLDOUT_CACHE.resolve()),
        "holdout_cache_sha256": holdout_cache_sha256,
        "split": split_audit,
        "holdout_read_once_after_checkpoint_and_rule_lock": True,
        "p19e8_holdout_used_only_for_exclusion": True,
        "fresh640_read": False,
        "confirm_read": False,
        "sealed_test_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
