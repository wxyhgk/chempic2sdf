#!/usr/bin/env python3
"""Capacity-test an observable-feature router on P19 calibration640."""

from __future__ import annotations

from collections import Counter
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

from chemvector.components.graph.correction.output_route_router import (
    OutputRouteRouter,
    OutputRouteRouterConfig,
    ROUTE_ACTIONS,
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
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e6_three_route_oracle as p19e6,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p19e7-output-route-capacity640"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = OUTPUT / "capacity-cache.pt"
CALIBRATION_MANIFEST = p19e4.CALIBRATION_MANIFEST
CALIBRATION_MANIFEST_SHA256 = p19e5.CALIBRATION_MANIFEST_SHA256
NEW_CHECKPOINT = p19e5.NEW_CHECKPOINT
NEW_CHECKPOINT_SHA256 = p19e5.NEW_CHECKPOINT_SHA256
OLD_CHECKPOINT = p19e5.OLD_CHECKPOINT
OLD_CHECKPOINT_SHA256 = p19e5.OLD_CHECKPOINT_SHA256
P19E5_AUDIT = p19e5.OUTPUT / "audit.json"
P19E5_AUDIT_SHA256 = "41fc704c8ed96d0e994ce85fba841787f9e62420eb45788371608b783a9e5592"
P19E6_AUDIT = p19e6.OUTPUT / "audit.json"
P19E6_AUDIT_SHA256 = "e77828f9e64e28b2800a7ea62ba1bfc267c9ed6094cd8e5fa018e04c344a4069"
ALPHA_NEW = 0.55
FUSION_THRESHOLD = -0.7528127431869507
NEW_THRESHOLD = p19e5.NEW_REFERENCE_THRESHOLD
OLD_THRESHOLD = p19e5.OLD_REFERENCE_THRESHOLD
STEPS = 5_000
EVALUATE_EVERY = 250
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
SEED = 20260810


def infer_atom_count(pair_count: int) -> int:
    if pair_count < 1:
        raise ValueError("P19-E7 pair count must be positive")
    atom_count = int((1.0 + (1.0 + 8.0 * pair_count) ** 0.5) / 2.0)
    if atom_count * (atom_count - 1) // 2 != pair_count:
        raise ValueError("P19-E7 pair count is not nC2")
    return atom_count


def _margin_statistics(name: str, values: torch.Tensor) -> tuple[list[str], list[float]]:
    quantiles = torch.quantile(
        values.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    )
    names = [
        f"{name}_mean",
        f"{name}_std",
        f"{name}_min",
        f"{name}_max",
        f"{name}_q10",
        f"{name}_q25",
        f"{name}_q50",
        f"{name}_q75",
        f"{name}_q90",
    ]
    values_out = [
        float(values.mean()),
        float(values.std(unbiased=False)),
        float(values.min()),
        float(values.max()),
        *[float(value) for value in quantiles],
    ]
    for radius in (0.25, 0.5, 1.0, 2.0):
        names.append(f"{name}_within_{radius:g}")
        values_out.append(float((values.abs() <= radius).sum()))
    return names, values_out


def _graph_statistics(
    name: str, selected: torch.Tensor, atom_count: int
) -> tuple[list[str], list[float]]:
    pairs = torch.combinations(torch.arange(atom_count), r=2)
    if len(pairs) != len(selected):
        raise ValueError("P19-E7 graph pair universe differs")
    edges = pairs[selected]
    degree = torch.zeros(atom_count, dtype=torch.float32)
    adjacency = [set() for _ in range(atom_count)]
    for begin, end in edges.tolist():
        degree[begin] += 1.0
        degree[end] += 1.0
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    seen = set()
    components = 0
    for atom_index in range(atom_count):
        if atom_index in seen:
            continue
        components += 1
        stack = [atom_index]
        seen.add(atom_index)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    edge_count = int(selected.sum())
    names = [
        f"{name}_edges",
        f"{name}_degree_mean",
        f"{name}_degree_std",
        f"{name}_degree_max",
        f"{name}_isolated",
        f"{name}_degree_gt4",
        f"{name}_components",
        f"{name}_cycle_rank",
    ]
    values = [
        float(edge_count),
        float(degree.mean()),
        float(degree.std(unbiased=False)),
        float(degree.max()),
        float((degree == 0).sum()),
        float((degree > 4).sum()),
        float(components),
        float(edge_count - atom_count + components),
    ]
    return names, values


def build_observable_features(
    new_probabilities: torch.Tensor,
    old_probabilities: torch.Tensor,
) -> tuple[list[str], torch.Tensor, dict[str, torch.Tensor]]:
    if new_probabilities.shape != old_probabilities.shape or new_probabilities.ndim != 1:
        raise ValueError("P19-E7 old/new probability shapes differ")
    atom_count = infer_atom_count(len(new_probabilities))
    new_reference = torch.logit(torch.tensor(NEW_THRESHOLD))
    old_reference = torch.logit(torch.tensor(OLD_THRESHOLD))
    new_margin = p19e5._logit(new_probabilities) - new_reference
    old_margin = p19e5._logit(old_probabilities) - old_reference
    fusion_score = ALPHA_NEW * new_margin + (1.0 - ALPHA_NEW) * old_margin
    fusion_margin = fusion_score - FUSION_THRESHOLD
    selected = {
        "fusion": fusion_margin >= 0.0,
        "new": new_margin >= 0.0,
        "old": old_margin >= 0.0,
    }
    names = ["atom_count", "pair_count"]
    values = [float(atom_count), float(len(new_probabilities))]
    for name, margin in (
        ("new_margin", new_margin),
        ("old_margin", old_margin),
        ("fusion_margin", fusion_margin),
    ):
        current_names, current_values = _margin_statistics(name, margin)
        names.extend(current_names)
        values.extend(current_values)
    for name in ("fusion", "new", "old"):
        names.append(f"{name}_selected")
        values.append(float(selected[name].sum()))
    for left, right in (("new", "old"), ("new", "fusion"), ("old", "fusion")):
        names.append(f"{left}_{right}_symmetric_difference")
        values.append(float((selected[left] != selected[right]).sum()))
    any_disagreement = (
        (selected["new"] != selected["old"])
        | (selected["new"] != selected["fusion"])
    )
    names.append("any_route_disagreement")
    values.append(float(any_disagreement.sum()))
    for name in ("fusion", "new", "old"):
        current_names, current_values = _graph_statistics(
            name, selected[name], atom_count
        )
        names.extend(current_names)
        values.extend(current_values)
    if bool(any_disagreement.any()):
        for name, margin in (
            ("new", new_margin),
            ("old", old_margin),
            ("fusion", fusion_margin),
        ):
            current = margin[any_disagreement]
            names.extend((f"disagreement_{name}_mean", f"disagreement_{name}_max_abs"))
            values.extend((float(current.mean()), float(current.abs().max())))
    else:
        for name in ("new", "old", "fusion"):
            names.extend((f"disagreement_{name}_mean", f"disagreement_{name}_max_abs"))
            values.extend((0.0, 0.0))
    return names, torch.tensor(values, dtype=torch.float32), selected


def _route_label(route_exact: torch.Tensor) -> int:
    fusion_exact, new_exact, old_exact = route_exact.tolist()
    if fusion_exact:
        return 0
    if new_exact:
        return 1
    if old_exact:
        return 2
    return 0


def build_capacity_records(
    new_records: list[dict], old_records: list[dict]
) -> tuple[list[dict], list[str]]:
    new_by_key = {str(record["sample_key"]): record for record in new_records}
    old_by_key = {str(record["sample_key"]): record for record in old_records}
    if set(new_by_key) != set(old_by_key):
        raise ValueError("P19-E7 old/new sample keys differ")
    records = []
    feature_names = None
    for key in sorted(new_by_key):
        new = new_by_key[key]
        old = old_by_key[key]
        if not torch.equal(new["presence"], old["presence"]):
            raise ValueError("P19-E7 old/new target pairs differ")
        names, features, selected = build_observable_features(
            new["probabilities"], old["probabilities"]
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("P19-E7 feature schema differs across images")
        presence = new["presence"]
        route_exact = torch.tensor(
            [
                torch.equal(selected["fusion"], presence),
                torch.equal(selected["new"], presence),
                torch.equal(selected["old"], presence),
            ],
            dtype=torch.bool,
        )
        records.append(
            {
                "sample_key": key,
                "variant": new["variant"],
                "stress_operation": new["stress_operation"],
                "features": features,
                "route_exact": route_exact,
                "target_action": _route_label(route_exact),
                "routes_observably_disagree": bool(
                    (selected["new"] != selected["old"]).any()
                    or (selected["new"] != selected["fusion"]).any()
                ),
            }
        )
    if feature_names is None:
        raise RuntimeError("P19-E7 built no capacity records")
    return records, feature_names


@torch.inference_mode()
def evaluate_router(
    router: OutputRouteRouter,
    normalized_features: torch.Tensor,
    records: list[dict],
) -> dict:
    router.eval()
    actions = router(normalized_features).argmax(-1).cpu()
    rows = []
    for index, record in enumerate(records):
        action = int(actions[index])
        fusion_exact = bool(record["route_exact"][0])
        selected_exact = bool(record["route_exact"][action])
        rows.append(
            {
                "variant": record["variant"],
                "stress_operation": record["stress_operation"],
                "action": ROUTE_ACTIONS[action],
                "target_action": ROUTE_ACTIONS[record["target_action"]],
                "selected_exact": selected_exact,
                "fusion_exact": fusion_exact,
                "rescue": selected_exact and not fusion_exact,
                "harmful": fusion_exact and not selected_exact,
                "target_correct": action == record["target_action"],
                "routes_observably_disagree": record["routes_observably_disagree"],
            }
        )

    def summary(current: list[dict]) -> dict:
        return {
            "images": len(current),
            "fusion_exact": sum(int(row["fusion_exact"]) for row in current),
            "selected_exact": sum(int(row["selected_exact"]) for row in current),
            "rescue": sum(int(row["rescue"]) for row in current),
            "harmful": sum(int(row["harmful"]) for row in current),
            "target_correct": sum(int(row["target_correct"]) for row in current),
            "observable_disagreement": sum(
                int(row["routes_observably_disagree"]) for row in current
            ),
            "actions": dict(Counter(row["action"] for row in current)),
        }

    return {
        "overall": summary(rows),
        "by_variant": {
            variant: summary([row for row in rows if row["variant"] == variant])
            for variant in ("coordgen", "original", "rdkit", "stress")
        },
        "stress_by_operation": {
            operation: summary(
                [row for row in rows if row["stress_operation"] == operation]
            )
            for operation in sorted(
                {
                    str(row["stress_operation"])
                    for row in rows
                    if row["stress_operation"] is not None
                }
            )
        },
    }


def _metric_key(metrics: dict) -> tuple[int, int, int, int]:
    overall = metrics["overall"]
    return (
        -int(overall["harmful"]),
        int(overall["selected_exact"]),
        int(overall["rescue"]),
        int(overall["target_correct"]),
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E7 output is not empty: {OUTPUT}")
    for path, expected in (
        (CALIBRATION_MANIFEST, CALIBRATION_MANIFEST_SHA256),
        (NEW_CHECKPOINT, NEW_CHECKPOINT_SHA256),
        (OLD_CHECKPOINT, OLD_CHECKPOINT_SHA256),
        (P19E5_AUDIT, P19E5_AUDIT_SHA256),
        (P19E6_AUDIT, P19E6_AUDIT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E7 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E7 requires CUDA")
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

    rows = p18e0._jsonl(CALIBRATION_MANIFEST)
    samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(rows), accepted_split="train"
        )
    )
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
    new_records, new_digest = p19e4.collect_probabilities(
        new_model, vision, atom, samples, device
    )
    old_records, old_digest = p19e4.collect_probabilities(
        old_model, vision, atom, samples, device
    )
    records, feature_names = build_capacity_records(new_records, old_records)
    features = torch.stack([record["features"] for record in records])
    labels = torch.tensor([record["target_action"] for record in records])
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(1e-4)
    normalized = ((features - mean) / std).to(device)
    labels_device = labels.to(device)
    route_exact = torch.stack([record["route_exact"] for record in records])
    fusion_exact = int(route_exact[:, 0].sum())
    oracle_exact = int(route_exact.any(-1).sum())
    class_counts = torch.bincount(labels, minlength=len(ROUTE_ACTIONS)).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).sqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p19e7-output-route-capacity-cache-0.1",
            "feature_names": feature_names,
            "features": features,
            "feature_mean": mean,
            "feature_std": std,
            "labels": labels,
            "route_exact": route_exact,
            "sample_keys": [record["sample_key"] for record in records],
            "variants": [record["variant"] for record in records],
            "stress_operations": [record["stress_operation"] for record in records],
            "observable_disagreement": torch.tensor(
                [record["routes_observably_disagree"] for record in records]
            ),
            "source_digests": {"new": new_digest, "old": old_digest},
        },
    )
    cache_sha256 = file_sha256(CACHE)
    config = OutputRouteRouterConfig(input_dim=features.size(1), hidden_dim=128)
    router = OutputRouteRouter(config).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial = evaluate_router(router, normalized, records)
    best_metrics = initial
    best_key = _metric_key(initial)
    _atomic_torch_save(
        OUTPUT / "best.pt",
        {
            "checkpoint_schema_version": "g3-p19e7-output-route-router-0.1",
            "model_type": MODEL_TYPE,
            "config": asdict(config),
            "router": router.state_dict(),
            "feature_names": feature_names,
            "feature_mean": mean,
            "feature_std": std,
            "global_step": 0,
            "metrics": initial,
            "cache_sha256": cache_sha256,
        },
    )
    history = []
    running = 0.0
    for step in range(1, STEPS + 1):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        logits = router(normalized)
        loss = F.cross_entropy(logits.float(), labels_device, weight=class_weights)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            router.parameters(), MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P19-E7 gradient is not finite")
        optimizer.step()
        running += float(loss.detach())
        if step % EVALUATE_EVERY == 0:
            metrics = evaluate_router(router, normalized, records)
            key = _metric_key(metrics)
            selected = key > best_key
            if selected:
                best_key = key
                best_metrics = metrics
                _atomic_torch_save(
                    OUTPUT / "best.pt",
                    {
                        "checkpoint_schema_version": "g3-p19e7-output-route-router-0.1",
                        "model_type": MODEL_TYPE,
                        "config": asdict(config),
                        "router": router.state_dict(),
                        "feature_names": feature_names,
                        "feature_mean": mean,
                        "feature_std": std,
                        "global_step": step,
                        "metrics": metrics,
                        "cache_sha256": cache_sha256,
                    },
                )
            history.append(
                {
                    "step": step,
                    "selected_best": selected,
                    "loss": running / EVALUATE_EVERY,
                    "metrics": metrics,
                }
            )
            running = 0.0
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
    }
    target_disagreement = sum(
        int(record["routes_observably_disagree"]) for record in records
    )
    checks = {
        "capacity_images_640": len(records) == 640,
        "route_oracle_headroom_min10": oracle_exact - fusion_exact >= 10,
        "observable_disagreement_present": target_disagreement >= 20,
        "both_alternative_actions_present": (
            class_counts[1] > 0 and class_counts[2] > 0
        ),
        "router_zero_harmful": best_metrics["overall"]["harmful"] == 0,
        "router_within2_of_oracle": (
            best_metrics["overall"]["selected_exact"] >= oracle_exact - 2
        ),
        "router_rescue_within2_of_headroom": (
            best_metrics["overall"]["rescue"] >= oracle_exact - fusion_exact - 2
        ),
        "frozen_sources_unchanged": states_before == states_after,
    }
    decision = (
        "output_route_capacity640_go"
        if all(checks.values())
        else "output_route_capacity640_no_go"
    )
    best_payload = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    result = {
        "schema_version": "g3-p19e7-output-route-capacity640-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "initial": initial,
        "best": best_metrics,
        "history": history,
        "capacity": {
            "fusion_exact": fusion_exact,
            "oracle_exact": oracle_exact,
            "oracle_headroom": oracle_exact - fusion_exact,
            "observable_disagreement": target_disagreement,
            "class_counts": {
                ROUTE_ACTIONS[index]: int(class_counts[index])
                for index in range(len(ROUTE_ACTIONS))
            },
        },
        "feature_dim": len(feature_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "best_step": int(best_payload["global_step"]),
        "best_checkpoint": str((OUTPUT / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(OUTPUT / "best.pt"),
        "cache": str(CACHE.resolve()),
        "cache_sha256": cache_sha256,
        "training_data_is_consumed_calibration": True,
        "capacity_only": True,
        "eval640_read": False,
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
