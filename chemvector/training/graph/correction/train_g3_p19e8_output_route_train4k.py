#!/usr/bin/env python3
"""Train the P19 output router on isolated train/calibration/holdout splits."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p4e3_global_peak_rank_pilot20k as p4e3
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.correction import (
    train_g3_p19e7_output_route_capacity640 as p19e7,
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


MODEL_TYPE = "chemvector-g3-p19e8-output-route-train4k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = OUTPUT / "router-cache.pt"
HOLDOUT_CACHE = OUTPUT / "holdout-cache.pt"
TRAIN_MANIFEST = OUTPUT / "train4000.jsonl"
CALIBRATION_MANIFEST = OUTPUT / "calibration640.jsonl"
HOLDOUT_MANIFEST = OUTPUT / "holdout640.jsonl"
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
NEW_CHECKPOINT = p19e5.NEW_CHECKPOINT
NEW_CHECKPOINT_SHA256 = p19e5.NEW_CHECKPOINT_SHA256
OLD_CHECKPOINT = p19e5.OLD_CHECKPOINT
OLD_CHECKPOINT_SHA256 = p19e5.OLD_CHECKPOINT_SHA256
P19E7_CHECKPOINT = p19e7.OUTPUT / "best.pt"
P19E7_CHECKPOINT_SHA256 = (
    "9cd4f454ab3fe0f511cbfd2ed4e4e1e264960ffb173cb1afed933071d8871e86"
)
TRAIN_MOLECULES = 1_000
CALIBRATION_MOLECULES = 160
HOLDOUT_MOLECULES = 160
TRAIN_IMAGES = TRAIN_MOLECULES * 4
CALIBRATION_IMAGES = CALIBRATION_MOLECULES * 4
HOLDOUT_IMAGES = HOLDOUT_MOLECULES * 4
VARIANTS = ("coordgen", "original", "rdkit", "stress")
STEPS = 3_000
MINIBATCH_SIZE = 128
EVALUATE_EVERY = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 20260810


def select_router_splits(
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
            raise ValueError(f"P19-E8 incomplete molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P19-E8 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        if (
            molecule_id in excluded_molecules
            or not scaffold
            or scaffold in excluded_scaffolds
        ):
            continue
        digest = hashlib.sha256(
            f"{SEED}\0p19e8\0{molecule_id}\0{scaffold}".encode()
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
        raise ValueError("P19-E8 has too few unique-scaffold molecules")
    boundaries = {
        "train": (0, TRAIN_MOLECULES),
        "calibration": (
            TRAIN_MOLECULES,
            TRAIN_MOLECULES + CALIBRATION_MOLECULES,
        ),
        "holdout": (
            TRAIN_MOLECULES + CALIBRATION_MOLECULES,
            required,
        ),
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
        name: [
            row
            for row in full_rows
            if str(row["molecule_id"]) in split_molecules[name]
        ]
        for name in boundaries
    }
    expected_images = {
        "train": TRAIN_IMAGES,
        "calibration": CALIBRATION_IMAGES,
        "holdout": HOLDOUT_IMAGES,
    }
    if any(len(splits[name]) != expected_images[name] for name in splits):
        raise RuntimeError("P19-E8 split image count differs")
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
            "train_calibration": len(
                split_molecules["train"] & split_molecules["calibration"]
            ),
            "train_holdout": len(
                split_molecules["train"] & split_molecules["holdout"]
            ),
            "calibration_holdout": len(
                split_molecules["calibration"] & split_molecules["holdout"]
            ),
        },
        "cross_split_scaffold_overlap": {
            "train_calibration": len(
                split_scaffolds["train"] & split_scaffolds["calibration"]
            ),
            "train_holdout": len(
                split_scaffolds["train"] & split_scaffolds["holdout"]
            ),
            "calibration_holdout": len(
                split_scaffolds["calibration"] & split_scaffolds["holdout"]
            ),
        },
        "sample_keys_sha256": {
            name: hashlib.sha256(
                "".join(f"{row['sample_key']}\n" for row in rows).encode()
            ).hexdigest()
            for name, rows in splits.items()
        },
    }
    return splits, audit


def _records_to_tensors(records: list[dict]) -> dict:
    return {
        "features": torch.stack([record["features"] for record in records]),
        "labels": torch.tensor([record["target_action"] for record in records]),
        "route_exact": torch.stack([record["route_exact"] for record in records]),
        "observable_disagreement": torch.tensor(
            [record["routes_observably_disagree"] for record in records]
        ),
        "sample_keys": [record["sample_key"] for record in records],
        "variants": [record["variant"] for record in records],
        "stress_operations": [record["stress_operation"] for record in records],
    }


def _tensor_records(cache: dict) -> list[dict]:
    return [
        {
            "sample_key": cache["sample_keys"][index],
            "variant": cache["variants"][index],
            "stress_operation": cache["stress_operations"][index],
            "route_exact": cache["route_exact"][index],
            "target_action": int(cache["labels"][index]),
            "routes_observably_disagree": bool(
                cache["observable_disagreement"][index]
            ),
        }
        for index in range(len(cache["labels"]))
    ]


def _actions_from_logits(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probabilities = logits.softmax(-1)
    proposed = probabilities.argmax(-1)
    confidence = probabilities.gather(1, proposed[:, None]).squeeze(1)
    return torch.where(
        (proposed != 0) & (confidence >= threshold),
        proposed,
        torch.zeros_like(proposed),
    )


def evaluate_actions(actions: torch.Tensor, records: list[dict]) -> dict:
    class _FixedRouter(torch.nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.values = values

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            logits = features.new_full((len(features), len(ROUTE_ACTIONS)), -100.0)
            logits[torch.arange(len(features), device=features.device), self.values] = 100.0
            return logits

    return p19e7.evaluate_router(
        _FixedRouter(actions.to(actions.device)),
        torch.zeros(len(actions), 1, device=actions.device),
        records,
    )


@torch.inference_mode()
def calibrate_action_threshold(
    router: OutputRouteRouter,
    normalized_features: torch.Tensor,
    records: list[dict],
) -> tuple[float, dict]:
    router.eval()
    logits = router(normalized_features)
    probabilities = logits.softmax(-1)
    proposed = probabilities.argmax(-1)
    confidence = probabilities.gather(1, proposed[:, None]).squeeze(1)
    candidates = {1.0}
    for value in confidence[proposed != 0].detach().cpu().tolist():
        candidates.add(float(np.nextafter(value, -np.inf)))
        candidates.add(float(value))
    best_threshold = 1.0
    best_metrics = None
    best_key = None
    for threshold in sorted(candidates):
        actions = _actions_from_logits(logits, threshold)
        metrics = evaluate_actions(actions, records)
        overall = metrics["overall"]
        key = (
            -int(overall["harmful"]),
            int(overall["selected_exact"]),
            int(overall["rescue"]),
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("P19-E8 threshold calibration failed")
    return best_threshold, best_metrics


def _checkpoint(
    router: OutputRouteRouter,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    threshold: float,
    calibration_metrics: dict,
    feature_names: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    cache_sha256: str,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p19e8-output-route-router-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(router.config),
        "router": router.state_dict(),
        "optimizer": optimizer.state_dict(),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "global_step": step,
        "action_threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "cache_sha256": cache_sha256,
        "training_args": {
            "steps": STEPS,
            "minibatch_size": MINIBATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E8 output is not empty: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256),
        (OLD_DEV_MANIFEST, OLD_DEV_MANIFEST_SHA256),
        (USED_CALIBRATION_MANIFEST, USED_CALIBRATION_MANIFEST_SHA256),
        (USED_EVAL_MANIFEST, USED_EVAL_MANIFEST_SHA256),
        (NEW_CHECKPOINT, NEW_CHECKPOINT_SHA256),
        (OLD_CHECKPOINT, OLD_CHECKPOINT_SHA256),
        (P19E7_CHECKPOINT, P19E7_CHECKPOINT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E8 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E8 requires CUDA")
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
    ):
        excluded_rows.extend(p18e0._jsonl(path))
    splits, split_audit = select_router_splits(full_rows, excluded_rows)
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
    # Holdout image bytes and targets remain unread until both the checkpoint
    # and the calibration-derived action threshold have been locked.
    for split_name in ("train", "calibration"):
        samples = list(
            p1a.iter_manifest_samples(
                p18e0._manifest_records(splits[split_name]), accepted_split="train"
            )
        )
        new_records, _ = p19e4.collect_probabilities(
            new_model, vision, atom, samples, device
        )
        old_records, _ = p19e4.collect_probabilities(
            old_model, vision, atom, samples, device
        )
        records, names = p19e7.build_capacity_records(new_records, old_records)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("P19-E8 feature schema differs across splits")
        caches[split_name] = _records_to_tensors(records)
    if feature_names is None:
        raise RuntimeError("P19-E8 built no feature schema")
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p19e8-output-route-cache-0.1",
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
    train_records = _tensor_records(train)
    calibration_records = _tensor_records(calibration)
    mean = train["features"].mean(0)
    std = train["features"].std(0, unbiased=False).clamp_min(1e-4)
    normalized_train = ((train["features"] - mean) / std).to(device)
    normalized_calibration = ((calibration["features"] - mean) / std).to(device)
    train_labels = train["labels"].to(device)
    class_counts = torch.bincount(
        train["labels"], minlength=len(ROUTE_ACTIONS)
    ).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).sqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)
    config = OutputRouteRouterConfig(
        input_dim=len(feature_names), hidden_dim=128
    )
    router = OutputRouteRouter(config).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial_threshold, initial_calibration = calibrate_action_threshold(
        router, normalized_calibration, calibration_records
    )
    best_key = p19e7._metric_key(initial_calibration)
    best_metrics = initial_calibration
    _atomic_torch_save(
        OUTPUT / "best.pt",
        _checkpoint(
            router,
            optimizer,
            step=0,
            threshold=initial_threshold,
            calibration_metrics=initial_calibration,
            feature_names=feature_names,
            mean=mean,
            std=std,
            cache_sha256=cache_sha256,
        ),
    )
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(train_labels), generator=generator).tolist()
    offset = 0
    running = 0.0
    history = []
    for step in range(1, STEPS + 1):
        if offset + MINIBATCH_SIZE > len(order):
            order = torch.randperm(len(train_labels), generator=generator).tolist()
            offset = 0
        indices = torch.tensor(
            order[offset : offset + MINIBATCH_SIZE], device=device
        )
        offset += MINIBATCH_SIZE
        router.train()
        optimizer.zero_grad(set_to_none=True)
        logits = router(normalized_train[indices])
        loss = F.cross_entropy(
            logits.float(), train_labels[indices], weight=class_weights
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            router.parameters(), MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P19-E8 gradient is not finite")
        optimizer.step()
        running += float(loss.detach())
        if step % EVALUATE_EVERY == 0:
            threshold, metrics = calibrate_action_threshold(
                router, normalized_calibration, calibration_records
            )
            key = p19e7._metric_key(metrics)
            selected = key > best_key
            if selected:
                best_key = key
                best_metrics = metrics
                _atomic_torch_save(
                    OUTPUT / "best.pt",
                    _checkpoint(
                        router,
                        optimizer,
                        step=step,
                        threshold=threshold,
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
                    "action_threshold": threshold,
                    "calibration": metrics,
                }
            )
            running = 0.0

    best_payload = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    best_router = OutputRouteRouter(
        OutputRouteRouterConfig(**best_payload["config"])
    ).to(device)
    best_router.load_state_dict(best_payload["router"], strict=True)
    best_router.eval()

    holdout_samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(splits["holdout"]), accepted_split="train"
        )
    )
    holdout_new_records, _ = p19e4.collect_probabilities(
        new_model, vision, atom, holdout_samples, device
    )
    holdout_old_records, _ = p19e4.collect_probabilities(
        old_model, vision, atom, holdout_samples, device
    )
    holdout_records_raw, holdout_feature_names = p19e7.build_capacity_records(
        holdout_new_records, holdout_old_records
    )
    if holdout_feature_names != feature_names:
        raise RuntimeError("P19-E8 holdout feature schema differs")
    holdout = _records_to_tensors(holdout_records_raw)
    _atomic_torch_save(
        HOLDOUT_CACHE,
        {
            "schema_version": "g3-p19e8-output-route-holdout-cache-0.1",
            "feature_names": feature_names,
            "split": holdout,
            "checkpoint_sha256_at_first_read": file_sha256(OUTPUT / "best.pt"),
            "action_threshold_at_first_read": float(
                best_payload["action_threshold"]
            ),
        },
    )
    holdout_cache_sha256 = file_sha256(HOLDOUT_CACHE)
    holdout_records = _tensor_records(holdout)
    normalized_holdout = ((holdout["features"] - mean) / std).to(device)
    with torch.inference_mode():
        holdout_logits = best_router(normalized_holdout)
        holdout_actions = _actions_from_logits(
            holdout_logits, float(best_payload["action_threshold"])
        )
    holdout_metrics = evaluate_actions(holdout_actions, holdout_records)
    calibration_fusion = best_metrics["overall"]["fusion_exact"]
    holdout_fusion = holdout_metrics["overall"]["fusion_exact"]
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "new": module_state_sha256(new_model),
        "old": module_state_sha256(old_model),
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
        "train_both_alternative_actions_min20": (
            class_counts[1] >= 20 and class_counts[2] >= 20
        ),
        "calibration_zero_harmful": best_metrics["overall"]["harmful"] == 0,
        "calibration_rescue_min3": best_metrics["overall"]["rescue"] >= 3,
        "holdout_zero_harmful": holdout_metrics["overall"]["harmful"] == 0,
        "holdout_rescue_min3": holdout_metrics["overall"]["rescue"] >= 3,
        "holdout_exact_improves_fusion": (
            holdout_metrics["overall"]["selected_exact"] > holdout_fusion
        ),
        "frozen_sources_unchanged": (
            states_before == states_after_cache == states_after
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    decision = (
        "output_route_train4k_go"
        if all(checks.values())
        else "output_route_train4k_no_go"
    )
    result = {
        "schema_version": "g3-p19e8-output-route-train4k-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "capacity": {
            "train_class_counts": {
                ROUTE_ACTIONS[index]: int(class_counts[index])
                for index in range(len(ROUTE_ACTIONS))
            },
            "calibration_fusion_exact": calibration_fusion,
            "holdout_fusion_exact": holdout_fusion,
        },
        "best_step": int(best_payload["global_step"]),
        "action_threshold": float(best_payload["action_threshold"]),
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
        "holdout_read_once_after_checkpoint_and_threshold_lock": True,
        "used_p19e4_eval_for_selection": False,
        "used_p19e4_eval_only_for_exclusion": True,
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
