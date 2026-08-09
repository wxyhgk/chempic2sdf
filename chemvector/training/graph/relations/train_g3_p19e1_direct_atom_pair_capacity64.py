#!/usr/bin/env python3
"""Capacity-test a newly initialized Direct AtomPair relation head on 64 images."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.joint.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    enumerate_unordered_atom_pairs,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e0_direct_atom_pair_oracle_ladder as p19e0,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p4e3_global_peak_rank_pilot20k as p4e3
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p19e1-direct-atom-pair-capacity64"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
CACHE = OUTPUT / "capacity-cache.pt"
PILOT_MANIFEST = p4e3.P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = p4e3.PILOT_MANIFEST_SHA256
FRESH640_MANIFEST = p18e0.SELECTION_MANIFEST
FRESH640_MANIFEST_SHA256 = p19e0.E0_MANIFEST_SHA256
SOURCE_CHECKPOINT = p19e0.SOURCE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = p19e0.SOURCE_CHECKPOINT_SHA256
VISION_CHECKPOINT = p19e0.VISION_CHECKPOINT
VISION_CHECKPOINT_SHA256 = p19e0.VISION_CHECKPOINT_SHA256
OLD_DIRECT_CHECKPOINT = p19e0.DIRECT_CHECKPOINT
OLD_DIRECT_CHECKPOINT_SHA256 = p19e0.DIRECT_CHECKPOINT_SHA256
CAPACITY_IMAGES = 64
EASY_IMAGES = 32
CROSSED_HARD_IMAGES = 16
NEAR_NONBONDED_HARD_IMAGES = 16
HARD_IMAGES = CROSSED_HARD_IMAGES + NEAR_NONBONDED_HARD_IMAGES
MAXIMUM_ATOMS = 128
MAXIMUM_BONDS = 192
BUILD_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 1
STEPS = 3_000
EVALUATE_EVERY = 250
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
THRESHOLD = 0.5
OLD_THRESHOLD = 0.85
SEED = 20260810
CONFIG = DirectAtomPairBondConfig()


def build_pair_targets(
    pair_indices: torch.Tensor,
    target_refs: torch.Tensor,
    target_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map canonical GT Atom identities to the complete unordered pair universe."""

    if target_refs.shape != (len(target_types), 2):
        raise ValueError("P19-E1 target Bond shapes differ")
    lookup = {
        tuple(sorted((int(ref[0]), int(ref[1])))): int(target_types[index])
        for index, ref in enumerate(target_refs.tolist())
    }
    if len(lookup) != len(target_refs):
        raise ValueError("P19-E1 target graph contains duplicate AtomPair bonds")
    presence = torch.zeros(len(pair_indices), dtype=torch.bool)
    chemical = torch.zeros(len(pair_indices), dtype=torch.long)
    for row, pair in enumerate(pair_indices.tolist()):
        value = lookup.get(tuple(sorted((int(pair[0]), int(pair[1])))))
        if value is not None:
            presence[row] = True
            chemical[row] = value
    if int(presence.sum()) != len(target_refs):
        raise ValueError("P19-E1 all-pair universe does not cover every target Bond")
    return presence, chemical


def _sample_metrics(
    probabilities: torch.Tensor,
    predicted_types: torch.Tensor,
    presence: torch.Tensor,
    chemical: torch.Tensor,
    *,
    threshold: float,
) -> dict:
    selected = probabilities >= threshold
    true_positive = selected & presence
    false_positive = selected & ~presence
    false_negative = ~selected & presence
    exact = bool(torch.equal(selected, presence))
    typed_exact = exact and bool(
        torch.equal(predicted_types[presence], chemical[presence])
    )
    return {
        "target_bonds": int(presence.sum()),
        "selected_bonds": int(selected.sum()),
        "true_positive": int(true_positive.sum()),
        "false_positive": int(false_positive.sum()),
        "false_negative": int(false_negative.sum()),
        "skeleton_exact": exact,
        "typed_exact": typed_exact,
    }


def balanced_presence_loss(
    logits: torch.Tensor,
    presence_targets: torch.Tensor,
    *,
    listwise_margin: float = 1.0,
    listwise_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Balance true/false pairs and require one image-wide decision margin."""

    positives = presence_targets.bool()
    negatives = ~positives
    if not bool(positives.any()) or not bool(negatives.any()):
        raise ValueError("P19-E1 presence loss needs positive and negative pairs")
    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[positives].float(), torch.ones_like(logits[positives]).float()
    )
    negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[negatives].float(), torch.zeros_like(logits[negatives]).float()
    )
    margin_loss = torch.relu(
        logits[negatives].max().float()
        - logits[positives].min().float()
        + listwise_margin
    )
    return {
        "loss": (positive_loss + negative_loss) * 0.5
        + listwise_weight * margin_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "listwise_margin_loss": margin_loss,
    }


def _summary(rows: list[dict]) -> dict:
    fields = (
        "target_bonds",
        "selected_bonds",
        "true_positive",
        "false_positive",
        "false_negative",
        "skeleton_exact",
        "typed_exact",
    )
    totals = {field: sum(int(row[field]) for row in rows) for field in fields}
    precision = totals["true_positive"] / max(totals["selected_bonds"], 1)
    recall = totals["true_positive"] / max(totals["target_bonds"], 1)
    return {
        "images": len(rows),
        **totals,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "skeleton_exact_rate": totals["skeleton_exact"] / max(len(rows), 1),
        "typed_exact_rate": totals["typed_exact"] / max(len(rows), 1),
    }


def _capacity_summary(rows: list[dict]) -> dict:
    return {
        "overall": _summary(rows),
        "by_difficulty": {
            difficulty: _summary(
                [row for row in rows if row["difficulty"] == difficulty]
            )
            for difficulty in ("easy", "hard")
        },
        "hard_by_operation": {
            operation: _summary(
                [
                    row
                    for row in rows
                    if row["difficulty"] == "hard"
                    and row["stress_operation"] == operation
                ]
            )
            for operation in ("crossed_bonds", "near_nonbonded")
        },
    }


def _selection_ready(easy: list[dict], hard: dict[str, list[dict]]) -> bool:
    return (
        len(easy) >= EASY_IMAGES
        and len(hard["crossed_bonds"]) >= CROSSED_HARD_IMAGES
        and len(hard["near_nonbonded"]) >= NEAR_NONBONDED_HARD_IMAGES
    )


def _record_from_batch(
    *,
    sample: dict,
    batch: dict,
    batch_index: int,
    feature_map: torch.Tensor,
    atom_features: torch.Tensor,
    atom_scores: torch.Tensor,
    pairs: torch.Tensor,
    probabilities: torch.Tensor,
    predicted_types: torch.Tensor,
    pair_rows: torch.Tensor,
    scaffold: str,
) -> tuple[dict, dict]:
    atom_mask = batch["atom_mask"][batch_index]
    atom_count = int(atom_mask.sum())
    target_mask = batch["bond_mask"][batch_index]
    local_pairs = pairs[pair_rows].cpu()
    presence, chemical = build_pair_targets(
        local_pairs,
        batch["bond_atom_refs"][batch_index][target_mask],
        batch["bond_types"][batch_index][target_mask],
    )
    metrics = _sample_metrics(
        probabilities[pair_rows.cpu()],
        predicted_types[pair_rows.cpu()],
        presence,
        chemical,
        threshold=OLD_THRESHOLD,
    )
    operation = str(
        sample["render"].get("render_parameters", {}).get(
            "stress_operation", "unknown"
        )
    )
    record = {
        "sample_key": str(sample["key"]),
        "molecule_id": str(sample["chemmol"]["molecule_id"]),
        "scaffold": scaffold,
        "variant": str(sample["render"]["variant"]),
        "stress_operation": operation,
        "feature_map": feature_map[batch_index].half().cpu().clone(),
        "atom_features": atom_features[batch_index].half().cpu().clone(),
        "atom_anchors": batch["atom_anchors"][batch_index].float().cpu().clone(),
        "atom_scores": atom_scores[batch_index].float().cpu().clone(),
        "atom_mask": atom_mask.cpu().clone(),
        "pair_indices": local_pairs,
        "presence_targets": presence,
        "chemical_targets": chemical,
        "old_probabilities": probabilities[pair_rows.cpu()].clone(),
        "old_predicted_types": predicted_types[pair_rows.cpu()].clone(),
        "atom_count": atom_count,
    }
    return record, metrics


@torch.inference_mode()
def build_capacity_cache(device: torch.device) -> tuple[list[dict], dict]:
    rows = [
        row
        for row in p18e0._jsonl(PILOT_MANIFEST)
        if str(row["variant"]) == "stress"
    ]
    rows_by_key = {str(row["sample_key"]): row for row in rows}
    manifest = p18e0._manifest_records(rows)
    source = p1a.iter_manifest_samples(manifest, accepted_split="train")
    source_payload = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if Path(source_payload["training_args"]["grounded_graph_checkpoint"]).resolve() != (
        VISION_CHECKPOINT.resolve()
    ):
        raise ValueError("P19-E1 source is bound to another vision checkpoint")
    vision, atom, bond, set_head, _ = load_components(
        str(SOURCE_CHECKPOINT), str(VISION_CHECKPOINT), device
    )
    old_payload = torch.load(
        OLD_DIRECT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if old_payload["config"] != asdict(CONFIG):
        raise ValueError("P19-E1 old Direct AtomPair config differs")
    old_model = DirectAtomPairBond(CONFIG).to(device)
    old_model.load_state_dict(old_payload["model"], strict=True)
    if float(old_payload["validation"]["threshold"]) != OLD_THRESHOLD:
        raise ValueError("P19-E1 old Direct AtomPair threshold differs")
    for module in (vision, atom, bond, set_head, old_model):
        module.eval().requires_grad_(False)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "old_direct": module_state_sha256(old_model),
    }
    collator = G3StructuralCollator(
        maximum_atoms=MAXIMUM_ATOMS,
        maximum_bonds=MAXIMUM_BONDS,
        image_size=768,
    )
    easy: list[dict] = []
    hard = {"crossed_bonds": [], "near_nonbonded": []}
    scanned = 0
    pending: list[dict] = []

    def process(samples: list[dict]) -> None:
        nonlocal scanned
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        anchors = batch["atom_anchors"].to(device)
        mask = batch["atom_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feature_map = vision.forward_stroke_resolution(pixels)
            atom_output = atom(feature_map)
            atom_features, atom_scores = p19e0._sample_atom_features(
                atom_output, anchors, mask
            )
            pairs, sample_indices = enumerate_unordered_atom_pairs(mask)
            old_output = old_model(
                feature_map,
                atom_features,
                anchors,
                atom_scores,
                pairs,
                sample_indices,
            )
        probabilities = old_output["presence_logits"].sigmoid().float().cpu()
        predicted_types = old_output["chemical_logits"].argmax(-1).cpu()
        for index, sample in enumerate(samples):
            scanned += 1
            pair_rows = torch.where(sample_indices == index)[0]
            record, metrics = _record_from_batch(
                sample=sample,
                batch=batch,
                batch_index=index,
                feature_map=feature_map,
                atom_features=atom_features,
                atom_scores=atom_scores,
                pairs=pairs,
                probabilities=probabilities,
                predicted_types=predicted_types,
                pair_rows=pair_rows,
                scaffold=str(rows_by_key[str(sample["key"])].get("scaffold", "")),
            )
            operation = record["stress_operation"]
            if metrics["skeleton_exact"] and len(easy) < EASY_IMAGES:
                record["difficulty"] = "easy"
                record["old_metrics"] = metrics
                easy.append(record)
            elif (
                not metrics["skeleton_exact"]
                and operation in hard
                and len(hard[operation])
                < (
                    CROSSED_HARD_IMAGES
                    if operation == "crossed_bonds"
                    else NEAR_NONBONDED_HARD_IMAGES
                )
            ):
                record["difficulty"] = "hard"
                record["old_metrics"] = metrics
                hard[operation].append(record)

    for sample in source:
        pending.append(sample)
        if len(pending) == BUILD_BATCH_SIZE:
            process(pending)
            pending = []
            if _selection_ready(easy, hard):
                break
    if pending and not _selection_ready(easy, hard):
        process(pending)
    if not _selection_ready(easy, hard):
        raise RuntimeError(
            "P19-E1 could not build capacity64: "
            f"easy={len(easy)} crossed={len(hard['crossed_bonds'])} "
            f"near={len(hard['near_nonbonded'])}"
        )
    records = (
        easy[:EASY_IMAGES]
        + hard["crossed_bonds"][:CROSSED_HARD_IMAGES]
        + hard["near_nonbonded"][:NEAR_NONBONDED_HARD_IMAGES]
    )
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "old_direct": module_state_sha256(old_model),
    }
    fresh_rows = p18e0._jsonl(FRESH640_MANIFEST)
    fresh_molecules = {str(row["molecule_id"]) for row in fresh_rows}
    fresh_scaffolds = {
        str(row.get("scaffold", "")) for row in fresh_rows if row.get("scaffold")
    }
    selected_molecules = {record["molecule_id"] for record in records}
    selected_scaffolds = {
        record["scaffold"] for record in records if record["scaffold"]
    }
    sample_digest = hashlib.sha256(
        "".join(f"{record['sample_key']}\n" for record in records).encode()
    ).hexdigest()
    audit = {
        "scanned_train_stress_images": scanned,
        "selected_images": len(records),
        "difficulty": dict(Counter(record["difficulty"] for record in records)),
        "operations": dict(Counter(record["stress_operation"] for record in records)),
        "hard_operations": dict(
            Counter(
                record["stress_operation"]
                for record in records
                if record["difficulty"] == "hard"
            )
        ),
        "old_baseline": _capacity_summary(
            [
                {
                    "difficulty": record["difficulty"],
                    "stress_operation": record["stress_operation"],
                    **record["old_metrics"],
                }
                for record in records
            ]
        ),
        "sample_keys_sha256": sample_digest,
        "selected_unique_molecules": len(selected_molecules),
        "fresh640_molecule_overlap": len(selected_molecules & fresh_molecules),
        "fresh640_scaffold_overlap": len(selected_scaffolds & fresh_scaffolds),
        "all_pair_counts_exact": all(
            len(record["pair_indices"])
            == record["atom_count"] * (record["atom_count"] - 1) // 2
            for record in records
        ),
        "all_pair_target_coverage": all(
            int(record["presence_targets"].sum())
            == int(record["old_metrics"]["target_bonds"])
            for record in records
        ),
        "frozen_states_unchanged": states_before == states_after,
    }
    del vision, atom, bond, set_head, old_model
    gc.collect()
    torch.cuda.empty_cache()
    return records, audit


def _stack(records: list[dict], device: torch.device) -> dict:
    feature_maps = torch.stack([record["feature_map"] for record in records]).to(
        device
    )
    atom_features = torch.stack(
        [record["atom_features"] for record in records]
    ).to(device)
    atom_anchors = torch.stack([record["atom_anchors"] for record in records]).to(
        device
    )
    atom_scores = torch.stack([record["atom_scores"] for record in records]).to(
        device
    )
    pairs = []
    samples = []
    presence = []
    chemical = []
    for sample_index, record in enumerate(records):
        current = record["pair_indices"].to(device)
        pairs.append(current)
        samples.append(
            torch.full(
                (len(current),), sample_index, device=device, dtype=torch.long
            )
        )
        presence.append(record["presence_targets"].to(device))
        chemical.append(record["chemical_targets"].to(device))
    return {
        "feature_maps": feature_maps,
        "atom_features": atom_features,
        "atom_anchors": atom_anchors,
        "atom_scores": atom_scores,
        "pair_indices": torch.cat(pairs),
        "sample_indices": torch.cat(samples),
        "presence_targets": torch.cat(presence),
        "chemical_targets": torch.cat(chemical),
    }


def _forward(model: DirectAtomPairBond, batch: dict) -> dict:
    return model(
        batch["feature_maps"],
        batch["atom_features"],
        batch["atom_anchors"],
        batch["atom_scores"],
        batch["pair_indices"],
        batch["sample_indices"],
    )


@torch.inference_mode()
def evaluate(
    model: DirectAtomPairBond,
    records: list[dict],
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    prediction_digest = hashlib.sha256()
    for record in records:
        batch = _stack([record], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(model, batch)
        logits = output["presence_logits"].float().cpu()
        predicted_types = output["chemical_logits"].argmax(-1).cpu()
        prediction_digest.update(record["sample_key"].encode())
        prediction_digest.update(logits.numpy().tobytes())
        prediction_digest.update(predicted_types.numpy().tobytes())
        rows.append(
            {
                "sample_key": record["sample_key"],
                "difficulty": record["difficulty"],
                "stress_operation": record["stress_operation"],
                **_sample_metrics(
                    logits.sigmoid(),
                    predicted_types,
                    record["presence_targets"],
                    record["chemical_targets"],
                    threshold=THRESHOLD,
                ),
            }
        )
    return {
        **_capacity_summary(rows),
        "prediction_sha256": prediction_digest.hexdigest(),
    }


def _metric_key(metrics: dict) -> tuple[int, int, int, int, int]:
    overall = metrics["overall"]
    return (
        int(metrics["by_difficulty"]["easy"]["skeleton_exact"]),
        int(metrics["by_difficulty"]["hard"]["skeleton_exact"]),
        int(overall["true_positive"]),
        -int(overall["false_positive"]),
        -int(overall["false_negative"]),
    )


def _checkpoint(
    model: DirectAtomPairBond,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict,
    cache_audit: dict,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p19e1-direct-atom-pair-capacity64-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "threshold": THRESHOLD,
        "metrics": metrics,
        "cache_audit": cache_audit,
        "initialized_from_old_direct": False,
        "source": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "vision_checkpoint": str(VISION_CHECKPOINT.resolve()),
            "vision_checkpoint_sha256": VISION_CHECKPOINT_SHA256,
            "selection_only_old_direct_checkpoint": str(
                OLD_DIRECT_CHECKPOINT.resolve()
            ),
            "selection_only_old_direct_checkpoint_sha256": (
                OLD_DIRECT_CHECKPOINT_SHA256
            ),
            "pilot_manifest": str(PILOT_MANIFEST.resolve()),
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        },
        "training_args": {
            "steps": STEPS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "threshold": THRESHOLD,
            "seed": SEED,
            "all_unordered_pairs_per_selected_image": True,
            "presence_only": True,
            "balanced_positive_negative_loss": True,
            "listwise_margin": 1.0,
            "listwise_weight": 0.5,
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E1 output is not empty: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (FRESH640_MANIFEST, FRESH640_MANIFEST_SHA256),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256),
        (VISION_CHECKPOINT, VISION_CHECKPOINT_SHA256),
        (OLD_DIRECT_CHECKPOINT, OLD_DIRECT_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E1 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E1 requires CUDA BF16")
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

    OUTPUT.mkdir(parents=True, exist_ok=False)
    records, cache_audit = build_capacity_cache(device)
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p19e1-direct-atom-pair-capacity-cache-0.1",
            "records": records,
            "audit": cache_audit,
        },
    )
    cache_audit = {
        **cache_audit,
        "path": str(CACHE.resolve()),
        "sha256": file_sha256(CACHE),
    }
    _atomic_json(OUTPUT / "cache-audit.json", cache_audit)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = DirectAtomPairBond(CONFIG).to(device)
    model.chemical.requires_grad_(False)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial = evaluate(model, records, device)
    _atomic_json(OUTPUT / "initial-metrics.json", initial)
    best_metrics = initial
    best_key = _metric_key(initial)
    _atomic_torch_save(
        OUTPUT / "best.pt",
        _checkpoint(
            model,
            optimizer,
            step=0,
            metrics=initial,
            cache_audit=cache_audit,
        ),
    )
    history = []
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(records), generator=generator).tolist()
    offset = 0
    running = Counter()
    for step in range(1, STEPS + 1):
        if offset + TRAIN_BATCH_SIZE > len(order):
            order = torch.randperm(len(records), generator=generator).tolist()
            offset = 0
        indices = order[offset : offset + TRAIN_BATCH_SIZE]
        offset += TRAIN_BATCH_SIZE
        batch = _stack([records[index] for index in indices], device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(model, batch)
            losses = balanced_presence_loss(
                output["presence_logits"], batch["presence_targets"]
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P19-E1 gradient is not finite")
        optimizer.step()
        for name, value in losses.items():
            running[name] += float(value.detach())
        if step % EVALUATE_EVERY == 0 or step == STEPS:
            metrics = evaluate(model, records, device)
            key = _metric_key(metrics)
            selected = key > best_key
            if selected:
                best_key = key
                best_metrics = metrics
                _atomic_torch_save(
                    OUTPUT / "best.pt",
                    _checkpoint(
                        model,
                        optimizer,
                        step=step,
                        metrics=metrics,
                        cache_audit=cache_audit,
                    ),
                )
            row = {
                "step": step,
                "selected_best": selected,
                "metrics": metrics,
                "mean_losses": {
                    name: value / EVALUATE_EVERY
                    for name, value in running.items()
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            running.clear()
            print(json.dumps(row, separators=(",", ":")), flush=True)

    final = evaluate(model, records, device)
    best_payload = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    best_model = DirectAtomPairBond(CONFIG).to(device)
    best_model.load_state_dict(best_payload["model"], strict=True)
    first_best_replay = evaluate(best_model, records, device)
    second_best_replay = evaluate(best_model, records, device)
    best_metrics = first_best_replay
    checks = {
        "capacity64_selected": cache_audit["selected_images"] == CAPACITY_IMAGES,
        "easy32_selected": cache_audit["difficulty"].get("easy", 0) == EASY_IMAGES,
        "crossed_hard16_selected": (
            cache_audit["hard_operations"].get("crossed_bonds", 0)
            == CROSSED_HARD_IMAGES
        ),
        "near_nonbonded_hard16_selected": (
            cache_audit["hard_operations"].get("near_nonbonded", 0)
            == NEAR_NONBONDED_HARD_IMAGES
        ),
        "all_pair_target_coverage": cache_audit["all_pair_target_coverage"],
        "all_pair_counts_exact": cache_audit["all_pair_counts_exact"],
        "selected_molecules_unique64": (
            cache_audit["selected_unique_molecules"] == CAPACITY_IMAGES
        ),
        "fresh640_molecule_overlap_zero": (
            cache_audit["fresh640_molecule_overlap"] == 0
        ),
        "fresh640_scaffold_overlap_zero": (
            cache_audit["fresh640_scaffold_overlap"] == 0
        ),
        "frozen_states_unchanged": cache_audit["frozen_states_unchanged"],
        "best_checkpoint_repeat_exact": (
            first_best_replay["prediction_sha256"]
            == second_best_replay["prediction_sha256"]
        ),
        "easy_zero_break_32_of32": (
            best_metrics["by_difficulty"]["easy"]["skeleton_exact"]
            == EASY_IMAGES
        ),
        "hard_rescue_min28_of32": (
            best_metrics["by_difficulty"]["hard"]["skeleton_exact"] >= 28
        ),
        "overall_exact_min60_of64": (
            best_metrics["overall"]["skeleton_exact"] >= 60
        ),
        "crossed_exact_min14_of16": (
            best_metrics["hard_by_operation"]["crossed_bonds"]["skeleton_exact"]
            >= 14
        ),
        "near_nonbonded_exact_min14_of16": (
            best_metrics["hard_by_operation"]["near_nonbonded"]["skeleton_exact"]
            >= 14
        ),
        "edge_precision_min995": (
            best_metrics["overall"]["edge_precision"] >= 0.995
        ),
        "edge_recall_min995": (
            best_metrics["overall"]["edge_recall"] >= 0.995
        ),
    }
    decision = (
        "direct_atom_pair_capacity64_go"
        if all(checks.values())
        else "direct_atom_pair_capacity64_no_go"
    )
    _atomic_torch_save(
        OUTPUT / "last.pt",
        _checkpoint(
            model,
            optimizer,
            step=STEPS,
            metrics=final,
            cache_audit=cache_audit,
        ),
    )
    result = {
        "schema_version": "g3-p19e1-direct-atom-pair-capacity64-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "initial": initial,
        "best": best_metrics,
        "final": final,
        "history": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "cache": cache_audit,
        "best_checkpoint": str((OUTPUT / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(OUTPUT / "best.pt"),
        "initialized_from_old_direct": False,
        "gt_atom_oracle_conditioned": True,
        "gt_bond_fields_forwarded": False,
        "chemical_loss_trained": False,
        "dense_bond_forward": False,
        "incidence_forward": False,
        "distance_or_neighbor_pruning": False,
        "sealed_test_read": False,
        "confirm_read": False,
        "fresh_dev_labels_used_for_training_or_selection": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
