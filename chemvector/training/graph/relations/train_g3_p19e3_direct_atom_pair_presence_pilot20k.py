#!/usr/bin/env python3
"""Train scratch Direct AtomPair presence on pilot20k with isolated dev640."""

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
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e2_direct_atom_pair_fresh640 as p19e2,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p4e3_global_peak_rank_pilot20k as p4e3
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.relations import (
    train_g3_p19e1_direct_atom_pair_capacity64 as p19e1,
)
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p19e3-direct-atom-pair-presence-pilot20k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
PILOT_MANIFEST = p4e3.P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = p4e3.PILOT_MANIFEST_SHA256
FULL_MANIFEST = p4e3.P4E0_ROOT / "full_train.jsonl"
FULL_MANIFEST_SHA256 = (
    "c1c787a646b5827ae72accd62dbe791ba2beb0a866d88741be3d79d26d114795"
)
DEV_MANIFEST = OUTPUT / "internal-dev640.jsonl"
SOURCE_CHECKPOINT = p19e0.SOURCE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = p19e0.SOURCE_CHECKPOINT_SHA256
VISION_CHECKPOINT = p19e0.VISION_CHECKPOINT
VISION_CHECKPOINT_SHA256 = p19e0.VISION_CHECKPOINT_SHA256
OLD_DIRECT_CHECKPOINT = p19e0.DIRECT_CHECKPOINT
OLD_DIRECT_CHECKPOINT_SHA256 = p19e0.DIRECT_CHECKPOINT_SHA256
TRAIN_IMAGES = 20_000
DEV_MOLECULES = 160
DEV_IMAGES = DEV_MOLECULES * 4
MAXIMUM_ATOMS = 128
MAXIMUM_BONDS = 192
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
EVALUATE_EVERY_IMAGES = 5_000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
THRESHOLD = 0.5
OLD_THRESHOLD = 0.85
SEED = 20260810
CONFIG = DirectAtomPairBondConfig()
VARIANTS = ("coordgen", "original", "rdkit", "stress")


def select_internal_dev640(
    pilot_rows: list[dict], full_rows: list[dict]
) -> tuple[list[dict], dict]:
    """Select full-train molecules with no pilot molecule or scaffold overlap."""

    pilot_molecules = {str(row["molecule_id"]) for row in pilot_rows}
    pilot_scaffolds = {
        str(row.get("scaffold", "")) for row in pilot_rows if row.get("scaffold")
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in full_rows:
        grouped[str(row["molecule_id"])].append(row)
    candidates = []
    for molecule_id, rows in grouped.items():
        scaffolds = {str(row.get("scaffold", "")) for row in rows}
        variants = {str(row["variant"]) for row in rows}
        if len(rows) != len(VARIANTS) or variants != set(VARIANTS):
            raise ValueError(f"P19-E3 incomplete molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P19-E3 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        if (
            molecule_id in pilot_molecules
            or not scaffold
            or scaffold in pilot_scaffolds
        ):
            continue
        digest = hashlib.sha256(
            f"{SEED}\0p19e3-dev\0{molecule_id}\0{scaffold}".encode()
        ).hexdigest()
        candidates.append((digest, molecule_id, scaffold))
    candidates.sort()
    if len(candidates) < DEV_MOLECULES:
        raise ValueError("P19-E3 has too few scaffold-isolated dev molecules")
    selected = candidates[:DEV_MOLECULES]
    selected_molecules = {item[1] for item in selected}
    rows = [
        row for row in full_rows if str(row["molecule_id"]) in selected_molecules
    ]
    selected_scaffolds = {item[2] for item in selected}
    if len(rows) != DEV_IMAGES:
        raise RuntimeError("P19-E3 dev640 image count differs")
    audit = {
        "policy": "smallest SHA256(seed,p19e3-dev,molecule_id,scaffold)",
        "eligible_molecules": len(candidates),
        "selected_molecules": len(selected_molecules),
        "selected_images": len(rows),
        "variants": dict(Counter(str(row["variant"]) for row in rows)),
        "pilot_molecule_overlap": len(selected_molecules & pilot_molecules),
        "pilot_scaffold_overlap": len(selected_scaffolds & pilot_scaffolds),
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{row['sample_key']}\n" for row in rows).encode()
        ).hexdigest(),
    }
    return rows, audit


def balanced_presence_loss_by_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    losses = []
    for sample_index in sample_indices.unique(sorted=True).tolist():
        rows = sample_indices == sample_index
        losses.append(p19e1.balanced_presence_loss(logits[rows], targets[rows]))
    if not losses:
        raise ValueError("P19-E3 batch has no AtomPair rows")
    return {
        name: torch.stack([loss[name] for loss in losses]).mean()
        for name in losses[0]
    }


def _pair_targets_for_batch(
    pairs: torch.Tensor,
    sample_indices: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    presence = []
    chemical = []
    target_bonds = 0
    for batch_index in range(len(batch["keys"])):
        rows = torch.where(sample_indices == batch_index)[0]
        target_mask = batch["bond_mask"][batch_index]
        current_presence, current_chemical = p19e1.build_pair_targets(
            pairs[rows].cpu(),
            batch["bond_atom_refs"][batch_index][target_mask],
            batch["bond_types"][batch_index][target_mask],
        )
        presence.append(current_presence)
        chemical.append(current_chemical)
        target_bonds += int(target_mask.sum())
    return (
        torch.cat(presence).to(device),
        torch.cat(chemical).to(device),
        target_bonds,
    )


def _summary(rows: list[dict]) -> dict:
    result = p19e1._summary(rows)
    result["atom_exact"] = len(rows)
    result["atom_exact_rate"] = 1.0 if rows else 0.0
    return result


@torch.inference_mode()
def evaluate_gt_atom(
    model: DirectAtomPairBond,
    vision,
    atom,
    samples: list[dict],
    *,
    threshold: float,
    device: torch.device,
) -> dict:
    model.eval()
    collator = G3StructuralCollator(
        maximum_atoms=MAXIMUM_ATOMS,
        maximum_bonds=MAXIMUM_BONDS,
        image_size=768,
    )
    rows = []
    digest = hashlib.sha256()
    for start in range(0, len(samples), EVAL_BATCH_SIZE):
        current = samples[start : start + EVAL_BATCH_SIZE]
        batch = collator(current)
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
            output = model(
                feature_map,
                atom_features,
                anchors,
                atom_scores,
                pairs,
                sample_indices,
            )
        probabilities = output["presence_logits"].sigmoid().float().cpu()
        predicted_types = output["chemical_logits"].argmax(-1).cpu()
        for index, sample in enumerate(current):
            pair_rows = torch.where(sample_indices == index)[0]
            target_mask = batch["bond_mask"][index]
            presence, chemical = p19e1.build_pair_targets(
                pairs[pair_rows].cpu(),
                batch["bond_atom_refs"][index][target_mask],
                batch["bond_types"][index][target_mask],
            )
            current_probabilities = probabilities[pair_rows.cpu()]
            current_types = predicted_types[pair_rows.cpu()]
            key = str(sample["key"])
            operation = str(
                sample["render"].get("render_parameters", {}).get(
                    "stress_operation", "unknown"
                )
            )
            digest.update(key.encode())
            digest.update(current_probabilities.numpy().tobytes())
            rows.append(
                {
                    "sample_key": key,
                    "variant": str(sample["render"]["variant"]),
                    "stress_operation": operation,
                    **p19e1._sample_metrics(
                        current_probabilities,
                        current_types,
                        presence,
                        chemical,
                        threshold=threshold,
                    ),
                }
            )
    return {
        "overall": _summary(rows),
        "by_variant": {
            variant: _summary([row for row in rows if row["variant"] == variant])
            for variant in VARIANTS
        },
        "stress_by_operation": {
            operation: _summary(
                [row for row in rows if row["stress_operation"] == operation]
            )
            for operation in sorted(
                {
                    row["stress_operation"]
                    for row in rows
                    if row["variant"] == "stress"
                }
            )
        },
        "prediction_sha256": digest.hexdigest(),
        "rows": rows,
    }


def _metric_key(metrics: dict) -> tuple[int, int, int, int, int]:
    overall = metrics["overall"]
    return (
        int(overall["skeleton_exact"]),
        int(metrics["by_variant"]["stress"]["skeleton_exact"]),
        int(overall["true_positive"]),
        -int(overall["false_positive"]),
        -int(overall["false_negative"]),
    )


def capability_checks(
    baseline: dict,
    best: dict,
    paired: dict,
    *,
    frozen_states_unchanged: bool,
    repeat_exact: bool,
    split_audit: dict,
) -> dict[str, bool]:
    old_nonstress = sum(
        int(baseline["by_variant"][variant]["skeleton_exact"])
        for variant in ("coordgen", "original", "rdkit")
    )
    new_nonstress = sum(
        int(best["by_variant"][variant]["skeleton_exact"])
        for variant in ("coordgen", "original", "rdkit")
    )
    return {
        "dev640_selected": split_audit["selected_images"] == DEV_IMAGES,
        "dev_pilot_molecule_overlap_zero": (
            split_audit["pilot_molecule_overlap"] == 0
        ),
        "dev_pilot_scaffold_overlap_zero": (
            split_audit["pilot_scaffold_overlap"] == 0
        ),
        "all_frozen_states_unchanged": frozen_states_unchanged,
        "best_checkpoint_repeat_exact": repeat_exact,
        "overall_no_regression": (
            best["overall"]["skeleton_exact"]
            >= baseline["overall"]["skeleton_exact"]
        ),
        "nonstress_no_regression": new_nonstress >= old_nonstress,
        "stress_gain_min8": (
            best["by_variant"]["stress"]["skeleton_exact"]
            >= baseline["by_variant"]["stress"]["skeleton_exact"] + 8
        ),
        "crossed_gain_min2": (
            best["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            >= baseline["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            + 2
        ),
        "near_nonbonded_gain_min2": (
            best["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            >= baseline["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            + 2
        ),
        "paired_rescue_exceeds_break": (
            paired["overall"]["rescue"] > paired["overall"]["break"]
        ),
    }


def _checkpoint(
    model: DirectAtomPairBond,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    samples_seen: int,
    metrics: dict,
    split_audit: dict,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p19e3-direct-atom-pair-presence-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": samples_seen,
        "threshold": THRESHOLD,
        "metrics": {key: value for key, value in metrics.items() if key != "rows"},
        "initialized_from_old_direct": False,
        "chemical_loss_trained": False,
        "split_audit": split_audit,
        "training_args": {
            "train_images": TRAIN_IMAGES,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "evaluate_every_images": EVALUATE_EVERY_IMAGES,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "threshold": THRESHOLD,
            "seed": SEED,
            "all_unordered_pairs": True,
            "presence_only": True,
        },
        "sources": {
            "pilot_manifest": str(PILOT_MANIFEST.resolve()),
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "full_manifest": str(FULL_MANIFEST.resolve()),
            "full_manifest_sha256": FULL_MANIFEST_SHA256,
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "vision_checkpoint": str(VISION_CHECKPOINT.resolve()),
            "vision_checkpoint_sha256": VISION_CHECKPOINT_SHA256,
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E3 output is not empty: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256),
        (VISION_CHECKPOINT, VISION_CHECKPOINT_SHA256),
        (OLD_DIRECT_CHECKPOINT, OLD_DIRECT_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E3 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E3 requires CUDA BF16")
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

    pilot_rows = p18e0._jsonl(PILOT_MANIFEST)
    full_rows = p18e0._jsonl(FULL_MANIFEST)
    dev_rows, split_audit = select_internal_dev640(pilot_rows, full_rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    DEV_MANIFEST.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    split_audit = {
        **split_audit,
        "dev_manifest": str(DEV_MANIFEST.resolve()),
        "dev_manifest_sha256": file_sha256(DEV_MANIFEST),
    }
    _atomic_json(OUTPUT / "split-audit.json", split_audit)

    source_payload = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if Path(source_payload["training_args"]["grounded_graph_checkpoint"]).resolve() != (
        VISION_CHECKPOINT.resolve()
    ):
        raise ValueError("P19-E3 source is bound to another vision checkpoint")
    vision, atom, bond, set_head, _ = load_components(
        str(SOURCE_CHECKPOINT), str(VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    frozen_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
    }
    old_payload = torch.load(
        OLD_DIRECT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    old_model = DirectAtomPairBond(
        DirectAtomPairBondConfig(**old_payload["config"])
    ).to(device)
    old_model.load_state_dict(old_payload["model"], strict=True)
    old_model.eval().requires_grad_(False)
    if float(old_payload["validation"]["threshold"]) != OLD_THRESHOLD:
        raise ValueError("P19-E3 old baseline threshold differs")

    dev_samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(dev_rows), accepted_split="train"
        )
    )
    baseline = evaluate_gt_atom(
        old_model,
        vision,
        atom,
        dev_samples,
        threshold=OLD_THRESHOLD,
        device=device,
    )
    del old_model
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
    initial = evaluate_gt_atom(
        model,
        vision,
        atom,
        dev_samples,
        threshold=THRESHOLD,
        device=device,
    )
    _atomic_json(
        OUTPUT / "baseline.json",
        {
            "old_v0103": {key: value for key, value in baseline.items() if key != "rows"},
            "scratch_step0": {key: value for key, value in initial.items() if key != "rows"},
        },
    )
    best_metrics = initial
    best_key = _metric_key(initial)
    _atomic_torch_save(
        OUTPUT / "best.pt",
        _checkpoint(
            model,
            optimizer,
            step=0,
            samples_seen=0,
            metrics=initial,
            split_audit=split_audit,
        ),
    )

    collator = G3StructuralCollator(
        maximum_atoms=MAXIMUM_ATOMS,
        maximum_bonds=MAXIMUM_BONDS,
        image_size=768,
    )
    source = p1a.iter_manifest_samples(
        p18e0._manifest_records(pilot_rows), accepted_split="train"
    )
    pending = []
    samples_seen = 0
    step = 0
    target_bonds_seen = 0
    pair_rows_seen = 0
    running = Counter()
    history = []

    def train_batch(samples: list[dict]) -> None:
        nonlocal samples_seen, step, target_bonds_seen, pair_rows_seen, best_metrics, best_key
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        anchors = batch["atom_anchors"].to(device)
        mask = batch["atom_mask"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            feature_map = vision.forward_stroke_resolution(pixels)
            atom_output = atom(feature_map)
            atom_features, atom_scores = p19e0._sample_atom_features(
                atom_output, anchors, mask
            )
        pairs, sample_indices = enumerate_unordered_atom_pairs(mask)
        presence, _, target_bonds = _pair_targets_for_batch(
            pairs, sample_indices, batch, device
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(
                feature_map,
                atom_features,
                anchors,
                atom_scores,
                pairs,
                sample_indices,
            )
            losses = balanced_presence_loss_by_sample(
                output["presence_logits"], presence, sample_indices
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P19-E3 gradient is not finite")
        optimizer.step()
        step += 1
        samples_seen += len(samples)
        target_bonds_seen += target_bonds
        pair_rows_seen += len(pairs)
        for name, value in losses.items():
            running[name] += float(value.detach()) * len(samples)
        if samples_seen % EVALUATE_EVERY_IMAGES == 0:
            metrics = evaluate_gt_atom(
                model,
                vision,
                atom,
                dev_samples,
                threshold=THRESHOLD,
                device=device,
            )
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
                        samples_seen=samples_seen,
                        metrics=metrics,
                        split_audit=split_audit,
                    ),
                )
            row = {
                "step": step,
                "samples_seen": samples_seen,
                "selected_best": selected,
                "metrics": {key: value for key, value in metrics.items() if key != "rows"},
                "mean_losses": {
                    name: value / EVALUATE_EVERY_IMAGES
                    for name, value in running.items()
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            running.clear()
            print(json.dumps(row, separators=(",", ":")), flush=True)

    for sample in source:
        pending.append(sample)
        if len(pending) == TRAIN_BATCH_SIZE:
            train_batch(pending)
            pending = []
    if pending:
        train_batch(pending)
    if samples_seen != TRAIN_IMAGES:
        raise RuntimeError(f"P19-E3 training exposure differs: {samples_seen}")

    final = evaluate_gt_atom(
        model,
        vision,
        atom,
        dev_samples,
        threshold=THRESHOLD,
        device=device,
    )
    _atomic_torch_save(
        OUTPUT / "last.pt",
        _checkpoint(
            model,
            optimizer,
            step=step,
            samples_seen=samples_seen,
            metrics=final,
            split_audit=split_audit,
        ),
    )
    best_payload = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    best_model = DirectAtomPairBond(CONFIG).to(device)
    best_model.load_state_dict(best_payload["model"], strict=True)
    best_model.eval().requires_grad_(False)
    first_replay = evaluate_gt_atom(
        best_model,
        vision,
        atom,
        dev_samples,
        threshold=THRESHOLD,
        device=device,
    )
    second_replay = evaluate_gt_atom(
        best_model,
        vision,
        atom,
        dev_samples,
        threshold=THRESHOLD,
        device=device,
    )
    best_metrics = first_replay
    paired = p19e2.paired_summary(baseline["rows"], best_metrics["rows"])
    paired_rows = paired.pop("rows")
    frozen_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
    }
    checks = capability_checks(
        baseline,
        best_metrics,
        paired,
        frozen_states_unchanged=frozen_before == frozen_after,
        repeat_exact=(
            first_replay["prediction_sha256"]
            == second_replay["prediction_sha256"]
        ),
        split_audit=split_audit,
    )
    decision = (
        "direct_atom_pair_presence_pilot20k_go"
        if all(checks.values())
        else "direct_atom_pair_presence_pilot20k_no_go"
    )
    _atomic_json(OUTPUT / "best-rows.json", best_metrics.pop("rows"))
    _atomic_json(OUTPUT / "paired-rows.json", paired_rows)
    baseline.pop("rows")
    final.pop("rows")
    result = {
        "schema_version": "g3-p19e3-direct-atom-pair-presence-pilot20k-result-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "baseline_old_v0103": baseline,
        "best": best_metrics,
        "final": final,
        "paired_vs_v0103": paired,
        "history": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "training": {
            "samples_seen": samples_seen,
            "steps": step,
            "pair_rows_seen": pair_rows_seen,
            "target_bonds_seen": target_bonds_seen,
            "all_target_bonds_covered": True,
        },
        "split": split_audit,
        "best_checkpoint": str((OUTPUT / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(OUTPUT / "best.pt"),
        "best_step": int(best_payload["global_step"]),
        "best_samples_seen": int(best_payload["samples_seen"]),
        "initialized_from_old_direct": False,
        "gt_atom_oracle_conditioned": True,
        "chemical_loss_trained": False,
        "dense_bond_forward": False,
        "incidence_forward": False,
        "distance_or_neighbor_pruning": False,
        "fresh640_read": False,
        "sealed_test_read": False,
        "confirm_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
