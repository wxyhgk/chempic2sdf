#!/usr/bin/env python3
"""Calibrate P19-E3 once, then replay on a scaffold-isolated eval640."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from chemvector.training.graph.relations import (
    train_g3_p19e3_direct_atom_pair_presence_pilot20k as p19e3,
)
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e4-presence-threshold-calibration"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = p19e3.OUTPUT / "best.pt"
CHECKPOINT_SHA256 = "ab25e9de7294e48baa22797a5800c9208de4af09d9fd5ef71410773cf9eaf1e0"
P19E3_RESULT = p19e3.OUTPUT / "result.json"
P19E3_RESULT_SHA256 = "e92bc27a602838fdfba543887a1e010e87abc73eac93f3e9f1bb5bd85c962a0c"
PILOT_MANIFEST = p4e3.P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = p4e3.PILOT_MANIFEST_SHA256
FULL_MANIFEST = p4e3.P4E0_ROOT / "full_train.jsonl"
FULL_MANIFEST_SHA256 = p19e3.FULL_MANIFEST_SHA256
OLD_DEV_MANIFEST = p19e3.DEV_MANIFEST
OLD_DEV_MANIFEST_SHA256 = "4e2cefe593ad81c856ef254475309db5fbf33b17361fc24d9a538bc0380ab6f7"
OLD_DIRECT_CHECKPOINT = p19e0.DIRECT_CHECKPOINT
OLD_DIRECT_CHECKPOINT_SHA256 = p19e0.DIRECT_CHECKPOINT_SHA256
CALIBRATION_MANIFEST = OUTPUT / "calibration640.jsonl"
EVAL_MANIFEST = OUTPUT / "eval640.jsonl"
MOLECULES_PER_SPLIT = 160
IMAGES_PER_SPLIT = 640
MAXIMUM_ATOMS = 128
MAXIMUM_BONDS = 192
BATCH_SIZE = 4
OLD_THRESHOLD = 0.85
SEED = 20260810
VARIANTS = ("coordgen", "original", "rdkit", "stress")


def select_calibration_eval(
    pilot_rows: list[dict],
    old_dev_rows: list[dict],
    full_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    excluded_molecules = {
        str(row["molecule_id"]) for row in pilot_rows + old_dev_rows
    }
    excluded_scaffolds = {
        str(row.get("scaffold", ""))
        for row in pilot_rows + old_dev_rows
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
            raise ValueError(f"P19-E4 incomplete molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P19-E4 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        if (
            molecule_id in excluded_molecules
            or not scaffold
            or scaffold in excluded_scaffolds
        ):
            continue
        digest = hashlib.sha256(
            f"{SEED}\0p19e4\0{molecule_id}\0{scaffold}".encode()
        ).hexdigest()
        candidates.append((digest, molecule_id, scaffold))
    candidates.sort()

    def choose(blocked_scaffolds: set[str]) -> list[tuple[str, str, str]]:
        selected = []
        used = set(blocked_scaffolds)
        for candidate in candidates:
            if candidate[2] in used:
                continue
            selected.append(candidate)
            used.add(candidate[2])
            if len(selected) == MOLECULES_PER_SPLIT:
                break
        if len(selected) != MOLECULES_PER_SPLIT:
            raise ValueError("P19-E4 has too few unique-scaffold molecules")
        return selected

    calibration = choose(set())
    calibration_molecules = {item[1] for item in calibration}
    calibration_scaffolds = {item[2] for item in calibration}
    candidates = [item for item in candidates if item[1] not in calibration_molecules]
    evaluation = choose(calibration_scaffolds)
    eval_molecules = {item[1] for item in evaluation}
    eval_scaffolds = {item[2] for item in evaluation}
    calibration_rows = [
        row for row in full_rows if str(row["molecule_id"]) in calibration_molecules
    ]
    eval_rows = [
        row for row in full_rows if str(row["molecule_id"]) in eval_molecules
    ]
    if len(calibration_rows) != IMAGES_PER_SPLIT or len(eval_rows) != IMAGES_PER_SPLIT:
        raise RuntimeError("P19-E4 split image count differs")
    audit = {
        "eligible_molecules": len(grouped),
        "calibration_molecules": len(calibration_molecules),
        "calibration_scaffolds": len(calibration_scaffolds),
        "calibration_images": len(calibration_rows),
        "eval_molecules": len(eval_molecules),
        "eval_scaffolds": len(eval_scaffolds),
        "eval_images": len(eval_rows),
        "calibration_eval_molecule_overlap": len(
            calibration_molecules & eval_molecules
        ),
        "calibration_eval_scaffold_overlap": len(
            calibration_scaffolds & eval_scaffolds
        ),
        "calibration_excluded_scaffold_overlap": len(
            calibration_scaffolds & excluded_scaffolds
        ),
        "eval_excluded_scaffold_overlap": len(eval_scaffolds & excluded_scaffolds),
        "calibration_sample_keys_sha256": hashlib.sha256(
            "".join(f"{row['sample_key']}\n" for row in calibration_rows).encode()
        ).hexdigest(),
        "eval_sample_keys_sha256": hashlib.sha256(
            "".join(f"{row['sample_key']}\n" for row in eval_rows).encode()
        ).hexdigest(),
    }
    return calibration_rows, eval_rows, audit


@torch.inference_mode()
def collect_probabilities(
    model: DirectAtomPairBond,
    vision,
    atom,
    samples: list[dict],
    device: torch.device,
) -> tuple[list[dict], str]:
    model.eval()
    collator = G3StructuralCollator(
        maximum_atoms=MAXIMUM_ATOMS,
        maximum_bonds=MAXIMUM_BONDS,
        image_size=768,
    )
    records = []
    digest = hashlib.sha256()
    for start in range(0, len(samples), BATCH_SIZE):
        current = samples[start : start + BATCH_SIZE]
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
        for index, sample in enumerate(current):
            rows = torch.where(sample_indices == index)[0]
            target_mask = batch["bond_mask"][index]
            presence, _ = p19e1.build_pair_targets(
                pairs[rows].cpu(),
                batch["bond_atom_refs"][index][target_mask],
                batch["bond_types"][index][target_mask],
            )
            current_probabilities = probabilities[rows.cpu()].clone()
            variant = str(sample["render"]["variant"])
            operation = (
                str(
                    sample["render"].get("render_parameters", {}).get(
                        "stress_operation", "unknown"
                    )
                )
                if variant == "stress"
                else None
            )
            key = str(sample["key"])
            digest.update(key.encode())
            digest.update(current_probabilities.numpy().tobytes())
            records.append(
                {
                    "sample_key": key,
                    "variant": variant,
                    "stress_operation": operation,
                    "probabilities": current_probabilities,
                    "presence": presence,
                }
            )
    return records, digest.hexdigest()


def metrics_at_threshold(records: list[dict], threshold: float) -> dict:
    rows = []
    for record in records:
        selected = record["probabilities"] >= threshold
        presence = record["presence"]
        true_positive = selected & presence
        false_positive = selected & ~presence
        false_negative = ~selected & presence
        rows.append(
            {
                "sample_key": record["sample_key"],
                "variant": record["variant"],
                "stress_operation": record["stress_operation"],
                "target_bonds": int(presence.sum()),
                "selected_bonds": int(selected.sum()),
                "true_positive": int(true_positive.sum()),
                "false_positive": int(false_positive.sum()),
                "false_negative": int(false_negative.sum()),
                "skeleton_exact": bool(torch.equal(selected, presence)),
                "typed_exact": False,
            }
        )

    def summary(current: list[dict]) -> dict:
        value = p19e1._summary(current)
        value["atom_exact"] = len(current)
        value["atom_exact_rate"] = 1.0 if current else 0.0
        return value

    return {
        "threshold": threshold,
        "overall": summary(rows),
        "by_variant": {
            variant: summary([row for row in rows if row["variant"] == variant])
            for variant in VARIANTS
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
        "rows": rows,
    }


def select_global_threshold(records: list[dict]) -> tuple[float, dict]:
    candidates = {0.5, OLD_THRESHOLD}
    intervals = []
    for record in records:
        positives = record["probabilities"][record["presence"]]
        negatives = record["probabilities"][~record["presence"]]
        if not len(positives) or not len(negatives):
            raise ValueError("P19-E4 threshold selection needs both pair classes")
        lower = float(negatives.max())
        upper = float(positives.min())
        intervals.append((lower, upper))
        candidates.add(float(np.nextafter(lower, math.inf)))
        candidates.add(upper)
    best_threshold = None
    best_metrics = None
    best_key = None
    for threshold in sorted(value for value in candidates if 0.0 <= value <= 1.0):
        exact = sum(int(lower < threshold <= upper) for lower, upper in intervals)
        if best_key is not None and exact < best_key[0]:
            continue
        metrics = metrics_at_threshold(records, threshold)
        overall = metrics["overall"]
        key = (
            exact,
            float(overall["edge_f1"]),
            float(overall["edge_precision"]),
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("P19-E4 threshold selection failed")
    return best_threshold, {
        "candidate_thresholds": len(candidates),
        "images_with_nonempty_exact_interval": sum(
            int(lower < upper) for lower, upper in intervals
        ),
        "selected": {key: value for key, value in best_metrics.items() if key != "rows"},
        "fixed_050": {
            key: value
            for key, value in metrics_at_threshold(records, 0.5).items()
            if key != "rows"
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E4 output is not empty: {OUTPUT}")
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256),
        (P19E3_RESULT, P19E3_RESULT_SHA256),
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (FULL_MANIFEST, FULL_MANIFEST_SHA256),
        (OLD_DEV_MANIFEST, OLD_DEV_MANIFEST_SHA256),
        (OLD_DIRECT_CHECKPOINT, OLD_DIRECT_CHECKPOINT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E4 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E4 requires CUDA BF16")
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
    old_dev_rows = p18e0._jsonl(OLD_DEV_MANIFEST)
    full_rows = p18e0._jsonl(FULL_MANIFEST)
    calibration_rows, eval_rows, split = select_calibration_eval(
        pilot_rows, old_dev_rows, full_rows
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for path, rows in (
        (CALIBRATION_MANIFEST, calibration_rows),
        (EVAL_MANIFEST, eval_rows),
    ):
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    split = {
        **split,
        "calibration_manifest_sha256": file_sha256(CALIBRATION_MANIFEST),
        "eval_manifest_sha256": file_sha256(EVAL_MANIFEST),
    }
    _atomic_json(OUTPUT / "split.json", split)

    vision, atom, bond, set_head, _ = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = DirectAtomPairBond(DirectAtomPairBondConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    old_payload = torch.load(
        OLD_DIRECT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    old_model = DirectAtomPairBond(
        DirectAtomPairBondConfig(**old_payload["config"])
    ).to(device)
    old_model.load_state_dict(old_payload["model"], strict=True)
    old_model.eval().requires_grad_(False)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "model": module_state_sha256(model),
        "old_model": module_state_sha256(old_model),
    }
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
    calibration_records, calibration_digest = collect_probabilities(
        model, vision, atom, calibration_samples, device
    )
    selected_threshold, selection = select_global_threshold(calibration_records)
    eval_records, eval_digest = collect_probabilities(
        model, vision, atom, eval_samples, device
    )
    old_eval_records, old_eval_digest = collect_probabilities(
        old_model, vision, atom, eval_samples, device
    )
    evaluated = metrics_at_threshold(eval_records, selected_threshold)
    fixed_050 = metrics_at_threshold(eval_records, 0.5)
    baseline = metrics_at_threshold(old_eval_records, OLD_THRESHOLD)
    paired = p19e2.paired_summary(baseline["rows"], evaluated["rows"])
    paired_rows = paired.pop("rows")
    old_nonstress = sum(
        baseline["by_variant"][variant]["skeleton_exact"]
        for variant in ("coordgen", "original", "rdkit")
    )
    new_nonstress = sum(
        evaluated["by_variant"][variant]["skeleton_exact"]
        for variant in ("coordgen", "original", "rdkit")
    )
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "model": module_state_sha256(model),
        "old_model": module_state_sha256(old_model),
    }
    checks = {
        "calibration640_selected": split["calibration_images"] == IMAGES_PER_SPLIT,
        "eval640_selected": split["eval_images"] == IMAGES_PER_SPLIT,
        "calibration_eval_molecule_overlap_zero": (
            split["calibration_eval_molecule_overlap"] == 0
        ),
        "calibration_eval_scaffold_overlap_zero": (
            split["calibration_eval_scaffold_overlap"] == 0
        ),
        "excluded_scaffold_overlap_zero": (
            split["calibration_excluded_scaffold_overlap"] == 0
            and split["eval_excluded_scaffold_overlap"] == 0
        ),
        "calibration_exact_improves_fixed050": (
            selection["selected"]["overall"]["skeleton_exact"]
            > selection["fixed_050"]["overall"]["skeleton_exact"]
        ),
        "eval_overall_no_regression": (
            evaluated["overall"]["skeleton_exact"]
            >= baseline["overall"]["skeleton_exact"]
        ),
        "eval_nonstress_no_regression": new_nonstress >= old_nonstress,
        "eval_stress_gain_min8": (
            evaluated["by_variant"]["stress"]["skeleton_exact"]
            >= baseline["by_variant"]["stress"]["skeleton_exact"] + 8
        ),
        "eval_crossed_gain_min2": (
            evaluated["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            >= baseline["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            + 2
        ),
        "eval_near_nonbonded_gain_min2": (
            evaluated["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            >= baseline["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            + 2
        ),
        "eval_paired_rescue_exceeds_break": (
            paired["overall"]["rescue"] > paired["overall"]["break"]
        ),
        "all_frozen_states_unchanged": states_before == states_after,
    }
    decision = (
        "presence_threshold_calibration_go"
        if all(checks.values())
        else "presence_threshold_calibration_no_go"
    )
    _atomic_json(OUTPUT / "paired-rows.json", paired_rows)
    _atomic_json(OUTPUT / "eval-rows.json", evaluated.pop("rows"))
    fixed_050.pop("rows")
    baseline.pop("rows")
    report = {
        "schema_version": "g3-p19e4-presence-threshold-calibration-0.1",
        "decision": decision,
        "checks": checks,
        "selected_threshold": selected_threshold,
        "calibration": selection,
        "eval_selected_threshold": evaluated,
        "eval_fixed_050": fixed_050,
        "eval_old_v0103": baseline,
        "eval_paired_vs_v0103": paired,
        "digests": {
            "calibration_prediction_sha256": calibration_digest,
            "eval_prediction_sha256": eval_digest,
            "old_eval_prediction_sha256": old_eval_digest,
        },
        "split": split,
        "training": False,
        "threshold_selected_on_calibration_only": True,
        "eval_labels_read_before_threshold_lock": False,
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
