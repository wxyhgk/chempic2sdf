#!/usr/bin/env python3
"""Audit whether P6-E3 copies peak rank instead of using structural pixels."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.structural_context_peak import StructuralContextPeakHead, anchor_centered_context_crops
from chemvector.components.recovery.structural_peak_selector import StructuralPeakSelector
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import extract_regional_maximum_objects
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p4e9_recovery_oracle as oracle
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e1_structural_context96_mixed as p6e1
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _autocast_context
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p6e4-selector-shortcut"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CHECKPOINT = p6e3.OUTPUT / "step-015000.pt"
CHECKPOINT_SHA256 = "f022ad9db63dd9e8b9ef13c3c8bf28ca800d379b7f7ede706d613629f35513e7"
P6E3_AUDIT = p6e3.OUTPUT / "audit.json"
P6E3_AUDIT_SHA256 = "63cab029cb19c6dddb77667af2ad4907899db19dd0fae7b44c926df80541018f"
ARMS = ("base", "full", "no_visual", "flat_peak")
MINIMUM_COPY_RATE = 0.90
MINIMUM_VISUAL_ABLATION_AGREEMENT = 0.90
SEED = p6e3.SEED


def _distance_metrics(values: list[float]) -> dict:
    return {
        "actions": len(values),
        "within_6px": sum(value <= 6.0 for value in values),
        "recovery_rate_6px": sum(value <= 6.0 for value in values) / len(values),
        "mae_pixels": float(np.mean(values)),
    }


def _paired(records: list[dict], split: str, arm: str) -> dict:
    current = [row for row in records if row["split"] == split]
    changed = [row for row in current if row[f"{arm}_index"] != row["base_index"]]
    return {
        "selection_agreement_with_base": 1.0 - len(changed) / len(current),
        "changed": len(changed),
        "rescue": sum(row["base_distance"] > 6.0 and row[f"{arm}_distance"] <= 6.0 for row in changed),
        "harmful": sum(row["base_distance"] <= 6.0 and row[f"{arm}_distance"] > 6.0 for row in changed),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E4 output already exists: {OUTPUT}")
    for path, expected in ((CHECKPOINT, CHECKPOINT_SHA256), (P6E3_AUDIT, P6E3_AUDIT_SHA256), (p6e3.P6E1_CHECKPOINT, p6e3.P6E1_CHECKPOINT_SHA256)):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E4 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P6-E4 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    peak_payload = torch.load(p6e3.P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device); peak_model.load_state_dict(peak_payload["model"], strict=True); peak_model.eval()
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = StructuralPeakSelector().to(device); model.load_state_dict(payload["model"], strict=True); model.eval()
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    records = []
    started = time.perf_counter()
    with torch.no_grad():
        for split in ("calibration", "holdout"):
            manifest, actions_by_key = p6e1.p5e1._action_manifest(split)
            for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=8, accepted_split="train"):
                objects, stroke = p1t.model_inputs(atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False)
                with _autocast_context(device, "bf16"):
                    field_output = field(objects, stroke)
                peaks = extract_regional_maximum_objects(decode_atom_center_field(field_output), maximum_objects=128)
                observations = recovery_data.build_local_recovery_observations(batch["pixel_values"].to(device), field_output, peaks)
                darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device)); center = field_output["center_logits"].detach().float()
                for batch_index, key_value in enumerate(batch["keys"]):
                    key = str(key_value)
                    for action in actions_by_key[key]:
                        matches = torch.where(observations["candidate_source_ids"][batch_index] == int(action["base_source_id"]))[0]
                        if len(matches) != 1: raise RuntimeError(f"P6-E4 Oracle source replay differs: {key}")
                        index = int(matches[0]); source = observations["candidate_anchors"][batch_index,index]
                        image = anchor_centered_context_crops(darkness[batch_index:batch_index+1], source[None,None])[0]
                        score = anchor_centered_context_crops(center[batch_index:batch_index+1], source[None,None])[0]
                        source_score = observations["candidate_scores"][batch_index,index:index+1]; source_rank = observations["candidate_ranks"][batch_index,index:index+1]
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            heatmap = peak_model(image, score, source_score, source_rank)
                        offsets, peak_logits, mask = p6e3.candidate_tensors(heatmap)
                        arm_inputs = {
                            "full": (image, score, peak_logits),
                            "no_visual": (torch.zeros_like(image), torch.zeros_like(score), peak_logits),
                            "flat_peak": (image, score, torch.zeros_like(peak_logits)),
                        }
                        indices = {"base": 0}
                        for arm, (arm_image, arm_score, arm_peaks) in arm_inputs.items():
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                values = model(arm_image, arm_score, offsets, arm_peaks, mask, source_score, source_rank)
                            indices[arm] = int(values[0].argmax())
                        target = source.new_tensor(action["target_anchor"]) - source
                        distances = {arm: float(torch.linalg.vector_norm((offsets[0,current] - target) * 768)) for arm, current in indices.items()}
                        records.append({"split":split,"sample_key":key,"base_index":0,"base_distance":distances["base"], **{f"{arm}_index":indices[arm] for arm in ("full","no_visual","flat_peak")}, **{f"{arm}_distance":distances[arm] for arm in ("full","no_visual","flat_peak")}})
    summaries = {}
    for split, expected in (("calibration",31),("holdout",67)):
        current = [row for row in records if row["split"] == split]
        if len(current) != expected: raise RuntimeError(f"P6-E4 {split} action count differs")
        summaries[split] = {
            "arms": {arm: _distance_metrics([row[f"{arm}_distance" if arm != "base" else "base_distance"] for row in current]) for arm in ARMS},
            "paired_to_base": {arm: _paired(records, split, arm) for arm in ("full","no_visual","flat_peak")},
            "full_no_visual_selection_agreement": sum(row["full_index"] == row["no_visual_index"] for row in current) / len(current),
        }
    checks = {f"{split}_full_copies_base_min90pct": summaries[split]["paired_to_base"]["full"]["selection_agreement_with_base"] >= MINIMUM_COPY_RATE for split in ("calibration","holdout")}
    checks.update({f"{split}_full_no_visual_agreement_min90pct": summaries[split]["full_no_visual_selection_agreement"] >= MINIMUM_VISUAL_ABLATION_AGREEMENT for split in ("calibration","holdout")})
    decision = "peak_rank_shortcut_confirmed" if all(checks.values()) else "peak_rank_shortcut_not_confirmed"
    OUTPUT.mkdir(parents=True); oracle._atomic_jsonl(OUTPUT / "records.jsonl", records)
    audit = {"schema_version":"g3-p6e4-selector-shortcut-0.1","decision":decision,"checks":checks,"summaries":summaries,"protocol":{"training_performed":False,"arms":ARMS,"minimum_copy_rate":MINIMUM_COPY_RATE,"minimum_visual_ablation_agreement":MINIMUM_VISUAL_ABLATION_AGREEMENT,"dev_confirm_external_sealed_read":False},"lineage":{"p6e3_checkpoint_sha256":CHECKPOINT_SHA256,"p6e3_audit_sha256":P6E3_AUDIT_SHA256,"vision_checkpoint_sha256":vision_sha},"elapsed_seconds":time.perf_counter()-started}
    _atomic_json(OUTPUT / "audit.json", audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
