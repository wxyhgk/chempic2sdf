#!/usr/bin/env python3
"""Build scaffold-isolated hard/easy caches for the P6 residual cycle."""

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

from chemvector.components.recovery.structural_context_peak import StructuralContextPeakHead, anchor_centered_context_crops
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p4e9_recovery_oracle as oracle
from chemvector.evaluation.audits.atoms import build_g3_p5e3_refine_capacity_full_train as p5e3
from chemvector.evaluation.audits.atoms import build_g3_p5e8_generic_direction_capacity as p5e8
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save, _autocast_context
from chemvector.g3_structural_data import file_sha256
from chemvector.data import structure_labels


AUDIT_NAME = "chemvector-g3-p6e8-residual-cycle-data"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
ELIGIBLE_MANIFEST = p5e3.OUTPUT / "eligible_train.jsonl"
ELIGIBLE_MANIFEST_SHA256 = "a8e62d8698ed53113d8b4cbc62cd91b0a1cad75eb1defd76d2bd23145ebac8df"
P6E1_CHECKPOINT = p6e3.P6E1_CHECKPOINT
P6E1_CHECKPOINT_SHA256 = p6e3.P6E1_CHECKPOINT_SHA256
HARD_CACHE = OUTPUT / "hard.pt"
EASY_CACHE = OUTPUT / "easy.pt"
CAP_OPTIONS = (1.0, 2.0, 4.0, 8.0)
EASY_HASH_MODULUS = 64
MINIMUM_HARD = 4_000
MINIMUM_MOLECULES = 2_000
MINIMUM_SCAFFOLDS = 1_500
SEED = 20260868


class ResidualCycleCollator(p1a.G3AtomCollator):
    def __call__(self, samples: list[dict]) -> dict:
        batch = super().__call__(samples)
        roles, degrees = [], []
        for sample in samples:
            labels = structure_labels(sample)
            atoms = sorted(labels["atoms"], key=lambda row: int(row["atom_id"]))
            id_to_index = {int(atom["atom_id"]): index for index, atom in enumerate(atoms)}
            current_degrees = [0] * len(atoms)
            for bond in labels["bonds"]:
                for name in ("begin_atom", "end_atom"):
                    atom_id = int(bond[name])
                    if atom_id in id_to_index:
                        current_degrees[id_to_index[atom_id]] += 1
            roles.append(["TEXT" if atom["visible_text"] else "IMPLICIT" for atom in atoms])
            degrees.append(current_degrees)
        batch["atom_roles"] = roles
        batch["atom_degrees"] = degrees
        return batch


def scaffold_split(scaffold: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"{SEED}:split:{scaffold}".encode()).digest()[:8], "big") % 10
    return "train" if value < 8 else "calibration" if value == 8 else "holdout"


def degree_bucket(degree: int) -> str:
    return "degree1" if degree <= 1 else "degree2" if degree == 2 else "degree3plus"


def choose_residual_cap(required: list[float]) -> float | None:
    if not required:
        return None
    values = torch.tensor(required)
    for cap in CAP_OPTIONS:
        if int((values <= cap).sum()) * 100 >= len(values) * 95:
            return cap
    return None


def _empty_cache() -> dict[str, list]:
    return defaultdict(list)


def _append_cache(cache: dict[str, list], images, scores, offsets, logits, mask, rows: list[int], metadata: list[dict]) -> None:
    if not rows:
        return
    indices = torch.tensor(rows, device=images.device)
    cache["image_crops"].append(images[indices].cpu().half())
    cache["score_crops"].append(scores[indices].cpu().half())
    cache["candidate_offsets"].append(offsets[indices].cpu().half())
    cache["candidate_peak_logits"].append(logits[indices].cpu().half())
    cache["candidate_mask"].append(mask[indices].cpu())
    cache["offset_targets"].append(torch.tensor([metadata[row]["offset_target"] for row in rows], dtype=torch.float32))
    for name in ("sample_key", "molecule_id", "scaffold", "split", "variant", "target_role", "degree_bucket", "target_index", "required_residual_cap"):
        cache[name].extend([metadata[row][name] for row in rows])


def _finish_cache(cache: dict[str, list], lineage: dict) -> dict:
    tensors = ("image_crops", "score_crops", "candidate_offsets", "candidate_peak_logits", "candidate_mask", "offset_targets")
    result = {name: torch.cat(cache[name]) for name in tensors}
    result.update({name: values for name, values in cache.items() if name not in tensors})
    result.update(lineage)
    return result


def _easy_selected(sample_key: str, target_index: int) -> bool:
    digest = hashlib.sha256(f"{SEED}:easy:{sample_key}:{target_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % EASY_HASH_MODULUS == 0


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E8 output already exists: {OUTPUT}")
    for path, expected in ((ELIGIBLE_MANIFEST, ELIGIBLE_MANIFEST_SHA256), (P6E1_CHECKPOINT, P6E1_CHECKPOINT_SHA256), (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256)):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E8 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P6-E8 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rows = [json.loads(line) for line in ELIGIBLE_MANIFEST.read_text().splitlines() if line]
    metadata_by_key = {str(row["sample_key"]): row for row in rows}
    manifest, manifest_audit = load_g3_sample_manifest(ELIGIBLE_MANIFEST, expected_sha256=ELIGIBLE_MANIFEST_SHA256, expected_samples=p5e3.EXPECTED_ELIGIBLE_IMAGES, expected_molecules=p5e3.EXPECTED_ELIGIBLE_MOLECULES, expected_variants=p5e8.cycle.VARIANTS)
    device = torch.device("cuda")
    payload = torch.load(P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device); peak_model.load_state_dict(payload["model"], strict=True); peak_model.eval()
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = ResidualCycleCollator(maximum_atoms=128, image_size=768)
    hard_cache, easy_cache = _empty_cache(), _empty_cache()
    counts = Counter(); hard_variants = Counter(); hard_splits = Counter(); hard_strata = Counter(); started = time.perf_counter(); processed = 0
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=8, accepted_split="train"):
            objects, stroke = p1t.model_inputs(atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False)
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device)); center = field_output["center_logits"].detach().float()
            for batch_index, key_value in enumerate(batch["keys"]):
                key = str(key_value); base_meta = metadata_by_key[key]
                anchors = batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float().to(device)
                sources=[]; targets=[]; atom_indices=[]; item_meta=[]
                for target_index, target in enumerate(anchors):
                    offset = p5e8.deterministic_offset(key, target_index).to(device); source = target - offset; pixels = source * 768
                    if bool((pixels < 48).any() or (pixels > 720).any()):
                        counts["context_outside"] += 1; continue
                    distances = torch.linalg.vector_norm((anchors - source) * 768, dim=1); order = distances.argsort(stable=True)
                    if int(order[0]) != target_index:
                        counts["another_nearest"] += 1; continue
                    if len(order) > 1 and float(distances[order[1]] - distances[order[0]]) < 1.0:
                        counts["margin"] += 1; continue
                    split = scaffold_split(str(base_meta["scaffold"])); role = batch["atom_roles"][batch_index][target_index]; degree = degree_bucket(batch["atom_degrees"][batch_index][target_index])
                    sources.append(source); targets.append(offset); atom_indices.append(target_index)
                    item_meta.append({"sample_key":key,"molecule_id":str(base_meta["molecule_id"]),"scaffold":str(base_meta["scaffold"]),"split":split,"variant":str(base_meta["variant"]),"target_role":role,"degree_bucket":degree,"target_index":target_index,"offset_target":[float(value) for value in offset],"required_residual_cap":0.0})
                if not sources:
                    continue
                source_tensor = torch.stack(sources); images = anchor_centered_context_crops(darkness[batch_index:batch_index+1], source_tensor[None])[0]; score_crops = anchor_centered_context_crops(center[batch_index:batch_index+1], source_tensor[None])[0]
                zeros = torch.zeros(len(sources), device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    heatmaps = peak_model(images, score_crops, zeros, zeros)
                candidate_offsets, candidate_logits, candidate_mask = p6e3.candidate_tensors(heatmaps)
                target_tensor = torch.stack(targets); distances = torch.linalg.vector_norm((candidate_offsets - target_tensor[:,None]) * 768, dim=-1).masked_fill(~candidate_mask, torch.inf)
                valid_logits = candidate_logits.float().masked_fill(~candidate_mask, 0.0); count = candidate_mask.sum(dim=1, keepdim=True).float(); mean = valid_logits.sum(dim=1, keepdim=True) / count; variance = (((valid_logits - mean).square() * candidate_mask).sum(dim=1, keepdim=True) / count).sqrt().clamp_min(1e-4); prior = (valid_logits - mean) / variance
                hard_rows=[]; easy_rows=[]
                for row, target_index in enumerate(atom_indices):
                    counts["valid"] += 1
                    compatible = candidate_mask[row] & (distances[row] <= 6.0)
                    if not bool(compatible.any()):
                        counts["unreachable"] += 1; continue
                    if float(distances[row,0]) > 6.0:
                        best_positive = float(prior[row].masked_fill(~compatible, -torch.inf).max()); required = max(0.0, (float(prior[row,0]) - best_positive) * 0.5); item_meta[row]["required_residual_cap"] = required; hard_rows.append(row)
                        hard_variants[item_meta[row]["variant"]] += 1; hard_splits[item_meta[row]["split"]] += 1; hard_strata[(item_meta[row]["split"], item_meta[row]["variant"], item_meta[row]["target_role"], item_meta[row]["degree_bucket"])] += 1
                    else:
                        counts["base_correct"] += 1
                        if _easy_selected(key, target_index): easy_rows.append(row)
                _append_cache(hard_cache, images, score_crops, candidate_offsets, candidate_logits, candidate_mask, hard_rows, item_meta)
                _append_cache(easy_cache, images, score_crops, candidate_offsets, candidate_logits, candidate_mask, easy_rows, item_meta)
            processed += len(batch["keys"])
            if processed % 4_000 == 0:
                print(json.dumps({"processed":processed,"hard":sum(hard_splits.values()),"easy":sum(len(value) if isinstance(value,list) else 0 for name,value in easy_cache.items() if name=="sample_key"),"elapsed_seconds":time.perf_counter()-started}), flush=True)
    lineage={"p6e1_checkpoint_sha256":P6E1_CHECKPOINT_SHA256,"eligible_manifest_sha256":ELIGIBLE_MANIFEST_SHA256,"vision_checkpoint_sha256":vision_sha}
    hard = _finish_cache(hard_cache, lineage); easy = _finish_cache(easy_cache, lineage)
    train_required=[float(value) for value,split in zip(hard["required_residual_cap"],hard["split"],strict=True) if split=="train"]
    selected_cap=choose_residual_cap(train_required); molecules=set(hard["molecule_id"]); scaffolds=set(hard["scaffold"])
    split_scaffolds={split:{scaffold for scaffold,current in zip(hard["scaffold"],hard["split"],strict=True) if current==split} for split in ("train","calibration","holdout")}
    overlap=bool((split_scaffolds["train"]&split_scaffolds["calibration"])|(split_scaffolds["train"]&split_scaffolds["holdout"])|(split_scaffolds["calibration"]&split_scaffolds["holdout"]))
    checks={"hard_min4000":len(hard["offset_targets"])>=MINIMUM_HARD,"hard_molecules_min2000":len(molecules)>=MINIMUM_MOLECULES,"hard_scaffolds_min1500":len(scaffolds)>=MINIMUM_SCAFFOLDS,"all_splits_nonempty":all(hard_splits[split]>0 for split in ("train","calibration","holdout")),"scaffold_overlap_zero":not overlap,"residual_cap_selected":selected_cap is not None,"easy_at_least_hard":len(easy["offset_targets"])>=len(hard["offset_targets"]),"dev_confirm_external_sealed_not_read":True}
    decision="residual_cycle_data_go" if all(checks.values()) else "residual_cycle_data_no_go"
    OUTPUT.mkdir(parents=True); _atomic_torch_save(HARD_CACHE,hard); _atomic_torch_save(EASY_CACHE,easy)
    audit={"schema_version":"g3-p6e8-residual-cycle-data-0.1","decision":decision,"checks":checks,"selected_residual_cap":selected_cap,"capacity":{"images":processed,"hard":len(hard["offset_targets"]),"easy":len(easy["offset_targets"]),"molecules":len(molecules),"scaffolds":len(scaffolds),"hard_splits":dict(sorted(hard_splits.items())),"hard_variants":dict(sorted(hard_variants.items())),"hard_strata":{str(key):value for key,value in sorted(hard_strata.items())},"counts":dict(sorted(counts.items()))},"protocol":{"split":"sha256 scaffold 80/10/10","easy_hash_modulus":EASY_HASH_MODULUS,"cap_options":CAP_OPTIONS,"cap_train_coverage":0.95,"dev_confirm_external_sealed_read":False},"lineage":{**lineage,"manifest_audit":manifest_audit},"elapsed_seconds":time.perf_counter()-started}
    _atomic_json(OUTPUT/"audit.json",audit); print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
