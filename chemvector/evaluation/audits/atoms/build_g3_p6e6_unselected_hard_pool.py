#!/usr/bin/env python3
"""Scan unselected P5-E8 pairs for P6 top-1-wrong/top-8-reachable actions."""

from __future__ import annotations

from collections import Counter
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
from chemvector.evaluation.audits.atoms import build_g3_p5e8_generic_direction_capacity as p5e8
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _autocast_context
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p6e6-unselected-hard-pool"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
SELECTED_PAIRS = p5e8.OUTPUT / "selected_pairs.jsonl"
SELECTED_PAIRS_SHA256 = p6e3.p6e1.PAIRS_SHA256
MINIMUM_HARD_ACTIONS = 1_000
MINIMUM_HARD_MOLECULES = 500
MINIMUM_HARD_SCAFFOLDS = 500
SEED = 20260866


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if OUTPUT.exists(): raise RuntimeError(f"P6-E6 output already exists: {OUTPUT}")
    for path, expected in ((SELECTED_PAIRS,SELECTED_PAIRS_SHA256),(p6e3.P6E1_CHECKPOINT,p6e3.P6E1_CHECKPOINT_SHA256),(p5e8.FIT_MANIFEST,p5e8.FIT_MANIFEST_SHA256)):
        if file_sha256(path) != expected: raise ValueError(f"P6-E6 input SHA256 differs: {path}")
    if not torch.cuda.is_available(): raise RuntimeError("P6-E6 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda")
    selected_rows = _jsonl(SELECTED_PAIRS)
    selected={(str(row["sample_key"]),int(row["target_index"])) for row in selected_rows}
    scaffold_by_molecule = {
        str(row["molecule_id"]): str(row["scaffold"]) for row in selected_rows
    }
    manifest,_=load_g3_sample_manifest(p5e8.FIT_MANIFEST,expected_sha256=p5e8.FIT_MANIFEST_SHA256,expected_samples=14_000,expected_molecules=3_500,expected_variants=p5e8.cycle.VARIANTS)
    metadata={row.sample_key:row for row in manifest}
    payload=torch.load(p6e3.P6E1_CHECKPOINT,map_location="cpu",weights_only=False); peak_model=StructuralContextPeakHead().to(device); peak_model.load_state_dict(payload["model"],strict=True); peak_model.eval()
    field,atom_head,vision,frozen_block,vision_sha=p4e6._load_model(device); collator=p1a.G3AtomCollator(maximum_atoms=128,image_size=768)
    hard=[]; counts=Counter(); variants=Counter(); started=time.perf_counter(); processed_images=0
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(manifest,collator,batch_size=8,accepted_split="train"):
            objects,stroke=p1t.model_inputs(atom_head,vision,frozen_block,batch["pixel_values"],device=device,training=False)
            with _autocast_context(device,"bf16"): field_output=field(objects,stroke)
            darkness=recovery_data._grayscale_darkness(batch["pixel_values"].to(device)); center=field_output["center_logits"].detach().float()
            for batch_index,key_value in enumerate(batch["keys"]):
                key=str(key_value); anchors=batch["atom_anchors"][batch_index][batch["atom_mask"][batch_index]].float().to(device)
                sources=[]; targets=[]; indices=[]
                for target_index,target in enumerate(anchors):
                    if (key,target_index) in selected: continue
                    offset=p5e8.deterministic_offset(key,target_index).to(device); source=target-offset; pixels=source*768
                    if bool((pixels<16).any() or (pixels>752).any()): counts["crop_outside"]+=1; continue
                    distances=torch.linalg.vector_norm((anchors-source)*768,dim=1); order=distances.argsort(stable=True)
                    if int(order[0])!=target_index: counts["another_nearest"]+=1; continue
                    if len(order)>1 and float(distances[order[1]]-distances[order[0]])<1.0: counts["margin"]+=1; continue
                    sources.append(source); targets.append(offset); indices.append(target_index)
                if not sources: continue
                source_tensor=torch.stack(sources)
                images=anchor_centered_context_crops(darkness[batch_index:batch_index+1],source_tensor[None])[0]
                scores=anchor_centered_context_crops(center[batch_index:batch_index+1],source_tensor[None])[0]
                zeros=torch.zeros(len(sources),device=device)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16): logits=peak_model(images,scores,zeros,zeros)
                candidate_offsets,_,mask=p6e3.candidate_tensors(logits)
                target_tensor=torch.stack(targets); distances=torch.linalg.vector_norm((candidate_offsets-target_tensor[:,None])*768,dim=-1).masked_fill(~mask,torch.inf)
                for row,target_index in enumerate(indices):
                    counts["unselected_valid"]+=1
                    if float(distances[row,0])<=6: counts["base_correct"]+=1; continue
                    if float(distances[row].min())>6: counts["unreachable"]+=1; continue
                    record=metadata[key]
                    scaffold = scaffold_by_molecule.get(record.molecule_id)
                    if scaffold is None:
                        counts["missing_scaffold_metadata"] += 1
                        continue
                    hard.append({"sample_key":key,"molecule_id":record.molecule_id,"scaffold":scaffold,"variant":record.variant,"target_index":target_index,"source_anchor":[float(v) for v in sources[row]],"target_anchor":[float(v) for v in anchors[target_index]],"offset_target":[float(v) for v in targets[row]],"base_distance_pixels":float(distances[row,0]),"oracle_top8_distance_pixels":float(distances[row].min())}); variants[record.variant]+=1
            processed_images+=len(batch["keys"])
            if processed_images%2000<8: print(json.dumps({"images":processed_images,"hard":len(hard),"elapsed_seconds":time.perf_counter()-started}),flush=True)
    molecules={row["molecule_id"] for row in hard}; scaffolds={row["scaffold"] for row in hard}
    checks={"hard_actions_min1000":len(hard)>=MINIMUM_HARD_ACTIONS,"hard_molecules_min500":len(molecules)>=MINIMUM_HARD_MOLECULES,"hard_scaffolds_min500":len(scaffolds)>=MINIMUM_HARD_SCAFFOLDS,"dev_confirm_external_sealed_not_read":True}
    decision="unselected_hard_pool_capacity_go" if all(checks.values()) else "unselected_hard_pool_capacity_no_go"
    OUTPUT.mkdir(parents=True); oracle._atomic_jsonl(OUTPUT/"hard_actions.jsonl",hard)
    audit={"schema_version":"g3-p6e6-unselected-hard-pool-0.1","decision":decision,"checks":checks,"capacity":{"hard_actions":len(hard),"molecules":len(molecules),"scaffolds":len(scaffolds),"variants":dict(sorted(variants.items())),"counts":dict(sorted(counts.items()))},"protocol":{"training_performed":False,"selected_pairs_excluded":len(selected),"hard_definition":"P6-E1 top1 >6px and target-free top8 <=6px","dev_confirm_external_sealed_read":False},"lineage":{"selected_pairs_sha256":SELECTED_PAIRS_SHA256,"p6e1_checkpoint_sha256":p6e3.P6E1_CHECKPOINT_SHA256,"fit_manifest_sha256":p5e8.FIT_MANIFEST_SHA256,"vision_checkpoint_sha256":vision_sha},"elapsed_seconds":time.perf_counter()-started}
    _atomic_json(OUTPUT/"audit.json",audit); print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
