#!/usr/bin/env python3
"""Small CPU/GPU parity audit for frozen r17 atom proposal masks."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / "outputs/audits/chemvector-r17-cpu-gpu-atom-parity-m128/report.json"


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    keys = []
    for split in ("train", "dev", "fresh_holdout"):
        by_profile = {p: [r for r in cache["records"][split] if r["sample_key"].endswith("__" + p)] for p in ("oa00","oa11","oa20","oa33")}
        keys.extend([r["sample_key"] for p in by_profile.values() for r in p[:4]])
    cpu_model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    cpu_model.eval().requires_grad_(False)
    gpu_model = None
    if torch.cuda.is_available():
        gpu_model, _, _ = load_bundle_for_probe(INITIAL, SHA)
        gpu_model.cuda().eval().requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for key in keys:
            pixels=image_tensor(Image.open(SOURCE/f"{key}.png").convert("RGB"))[None]
            co=structure_forward(cpu_model,pixels,config,-0.25)[6]
            cpu_mask=co["atom_mask"][0].cpu(); cpu_points=co["atom_geometry"][0,cpu_mask,:2].cpu()
            if gpu_model is not None:
                go=structure_forward(gpu_model,pixels.cuda(),config,-0.25)[6]
                gpu_mask=go["atom_mask"][0].cpu(); gpu_points=go["atom_geometry"][0,gpu_mask,:2].cpu()
                common=int((cpu_mask & gpu_mask).sum()); union=int((cpu_mask | gpu_mask).sum())
                point_delta=float(torch.cdist(cpu_points,gpu_points).min(-1).values.mean()*768.0) if len(cpu_points) and len(gpu_points) else None
            else:
                gpu_mask=cpu_mask; common=int(cpu_mask.sum()); union=common; point_delta=0.0
            rows.append({"sample_key":key,"cpu_selected":int(cpu_mask.sum()),"gpu_selected":int(gpu_mask.sum()),"mask_intersection":common,"mask_union":union,"jaccard":common/max(union,1),"mean_nearest_point_delta_px":point_delta})
    result={"schema_version":"chemvector-r17-cpu-gpu-atom-parity-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":gpu_model is not None,"sealed_test_read":False,"production_checkpoint_written":False,"rows":rows,"summary":{"images":len(rows),"mean_jaccard":sum(r['jaccard'] for r in rows)/max(len(rows),1),"mean_point_delta_px":sum(r['mean_nearest_point_delta_px'] or 0 for r in rows)/max(len(rows),1),"cpu_gpu_available":gpu_model is not None},"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"parity_informational_only":True,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["summary"],indent=2))

if __name__=="__main__": run()
