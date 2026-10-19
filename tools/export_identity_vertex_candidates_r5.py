"""Export scored, target-independent candidates from the reviewed r5 state."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from chemvector.core.model import ChemVectorConfig, ChemVertexDetector
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.identity_vertex_decoder import decode_dense

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render'
STATE=ROOT/'outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/vertex-state.pt'
OUT=ROOT/'outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/candidates-scored.jsonl'
SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x]
    model,cfg,payload=load_bundle_for_probe(INITIAL,SHA); model.cuda().eval().requires_grad_(False)
    head=ChemVertexDetector(ChemVectorConfig(vision_dims=(96,192,384,512),vertex_feature_dim=64)).cuda()
    state=torch.load(STATE,map_location='cpu',weights_only=False); head.load_state_dict(state['head_state_dict']); model.vision.stages[0][-1].load_state_dict(state['vision_stride4_block_state_dict']); head.eval()
    with OUT.open('x') as f:
      for row in rows:
        sample=load_sample(DATA,row['sample_key'],purpose='diagnostic'); pixels=image_tensor(sample['image'])[None].cuda()
        with torch.inference_mode():
          points=decode_dense(head(model.vision.forward_stroke_resolution(pixels)),threshold=.05,maximum_vertices=128,nms_kernel_size=3,with_scores=True)[0]
        json.dump({'sample_key':row['sample_key'],'source':'shadow','candidates':[{'x':float(x),'y':float(y),'score':float(s)} for x,y,s in points]},f,sort_keys=True); f.write('\n')
    print(OUT)
if __name__=='__main__':run()
