#!/usr/bin/env python3
"""Export target-free frozen atom/bond outputs for the known-clean cohort."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import sys
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
INITIAL=ROOT/'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
MANIFEST=ROOT/os.environ.get('R17_GRAPH_MANIFEST','outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/manifest.json')
SOURCE=ROOT/'outputs/datasets/structure-expansion-batch-12272-20260923/samples'
OUT=ROOT/os.environ.get('R17_GRAPH_ARTIFACT_OUT','outputs/audits/chemvector-r17-known-clean-graph-artifact-v1')

from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(t): return t.detach().cpu().tolist()

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL)!=SHA: raise RuntimeError('initial checkpoint changed')
    rows=json.loads(MANIFEST.read_text())['rows']
    model,config,payload=load_bundle_for_probe(INITIAL,SHA)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval().requires_grad_(False)
    OUT.mkdir(parents=False); artifact=OUT/'graph-artifact.jsonl'
    with artifact.open('w') as stream, torch.inference_mode():
        for index,row in enumerate(rows,1):
            key=row['sample_key']; image=Image.open(SOURCE/(key+'.png')).convert('RGB')
            outputs=structure_forward(model,image_tensor(image)[None].to(device),config,payload['provenance']['atom_threshold'])
            raw, bonds, set_output=outputs[3],outputs[4],outputs[5]
            raw_mask=raw['atom_mask'][0].bool()
            keep=(raw_mask & (set_output['atom_presence_logits'][0] >= payload['provenance']['atom_threshold']))
            bond_mask=bonds['bond_mask'][0].bool()
            record={'sample_key':key,'split':row['split'],'profile_id':row['profile_id'],
                    'atom_points':serial(raw['atom_geometry'][0,:,:2]),
                    'atom_keep_mask':serial(keep),
                    'atom_raw_mask':serial(raw_mask),
                    'bond_endpoints':serial(bonds['bond_endpoints'][0]),
                    'bond_presence_logits':serial(bonds['bond_scores'][0]),
                    'bond_type_ids':serial(torch.argmax(bonds['bond_types'][0],dim=-1)),
                    'bond_type_logits':serial(bonds['bond_types'][0]),
                    'bond_mask':serial(bond_mask)}
            stream.write(json.dumps(record,separators=(',',':'))+'\n')
            if index%256==0: print(json.dumps({'exported':index}),flush=True)
    report={'schema_version':'chemvector-r17-known-clean-graph-artifact-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':device.type=='cuda','sealed_test_read':False,'production_checkpoint_written':False,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'manifest_sha256':digest(MANIFEST),'artifact_sha256':digest(artifact),'rows':len(rows),'device':str(device),'schema_fields':['sample_key','split','profile_id','atom_points','atom_keep_mask','atom_raw_mask','bond_endpoints','bond_presence_logits','bond_type_ids','bond_type_logits','bond_mask'],'target_free':True,'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'artifact_complete':len(rows)>0,'target_free':True,'production_go':False}}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'output':str(OUT),'report':report},indent=2))
if __name__=='__main__': run()
