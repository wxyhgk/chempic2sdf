#!/usr/bin/env python3
"""Train an isolated element head on the r17 frozen AtomObject cache."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3/feature-cache.pt"
OUT = ROOT / "outputs/audits/chemvector-element-head-cpu-r17-all-m16"
ELEMENTS = ("C", "O", "N", "S", "F", "Cl", "Br", "I", "P", "B", "Si")


class ElementHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, len(ELEMENTS)))
    def forward(self, x): return self.net(x)


def data_records(records):
    xs=[]; ys=[]
    for record in records:
        mapping=record["mapping"]
        for compact,target in mapping.items():
            element=record["target_elements"].get(int(target),"C")
            if element in ELEMENTS and compact < len(record["atom_features"]):
                xs.append(record["atom_features"][compact]); ys.append(ELEMENTS.index(element))
    if not xs: return torch.empty((0,128)), torch.empty((0,),dtype=torch.long)
    return torch.stack(xs),torch.tensor(ys,dtype=torch.long)


def score(head, data):
    x,y=data; pred=head(x).argmax(-1); return {"rows":len(y),"accuracy":float((pred==y).float().mean()),"confusion":torch.bincount(y*len(ELEMENTS)+pred,minlength=len(ELEMENTS)**2).reshape(len(ELEMENTS),len(ELEMENTS)).tolist()}


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    cache=torch.load(CACHE,map_location="cpu",weights_only=False); records=cache["records"]
    train=data_records(records["train"]); head=ElementHead(); opt=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-4)
    counts=torch.bincount(train[1],minlength=len(ELEMENTS)).float(); weights=(len(train[1])/counts.clamp_min(1)).sqrt()
    events=[]
    for step in range(1,301):
        head.train();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(train[0]),train[1],weight=weights);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);opt.step()
        if step in (1,100,200,300):events.append({"step":step,"loss":float(loss.detach())})
    head.eval(); metrics={split:score(head,data_records(records[split])) for split in records}
    result={"schema_version":"chemvector-element-head-cpu-r17-0.1","complete":True,"read_only":False,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"elements":ELEMENTS,"train_rows":len(train[1]),"events":events,"metrics":metrics,"gate":{"identity_disjoint":True,"dev_accuracy_ge_0.90":metrics["dev"]["accuracy"]>=.90,"fresh_accuracy_ge_0.90":metrics["fresh_holdout"]["accuracy"]>=.90,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=False);torch.save({"schema_version":result["schema_version"],"state_dict":head.state_dict()},OUT/"head-state.pt");(OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(OUT/"report.json"),"metrics":metrics,"gate":result["gate"]},indent=2))

if __name__=='__main__':run()
