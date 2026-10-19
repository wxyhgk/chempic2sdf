#!/usr/bin/env python3
"""Tiny CPU bond-type residual A/B after the r10 presence trial."""
from __future__ import annotations
from collections import Counter
import copy, hashlib, json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA, _extract, rows, DirectPairResidual
from tools.train_direct_graph_pair_scorer_cpu_r10_coverage import choose_train
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one, finish

TRIAL = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"
STATE = TRIAL / "scorer-state.pt"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r10"
REPORT = OUT / "report.json"
TYPE_STATE = OUT / "type-state.pt"
STEPS = 300
THRESHOLD = 0.72

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

class TypeResidual(nn.Module):
    def __init__(self, dim=262):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(BOND_TYPES))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x): return self.out(self.norm(x))

def positive_type_rows(items):
    feats=[]; labels=[]; counts=Counter()
    for item in items:
        pairs=item["pairs"]; slots=item["slots"]; mapping=item["mapping"]; types=item["target_type_by_edge"]
        for i in torch.where(item["labels"] > .5)[0].tolist():
            left,right=(int(v) for v in pairs[i].tolist())
            li=int((slots==left).nonzero(as_tuple=False)[0]); ri=int((slots==right).nonzero(as_tuple=False)[0])
            if li not in mapping or ri not in mapping: continue
            edge=tuple(sorted((mapping[li],mapping[ri]))); name=types.get(edge)
            if name not in BOND_TYPES: continue
            feats.append(item["features"][i]); labels.append(BOND_TYPES.index(name)); counts[name]+=1
    if not feats: raise RuntimeError("no positive type rows")
    return torch.stack(feats), torch.tensor(labels,dtype=torch.long), counts

def apply_type(item, scorer, typer):
    with torch.inference_mode():
        # Presence scorer is frozen r10; type residual is the only A/B change.
        item = dict(item)
        item["chemical_logits"] = item["chemical_logits"] + typer(item["features"])
        return item

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != SHA: raise RuntimeError("production checkpoint changed")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261009)
    state=torch.load(STATE,map_location="cpu",weights_only=False); presence=DirectPairResidual(262); presence.load_state_dict(state["state_dict"]); presence.eval()
    model,config,_=load_bundle_for_probe(INITIAL,SHA); model.eval().requires_grad_(False)
    all_rows=rows(); train_rows=choose_train(all_rows); holdout_rows=sorted((r for r in all_rows if r.get("split")=="fresh_holdout"),key=lambda r:r["sample_key"])
    train=[_extract(r,model,config,need_labels=True) for r in train_rows]; hold=[_extract(r,model,config,need_labels=True) for r in holdout_rows]
    x,y,counts=positive_type_rows(train); typer=TypeResidual(x.size(1)); opt=torch.optim.AdamW(typer.parameters(),lr=2e-3,weight_decay=1e-4)
    weights=torch.ones(len(BOND_TYPES));
    for k,v in counts.items(): weights[BOND_TYPES.index(k)] = (sum(counts.values())/max(v,1))**0.5
    loss_events=[]
    for step in range(1,STEPS+1):
        typer.train(); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(typer(x),y,weight=weights); loss.backward(); torch.nn.utils.clip_grad_norm_(typer.parameters(),1.0); opt.step()
        if step in (1,100,200,300): loss_events.append({"step":step,"loss":float(loss.detach())})
    typer.eval(); base_total=Counter(); typed_total=Counter(); typed_profiles={}; base_profiles={}
    for item in hold:
        base_total.update(one(item,presence,THRESHOLD)); typed_total.update(one(apply_type(item,presence,typer),presence,THRESHOLD))
    for profile in sorted({i["row"]["profile_id"] for i in hold}):
        b=Counter(); t=Counter()
        for item in hold:
            if item["row"]["profile_id"]==profile:
                b.update(one(item,presence,THRESHOLD)); t.update(one(apply_type(item,presence,typer),presence,THRESHOLD))
        base_profiles[profile]=finish(b); typed_profiles[profile]=finish(t)
    after=digest(INITIAL); result={"schema_version":"chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r10-0.1","complete":True,"read_only":False,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"train_images":len(train_rows),"holdout_images":len(holdout_rows),"train_type_rows":len(y),"train_type_counts":dict(counts),"type_sidecar":{"class":"TypeResidual","parameters":sum(p.numel() for p in typer.parameters()),"steps":STEPS,"loss_events":loss_events,"class_weights":weights.tolist()},"baseline_holdout":finish(base_total),"typed_holdout":finish(typed_total),"baseline_profiles":base_profiles,"typed_profiles":typed_profiles,"initial_checkpoint_sha256":SHA,"initial_checkpoint_sha256_after":after,"gate":{"checkpoint_unchanged":after==SHA,"sealed_test_read":False,"strict_sanitize_improvement":finish(typed_total)["oracle_sanitize_rate"]>finish(base_total)["oracle_sanitize_rate"],"oracle_sanitize_ge_0.90":finish(typed_total)["oracle_sanitize_rate"]>=.90,"all_profile_sanitize_ge_0.85":all(v["oracle_sanitize_rate"]>=.85 for v in typed_profiles.values()),"production_go":False,"next":"CPU A/B only; no GPU unless chemistry and graph gates pass"}}
    OUT.mkdir(parents=True); torch.save({"schema_version":result["schema_version"],"state_dict":typer.state_dict(),"initial_checkpoint_sha256":SHA},TYPE_STATE); REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(REPORT),"baseline":result["baseline_holdout"],"typed":result["typed_holdout"],"gate":result["gate"]},indent=2,ensure_ascii=False))

if __name__=="__main__": run()
