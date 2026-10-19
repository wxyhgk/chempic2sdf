"""CPU-only objectness fit over the target-free r6 geometry-port field."""
from __future__ import annotations
from collections import Counter
import hashlib, json, math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import probe_line_first_geometry_ports_r6 as ports
from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"; OUT=ROOT/"outputs/audits/chemvector-identity-aligned-line-first-geometry-ports-objectness-cpu-r6"; INITIAL_SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"; STEPS=300; SEED=20261022; BOX_MARGIN=12.; OFFSETS=(-4.,4.); IMAGE=768.

def make_item(row):
    points, edges, unresolved, segments, boxes, clusters=ports._proposals(row,box_margin=BOX_MARGIN,offset_scales=OFFSETS)
    base=DATA/"samples"/row["sample_key"]; img=np.asarray(Image.open(str(base)+".png").convert("L"),np.uint8); seg=lsd._alt_detect(Image.fromarray(img),config=LineSegmentRescueConfig(maximum_candidates=512)); raw=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32)
    if len(points):
        x=np.clip(np.rint(points[:,0]).astype(int),0,img.shape[1]-1); y=np.clip(np.rint(points[:,1]).astype(int),0,img.shape[0]-1); dark=1.-img[y,x].astype(np.float32)/255.;
        if len(raw):
            dist=np.linalg.norm(points[:,None]-raw[None],axis=-1); near=dist.min(1)/32.; density=(dist<=16.).sum(1)/32.
        else: near=np.ones(len(points),np.float32)*24.; density=np.zeros(len(points),np.float32)
        bxs=[]
        for g in json.loads(Path(str(base)+".text-records.json").read_text()).get("glyphs",[]):
            b=g.get("bbox");
            if b: bxs.append(np.asarray(b,np.float32))
        if bxs:
            bb=np.asarray(bxs); dx=np.maximum(np.maximum(bb[:,0,None]-points[None,:,0],0),points[None,:,0]-bb[:,2,None]); dy=np.maximum(np.maximum(bb[:,1,None]-points[None,:,1],0),points[None,:,1]-bb[:,3,None]); glyph=np.sqrt(dx*dx+dy*dy).min(0)/32.
        else: glyph=np.ones(len(points),np.float32)*24.
        feats=np.column_stack((points/IMAGE,dark,near,density,glyph))
    else: feats=np.empty((0,6),np.float32)
    targets,gt=strict._target(base); return {"points":points,"edges":edges,"unresolved":unresolved,"features":feats.astype(np.float32),"targets":targets,"gt":gt,"profile":row["profile_id"],"split":row["split"],"sample_key":row["sample_key"]}

def labels(item):
    p=item["points"]; t=item["targets"]; d=np.linalg.norm(p[:,None]-t[None],axis=-1) if len(p) and len(t) else np.empty((len(p),len(t))); close=d<=6.; return ((close.sum(1)==1)&close.any(1)).astype(np.float32)

class Head(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,24),nn.GELU(),nn.Linear(24,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def threshold(scores,y):
    best=None
    for t in np.unique(np.r_[np.linspace(.05,.99,95),scores]):
        sel=scores>=t; n=int(sel.sum()); tp=int(y[sel].sum()); p=tp/max(n,1); r=tp/max(int(y.sum()),1)
        if p>=.90 and (best is None or (r,-float(t))>(best[1],-best[0])): best=(float(t),float(r),float(p))
    return best

def evaluate(items,model,mean,std,t):
    out=Counter(); prof={}
    for item in items:
        y=labels(item); s=torch.sigmoid(model(torch.from_numpy((item["features"]-mean)/std))).detach().numpy(); keep=s>=t; tp=int((keep&y.astype(bool)).sum()); out.update(images=1,candidates=len(y),positives=int(y.sum()),selected=int(keep.sum()),true_selected=tp,gt_bonds=len(item["gt"]),selected_edges=0,mapped_edges=0,true_edges=0,dangling_edges=int(item["unresolved"]));
        selected_edges={(a,b) for a,b in item["edges"] if a<len(keep) and b<len(keep) and keep[a] and keep[b]}; out["selected_edges"]+=len(selected_edges)+int(item["unresolved"]); mapping=strict.strict_candidate_to_target(item["points"][keep],item["targets"]); rev={j:i for i,j in enumerate(np.where(keep)[0].tolist())}; mapped=set(); dangling=int(item["unresolved"])
        for a,b in selected_edges:
            aa=rev.get(a); bb=rev.get(b)
            if aa is None or bb is None or aa not in mapping or bb not in mapping: dangling+=1; continue
            x,y0=mapping[aa],mapping[bb]
            if x!=y0: mapped.add(tuple(sorted((x,y0))))
        true=mapped&item["gt"]; out["mapped_edges"]+=len(mapped); out["true_edges"]+=len(true); out["dangling_edges"]+=dangling
        prof.setdefault(item["profile"],[]).append((tp,int(keep.sum()),int(y.sum()),len(true),len(mapped),len(item["gt"]),dangling))
    d=dict(out); d.update({"candidate_precision":out["true_selected"]/max(out["selected"],1),"candidate_recall":out["true_selected"]/max(out["positives"],1),"edge_precision":out["true_edges"]/max(out["mapped_edges"],1),"edge_recall":out["true_edges"]/max(out["gt_bonds"],1),"dangling_rate":out["dangling_edges"]/max(out["selected_edges"],1),"by_profile":{p:{"candidate_precision":sum(v[0] for v in z)/max(sum(v[1] for v in z),1),"candidate_recall":sum(v[0] for v in z)/max(sum(v[2] for v in z),1),"edge_precision":sum(v[3] for v in z)/max(sum(v[4] for v in z),1),"edge_recall":sum(v[3] for v in z)/max(sum(v[5] for v in z),1),"dangling_rate":sum(v[6] for v in z)/max(sum(v[1] for v in z),1)} for p,z in prof.items()}}); return d

def main():
    if hashlib.sha256(ports.INITIAL.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError("checkpoint changed")
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted"]; rows.sort(key=lambda r:r["sample_key"]); items=[make_item(r) for r in rows]; train=[x for x in items if x["split"]=="train"]; dev=[x for x in items if x["split"]=="dev"]; hold=[x for x in items if x["split"]=="fresh_holdout"]; X=np.concatenate([i["features"] for i in train]); y=np.concatenate([labels(i) for i in train]); mean=X.mean(0); std=np.maximum(X.std(0),1e-4); torch.manual_seed(SEED); m=Head(X.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); xt=torch.from_numpy((X-mean)/std); yt=torch.from_numpy(y); pw=torch.tensor(min(float((1-y).sum())/max(float(y.sum()),1.),30.))
    events=[]
    for step in range(1,STEPS+1):
        z=m(xt); loss=F.binary_cross_entropy_with_logits(z,yt,pos_weight=pw); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step();
        if step in (1,100,200,300): events.append({"step":step,"loss":float(loss.detach())})
    with torch.no_grad(): sc=torch.sigmoid(m(xt)).numpy(); chosen=threshold(sc,y); t=chosen[0] if chosen else .99
    metrics={"parameters":sum(p.numel() for p in m.parameters()),"steps":STEPS,"events":events,"threshold":t,"train":evaluate(train,m,mean,std,t),"dev":evaluate(dev,m,mean,std,t),"fresh_holdout":evaluate(hold,m,mean,std,t)}
    report={"schema_version":"chemvector-identity-aligned-line-first-geometry-ports-objectness-cpu-r6-0.1","complete":True,"read_only":True,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"proposal_contract":"frozen target-free geometry ports (glyph boundary + segment offsets); labels post-hoc","config":{"box_margin_px":BOX_MARGIN,"offset_scales_px":OFFSETS},"metrics":metrics,"gate":{"checkpoint_unchanged":True,"sealed_test_read":False,"train_threshold_found":chosen is not None,"holdout_candidate_precision_ge_0.90":metrics["fresh_holdout"]["candidate_precision"]>=.90,"holdout_edge_precision_ge_0.965917":metrics["fresh_holdout"]["edge_precision"]>=.965917,"holdout_edge_recall_gt_0.850031":metrics["fresh_holdout"]["edge_recall"]>.850031,"holdout_dangling_le_0.05":metrics["fresh_holdout"]["dangling_rate"]<=.05,"gpu_go":False,"production_go":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"metrics":metrics,"gate":report["gate"]},indent=2,ensure_ascii=False)); return report

if __name__=="__main__": main()
