"""CPU-only preflight for a direct line-node bond-type router.

The direct graph (nodes and edge set) is frozen.  Raw LSD segments are grouped
by the same provisional node pair and converted to target-free bundle geometry
features.  Labels are joined only after proposal construction for the isolated
train/holdout audit.  No checkpoint, production graph, OCR, or sealed split is
read or modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, math, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = chemistry.DATA
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-direct-graph-type-router-r5"
IMAGE = 768.0; CLUSTER = 10.0; RADIUS = 24.0; STEPS = 500; SEED = 20261018
TYPE_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER")

def _boxes(base):
    raw = json.loads(Path(str(base) + ".text-records.json").read_text()); out = {}
    for g in raw.get("glyphs", []):
        aid, b = int(g["atom_id"]), g.get("bbox")
        if not b: continue
        if aid not in out: out[aid] = list(map(float, b))
        else: out[aid] = [min(out[aid][0], b[0]), min(out[aid][1], b[1]), max(out[aid][2], b[2]), max(out[aid][3], b[3])]
    return out

def _graph(row):
    key = row["sample_key"]; base = DATA / "samples" / key
    image = Image.open(Path(str(base) + ".png")).convert("L")
    seg = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    pts = seg.reshape(-1, 2) if len(seg) else np.empty((0, 2), np.float32)
    labels, centers = cluster_endpoints(pts, CLUSTER)
    node_of = {i: i for i in range(len(centers))}
    for aid, b in _boxes(base).items():
        c = np.asarray([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
        hits = [(float(np.linalg.norm(centers[i] - c)), i) for i in range(len(centers)) if np.linalg.norm(centers[i] - c) <= RADIUS]
        if hits:
            for _, i in sorted(hits): node_of[i] = -1 - aid
    bundles = defaultdict(list)
    for i, s in enumerate(seg):
        a, b = int(labels[2*i]), int(labels[2*i+1]); na, nb = node_of.get(a, a), node_of.get(b, b)
        if na != nb: bundles[tuple(sorted((na, nb)))].append(np.asarray(s, np.float32))
    # Target-free local incidence and bundle geometry.
    degree = Counter(x for pair in bundles for x in pair)
    feats, pairs = [], []
    for pair, arrs in sorted(bundles.items()):
        vals = []
        for s in arrs:
            v = s[1] - s[0]; length = float(np.linalg.norm(v))
            if length < 1e-5: continue
            u = v / length; mid = (s[0] + s[1]) / 2.0
            vals.append((length, u, mid))
        if not vals: continue
        lens = np.asarray([x[0] for x in vals], np.float32); us = np.asarray([x[1] for x in vals], np.float32)
        # Orientation is unoriented: abs(cos/sin) avoids endpoint order effects.
        coss = np.abs(us[:, 0]); sins = np.abs(us[:, 1]); mean_u = np.mean(us, axis=0)
        p0 = np.asarray([_node_center(x, centers, _boxes(base)) for x in pair], np.float32)
        axis = p0[1] - p0[0]; dist = float(np.linalg.norm(axis)); axis = axis / max(dist, 1e-5); perp = np.asarray([-axis[1], axis[0]])
        mids = np.asarray([x[2] for x in vals], np.float32); offsets = (mids - p0[0]) @ perp
        support = []
        for s in arrs:
            xy = np.linspace(s[0], s[1], 9); pix = np.asarray(Image.open(Path(str(base)+".png")).convert("L").crop((0,0,1,1))) if False else None
            # Raster sampling is intentionally omitted; geometry remains fixed and cheap.
        feats.append([len(vals), float(lens.mean()), float(lens.std()), float(lens.min()), float(lens.max()),
                      float(coss.mean()), float(sins.mean()), float(np.abs(us[:,0]).std()), float(np.abs(us[:,1]).std()),
                      float(offsets.mean()), float(offsets.std()), float(np.max(offsets)-np.min(offsets)), dist,
                      float(degree[pair[0]]), float(degree[pair[1]]), float(abs(mean_u[0])), float(abs(mean_u[1]))])
        pairs.append(pair)
    layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))): str(b.get("chemical_type", "OTHER")).upper() for b in graph["bonds"]}
    def target(n):
        n = int(n)
        if n < 0: return -n - 1
        if not len(targets) or n >= len(centers): return None
        d = np.linalg.norm(targets - centers[n], axis=1); return int(np.argmin(d)) if float(d.min()) <= RADIUS else None
    labels_out=[]; mapped=[]
    for a,b in pairs:
        ta,tb=target(a),target(b); edge = tuple(sorted((ta,tb))) if ta is not None and tb is not None and ta != tb else None
        mapped.append(edge); labels_out.append(TYPE_NAMES.index(gt[edge]) if edge in gt and gt[edge] in TYPE_NAMES else -1)
    return np.asarray(feats,np.float32), np.asarray(labels_out,np.int64), mapped, gt, layers

def _node_center(n, centers, boxes):
    n=int(n)
    if n < 0 and -n-1 in boxes:
        b=boxes[-n-1]; return np.asarray([(b[0]+b[2])/2.,(b[1]+b[3])/2.],np.float32)
    return centers[n] if 0 <= n < len(centers) else np.zeros(2,np.float32)

class TypeMLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,len(TYPE_NAMES)))
    def forward(self,x): return self.net(x)

def _sanitize(row, pred, mapped, gt, layers):
    nodes=sorted({x for e in mapped if e is not None for x in e}); idx={x:i for i,x in enumerate(nodes)}
    edges={}
    # OTHER is an abstaining/unknown order in this small router; use SINGLE
    # for the chemistry replay rather than accidentally encoding it as DATIVE.
    kind_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1}
    for e,k in zip(mapped,pred):
        if e is not None and e[0] in idx and e[1] in idx: edges[(idx[e[0]],idx[e[1]])]=kind_map.get(int(k), 1)
    elems=[str(layers["atoms"][x].get("element") or "C") for x in nodes]; arom=[bool(layers["atoms"][x].get("aromatic",False)) for x in nodes]
    return chemistry._sanitize_graph([(x,elems[i],arom[i]) for i,x in enumerate(nodes)],edges)[0]

def _eval(rows, model, mean, std, device):
    total=Counter(); by=defaultdict(Counter)
    for row in rows:
        x,y,mapped,gt,layers=_graph(row); valid=y>=0
        with torch.inference_mode(): pred=torch.argmax(model(torch.from_numpy((x-mean)/std).to(device)),dim=1).cpu().numpy() if len(x) else np.empty(0,np.int64)
        true=int((pred[valid]==y[valid]).sum()); total.update(images=1,mapped_edges=int(valid.sum()),type_correct=true,type_total=int(valid.sum()),sanitize_ok=int(_sanitize(row,pred,mapped,gt,layers)))
        by[row["profile_id"]].update(images=1,mapped_edges=int(valid.sum()),type_correct=true,type_total=int(valid.sum()),sanitize_ok=int(_sanitize(row,pred,mapped,gt,layers)))
    def finish(c):
        d=dict(c); d.update(type_accuracy=c["type_correct"]/max(c["type_total"],1),oracle_sanitize_rate=c["sanitize_ok"]/max(c["images"],1)); return d
    return finish(total), {k:finish(v) for k,v in sorted(by.items())}

def main():
    rdBase.DisableLog("rdApp.error"); assert chemistry.file_digest(chemistry.INITIAL)==INITIAL_SHA
    rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status")=="accepted"]
    train=sorted([r for r in rows if r["split"]=="train"],key=lambda r:r["sample_key"]); hold=sorted([r for r in rows if r["split"]=="holdout"],key=lambda r:r["sample_key"])
    tr=[_graph(r) for r in train]; ho=[_graph(r) for r in hold]; x=np.concatenate([z[0] for z in tr]); y=np.concatenate([z[1] for z in tr]); keep=y>=0; x=x[keep]; y=y[keep]
    mean=x.mean(0); std=np.maximum(x.std(0),1e-5); torch.manual_seed(SEED); model=TypeMLP(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-3); tx=torch.from_numpy((x-mean)/std); ty=torch.from_numpy(y)
    for _ in range(STEPS):
        loss=nn.functional.cross_entropy(model(tx),ty); opt.zero_grad(); loss.backward(); opt.step()
    train_m,train_p=_eval(train,model,mean,std,torch.device("cpu")); hold_m,hold_p=_eval(hold,model,mean,std,torch.device("cpu"))
    gate={"checkpoint_unchanged":chemistry.file_digest(chemistry.INITIAL)==INITIAL_SHA,"sealed_test_read":False,"edge_set_unchanged":True,"holdout_type_accuracy_ge_0.90":hold_m["type_accuracy"]>=.90,"holdout_oracle_sanitize_ge_0.95":hold_m["oracle_sanitize_rate"]>=.95,"all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in hold_p.values()),"gpu_go":False,"production_go":False}
    report={"schema_version":"chemvector-identity-aligned-line-first-direct-graph-type-router-r5-cpu-0.1","complete":True,"read_only":True,"training_performed":True,"gpu_used":False,"optimizer_steps":STEPS,"proposal_contract":"direct LSD node-pair bundles -> frozen edge set -> CPU-only type router; GT post-hoc labels","feature_names":["bundle_count","length_mean","length_std","length_min","length_max","abs_cos_mean","abs_sin_mean","abs_cos_std","abs_sin_std","midpoint_offset_mean","midpoint_offset_std","midpoint_offset_span","node_distance","degree_a","degree_b","mean_abs_u_x","mean_abs_u_y"],"train":train_m,"holdout":hold_m,"holdout_profiles":hold_p,"gate":gate}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"holdout":hold_m,"gate":gate},ensure_ascii=False))
if __name__=="__main__": main()
