#!/usr/bin/env python3
"""CPU label-only preflight for endpoint-conditioned vertex compatibility.

Frozen bond proposals and the reviewed r5 vertex candidates are generated once
from the production checkpoint.  Ground truth is used only after feature
construction to make in-memory labels; no target identity, target coordinate,
or target bond is written to the feature artifact.  This is a separability
gate, not a training run and it never writes a checkpoint.
"""
from __future__ import annotations

from collections import Counter
import hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/candidates-scored.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-compatibility-r5-cpu-preflight-v2/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0
PAIR_RADIUS = 32.0
TARGET_RADIUS = 6.0
ENDPOINT_RADIUS = 12.0
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
SEED = 20261017

class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def target_data(sample):
    atoms = sorted(sample["graph"]["atoms"], key=lambda x: x["atom_id"])
    by_id = {x["atom_id"]: x for x in sample["layers"]["atoms"]}
    points = np.asarray([by_id[x["atom_id"]]["center"] for x in atoms], np.float32)
    bonds = [(int(x["begin_atom"]), int(x["end_atom"])) for x in sample["graph"]["bonds"]]
    return points, bonds

def load_candidates():
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line)
        out[row["sample_key"]] = row["candidates"]
    return out

def make_rows(sample, candidates, bond_endpoints, bond_scores, bond_types, bond_mask):
    """Return target-independent feature rows and in-memory labels."""
    points, target_bonds = target_data(sample)
    cp = np.asarray([[x["x"] * IMAGE, x["y"] * IMAGE] for x in candidates], np.float32)
    cs = np.asarray([x["score"] for x in candidates], np.float32)
    rows, labels = [], []
    # Candidate/endpoint rows are generated without target filtering.  Target
    # geometry is consulted only below, after the feature vector is built, to
    # assign an in-memory diagnostic label.
    ep = bond_endpoints * IMAGE
    active = np.flatnonzero(bond_mask)
    for qi in active:
        proposal = ep[qi]
        best = []
        for bi, (a, b) in enumerate(target_bonds):
            t = points[[a, b]]
            d = min(np.linalg.norm(proposal - t, axis=1).max(), np.linalg.norm(proposal - t[::-1], axis=1).max())
            best.append((float(d), bi))
        best.sort()
        matched_atom = [None, None]
        if best and best[0][0] <= ENDPOINT_RADIUS and (len(best) == 1 or best[1][0] > ENDPOINT_RADIUS):
            bi = best[0][1]
            ta, tb = target_bonds[bi]
            expected = points[[ta, tb]]
            if np.linalg.norm(proposal[0] - expected[0]) <= ENDPOINT_RADIUS and np.linalg.norm(proposal[1] - expected[1]) <= ENDPOINT_RADIUS:
                matched_atom = [ta, tb]
            elif np.linalg.norm(proposal[0] - expected[1]) <= ENDPOINT_RADIUS and np.linalg.norm(proposal[1] - expected[0]) <= ENDPOINT_RADIUS:
                matched_atom = [tb, ta]
        for side, atom_id in enumerate(matched_atom):
            e = proposal[side]
            dist = np.linalg.norm(cp - e[None, :], axis=1)
            for ci in np.flatnonzero(dist <= PAIR_RADIUS):
                # Inputs contain only predicted candidate/endpoint geometry and
                # frozen bond confidence/type; no target-derived value.
                delta = (cp[ci] - e) / IMAGE
                partner = proposal[1 - side]
                plen = max(float(np.linalg.norm(proposal[1] - proposal[0]) / IMAGE), 1e-4)
                proj = float(np.dot(cp[ci] - e, partner - e) / (IMAGE * IMAGE * plen))
                type_vec = np.asarray(bond_types[qi, 4:], np.float32)
                feat = np.concatenate(([cs[ci], dist[ci] / IMAGE, delta[0], delta[1],
                                        float(bond_scores[qi]), plen, proj], type_vec)).astype(np.float32)
                # Label is computed after feature construction and kept in RAM.
                dtarget = np.linalg.norm(points - cp[ci][None, :], axis=1)
                close = np.flatnonzero(dtarget <= TARGET_RADIUS)
                label = int(atom_id is not None and len(close) == 1 and int(close[0]) == atom_id)
                rows.append(feat); labels.append(label)
    return rows, labels

def auprc(y, s):
    order = np.argsort(-s, kind="stable"); y = np.asarray(y)[order]
    pos = max(int(y.sum()), 1); tp = np.cumsum(y); fp = np.cumsum(1-y)
    return float(np.sum((tp / np.maximum(tp + fp, 1)) * y) / pos)

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint digest changed")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cand = load_candidates()
    if not cand or any(float(x["score"]) <= 0 for v in cand.values() for x in v):
        raise RuntimeError("scored candidate artifact is missing or contains zero scores")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    train = sorted([r for r in rows if r["split"] == "train"], key=lambda r:r["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([r for r in rows if r["split"] == "holdout"], key=lambda r:r["sample_key"])[:HOLDOUT_LIMIT]
    # This preflight is deliberately CPU-only; it is a label/separability gate,
    # not a GPU training allocation.
    device = torch.device("cpu")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    split_data = {}
    counts = {}
    for name, chosen in (("train", train), ("holdout", hold)):
        fx, yy = [], []
        for i, row in enumerate(chosen):
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            with torch.inference_mode():
                out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
            bonds = out[4]
            f, y = make_rows(sample, cand[row["sample_key"]], bonds["bond_endpoints"][0].detach().cpu().numpy(), bonds["bond_scores"][0].detach().cpu().numpy(), bonds["bond_types"][0].detach().cpu().numpy(), bonds["bond_mask"][0].detach().cpu().numpy().astype(bool))
            fx.extend(f); yy.extend(y)
        split_data[name] = (np.asarray(fx, np.float32), np.asarray(yy, np.float32))
        counts[name] = {"pairs": len(yy), "positive": int(sum(yy)), "negative": int(len(yy)-sum(yy))}
    x, y = split_data["train"]; xv, yv = split_data["holdout"]
    if len(x) == 0 or y.sum() == 0 or yv.sum() == 0: raise RuntimeError("fixed slice has no positives")
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    head = MLP(x.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, yt = torch.tensor((x-mean)/std), torch.tensor(y)
    rng = np.random.default_rng(SEED); pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
    for _ in range(300):
        ix=np.r_[rng.choice(pos,256,replace=True), rng.choice(neg,256,replace=True)]; rng.shuffle(ix)
        loss=F.binary_cross_entropy_with_logits(head(xt[torch.tensor(ix)]),yt[torch.tensor(ix)])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        st=head(torch.tensor((x-mean)/std)).numpy(); sv=head(torch.tensor((xv-mean)/std)).numpy()
    # One train-only calibrated operating point.  Select the lowest logit that
    # reaches the predeclared 0.85 precision target on train, breaking ties by
    # maximum recall; the resulting scalar is applied unchanged to holdout.
    def metric(yy, ss, threshold):
        sel=ss>=threshold; tp=int((sel & (yy==1)).sum()); fp=int((sel & (yy==0)).sum()); fn=int(((~sel)&(yy==1)).sum())
        return {"selected":int(sel.sum()),"tp":tp,"fp":fp,"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"auprc":auprc(yy,ss)}
    candidates_thresholds=np.unique(st)[::-1]
    eligible=[]
    for threshold in candidates_thresholds:
        m=metric(y,st,float(threshold))
        if m["precision"]>=.85: eligible.append((m["recall"],-float(threshold),float(threshold)))
    if not eligible: raise RuntimeError("train split cannot calibrate 0.85 precision")
    threshold=max(eligible)[2]
    train_metric, hold_metric = metric(y,st,threshold), metric(yv,sv,threshold)
    report={"schema_version":"chemvector-identity-aligned-endpoint-compatibility-cpu-preflight-0.2","complete":True,"training_performed":False,"sidecar_fit_in_cpu":True,"sealed_test_read":False,"deployment_authorized":False,"initial_sha256":INITIAL_SHA,"candidate_artifact_sha256":digest(CANDIDATES),"fixed_slice":{"train_images":len(train),"holdout_images":len(hold),"pair_radius_px":PAIR_RADIUS,"target_radius_px":TARGET_RADIUS,"endpoint_radius_px":ENDPOINT_RADIUS,"threshold_policy":"lowest train logit reaching precision >= 0.85; max recall tie-break"},"counts":counts,"calibrated_threshold":threshold,"metrics":{"train":train_metric,"holdout":hold_metric,"holdout_auprc":auprc(yv,sv)},"gate":{"candidate_artifact_valid":True,"train_precision_ge_085":train_metric["precision"]>=.85,"holdout_precision_ge_085":hold_metric["precision"]>=.85,"holdout_recall_ge_050":hold_metric["recall"]>=.50,"go":False,"next":"only if all calibrated separability gates pass, review one <=800-step isolated compatibility shadow; no production integration"}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report["metrics"]|{"gate":report["gate"]},indent=2)); return report

if __name__ == "__main__": run()
