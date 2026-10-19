"""CPU-only feasibility probe for a local bond-patch residual scorer.

The line-first detector, endpoint clustering and text aliasing are frozen.  A
small scorer is fitted on target-free geometric/raster features for the
already-proposed candidate edges; labels are joined only after proposal
construction and are used for reporting/training loss in this isolated audit.
No candidate is added, no production head/checkpoint is touched, and the
sealed test is never referenced.  This is deliberately a ceiling probe: if
the frozen proposal cannot reach a GT edge, a patch scorer cannot recover it.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib, json, math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-local-patch-cpu-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0
SEED = 20261020
STEPS = 300
MAX_TRAIN_IMAGES = 1000
CLUSTER_RADIUS = 10.0


def _boxes(base: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for glyph in json.loads(Path(str(base) + ".text-records.json").read_text()).get("glyphs", []):
        aid, box = int(glyph["atom_id"]), glyph.get("bbox")
        if not box:
            continue
        b = np.asarray(box, dtype=np.float32)
        if aid in out:
            old = out[aid]; out[aid] = np.array([min(old[0], b[0]), min(old[1], b[1]), max(old[2], b[2]), max(old[3], b[3])], np.float32)
        else:
            out[aid] = b
    return out


def _ink_features(image: np.ndarray, p0: np.ndarray, p1: np.ndarray, boxes: dict[int, np.ndarray]) -> np.ndarray:
    d = p1 - p0; length = float(np.linalg.norm(d)); unit = d / max(length, 1e-6)
    normal = np.array([-unit[1], unit[0]], np.float32)
    ts = np.linspace(0.10, 0.90, 17, dtype=np.float32)
    occ = []
    for off in (-5., -3., -1., 0., 1., 3., 5.):
        q = p0[None] + ts[:, None] * d[None] + off * normal[None]
        x = np.clip(np.rint(q[:, 0]).astype(int), 0, image.shape[1]-1)
        y = np.clip(np.rint(q[:, 1]).astype(int), 0, image.shape[0]-1)
        occ.append(float((image[y, x] < 200).mean()))
    center = (p0 + p1) / 2.
    glyph_dist = IMAGE
    for b in boxes.values():
        dx = max(float(b[0] - center[0]), 0., float(center[0] - b[2]))
        dy = max(float(b[1] - center[1]), 0., float(center[1] - b[3]))
        glyph_dist = min(glyph_dist, math.hypot(dx, dy))
    return np.asarray([
        length / IMAGE, math.cos(2 * math.atan2(float(d[1]), float(d[0]))),
        math.sin(2 * math.atan2(float(d[1]), float(d[0]))), center[0] / IMAGE, center[1] / IMAGE,
        *occ, glyph_dist / IMAGE,
    ], np.float32)


def _sample(row: dict) -> dict:
    key, base = row["sample_key"], DATA / "samples" / row["sample_key"]
    image = np.asarray(Image.open(Path(str(base) + ".png")).convert("L"))
    segments = lsd._alt_detect(Image.fromarray(image), config=LineSegmentRescueConfig(maximum_candidates=512))
    raw = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2), np.float32)
    labels, centers = cluster_endpoints(raw, CLUSTER_RADIUS)
    boxes = _boxes(base)
    alias = {i: i for i in range(len(centers))}; pred = [np.asarray(c, np.float32) for c in centers]
    for aid, b in boxes.items():
        c = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
        hits = [(float(np.linalg.norm(centers[i] - c)), i) for i in range(len(centers)) if np.linalg.norm(centers[i] - c) <= 24.]
        if hits:
            tid = -(aid + 1); pred.append(c)
            for _, i in sorted(hits): alias[i] = tid
    ids = sorted(set(alias.values())); id_to_i = {n: i for i, n in enumerate(ids)}
    pred2 = []
    for n in ids:
        if n < 0:
            b = boxes[-n - 1]; pred2.append(np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32))
        else: pred2.append(np.asarray(centers[n], np.float32))
    pred2 = np.asarray(pred2, np.float32)
    layers = json.loads(Path(str(base) + ".layers.json").read_text()); graph = json.loads(Path(str(base) + ".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    mapping = strict_candidate_to_target(pred2, targets)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    candidates: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for i in range(len(segments)):
        a, b = int(labels[2*i]), int(labels[2*i+1]); na, nb = alias.get(a, a), alias.get(b, b)
        if na == nb or na not in id_to_i or nb not in id_to_i: continue
        pair = tuple(sorted((id_to_i[na], id_to_i[nb])))
        candidates[pair].append(_ink_features(image, raw[2*i], raw[2*i+1], boxes))
    features, labels_out, records, reachable, typed = [], [], [], set(), set()
    # The map below is retained exclusively for post-hoc labels/metrics.  It
    # is never consulted while constructing candidate pairs or feature rows.
    bond_types = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))): str(b["chemical_type"]).upper()
                  for b in graph["bonds"]}
    for pair, vals in candidates.items():
        # Max support and multiplicity are target-independent.  Aggregate all
        # parallel segments into one candidate-pair row.
        arr = np.stack(vals); feat = np.concatenate([arr.max(0), arr.mean(0), np.asarray([len(vals) / 4.], np.float32)])
        features.append(feat)
        mapped = tuple(sorted((mapping[pair[0]], mapping[pair[1]]))) if pair[0] in mapping and pair[1] in mapping and mapping[pair[0]] != mapping[pair[1]] else None
        labels_out.append(int(mapped in gt if mapped is not None else False))
        if mapped is not None: reachable.add(mapped)
        if mapped is not None and mapped in gt: typed.add(mapped)
        records.append({"pair": pair, "feature": feat, "mapped": mapped,
                        "label": int(mapped in gt if mapped is not None else False),
                        "multiplicity": len(vals),
                        "expected_type": bond_types.get(mapped) if mapped is not None else None})
    return {"records": records, "gt": gt, "reachable": reachable,
            "mapped_nodes": len(mapping), "bond_types": bond_types,
            "profile": row.get("profile_id", "unknown"), "split": row["split"],
            "sample_key": key}


class TinyScorer(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 24), nn.GELU(), nn.LayerNorm(24), nn.Linear(24, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def _fit_eval(train: list[tuple[np.ndarray, int]], hold: list[tuple[np.ndarray, int]]) -> dict:
    torch.manual_seed(SEED); x = torch.tensor(np.stack([a for a, _ in train]), dtype=torch.float32); y = torch.tensor([b for _, b in train], dtype=torch.float32)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-4); xz = (x - mean) / std
    model = TinyScorer(x.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    pos = max(float(y.sum()), 1.); weight = torch.tensor(min(float((1-y).sum()) / pos, 20.))
    for _ in range(STEPS):
        loss = nn.functional.binary_cross_entropy_with_logits(model(xz), y, pos_weight=weight); opt.zero_grad(); loss.backward(); opt.step()
    def score(rows): return torch.sigmoid(model((torch.tensor(np.stack([a for a, _ in rows]), dtype=torch.float32)-mean)/std)).detach().numpy(), np.asarray([b for _, b in rows], int)
    ts, ty = score(train); hs, hy = score(hold)
    # Select lowest threshold attaining the strict edge precision floor on train.
    thresholds = np.unique(np.r_[np.linspace(0.05, .99, 95), ts]); admissible=[]
    for t in thresholds:
        sel = ts >= t; p = float((ty[sel].sum()) / max(sel.sum(), 1)); r = float(ty[sel].sum() / max(ty.sum(), 1)); admissible.append((p, r, float(t), int(sel.sum())))
    good = [z for z in admissible if z[0] >= .965917]; chosen = max(good, key=lambda z: (z[1], -z[2])) if good else max(admissible, key=lambda z: (z[0], z[1]))
    t = chosen[2]
    def metric(s, y):
        sel=s>=t; tp=int(y[sel].sum()); n=int(sel.sum()); return {"candidates":int(len(y)),"positives":int(y.sum()),"selected":n,"true_selected":tp,"edge_precision":tp/max(n,1),"reachable_edge_recall":tp/max(int(y.sum()),1)}
    return {"parameters":sum(p.numel() for p in model.parameters()),"steps":STEPS,"threshold":t,"train":metric(ts,ty),"holdout":metric(hs,hy),"train_threshold_precision":chosen[0],"train_threshold_recall":chosen[1], "model": model, "mean": mean, "std": std}


def _roundtrip(records_by_sample: list[dict], fit: dict) -> dict:
    """Evaluate complete candidate-pair graph assembly after thresholding.

    Only target-free record fields are scored.  Target mappings are joined in
    this function for metrics, and therefore cannot leak into training.
    """
    model, mean, std, threshold = fit["model"], fit["mean"], fit["std"], float(fit["threshold"])
    by_profile = defaultdict(lambda: {"images": 0, "gt_bonds": 0, "selected": 0,
                                      "mapped_selected": 0, "true_edges": 0,
                                      "dangling_endpoints": 0, "type_correct": 0,
                                      "type_evaluated": 0, "sanitize_ok": 0})
    for item in records_by_sample:
        rows = item["records"]
        if not rows:
            continue
        x = torch.tensor(np.stack([r["feature"] for r in rows]), dtype=torch.float32)
        with torch.inference_mode():
            score = torch.sigmoid(model((x - mean) / std)).numpy()
        selected = [r for r, s in zip(rows, score) if float(s) >= threshold]
        gt = item["gt"]
        mapped = [r["mapped"] for r in selected]
        mapped_edges = {m for m in mapped if m is not None}
        true = mapped_edges & gt
        # A candidate with either endpoint outside strict post-hoc ownership is
        # a dangling prediction.  This is an evaluation-only definition.
        dangling = sum(2 if r["mapped"] is None else 0 for r in selected)
        d = by_profile[item["profile"]]
        d["images"] += 1; d["gt_bonds"] += len(gt); d["selected"] += len(selected)
        d["mapped_selected"] += len(mapped_edges); d["true_edges"] += len(true)
        d["dangling_endpoints"] += dangling
        # Multiplicity is a deterministic patch-level type proxy, not a
        # learned type head.  Keep it explicit so type quality cannot be
        # confused with the keep/drop residual result.
        for r in selected:
            if r["mapped"] in gt:
                d["type_evaluated"] += 1
                m = r["multiplicity"]
                pred = "SINGLE" if m <= 1 else ("DOUBLE" if m == 2 else "TRIPLE")
                d["type_correct"] += int(pred == item["bond_types"].get(r["mapped"]))
        # Construct the predicted graph with target atom identities solely for
        # post-hoc chemistry sanity.  All predicted bonds use SINGLE: this
        # isolates structural residual feasibility from a type router.
        try:
            from rdkit import Chem
            mol = Chem.RWMol()
            for atom in item.get("atom_records", []):
                a = Chem.Atom(str(atom.get("element", "C"))); a.SetFormalCharge(int(atom.get("formal_charge", 0))); mol.AddAtom(a)
            for a, b in mapped_edges:
                if a != b and a < mol.GetNumAtoms() and b < mol.GetNumAtoms(): mol.AddBond(int(a), int(b), Chem.BondType.SINGLE)
            Chem.SanitizeMol(mol.GetMol()); d["sanitize_ok"] += 1
        except Exception:
            pass
    out = {}
    for profile, d in sorted(by_profile.items()):
        out[profile] = {**d,
            "edge_precision": d["true_edges"] / max(d["mapped_selected"], 1),
            "edge_recall": d["true_edges"] / max(d["gt_bonds"], 1),
            "dangling_rate": d["dangling_endpoints"] / max(2*d["selected"], 1),
            "type_accuracy": d["type_correct"] / max(d["type_evaluated"], 1),
            "sanitize_rate": d["sanitize_ok"] / max(d["images"], 1)}
    total = {k: sum(v[k] for v in by_profile.values()) for k in next(iter(by_profile.values()), {})}
    if total:
        total.update({"edge_precision": total["true_edges"] / max(total["mapped_selected"], 1),
                      "edge_recall": total["true_edges"] / max(total["gt_bonds"], 1),
                      "dangling_rate": total["dangling_endpoints"] / max(2*total["selected"], 1),
                      "type_accuracy": total["type_correct"] / max(total["type_evaluated"], 1),
                      "sanitize_rate": total["sanitize_ok"] / max(total["images"], 1)})
    return {"all": total, "by_profile": out}


def main() -> None:
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"]
    rows.sort(key=lambda r: r["sample_key"]); train_rows = [r for r in rows if r["split"] == "train"][:MAX_TRAIN_IMAGES]; hold_rows = [r for r in rows if r["split"] == "fresh_holdout"]
    train, hold = [], []; aggregate = {}; train_items = []; hold_items = []
    for i, row in enumerate(train_rows + hold_rows, 1):
        item = _sample(row)
        # Atom metadata is read only for post-hoc RDKit roundtrip.  It never
        # enters the target-free pair feature vector.
        graph = json.loads(Path(str(DATA / "samples" / row["sample_key"]) + ".graph.json").read_text())
        item["atom_records"] = graph["atoms"]
        aggregate[row["sample_key"]] = {"gt":len(item["gt"]),"reachable":len(item["reachable"]),"candidate_pairs":len(item["records"]),"mapped_nodes":item["mapped_nodes"],"split":row["split"]}
        pairs = [(r["feature"], r["label"]) for r in item["records"]]
        (train if row["split"] == "train" else hold).extend(pairs)
        (train_items if row["split"] == "train" else hold_items).append(item)
        if i % 200 == 0: print(json.dumps({"processed":i,"total":len(train_rows)+len(hold_rows)}), flush=True)
    # The scorer result is reported, but the conservative reachability ceiling
    # remains the decision criterion and prevents a misleading GPU go.
    fit = _fit_eval(train, hold)
    # Full graph roundtrip is deliberately evaluated once on the fixed fresh
    # holdout.  It is separate from pair-level fit metrics.
    roundtrip = _roundtrip(hold_items, fit)
    total_gt=sum(v["gt"] for v in aggregate.values() if v["split"]=="fresh_holdout"); total_reach=sum(v["reachable"] for v in aggregate.values() if v["split"]=="fresh_holdout")
    # Remove live torch objects before serializing the report.
    fit_report = {k:v for k,v in fit.items() if k not in {"model", "mean", "std"}}
    allm = roundtrip["all"]
    profile_gate = all(v["edge_recall"] >= .75 and v["sanitize_rate"] >= .90 for v in roundtrip["by_profile"].values()) if roundtrip["by_profile"] else False
    report={"schema_version":"chemvector-identity-aligned-line-first-local-patch-cpu-r5-0.2","complete":True,"read_only":True,"training_performed":True,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"proposal_contract":"frozen LSD endpoint clusters + glyph aliases -> frozen candidate pairs; local raster features only; labels joined post-hoc; no new nodes","split":{"train_images":len(train_rows),"fresh_holdout_images":len(hold_rows)},"fit":fit_report,"roundtrip":roundtrip,"holdout_ceiling":{"gt_bonds":total_gt,"reachable_candidate_bonds":total_reach,"edge_recall_ceiling":total_reach/max(total_gt,1)},"gate":{"edge_precision_ge_0.965917":allm.get("edge_precision",0)>=.965917,"edge_recall_gt_0.850031":allm.get("edge_recall",0)>.850031,"profile_recall_sanitize":profile_gate,"dangling_le_0.05":allm.get("dangling_rate",1)<=.05,"type_accuracy_ge_0.90":allm.get("type_accuracy",0)>=.90,"sanitize_ge_0.95":allm.get("sanitize_rate",0)>=.95,"gpu_go":False,"production_go":False,"reason":"local patch scorer cannot recover frozen unreachable endpoints and type/sanitize gates are evaluated explicitly"},"dataset_fingerprints":{"samples_jsonl_sha256":hashlib.sha256((DATA/"samples.jsonl").read_bytes()).hexdigest()}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); (OUT/"per_sample.json").write_text(json.dumps(aggregate,indent=2)+"\n"); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
