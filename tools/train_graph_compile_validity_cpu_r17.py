#!/usr/bin/env python3
"""Target-free graph-compile validity scorer preflight (r17).

This is an isolated CPU experiment.  Labels are generated only by compiling
frozen candidate graphs and their one-edge-deletion variants; no target graph
fields are read while fitting or routing.  A tiny MLP ranks deletion
transactions for invalid candidates.  Every proposed transaction still needs
an actual compiler success, so this is not an authorization or production
path.
"""
from __future__ import annotations

from collections import Counter
import hashlib, json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges
from tools.audit_r17_target_free_rollback import context_features, target_edges
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_element_context_head_cpu_r17 import ELEMENTS, ElementContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1-element-context-m64/feature-cache.pt"
AROMATIC_STATE = CACHE.parent / "sidecar-state.pt"
TYPE_STATE = ROOT / "outputs/audits/chemvector-type-residual-cpu-r17-all-m64/sidecar-state.pt"
ELEMENT_STATE = ROOT / "outputs/audits/chemvector-element-context-head-cpu-r17-all-m64/head-state.pt"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUT = ROOT / "outputs/audits/chemvector-graph-compile-validity-cpu-r17-v2"
ORDER = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.5}
CAPS = {"C": 4.0, "N": 3.0, "O": 2.0, "S": 6.0, "P": 5.0,
        "F": 1.0, "Cl": 1.0, "Br": 1.0, "I": 1.0, "B": 3.0, "Si": 4.0}


def digest(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def build(rec, ah, th, eh):
    nodes = sorted({int(a) for a, b, _ in rec["pairs"]} | {int(b) for a, b, _ in rec["pairs"]})
    local = {n: i for i, n in enumerate(nodes)}
    with torch.inference_mode():
        ap = ah(rec["features"]).reshape(-1).sigmoid()
        tl = th(rec["type_features"]); el = eh(context_features(rec, nodes))
    raw = {tuple(sorted((int(a), int(b)))) for (a, b, _), p in zip(rec["pairs"], ap) if float(p) >= .5}
    selected = cycle_edges(rec["pairs"], raw)
    tp = tl.argmax(-1).tolist(); ep = el.argmax(-1).tolist()
    tc = tl.softmax(-1).max(-1).values.tolist(); ec = el.softmax(-1).max(-1).values.tolist()
    edges = {}; info = {}; usage = [0.0] * len(nodes)
    for i, (a, b, _) in enumerate(rec["pairs"]):
        k = int(tp[i]); aromatic = tuple(sorted((int(a), int(b)))) in selected
        if k == 3: k = 0
        if aromatic: k = 3
        edge = tuple(sorted((local[int(a)], local[int(b)]))); edges[edge] = 4 if k == 3 else k + 1
        usage[edge[0]] += ORDER[k]; usage[edge[1]] += ORDER[k]
        conf = min(ec[nodes.index(int(a))], ec[nodes.index(int(b))]) * tc[i]
        conf *= float(ap[i]) if aromatic else float(1.0 - ap[i])
        info[edge] = {"confidence": conf, "pair_index": i}
    elements = [ELEMENTS[i] for i in ep]
    arom_nodes = {u for e in selected for u in e}
    return {"nodes": nodes, "edges": edges, "elements": elements,
            "arom_nodes": arom_nodes, "info": info, "usage": usage,
            "selected": selected}


def compile(g, edges=None):
    em = g["edges"] if edges is None else edges
    # Keep the historical compiler contract exactly: edge keys are local
    # indices while aromatic-node ids are original candidate ids.  This is
    # intentionally not "fixed" in a sidecar experiment; changing it would
    # make the A/B incomparable to the frozen rollback baseline.
    arom_nodes = set(g["arom_nodes"]) if edges is None else {
        u for u in g["arom_nodes"] if any(u in e and em[e] == 4 for e in em)
    }
    payload = [(n, g["elements"][i], n in arom_nodes) for i, n in enumerate(g["nodes"])]
    return sanitize(payload, em)


def vector(g, edges=None):
    em = g["edges"] if edges is None else edges
    n = max(len(g["nodes"]), 1); vals = [0.0] * n
    aromatic = 0; conf = []
    for e, kind in em.items():
        vals[e[0]] += ORDER[3] if kind == 4 else ORDER[kind - 1]
        vals[e[1]] += ORDER[3] if kind == 4 else ORDER[kind - 1]
        aromatic += int(kind == 4); conf.append(g["info"].get(e, {"confidence": 0.0})["confidence"])
    overflow = [max(0.0, vals[i] - CAPS.get(g["elements"][i], 4.0)) for i in range(n)]
    onehot = [sum(x == e for x in g["elements"]) / n for e in ELEMENTS]
    c = conf or [0.0]
    return torch.tensor([len(g["nodes"]) / 16, len(em) / 16, aromatic / 16,
                         min(c), sum(c) / len(c), max(c), min(vals, default=0) / 4,
                         sum(vals) / max(n, 1) / 4, max(vals, default=0) / 4,
                         max(overflow, default=0), sum(overflow), *onehot], dtype=torch.float32)


class ValidityHead(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def fit(head, xs, ys):
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3); events = []
    pos = ys > .5; neg = ~pos
    for step in range(1, 401):
        # balanced deterministic minibatch over all training graphs
        idx = torch.cat((torch.where(pos)[0], torch.where(neg)[0]))
        logits = head(xs[idx]); loss = F.binary_cross_entropy_with_logits(logits, ys[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step()
        if step in (1, 100, 200, 400): events.append({"step": step, "loss": float(loss.detach())})
    return events


def route(g, head, mean, std):
    base = dict(g["edges"]); current = dict(base); base_ok, base_err = compile(g, current)
    transactions = []; attempts = 0; abstain = False
    while not base_ok and current and len(transactions) < 8 and attempts < 64:
        trials = []
        for edge in current:
            trial = dict(current); trial.pop(edge)
            score = float(head(((vector(g, trial) - mean) / std).unsqueeze(0))[0].sigmoid())
            trials.append((score, -g["info"].get(edge, {"confidence": 0})["confidence"], edge, trial))
        accepted = False
        for score, _, edge, trial in sorted(trials, reverse=True):
            attempts += 1; ok, err = compile(g, trial)
            if ok:
                current = trial; base_ok, base_err = True, "none"; accepted = True
                transactions.append({"edge": list(edge), "validity_score": score, "attempts_before_accept": attempts}); break
        if not accepted: abstain = True; break
    return {"baseline_sanitize": bool(compile(g, base)[0]), "route_sanitize": bool(base_ok),
            "baseline_error": compile(g, base)[1], "route_error": base_err,
            "attempts": attempts, "removed": len(transactions), "abstain": abstain,
            "edges": current, "base_edges": base}


def edge_metrics(rows, split):
    total = tp = target = 0
    for row in rows:
        if row["split"] != split: continue
        pred, truth = row["pred_edges"], row["target_edges"]; total += len(pred); target += len(truth); tp += len(set(pred) & set(truth))
    return {"predicted": total, "target": target, "true_positive": tp,
            "precision": tp / max(total, 1), "recall": tp / max(target, 1)}


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != SHA: raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA: raise RuntimeError("cache checkpoint mismatch")
    ast = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False); tst = torch.load(TYPE_STATE, map_location="cpu", weights_only=False); est = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    ah = AromaticContextHead(526); ah.load_state_dict(ast["state_dict"]); ah.eval(); th = TypeContextHead(262); th.load_state_dict(tst["state_dict"]); th.eval(); eh = ElementContextHead(); eh.load_state_dict(est["state_dict"]); eh.eval()
    built = {s: [build(r, ah, th, eh) for r in rs if r["pairs"]] for s, rs in cache["records"].items()}
    # Self-supervised labels: compiler validity only, no target graph fields.
    train_x = []; train_y = []
    for g in built["train"]:
        train_x.append(vector(g)); train_y.append(float(compile(g)[0]))
        for edge in list(g["edges"]):
            trial = dict(g["edges"]); trial.pop(edge); train_x.append(vector(g, trial)); train_y.append(float(compile(g, trial)[0]))
    xs = torch.stack(train_x); ys = torch.tensor(train_y); mean = xs.mean(0); std = xs.std(0).clamp_min(1e-4); head = ValidityHead(xs.shape[1]); events = fit(head, (xs - mean) / std, ys); head.eval()
    fit_rows = []
    for split, gs in built.items():
        for g in gs:
            out = route(g, head, mean, std); target, _ = target_edges(g.get("sample_key", "")) if g.get("sample_key") else (set(), {})
            # sample key is attached below; post-hoc target join is kept separate
            fit_rows.append({"split": split, **out})
    summary = {}
    for split in built:
        rr = [r for r in fit_rows if r["split"] == split]; n = len(rr)
        summary[split] = {"images": n, "baseline_sanitize_rate": sum(r["baseline_sanitize"] for r in rr) / max(n, 1), "route_sanitize_rate": sum(r["route_sanitize"] for r in rr) / max(n, 1), "delta_sanitize": (sum(r["route_sanitize"] for r in rr) - sum(r["baseline_sanitize"] for r in rr)) / max(n, 1), "attempts": sum(r["attempts"] for r in rr), "removed": sum(r["removed"] for r in rr), "abstain_rate": sum(r["abstain"] for r in rr) / max(n, 1)}
    report = {"schema_version": "chemvector-graph-compile-validity-cpu-r17-0.1", "complete": True, "training_performed": True, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "objective": "self-supervised compiler-validity graph scorer", "train_graphs": len(built["train"]), "train_variants": len(train_y), "head_parameters": sum(p.numel() for p in head.parameters()), "events": events, "summary": summary, "policy": {"routing_target_free": True, "compiler_gate_required": True, "edge_set_frozen": True, "production_integration": "not_authorized"}, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL), "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False, "dev_sanitize_ge_rollback": summary["dev"]["route_sanitize_rate"] >= .94921875, "fresh_sanitize_ge_rollback": summary["fresh_holdout"]["route_sanitize_rate"] >= .953125, "production_go": False, "stop_condition": "No GPU/merge unless both held-out sanitize rates strictly exceed rollback and abstain does not increase >0.02."}}
    OUT.mkdir(parents=True); (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n"); torch.save({"state_dict": head.state_dict(), "mean": mean, "std": std, "initial_checkpoint_sha256": SHA}, OUT / "head-state.pt")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": report["gate"]}, indent=2)); return report


if __name__ == "__main__": run()
