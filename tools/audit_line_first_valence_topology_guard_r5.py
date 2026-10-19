#!/usr/bin/env python3
"""CPU-only, target-free valence/topology guard probe.

The guard is deliberately downstream of the frozen line-first edge/type
sidecar.  It sees only predicted candidate ownership, edge confidence, bond
type and graph topology.  Atom elements and bonds are joined *after* the
proposal for audit metrics; they are never consulted by the guard.  This is
therefore a safe probe of whether conservative removal of low-confidence
over-connected edges can contain the remaining RDKit valence failures.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_hungarian_roundtrip_r5 as base  # noqa: E402
from tools.shadow_line_first_edge_bond_type_r5 import EdgeBondMLP  # noqa: E402

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json"
INITIAL_SHA = base.audit.INITIAL_SHA
IMAGE = base.IMAGE
RADIUS, LAMBDA = 24.0, 8.0
TYPE_AROMATIC = 4


def _cycles(edges: dict[tuple[int, int], tuple[float, int]]) -> set[tuple[int, int]]:
    g = nx.Graph(); g.add_edges_from(edges)
    valid: set[tuple[int, int]] = set()
    for start in g.nodes:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in g.neighbors(node):
                if nxt == start and 5 <= len(path) <= 6:
                    valid.update(tuple(sorted((path[i], path[(i + 1) % len(path)]))) for i in range(len(path)))
                elif nxt not in path and len(path) < 6 and nxt >= start:
                    stack.append((nxt, path + [nxt]))
    return valid


def _raw(row, candidates, model, mean, std, threshold):
    key = row["sample_key"]
    features, centers, _ = base.audit._proposals(key)
    vals = candidates[key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in vals], dtype=np.float32)
    owner = base.assign(centers.astype(np.float64), points.astype(np.float64))
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    score = torch.sigmoid(edge_logits).numpy(); kind = (torch.argmax(type_logits, dim=1) + 1).numpy()
    selected: dict[tuple[int, int], tuple[float, int]] = {}; dangling = 0
    for i, (a, b) in enumerate(base.audit._candidate_keys(key)):
        if score[i] < threshold: continue
        if a not in owner or b not in owner: dangling += 1; continue
        ca, cb = owner[a], owner[b]
        if ca == cb: continue
        pair = tuple(sorted((ca, cb))); old = selected.get(pair)
        if old is None or score[i] > old[0]: selected[pair] = (float(score[i]), int(kind[i]))
    # Existing target-free aromatic cycle demotion is the baseline for this probe.
    ring = _cycles(selected)
    selected = {p: (s, 1 if t == TYPE_AROMATIC and p not in ring else t) for p, (s, t) in selected.items()}
    return selected, points, dangling


def guard(selected: dict[tuple[int, int], tuple[float, int]], *, low_confidence: float,
          max_passes: int = 4) -> tuple[dict[tuple[int, int], tuple[float, int]], dict[str, int]]:
    """Remove only low-confidence edges implicated by impossible topology.

    No element/target information is used.  Aromatic edges are treated as
    order 1.5 for the generic order budget.  An edge is removable only when
    its score is <= ``low_confidence``; high-confidence predictions are kept
    and surfaced as an unresolved case rather than silently edited.
    """
    edges = dict(selected); edits = Counter()
    order = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0}
    for _ in range(max_passes):
        changed = False
        inc: dict[int, list[tuple[tuple[int, int], float, int]]] = defaultdict(list)
        for pair, (score, kind) in edges.items():
            for node in pair: inc[node].append((pair, score, kind))
        # Generic, target-free hard budget: no atom can carry >4 bond order.
        for node, items in inc.items():
            total = sum(order.get(kind, 1.0) for _, _, kind in items)
            if total > 4.0 + 1e-6:
                removable = sorted(items, key=lambda x: (x[1], x[0]))
                for pair, score, kind in removable:
                    if total <= 4.0 + 1e-6: break
                    if pair not in edges: continue
                    if score > low_confidence: continue
                    del edges[pair]; total -= order.get(kind, 1.0); edits["order_budget_drop"] += 1; changed = True
        # A multiple bond plus >=2 neighbors is a common hallucinated dangling
        # endpoint.  Drop only a very low-confidence single edge; this keeps a
        # high-confidence nitrile C#C attachment untouched.
        inc = defaultdict(list)
        for pair, (score, kind) in edges.items():
            for node in pair: inc[node].append((pair, score, kind))
        for node, items in inc.items():
            high = [x for x in items if x[2] in (2, 3)]
            if not high or len(items) < 3: continue
            singles = sorted((x for x in items if x[2] == 1), key=lambda x: (x[1], x[0]))
            if singles and singles[0][1] <= low_confidence:
                pair, _, _ = singles[0]; del edges[pair]; edits["low_confidence_multiple_junction_drop"] += 1; changed = True
        if not changed: break
    edits["removed_edges"] = len(selected) - len(edges)
    return edges, dict(edits)


def _metrics(row, selected, points, dangling):
    key = row["sample_key"]; basepath = base.DATA / "samples" / key
    layers = json.loads(basepath.with_suffix(".layers.json").read_text()); graph = json.loads(basepath.with_suffix(".graph.json").read_text())
    targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    c2t = base.audit.strict_candidate_to_target(points[:, :2], targets)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    type_by = {}
    for bond in graph["bonds"]:
        e = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        name = str(bond.get("chemical_type", "OTHER")).upper()
        type_by[e] = base.audit.TYPE_NAMES.index(name) if name in base.audit.TYPE_NAMES else base.audit.TYPE_NAMES.index("OTHER")
    mapped = {}
    for (a, b), (_, kind) in selected.items():
        if a in c2t and b in c2t and c2t[a] != c2t[b]: mapped[tuple(sorted((c2t[a], c2t[b])))] = kind
    true = set(mapped) & gt; type_correct = sum(mapped[e] == type_by[e] for e in true)
    node_ids = sorted({x for p in selected for x in p})
    nodes = [(c, str(layers["atoms"][t].get("element") or "C"), bool(layers["atoms"][t].get("aromatic", False))) for c, t in sorted(c2t.items()) if c in node_ids]
    ni = {c: i for i, (c, _, _) in enumerate(nodes)}
    edges = {(ni[a], ni[b]): kind for (a, b), (_, kind) in selected.items() if a in ni and b in ni}
    ok, error, _ = base.audit._sanitize_graph(nodes, edges)
    return Counter(images=1, selected_edges=len(selected), dangling_edges=dangling, mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), type_correct=type_correct, type_total=len(true), sanitize_ok=int(ok), aromatic_errors=int(error == "aromatic"), valence_errors=int(error == "valence"), all_bonds_recovered=int(true == gt))


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1), "aromatic_error_rate": c["aromatic_errors"] / max(c["images"], 1), "valence_error_rate": c["valence_errors"] / max(c["images"], 1), "all_bonds_recovered_fraction": c["all_bonds_recovered"] / max(c["images"], 1)}


def run():
    rdBase.DisableLog("rdApp.error")
    if base.digest(base.audit.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates = base.audit._candidate_rows(); state = torch.load(base.audit.STATE, map_location="cpu", weights_only=False)
    model = EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std, threshold = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32), float(state["threshold"])
    rows = [json.loads(x) for x in (base.DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"]
    rows = sorted(rows, key=lambda x: x["sample_key"])
    raw = {r["sample_key"]: _raw(r, candidates, model, mean, std, threshold) for r in rows}
    sweeps = {}
    for tau in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        by_split = defaultdict(Counter); by_profile = defaultdict(Counter); edits = Counter()
        for r in rows:
            selected, points, dangling = raw[r["sample_key"]]; guarded, ev = guard(selected, low_confidence=tau); edits.update(ev)
            m = _metrics(r, guarded, points, dangling); by_split[r["split"]].update(m); by_profile[r["split"] + ":" + r["profile_id"]].update(m)
        sweeps[str(tau)] = {"by_split": {k: finish(v) for k, v in sorted(by_split.items())}, "by_profile": {k: finish(v) for k, v in sorted(by_profile.items())}, "edits": dict(edits)}
    reference = sweeps["0.4"]
    hold = reference["by_split"].get("holdout", {})
    hold_profiles = [value for key, value in reference["by_profile"].items() if key.startswith("holdout:")]
    report = {"schema_version": "chemvector-identity-aligned-line-first-valence-topology-guard-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": base.digest(base.audit.INITIAL), "proposal_contract": "LSD -> target-free Hungarian ownership -> frozen edge/type sidecar -> cycle demotion -> target-free valence/topology guard; labels post-hoc only", "config": {"ownership_radius_px": RADIUS, "ownership_score_lambda_px": LAMBDA, "confidence_sweep": [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80], "reference_tau": 0.4}, "sweeps": sweeps, "gate": {"checkpoint_unchanged": base.digest(base.audit.INITIAL) == INITIAL_SHA, "sealed_test_read": False, "full_holdout_edge_precision_ge_0.90": hold.get("edge_precision", 0) >= .90, "full_holdout_edge_recall_ge_0.75": hold.get("edge_recall", 0) >= .75, "full_holdout_type_accuracy_ge_0.90": hold.get("bond_type_accuracy", 0) >= .90, "full_holdout_dangling_le_0.05": hold.get("dangling_rate", 1) <= .05, "full_holdout_sanitize_ge_0.90": hold.get("oracle_sanitize_rate", 0) >= .90, "all_holdout_profiles_sanitize_ge_0.90": bool(hold_profiles) and all(value.get("oracle_sanitize_rate", 0) >= .90 for value in hold_profiles), "all_holdout_profiles_edge_recall_ge_0.75": bool(hold_profiles) and all(value.get("edge_recall", 0) >= .75 for value in hold_profiles), "production_go": False}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "tau_0.4": hold, "gate": report["gate"]}, ensure_ascii=False)); return report


if __name__ == "__main__": run()
