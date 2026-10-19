"""CPU probe for candidate-conditioned line-support ownership at the new-LSD seam.

The assignment cost augments center distance with median perpendicular distance
to incident segment lines and an endpoint projection penalty. Candidate
identity/score remains frozen; all labels are joined only after proposal
construction. No training, sealed split, or production mutation is allowed.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd  # noqa: E402
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-candidate-conditioned-ownership-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE, CENTER_RADIUS, CLUSTER = 768.0, 40.0, 10.0


def _incident_geometry(centers, segments, labels):
    lines = defaultdict(list)
    for i, (start, end) in enumerate(segments):
        for node, endpoint, other in ((int(labels[2 * i]), start, end), (int(labels[2 * i + 1]), end, start)):
            vec = other - endpoint; ll = float(np.linalg.norm(vec))
            if ll > 1e-6: lines[node].append((endpoint.astype(np.float64), vec.astype(np.float64), ll))
    return lines


def _assign(centers, points, segments, labels, *, beta, score_lambda):
    n, m = len(centers), len(points)
    if not n or not m: return {}
    d = np.linalg.norm(centers[:, None] - points[None, :, :2], axis=-1)
    lines = _incident_geometry(centers, segments, labels)
    cost = d.copy()
    for node in range(n):
        incident = lines.get(node, [])
        if not incident: continue
        for ci, candidate in enumerate(points[:, :2]):
            perpendicular, endpoint = [], []
            for start, vec, ll in incident:
                unit = vec / ll; projection_t = float(np.dot(candidate - start, unit))
                projection = start + projection_t * unit
                perpendicular.append(float(np.linalg.norm(candidate - projection)))
                # Penalize distance from the candidate to the observed endpoint;
                # this is target-free and keeps text/line seams local.
                endpoint.append(float(np.linalg.norm(candidate - start)))
            cost[node, ci] = float(np.median(perpendicular) + beta * np.median(endpoint) - score_lambda * points[ci, 2])
    # Keep all Hungarian prefix assignments; only a broad geometric radius can
    # suppress catastrophic assignments, never a post-hoc target test.
    rr, cc = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rr, cc) if d[r, c] <= CENTER_RADIUS}


def _finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def _one(row, candidates, beta, score_lambda):
    key = row["sample_key"]; _, centers, segments = chemistry._proposals(key)
    labels = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2))
    # labels are reconstructed with the same new-LSD cluster contract.
    from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
    raw_points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2))
    labels, _ = cluster_endpoints(raw_points, CLUSTER)
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in candidates[key]], dtype=np.float64)
    owners = _assign(centers.astype(np.float64), points, segments, labels, beta=beta, score_lambda=score_lambda)
    selected = {}; dangling = 0
    for a, b in chemistry._candidate_keys(key):
        if a not in owners or b not in owners: dangling += 1; continue
        ca, cb = owners[a], owners[b]
        if ca == cb: continue
        selected[tuple(sorted((ca, cb)))] = (1.0, 1)
    base = DATA / "samples" / key; layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32); c2t = chemistry.strict_candidate_to_target(points[:, :2].astype(np.float32), targets)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}; mapped = {tuple(sorted((c2t[a], c2t[b]))) for a, b in selected if a in c2t and b in c2t and c2t[a] != c2t[b]}; true = mapped & gt
    node_ids = sorted({z for p in selected for z in p}); nodes = [(c, str(layers["atoms"][t].get("element") or "C"), bool(layers["atoms"][t].get("aromatic", False))) for c, t in sorted(c2t.items()) if c in node_ids]; ni = {c: i for i, (c, _, _) in enumerate(nodes)}; edges = {(ni[a], ni[b]): 1 for a, b in selected if a in ni and b in ni}; ok, error, _ = chemistry._sanitize_graph(nodes, edges)
    return guard.Counter(images=1, selected_edges=len(selected), dangling_edges=dangling, mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), type_correct=0, type_total=0, sanitize_ok=int(ok), aromatic_errors=int(error == "aromatic"), valence_errors=int(error == "valence"))


def main():
    rdBase.DisableLog("rdApp.error")
    if chemistry.file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    new_lsd._install(); candidates = hybrid.candidate_rows(); rows = sorted([json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"], key=lambda x: x["sample_key"])
    sweeps = {}
    for beta in (0.0, .25, .50):
        for lam in (0.0, 8.0):
            totals, profiles = defaultdict(Counter), defaultdict(Counter)
            for row in rows:
                value = _one(row, candidates, beta, lam); totals[row["split"]].update(value); profiles[f"{row['split']}:{row['profile_id']}"].update(value)
            sweeps[f"beta{beta:g}_lambda{lam:g}"] = {"by_split": {k: _finish(v) for k, v in sorted(totals.items())}, "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    selected = max(sweeps, key=lambda k: (sweeps[k]["by_split"]["train"]["edge_recall"], sweeps[k]["by_split"]["train"]["edge_precision"]))
    hold, profiles = sweeps[selected]["by_split"]["holdout"], {k: v for k, v in sweeps[selected]["by_profile"].items() if k.startswith("holdout:")}
    gate = {"checkpoint_unchanged": chemistry.file_digest(chemistry.INITIAL) == INITIAL_SHA, "sealed_test_read": False, "holdout_edge_precision_ge_0.90": hold["edge_precision"] >= .90, "holdout_edge_recall_ge_0.75": hold["edge_recall"] >= .75, "holdout_sanitize_ge_0.90": hold["oracle_sanitize_rate"] >= .90, "all_profiles_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in profiles.values()), "all_profiles_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= .90 for v in profiles.values()), "production_go": False}
    report = {"schema_version": "chemvector-identity-aligned-line-first-lsd-candidate-conditioned-ownership-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "proposal_contract": "new LSD min2 STD scale.75 cluster r10 -> candidate-conditioned incident-line perpendicular/endpoint cost -> full Hungarian prefix -> post-hoc metrics", "config": {"minimum_length_px": 2.0, "refine": "STD", "scale": .75, "cluster_radius_px": CLUSTER, "center_radius_px": CENTER_RADIUS, "beta_sweep": [0., .10, .25, .50, 1.], "lambda_sweep": [0., 4., 8.], "selection_source": "train-only"}, "selected": selected, "sweeps": sweeps, "holdout": hold, "holdout_profiles": profiles, "gate": gate, "next": "NO-GO for GPU unless ownership cost improves both full holdout recall and profile floors"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "selected": selected, "holdout": hold, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__": main()
