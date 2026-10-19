"""CPU-only pseudo-candidate resampling probe at the new-LSD seam.

Unowned provisional nodes receive a low-score pseudo candidate at their own
LSD centre.  Original frozen candidates are not changed or overwritten.  The
probe measures only whether this can reduce ownership loss; pseudo identities
are joined to atom labels strictly after proposal construction for metrics.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-pseudo-candidate-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE, CLUSTER, MERGE, TAU = 768.0, 10.0, 20.0, .4


def _assign(centres, points, radius):
    if not len(centres) or not len(points): return {}
    d = np.linalg.norm(centres[:, None] - points[None, :, :2], axis=-1)
    cost = np.where(d <= radius, d - 8.0 * points[None, :, 2], 1e5 + d)
    rr, cc = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rr, cc) if d[r, c] <= radius}


def _rows(split=None):
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"]
    return sorted([r for r in rows if split is None or r["split"] == split], key=lambda x: x["sample_key"])


def _finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def _one(row, candidate_rows, pseudo_score, radius):
    key = row["sample_key"]
    _, centres, _ = chemistry._proposals(key)
    originals = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in candidate_rows[key]], dtype=np.float64)
    first = _assign(centres.astype(np.float64), originals, radius)
    used = set(first)
    extra = np.asarray([[*centres[node], pseudo_score] for node in range(len(centres)) if node not in used], dtype=np.float64)
    points = np.concatenate([originals, extra], axis=0) if len(extra) else originals
    owners = _assign(centres.astype(np.float64), points, radius)
    selected, dangling = {}, 0
    # Keep every new-LSD provisional edge; this is an ownership ceiling probe.
    for begin, end in chemistry._candidate_keys(key):
        if begin not in owners or end not in owners:
            dangling += 1; continue
        left, right = owners[begin], owners[end]
        if left == right: continue
        pair = tuple(sorted((left, right))); selected[pair] = (1.0, 1)
    selected, _ = guard.guard(selected, low_confidence=TAU)
    return guard._metrics(row, selected, points.astype(np.float32), dangling)


def main():
    rdBase.DisableLog("rdApp.error")
    if chemistry.file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    new_lsd._install()
    candidate_rows = hybrid.candidate_rows()
    rows = _rows()
    sweeps = {}
    for score in (0.00, .02, .05, .10):
        for radius in (24.0, 32.0, 40.0):
            split_totals, profiles = defaultdict(Counter), defaultdict(Counter)
            for row in rows:
                value = _one(row, candidate_rows, score, radius); split_totals[row["split"]].update(value); profiles[f"{row['split']}:{row['profile_id']}"].update(value)
            sweeps[f"score{score:g}_r{radius:g}"] = {"by_split": {k: _finish(v) for k, v in sorted(split_totals.items())}, "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    raw = sweeps["score0_r24"]["by_split"]["train"]
    eligible = {k: v for k, v in sweeps.items() if v["by_split"]["train"]["edge_precision"] >= raw["edge_precision"]}
    selected = max(eligible, key=lambda k: (eligible[k]["by_split"]["train"]["edge_recall"], eligible[k]["by_split"]["train"]["edge_precision"]))
    hold, hp = sweeps[selected]["by_split"]["holdout"], {k: v for k, v in sweeps[selected]["by_profile"].items() if k.startswith("holdout:")}
    gate = {"checkpoint_unchanged": chemistry.file_digest(chemistry.INITIAL) == INITIAL_SHA, "sealed_test_read": False, "holdout_edge_precision_ge_0.90": hold["edge_precision"] >= .90, "holdout_edge_recall_ge_0.75": hold["edge_recall"] >= .75, "holdout_dangling_le_0.05": hold["dangling_rate"] <= .05, "holdout_sanitize_ge_0.90": hold["oracle_sanitize_rate"] >= .90, "all_profiles_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in hp.values()), "all_profiles_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= .90 for v in hp.values()), "production_go": False}
    report = {"schema_version": "chemvector-identity-aligned-line-first-lsd-pseudo-candidate-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "proposal_contract": "new LSD min2 STD scale.75 cluster r10 -> frozen candidates + low-score pseudo candidates at unmatched provisional centres -> Hungarian ownership + topology guard; pseudo labels post-hoc only", "config": {"minimum_length_px": 2.0, "refine": "STD", "scale": .75, "cluster_radius_px": CLUSTER, "merge_px": MERGE, "guard_tau": TAU, "selection_source": "train-only precision-preserving recall"}, "selected": selected, "sweeps": sweeps, "holdout": hold, "holdout_profiles": hp, "gate": gate, "next": "NO-GO for GPU; pseudo-candidates cannot repair merged or text-occluded endpoints without increasing ownership risk"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected": selected, "holdout": hold, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__": main()
