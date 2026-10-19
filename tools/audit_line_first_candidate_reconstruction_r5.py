"""CPU-only candidate-coordinate reconstruction probe.

Frozen vertex candidate identities/scores are retained while coordinates are
optionally projected onto nearby LSD segment lines before ownership. This is
an isolated read-only audit: no target is used in proposal construction, no
training is performed, and production is untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as base  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as topology  # noqa: E402
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd  # noqa: E402
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments  # noqa: E402

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-candidate-reconstruction-r5/report.json"
IMAGE, RADIUS, LAMBDA, CLUSTER_RADIUS = 768.0, 24.0, 8.0, 10.0
MINIMUM_LENGTH, REFINE, SCALE, BORDER_MARGIN = 2.0, "STD", 0.75, 2.0


def _alt_detect(image, *, config=None):
    """Use the train-selected new-LSD proposal, not the production detector."""
    return detect_segments(
        image,
        minimum_length=MINIMUM_LENGTH,
        refine=REFINE,
        scale=SCALE,
        border_margin=BORDER_MARGIN,
        maximum_candidates=int(getattr(config, "maximum_candidates", 512)),
    )


def _install_new_lsd():
    # Patch only this audit's imported function globals. Production modules and
    # checkpoints remain untouched.
    base._proposals.__globals__["detect_raster_line_segments"] = _alt_detect
    base._proposals.__globals__["CLUSTER_RADIUS"] = CLUSTER_RADIUS
    base._candidate_keys.__globals__["detect_raster_line_segments"] = _alt_detect
    base._candidate_keys.__globals__["CLUSTER_RADIUS"] = CLUSTER_RADIUS


def _reconstruct(candidates, segments, *, perp_px, extension):
    if not len(candidates) or not len(segments):
        return candidates[:, :2].copy()
    output = candidates[:, :2].copy()
    for ci, point in enumerate(candidates[:, :2]):
        projections, weights = [], []
        for start, end in segments:
            vector = end - start; length2 = float(np.dot(vector, vector))
            if length2 <= 1e-6: continue
            t = float(np.dot(point - start, vector) / length2)
            if t < -extension or t > 1.0 + extension: continue
            projection = start + np.clip(t, -extension, 1.0 + extension) * vector
            perpendicular = float(np.linalg.norm(point - projection))
            if perpendicular > perp_px: continue
            projections.append(projection); weights.append(float(np.exp(-perpendicular / max(perp_px, 1.0))))
        if projections:
            output[ci] = np.average(np.asarray(projections), axis=0, weights=np.asarray(weights))
    return output


def _assign(centers, candidates):
    if not len(centers) or not len(candidates): return {}
    distances = np.linalg.norm(centers[:, None] - candidates[None, :, :2], axis=-1)
    cost = np.where(distances <= RADIUS, distances - LAMBDA * candidates[None, :, 2], 1e5 + distances)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if distances[r, c] <= RADIUS}


def _finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1),
            "all_bonds_recovered_fraction": c["all_bonds_recovered"] / max(c["images"], 1)}


def _evaluate(row, candidate_rows, model, mean, std, threshold, *, perp_px, extension):
    key = row["sample_key"]
    features, centers, segments = base._proposals(key)
    values = candidate_rows[key]
    original = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in values], dtype=np.float64)
    projected = original.copy(); projected[:, :2] = _reconstruct(original, segments, perp_px=perp_px, extension=extension)
    owners = _assign(centers.astype(np.float64), projected)
    with torch.inference_mode(): edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    scores = torch.sigmoid(edge_logits).numpy(); kinds = (torch.argmax(type_logits, dim=1) + 1).numpy()
    selected, dangling = {}, 0
    for i, (begin, end) in enumerate(base._candidate_keys(key)):
        if scores[i] < threshold: continue
        if begin not in owners or end not in owners: dangling += 1; continue
        left, right = owners[begin], owners[end]
        if left == right: continue
        pair = tuple(sorted((left, right))); old = selected.get(pair)
        if old is None or scores[i] > old[0]: selected[pair] = (float(scores[i]), int(kinds[i]))
    # Keep exactly the already-audited target-free chemistry policy.
    ring = topology._cycles(selected)
    selected = {p: (s, 1 if t == topology.TYPE_AROMATIC and p not in ring else t) for p, (s, t) in selected.items()}
    selected, edits = topology.guard(selected, low_confidence=0.4)
    basepath = base.DATA / "samples" / key
    layers = json.loads(basepath.with_suffix(".layers.json").read_text()); graph = json.loads(basepath.with_suffix(".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32)
    # Original coordinates are used only to score identity, never for ownership.
    c2t = base.strict_candidate_to_target(original[:, :2].astype(np.float32), targets)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    type_by = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        name = str(bond.get("chemical_type", "OTHER")).upper(); type_by[edge] = base.TYPE_NAMES.index(name) if name in base.TYPE_NAMES else base.TYPE_NAMES.index("OTHER")
    mapped = {}
    for (left, right), (_, kind) in selected.items():
        if left in c2t and right in c2t and c2t[left] != c2t[right]: mapped[tuple(sorted((c2t[left], c2t[right])))] = kind
    true = set(mapped) & gt; type_correct = sum(mapped[e] == type_by[e] for e in true)
    node_ids = sorted({x for p in selected for x in p})
    nodes = [(c, str(layers["atoms"][t].get("element") or "C"), bool(layers["atoms"][t].get("aromatic", False))) for c, t in sorted(c2t.items()) if c in node_ids]
    ni = {c: i for i, (c, _, _) in enumerate(nodes)}; edges = {(ni[a], ni[b]): k for (a, b), (_, k) in selected.items() if a in ni and b in ni}
    ok, error, _ = base._sanitize_graph(nodes, edges)
    return Counter(images=1, selected_edges=len(selected), dangling_edges=dangling, mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), type_correct=type_correct, type_total=len(true), sanitize_ok=int(ok), aromatic_errors=int(error == "aromatic"), valence_errors=int(error == "valence"), all_bonds_recovered=int(true == gt), reconstructed_candidates=int(np.sum(np.linalg.norm(projected[:, :2] - original[:, :2], axis=1) > 1e-6)), reconstruction_edits=int(edits.get("removed_edges", 0)))


def main():
    rdBase.DisableLog("rdApp.error")
    if base.file_digest(base.INITIAL) != base.INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    # P1 is explicitly evaluated at the newly selected LSD seam, rather than
    # the legacy detector hidden inside the original r5 preflight helpers.
    new_lsd._install()
    _install_new_lsd()
    candidate_rows = base._candidate_rows(); state = torch.load(base.STATE, map_location="cpu", weights_only=False)
    model = base.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std, threshold = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32), float(state["threshold"])
    rows = sorted([json.loads(x) for x in (base.DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"], key=lambda x: x["sample_key"])
    sweeps = {}
    for perp_px, extension in ((0.0, 0.0), (6.0, .25), (10.0, .25), (14.0, .5), (20.0, 1.0)):
        splits, profiles = defaultdict(Counter), defaultdict(Counter)
        for row in rows:
            v = _evaluate(row, candidate_rows, model, mean, std, threshold, perp_px=perp_px, extension=extension)
            splits[row["split"]].update(v); profiles[f"{row['split']}:{row['profile_id']}"].update(v)
        sweeps[f"perp{perp_px:g}_ext{extension:g}"] = {"by_split": {k: _finish(v) for k, v in sorted(splits.items())}, "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    raw = sweeps["perp0_ext0"]["by_split"]["train"]
    eligible = {k: v for k, v in sweeps.items() if v["by_split"]["train"]["edge_precision"] >= raw["edge_precision"]}
    selected = max(eligible, key=lambda k: (eligible[k]["by_split"]["train"]["edge_recall"], k)) if eligible else "perp0_ext0"
    hold = sweeps[selected]["by_split"]["holdout"]; profiles = {k: v for k, v in sweeps[selected]["by_profile"].items() if k.startswith("holdout:")}
    gate = {"checkpoint_unchanged": base.file_digest(base.INITIAL) == base.INITIAL_SHA, "sealed_test_read": False, "holdout_edge_precision_ge_0.90": hold["edge_precision"] >= .90, "holdout_edge_recall_ge_0.75": hold["edge_recall"] >= .75, "holdout_sanitize_ge_0.90": hold["oracle_sanitize_rate"] >= .90, "all_profiles_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in profiles.values()), "all_profiles_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= .90 for v in profiles.values()), "production_go": False}
    report = {"schema_version": "chemvector-identity-aligned-line-first-candidate-reconstruction-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": base.INITIAL_SHA, "proposal_contract": "LSD provisional nodes -> target-free local line projection of frozen candidate coordinates (identity/score unchanged) -> Hungarian ownership -> cycle/topology guard", "config": {"radius_px": RADIUS, "score_lambda_px": LAMBDA, "cluster_radius_px": CLUSTER_RADIUS, "sweep": [[0., 0.], [6., .25], [10., .25], [14., .5], [20., 1.]], "selection_source": "train-only precision-preserving recall"}, "selected": selected, "sweeps": sweeps, "holdout": hold, "holdout_profiles": profiles, "gate": gate, "next": "NO-GO for GPU unless a separate candidate contract improves full holdout and every profile; production remains frozen"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected": selected, "holdout": hold, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__": main()
