#!/usr/bin/env python3
"""Read-only graph/chemistry roundtrip audit for the strict line-first field.

The proposal is constructed entirely from LSD segments, frozen scored vertex
candidates and the isolated edge/type sidecar.  Distinct candidate edges are
serialized in memory; target atoms/bonds are joined only afterward for
metrics. RDKit is then run on (1) a candidate-only carbon topology and (2) an
oracle-element overlay of the same predicted topology, which separates
topological/valence failures from missing atom-identity supervision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.checkpoints import file_digest
from tools.shadow_line_first_edge_bond_type_r5 import DATA, INITIAL, INITIAL_SHA, STATE, EdgeBondMLP, TYPE_NAMES, _candidate_keys, _proposals
from tools.audit_line_first_candidate_alignment_r5 import align_nodes, read_jsonl, IMAGE
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5-v2/report.json"
ALIGN_RADIUS = 16.0
HOLDOUT_LIMIT = 80


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_rows() -> dict[str, list[dict[str, float]]]:
    out = {}; forbidden = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) & forbidden: raise ValueError(f"candidate contract failed line {line_no}")
        out[row["sample_key"]] = row["candidates"]
    if len(out) != 1170: raise ValueError("expected 1170 candidate rows")
    return out


def _bond_type(kind: int):
    return {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC, 5: Chem.BondType.DATIVE}.get(int(kind), Chem.BondType.SINGLE)


def _sanitize_graph(nodes: list[tuple[int, str, bool]], edges: dict[tuple[int, int], int]) -> tuple[bool, str, str | None]:
    """Build and sanitize a graph; return status, coarse error class, SMILES."""
    mol = Chem.RWMol(); local = {}
    for index, (_, element, aromatic) in enumerate(nodes):
        atom = Chem.Atom(element or "C"); atom.SetIsAromatic(bool(aromatic)); local[index] = mol.AddAtom(atom)
    try:
        for (a, b), kind in edges.items():
            if a == b or a not in local or b not in local: continue
            bond_type = _bond_type(kind)
            mol.AddBond(local[a], local[b], bond_type)
            if bond_type == Chem.BondType.AROMATIC:
                mol.GetBondBetweenAtoms(local[a], local[b]).SetIsAromatic(True)
        candidate = mol.GetMol(); Chem.SanitizeMol(candidate)
        return True, "none", Chem.MolToSmiles(candidate, canonical=True)
    except Exception as exc:  # RDKit exceptions are intentionally summarized only.
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        lowered = message.lower(); kind = "aromatic" if "aromatic" in lowered or "kekul" in lowered else ("valence" if "valence" in lowered else "other")
        return False, kind, None


def _evaluate(row: dict, candidates: dict, model: EdgeBondMLP, mean: np.ndarray, std: np.ndarray, threshold: float, device: torch.device) -> Counter:
    key = row["sample_key"]; features, centers, _ = _proposals(key); values = candidates[key]; points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32); node_map = align_nodes(centers.astype(np.float64), points.astype(np.float64), ALIGN_RADIUS); line_keys = _candidate_keys(key)
    with torch.inference_mode(): edge_logits, type_logits = model(torch.from_numpy((features - mean) / std).to(device))
    scores = torch.sigmoid(edge_logits).cpu().numpy(); kinds = (torch.argmax(type_logits, dim=1) + 1).cpu().numpy()
    selected = {}; dangling = 0; self_loops = 0
    for index, (a, b) in enumerate(line_keys):
        if scores[index] < threshold: continue
        if a not in node_map or b not in node_map: dangling += 1; continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb: self_loops += 1; continue
        pair = tuple(sorted((ca, cb))); old = selected.get(pair)
        if old is None or scores[index] > old[0]: selected[pair] = (float(scores[index]), int(kinds[index]))
    base = DATA / "samples" / key; layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text()); targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32); c2t = strict_candidate_to_target(points, targets); gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}; gt_types = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))): (TYPE_NAMES.index(str(x.get("chemical_type", "OTHER")).upper()) if str(x.get("chemical_type", "OTHER")).upper() in TYPE_NAMES else TYPE_NAMES.index("OTHER")) for x in graph["bonds"]}
    mapped = {}; for_oracle = {}
    for (ca, cb), (_, kind) in selected.items():
        if ca not in c2t or cb not in c2t: continue
        if c2t[ca] == c2t[cb]: self_loops += 1; continue
        edge = tuple(sorted((c2t[ca], c2t[cb]))); mapped[edge] = kind; for_oracle[edge] = kind
    true = set(mapped) & gt; type_correct = sum(mapped[e] == gt_types[e] for e in true)
    # Candidate-only topology: every selected candidate node is carbon.
    candidate_nodes = [(candidate, "C", False) for candidate in sorted(set(node_map.values()))]
    candidate_index = {candidate: i for i, candidate in enumerate(sorted(set(node_map.values())))}
    candidate_edges = {(candidate_index[a], candidate_index[b]): kind for (a, b), (_, kind) in selected.items() if a in candidate_index and b in candidate_index}
    c_ok, c_error, c_smiles = _sanitize_graph(candidate_nodes, candidate_edges)
    # Oracle-element overlay is post-hoc only and is not used to select edges.
    target_nodes = [(candidate, str(layers["atoms"][target]["element"] or "C"), bool(layers["atoms"][target].get("aromatic", False))) for candidate, target in sorted(c2t.items()) if candidate in set(node_map.values())]
    oracle_index = {candidate: i for i, (candidate, _, _) in enumerate(target_nodes)}
    oracle_edges = {(oracle_index[a], oracle_index[b]): kind for (a, b), (_, kind) in selected.items() if a in oracle_index and b in oracle_index}
    o_ok, o_error, o_smiles = _sanitize_graph(target_nodes, oracle_edges)
    isolated = sum(1 for candidate in candidate_index if not any(candidate in pair for pair in selected))
    return Counter({"images": 1, "line_nodes": len(centers), "aligned_nodes": len(node_map), "serialized_nodes": len(candidate_nodes), "line_edges": len(line_keys), "selected_edges": len(selected), "dangling_edges": dangling, "self_loops": self_loops, "isolated_nodes": isolated, "mapped_edges": len(mapped), "true_edges": len(true), "gt_bonds": len(gt), "type_correct": type_correct, "type_total": len(true), "candidate_sanitize_ok": int(c_ok), "candidate_aromatic_errors": int(c_error == "aromatic"), "candidate_valence_errors": int(c_error == "valence"), "candidate_other_errors": int(c_error == "other"), "oracle_sanitize_ok": int(o_ok), "oracle_aromatic_errors": int(o_error == "aromatic"), "oracle_valence_errors": int(o_error == "valence"), "oracle_other_errors": int(o_error == "other"), "all_bonds_recovered": int(true == gt)})


def _finish(c: Counter) -> dict:
    out = dict(c); out["edge_precision"] = c["true_edges"] / max(c["mapped_edges"], 1); out["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1); out["conditional_bond_type_accuracy"] = c["type_correct"] / max(c["type_total"], 1); out["dangling_rate"] = c["dangling_edges"] / max(c["selected_edges"], 1); out["self_loop_rate"] = c["self_loops"] / max(c["selected_edges"], 1); out["isolated_node_rate"] = c["isolated_nodes"] / max(c["serialized_nodes"], 1); out["candidate_sanitize_rate"] = c["candidate_sanitize_ok"] / max(c["images"], 1); out["oracle_sanitize_rate"] = c["oracle_sanitize_ok"] / max(c["images"], 1); out["all_bonds_recovered_fraction"] = c["all_bonds_recovered"] / max(c["images"], 1); return out


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates = _candidate_rows(); state = torch.load(STATE, map_location="cpu", weights_only=False); model = EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval(); device = torch.device("cpu"); mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32); threshold = float(state["threshold"])
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted"]; rows = sorted(rows, key=lambda x: x["sample_key"]); by_split = defaultdict(list)
    for row in rows: by_split[row["split"]].append(row)
    outputs = {}
    for split, chosen_all in by_split.items():
        chosen_sets = [(split, chosen_all[:HOLDOUT_LIMIT])] if split == "holdout" else [(split, chosen_all)]
        if split == "holdout":
            chosen_sets.append(("holdout_full", chosen_all))
        for output_split, chosen in chosen_sets:
            total = Counter(); profiles = defaultdict(Counter)
            for row in chosen:
                current = _evaluate(row, candidates, model, mean, std, threshold, device); total.update(current); profiles[row["profile_id"]].update(current)
            outputs[output_split] = {"images": len(chosen), "all": _finish(total), "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    after = file_digest(INITIAL); hold = outputs.get("holdout_full", {}).get("all", {}); gate = {"checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False, "production_go": False, "holdout_full_edge_precision_ge_0_90": hold.get("edge_precision", 0) >= .90, "holdout_full_edge_recall_ge_0_75": hold.get("edge_recall", 0) >= .75, "holdout_full_type_accuracy_ge_0_90": hold.get("conditional_bond_type_accuracy", 0) >= .90, "holdout_full_no_self_loop": hold.get("self_loop_rate", 1) == 0, "holdout_full_dangling_le_0_05": hold.get("dangling_rate", 1) <= .05, "holdout_full_oracle_sanitize_rate_ge_0_95": hold.get("oracle_sanitize_rate", 0) >= .95}; gate["roundtrip_shadow_worth_attempting"] = all(v for k, v in gate.items() if k not in {"sealed_test_read", "production_go"}) and not gate["sealed_test_read"]
    report = {"schema_version": "chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after, "sidecar_state_sha256": _digest(STATE), "candidate_artifact_sha256": _digest(CANDIDATES), "proposal_contract": "LSD -> provisional nodes -> target-free candidate alignment -> frozen edge/type score; GT only post-hoc", "fixed_threshold": threshold, "splits": outputs, "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "holdout": outputs.get("holdout", {}).get("all"), "gate": gate}, indent=2)); return report


if __name__ == "__main__": run()
