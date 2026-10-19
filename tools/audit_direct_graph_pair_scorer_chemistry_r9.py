#!/usr/bin/env python3
"""Post-hoc chemistry audit for the isolated r9 direct-pair GPU shadow.

This script never uses target chemistry during proposal/scoring.  It first
replays the frozen AtomObject/AtomPair route and the quarantined residual
sidecar, then joins target atoms and bond labels only for the audit.  RDKit is
run on the predicted topology twice: carbon-only and with oracle target
elements/aromatic flags overlaid.  The latter is explicitly an evaluation
diagnostic, not a training or routing input.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    DATA, SHA, BASE_PAIR_THRESHOLD, DirectPairResidual, _extract, rows,
)
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

SHADOW = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9"
REPORT = SHADOW / "report.json"
STATE = SHADOW / "sidecar-state.pt"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-chemistry-r9-v2"
MATCHED_THRESHOLD = 0.95


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sanitize(nodes: list[tuple[int, str, bool]], edges: dict[tuple[int, int], int]):
    mol = Chem.RWMol(); local = {}
    for i, (_, element, aromatic) in enumerate(nodes):
        atom = Chem.Atom(element or "C")
        atom.SetIsAromatic(bool(aromatic))
        local[i] = mol.AddAtom(atom)
    try:
        for (a, b), kind in edges.items():
            if a == b or a not in local or b not in local:
                continue
            bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}.get(int(kind), Chem.BondType.SINGLE)
            mol.AddBond(local[a], local[b], bt)
            if bt == Chem.BondType.AROMATIC:
                mol.GetBondBetweenAtoms(local[a], local[b]).SetIsAromatic(True)
        obj = mol.GetMol(); Chem.SanitizeMol(obj)
        return True, "none"
    except Exception as exc:
        msg = str(exc).lower()
        if "aromatic" in msg or "kekul" in msg:
            kind = "aromatic"
        elif "valence" in msg:
            kind = "valence"
        else:
            kind = "other"
        return False, kind


def one(item: dict, scorer: DirectPairResidual, threshold: float) -> Counter:
    key = item["row"]["sample_key"]
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda x: int(x["atom_id"]))
    target_elements = {i: str(a.get("element") or "C") for i, a in enumerate(atoms)}
    target_aromatic = {i: bool(a.get("aromatic", False)) for i, a in enumerate(atoms)}
    with torch.inference_mode():
        logits = item["base_logits"] if scorer is None else item["base_logits"] + scorer(item["features"])
    chosen = torch.where(logits.sigmoid() >= threshold)[0].tolist()
    pairs = item["pairs"]; slots = item["slots"]
    slot_to_compact = {int(slot): i for i, slot in enumerate(slots.tolist())}
    # Keep one edge per predicted candidate pair.  The frozen AtomPair route
    # has one row per unordered pair, so duplicates here indicate mapped alias
    # collapse rather than duplicate candidate rows.
    candidate_edges: dict[tuple[int, int], int] = {}
    dangling = 0; self_loops = 0
    chemical = item["chemical_logits"]
    for index in chosen:
        left, right = (int(v) for v in pairs[index].tolist())
        if left not in slot_to_compact or right not in slot_to_compact:
            dangling += 1; continue
        a, b = slot_to_compact[left], slot_to_compact[right]
        if a == b:
            self_loops += 1; continue
        kind = int(chemical[index].argmax()) + 1
        candidate_edges.setdefault(tuple(sorted((a, b))), kind)
    mapping = item["mapping"]  # compact candidate atom -> target atom
    mapped_edges: dict[tuple[int, int], int] = {}
    mapped_duplicates = 0
    for (a, b), kind in candidate_edges.items():
        if a not in mapping or b not in mapping:
            continue
        ta, tb = int(mapping[a]), int(mapping[b])
        if ta == tb:
            self_loops += 1; continue
        edge = tuple(sorted((ta, tb)))
        if edge in mapped_edges:
            mapped_duplicates += 1
        else:
            mapped_edges[edge] = kind
    # Candidate topology includes every endpoint of a selected non-dangling
    # edge.  Oracle overlay changes element/aromatic labels only.
    node_ids = sorted({x for edge in candidate_edges for x in edge})
    local = {node: i for i, node in enumerate(node_ids)}
    carbon_nodes = [(node, "C", False) for node in node_ids]
    carbon_edges = {(local[a], local[b]): kind for (a, b), kind in candidate_edges.items()}
    carbon_ok, carbon_error = sanitize(carbon_nodes, carbon_edges)
    oracle_nodes = []
    for node in node_ids:
        target = mapping.get(node)
        oracle_nodes.append((node, target_elements.get(int(target), "C") if target is not None else "C",
                             target_aromatic.get(int(target), False) if target is not None else False))
    oracle_edges = {(local[a], local[b]): kind for (a, b), kind in candidate_edges.items()}
    oracle_ok, oracle_error = sanitize(oracle_nodes, oracle_edges)
    return Counter({
        "images": 1, "selected_edges": len(chosen), "dangling_edges": dangling,
        "self_loops": self_loops, "candidate_edges": len(candidate_edges),
        "mapped_edges": len(mapped_edges), "mapped_duplicates": mapped_duplicates,
        "candidate_sanitize_ok": int(carbon_ok), "candidate_aromatic_errors": int(carbon_error == "aromatic"),
        "candidate_valence_errors": int(carbon_error == "valence"), "candidate_other_errors": int(carbon_error == "other"),
        "oracle_sanitize_ok": int(oracle_ok), "oracle_aromatic_errors": int(oracle_error == "aromatic"),
        "oracle_valence_errors": int(oracle_error == "valence"), "oracle_other_errors": int(oracle_error == "other"),
    })


def finish(c: Counter) -> dict:
    out = dict(c)
    out.update({
        "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
        "self_loop_rate": c["self_loops"] / max(c["selected_edges"], 1),
        "candidate_sanitize_rate": c["candidate_sanitize_ok"] / max(c["images"], 1),
        "oracle_sanitize_rate": c["oracle_sanitize_ok"] / max(c["images"], 1),
    })
    return out


def run():
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    if not REPORT.exists() or not STATE.exists():
        raise RuntimeError("GPU shadow artifacts missing")
    shadow = json.loads(REPORT.read_text())
    if shadow.get("sealed_test_read") is not False:
        raise RuntimeError("invalid shadow provenance")
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    scorer = DirectPairResidual(input_dim=262)
    scorer.load_state_dict(state["state_dict"]); scorer.eval()
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    holdout = sorted((r for r in rows() if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    items = [_extract(r, model, config, need_labels=True) for r in holdout]
    totals = {}
    profiles = {}
    for name, active_scorer, threshold in (("baseline", None, BASE_PAIR_THRESHOLD), ("shadow", scorer, MATCHED_THRESHOLD)):
        total = Counter(); by_profile = {}
        for item in items:
            current = one(item, active_scorer, threshold); total.update(current)
        for profile in sorted({i["row"]["profile_id"] for i in items}):
            c = Counter()
            for item in items:
                if item["row"]["profile_id"] == profile:
                    c.update(one(item, active_scorer, threshold))
            by_profile[profile] = finish(c)
        totals[name] = finish(total); profiles[name] = by_profile
    after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-chemistry-r9-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "dataset": str(DATA), "split": "fresh_holdout", "images": len(items),
        "thresholds": {"baseline": BASE_PAIR_THRESHOLD, "shadow": MATCHED_THRESHOLD},
        "threshold_source": "GPU shadow train-only calibration receipt; baseline frozen threshold",
        "proposal_contract": "frozen AtomObject -> unordered AtomPair -> frozen presence/type -> isolated residual; chemistry post-hoc only",
        "all": totals, "by_profile": profiles,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "shadow_report_sha256": digest(REPORT), "shadow_state_sha256": digest(STATE),
        "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False,
                 "oracle_sanitize_reported": True, "production_go": False,
                 "next": "use chemistry audit to choose train coverage/hard-negative CPU trial; do not merge sidecar"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "all": result["all"], "gate": result["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
