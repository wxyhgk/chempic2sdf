#!/usr/bin/env python3
"""Post-hoc target-element sanitize audit for the isolated r9 pair sidecar.

The direct-pair proposal and sidecar are run exactly as in the r9 shadow.  The
target element/aromatic fields are joined only after pair selection, solely to
measure an oracle-element RDKit sanitize rate.  No target field is fed to the
proposal, sidecar, threshold, or routing logic; sealed artifacts and the
production checkpoint are never touched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.shadow_direct_graph_pair_scorer_gpu_r9 import (
    ATOM_THRESHOLD,
    BASE_PAIR_THRESHOLD,
    DATA,
    SHA,
    DirectPairResidual,
    _profile_rows,
)

_REPLAY_SUFFIX = os.environ.get("R9_ORACLE_REPLAY_SUFFIX", "")
OUT = ROOT / ("outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-oracle-sanitize-r9" + _REPLAY_SUFFIX)
STATE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9/sidecar-state.pt"
IMAGE_SIZE = 768.0
MATCH_RADIUS = 16.0
SCORER_THRESHOLD = float(os.environ.get("R9_ORACLE_THRESHOLD", "0.95"))  # train-only calibration candidate


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target(key: str):
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    centers = torch.tensor([row["center"] for row in sorted(layers["atoms"], key=lambda x: int(x["atom_id"]))], dtype=torch.float32)
    return layers, graph, centers


def map_atoms(predicted: torch.Tensor, expected: torch.Tensor) -> dict[int, int]:
    if not len(predicted) or not len(expected):
        return {}
    d = torch.cdist(predicted.float() * IMAGE_SIZE, expected.float())
    candidates = [(float(d[pi, ti]), pi, ti) for pi in range(d.size(0)) for ti in range(d.size(1)) if float(d[pi, ti]) <= MATCH_RADIUS]
    out: dict[int, int] = {}; used: set[int] = set()
    for _, pi, ti in sorted(candidates):
        if pi not in out and ti not in used:
            out[pi] = ti; used.add(ti)
    return out


def sanitize(nodes: list[tuple[int, str, bool]], edges: dict[tuple[int, int], int]) -> tuple[bool, str]:
    mol = Chem.RWMol(); local = {}
    for index, (_, element, aromatic) in enumerate(nodes):
        atom = Chem.Atom(element or "C"); atom.SetIsAromatic(bool(aromatic)); local[index] = mol.AddAtom(atom)
    try:
        for (a, b), kind in edges.items():
            if a == b or a not in local or b not in local:
                continue
            bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}.get(int(kind), Chem.BondType.SINGLE)
            mol.AddBond(local[a], local[b], bt)
            if bt == Chem.BondType.AROMATIC:
                mol.GetBondBetweenAtoms(local[a], local[b]).SetIsAromatic(True)
        candidate = mol.GetMol(); Chem.SanitizeMol(candidate)
        return True, "none"
    except Exception as exc:
        message = str(exc).splitlines()[0].lower()
        return False, ("aromatic" if "aromatic" in message or "kekul" in message else "valence" if "valence" in message else "other")


def extract(row: dict, model, config, device: torch.device) -> dict[str, object]:
    key = row["sample_key"]; base = DATA / "samples" / key
    image = Image.open(base.with_suffix(".png")).convert("RGB")
    pixels = image_tensor(image)[None].to(device)
    with torch.inference_mode():
        outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
        selected = outputs[6]
        slots = torch.where(selected["atom_mask"][0])[0]
        pairs, sample_indices = enumerate_unordered_atom_pairs(selected["atom_mask"])
        direct = model.pair(outputs[0], selected["atom_features"], selected["atom_geometry"][..., :2], selected["atom_scores"], pairs, sample_indices)
        hidden = direct["hidden"].float()
        base_logits = direct["presence_logits"].float()
        all_points = selected["atom_geometry"][0, :, :2].float()
        pair_points = all_points[pairs]
        delta = pair_points[:, 1] - pair_points[:, 0] if len(pairs) else all_points.new_empty((0, 2))
        distance = torch.linalg.vector_norm(delta * IMAGE_SIZE, dim=-1) if len(pairs) else all_points.new_empty(0)
        midpoint = pair_points.mean(dim=1) if len(pairs) else all_points.new_empty((0, 2))
        scores = selected["atom_scores"][0, pairs].float() if len(pairs) else all_points.new_empty((0, 2))
        features = torch.cat((hidden, base_logits[:, None], midpoint, distance[:, None], scores), dim=-1)
        chemical = direct["chemical_logits"].float()
        points = selected["atom_geometry"][0, slots, :2].float()
    layers, graph, target_points = target(key)
    mapping = map_atoms(points.cpu(), target_points)
    return {"row": row, "pairs": pairs.cpu(), "slots": slots.cpu(), "base_logits": base_logits.cpu(), "features": features.cpu(), "chemical_logits": chemical.cpu(), "mapping": mapping, "layers": layers, "graph": graph, "target_atoms": len(target_points)}


def score(item: dict[str, object], scorer: DirectPairResidual | None, threshold: float) -> Counter:
    pairs, slots, base_logits = item["pairs"], item["slots"], item["base_logits"]
    if scorer is None:
        logits = base_logits
    else:
        with torch.inference_mode():
            logits = base_logits + scorer(item["features"])
    chosen = torch.where(logits.sigmoid() >= threshold)[0]
    mapping = item["mapping"]; chemical = item["chemical_logits"]
    edge_types: dict[tuple[int, int], tuple[float, int]] = {}; dangling = 0
    for index in chosen.tolist():
        left, right = [int(v) for v in pairs[index].tolist()]
        li = int((slots == left).nonzero(as_tuple=False)[0]); ri = int((slots == right).nonzero(as_tuple=False)[0])
        if li not in mapping or ri not in mapping:
            dangling += 1; continue
        edge = tuple(sorted((mapping[li], mapping[ri])))
        if edge[0] == edge[1]:
            continue
        kind = int(chemical[index].argmax()) + 1
        value = float(logits[index].sigmoid().detach())
        if edge not in edge_types or value > edge_types[edge][0]:
            edge_types[edge] = (value, kind)
    atoms = item["layers"]["atoms"]
    # Preserve all matched target nodes, including isolated ones, as the prior
    # line-first chemistry audits do. Elements are post-hoc oracle attributes.
    target_ids = sorted(set(mapping.values())); ni = {tid: i for i, tid in enumerate(target_ids)}
    nodes = [(tid, str(atoms[tid].get("element") or "C"), bool(atoms[tid].get("aromatic", False))) for tid in target_ids]
    edges = {(ni[a], ni[b]): kind for (a, b), (_, kind) in edge_types.items() if a in ni and b in ni}
    ok, error = sanitize(nodes, edges)
    return Counter(images=1, selected_edges=len(chosen), mapped_edges=len(edge_types), dangling_edges=dangling, oracle_sanitize_ok=int(ok), oracle_aromatic_errors=int(error == "aromatic"), oracle_valence_errors=int(error == "valence"), oracle_other_errors=int(error == "other"))


def finish(counter: Counter) -> dict[str, object]:
    out = dict(counter); out["oracle_sanitize_rate"] = counter["oracle_sanitize_ok"] / max(counter["images"], 1); out["dangling_rate"] = counter["dangling_edges"] / max(counter["selected_edges"], 1); return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    requested_device = os.environ.get("R9_ORACLE_DEVICE", "cuda:0")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA oracle audit but CUDA is unavailable")
    device = torch.device(requested_device)
    train_rows, holdout_rows = _profile_rows(__import__("tools.train_direct_graph_pair_scorer_cpu_r9", fromlist=["rows"]).rows())
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.to(device).eval().requires_grad_(False)
    payload = torch.load(STATE, map_location="cpu", weights_only=False); scorer = DirectPairResidual(input_dim=262); scorer.load_state_dict(payload["state_dict"]); scorer.eval()
    all_results = {}
    for name, chosen in (("train", train_rows), ("fresh_holdout", holdout_rows)):
        total = Counter(); profiles = {}
        for row in chosen:
            item = extract(row, model, config, device)
            current = score(item, scorer, SCORER_THRESHOLD); total.update(current)
            profiles.setdefault(row["profile_id"], Counter()).update(current)
        all_results[name] = {"images": len(chosen), "scored": finish(total), "profiles": {key: finish(value) for key, value in sorted(profiles.items())}}
    after = digest(INITIAL); hold = all_results["fresh_holdout"]["scored"]
    report = {"schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-oracle-sanitize-r9-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": device.type == "cuda", "device": str(device), "sealed_test_read": False, "production_checkpoint_written": False, "threshold": SCORER_THRESHOLD, "threshold_source": "r9 GPU shadow train-only calibration receipt", "proposal_contract": "frozen Atom candidates -> all unordered Atom pairs -> frozen DirectAtomPair hidden -> isolated residual sidecar; target elements/aromaticity joined only post-hoc sanitize audit", "splits": all_results, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after, "sidecar_state_sha256": digest(STATE), "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "holdout_oracle_sanitize_ge_0.95": hold["oracle_sanitize_rate"] >= 0.95, "all_profile_oracle_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= 0.90 for v in all_results["fresh_holdout"]["profiles"].values()), "holdout_dangling_le_0.01": hold["dangling_rate"] <= 0.01, "production_go": False}}
    OUT.mkdir(parents=True); (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "holdout": hold, "gate": report["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
