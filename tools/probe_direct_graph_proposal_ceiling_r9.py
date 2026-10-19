#!/usr/bin/env python3
"""Frozen direct-graph proposal ceiling on the r6 fresh holdout.

This probe intentionally bypasses LSD/line-first candidates.  The frozen
continuation bundle proposes Atom candidates from the image, evaluates every
unordered Atom pair with the existing direct pair head, and emits a graph by
thresholding pair presence.  Candidate selection never reads graph/layer
targets; targets are opened only after proposal construction for metrics.

The probe is read-only and deterministic.  It is a CPU gate: no optimizer,
checkpoint write, sealed-test read, OCR, or line-segment detector is involved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe


DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-proposal-ceiling-r9"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
# These are the frozen continuation thresholds, not values selected against
# this holdout.  Atom logits use the signed set-head contract; pair uses a
# probability threshold from the initial provenance.
ATOM_THRESHOLD = -0.25
PAIR_PROBABILITY_THRESHOLD = 0.85
ATOM_MATCH_RADIUS = 16.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (DATA / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == "fresh_holdout"]
    if len(rows) != 95:
        raise ValueError(f"Expected fixed 95-image fresh holdout, got {len(rows)}")
    return sorted(rows, key=lambda row: row["sample_key"])


def _target(sample_key: str) -> tuple[torch.Tensor, set[tuple[int, int]], list[str]]:
    base = DATA / "samples" / sample_key
    layers = json.loads((base.with_suffix(".layers.json")).read_text(encoding="utf-8"))
    graph = json.loads((base.with_suffix(".graph.json")).read_text(encoding="utf-8"))
    atoms = sorted(layers["atoms"], key=lambda row: int(row["atom_id"]))
    centers = torch.tensor([row["center"] for row in atoms], dtype=torch.float32)
    edges = {
        tuple(sorted((int(row["begin_atom"]), int(row["end_atom"]))))
        for row in graph["bonds"]
    }
    types = [str(row["chemical_type"]) for row in graph["bonds"]]
    # Keep explicit edge type lookup separate from the unordered edge set.
    return centers, edges, types


def _map_atoms(predicted: torch.Tensor, target: torch.Tensor) -> dict[int, int]:
    """Target-free proposal is already complete; this is post-hoc scoring only."""

    if not len(predicted) or not len(target):
        return {}
    distance = torch.cdist(predicted.float() * IMAGE_SIZE, target.float())
    rows: list[tuple[float, int, int]] = []
    for pi in range(distance.size(0)):
        for ti in range(distance.size(1)):
            d = float(distance[pi, ti])
            if d <= ATOM_MATCH_RADIUS:
                rows.append((d, pi, ti))
    mapping: dict[int, int] = {}
    used_targets: set[int] = set()
    for _, pi, ti in sorted(rows):
        if pi not in mapping and ti not in used_targets:
            mapping[pi] = ti
            used_targets.add(ti)
    return mapping


def _one(row: dict, model, config, provenance) -> dict[str, object]:
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(base.with_suffix(".png")).convert("RGB")
    pixels = image_tensor(image)[None]
    # No target tensors are passed to this path.  All candidate IDs, positions,
    # and pair decisions are produced from pixels and frozen model weights.
    with torch.inference_mode():
        outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
        raw_atoms, selected_atoms = outputs[3], outputs[6]
        raw_indices = torch.where(raw_atoms["atom_mask"][0])[0]
        selected_indices = torch.where(selected_atoms["atom_mask"][0])[0]
        selected_mask = selected_atoms["atom_mask"]
        pairs, sample_indices = enumerate_unordered_atom_pairs(selected_mask)
        direct = model.pair(
            outputs[0],
            selected_atoms["atom_features"],
            selected_atoms["atom_geometry"][..., :2],
            selected_atoms["atom_scores"],
            pairs,
            sample_indices,
        )
    probabilities = direct["presence_logits"].float().sigmoid()
    chosen = torch.where(probabilities >= PAIR_PROBABILITY_THRESHOLD)[0]
    predicted_points = selected_atoms["atom_geometry"][0, selected_indices, :2].float()
    target_points, target_edges, _ = _target(row["sample_key"])
    mapping = _map_atoms(predicted_points, target_points)
    predicted_edges = {
        tuple(sorted((int(pairs[index, 0]), int(pairs[index, 1]))))
        for index in chosen.tolist()
    }
    mapped_edges: set[tuple[int, int]] = set()
    dangling = 0
    for begin, end in predicted_edges:
        # pair indices index selected_atoms slots, not compact active offsets;
        # selected_indices is the active slot list used to build coordinates.
        begin_offset = int((selected_indices == begin).nonzero(as_tuple=False)[0])
        end_offset = int((selected_indices == end).nonzero(as_tuple=False)[0])
        if begin_offset not in mapping or end_offset not in mapping:
            dangling += 1
            continue
        mapped_edges.add(tuple(sorted((mapping[begin_offset], mapping[end_offset]))))
    true_edges = mapped_edges & target_edges
    type_correct = 0
    # Type scoring is intentionally conservative: only direct pair predictions
    # whose endpoints map to a true target edge are counted.
    target_type_by_edge = {}
    graph = json.loads((base.with_suffix(".graph.json")).read_text(encoding="utf-8"))
    for bond in graph["bonds"]:
        target_type_by_edge[tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))] = str(bond["chemical_type"])
    for index in chosen.tolist():
        begin, end = [int(v) for v in pairs[index].tolist()]
        bo = int((selected_indices == begin).nonzero(as_tuple=False)[0])
        eo = int((selected_indices == end).nonzero(as_tuple=False)[0])
        if bo not in mapping or eo not in mapping:
            continue
        edge = tuple(sorted((mapping[bo], mapping[eo])))
        if edge not in target_edges:
            continue
        predicted_type = BOND_TYPES[int(direct["chemical_logits"][index].argmax())]
        if predicted_type == target_type_by_edge.get(edge):
            type_correct += 1
    return {
        "images": 1,
        "target_atoms": int(len(target_points)),
        "raw_atom_candidates": int(len(raw_indices)),
        "selected_atom_candidates": int(len(selected_indices)),
        "matched_atoms": int(len(mapping)),
        "predicted_edges": int(len(predicted_edges)),
        "mapped_edges": int(len(mapped_edges)),
        "true_edges": int(len(true_edges)),
        "target_edges": int(len(target_edges)),
        "dangling_edges": int(dangling),
        "type_rows": int(len(true_edges)),
        "type_correct": int(type_correct),
    }


def _finish(counter: Counter) -> dict[str, object]:
    out = dict(counter)
    out.update(
        atom_precision=counter["matched_atoms"] / max(counter["selected_atom_candidates"], 1),
        atom_recall=counter["matched_atoms"] / max(counter["target_atoms"], 1),
        edge_precision=counter["true_edges"] / max(counter["mapped_edges"], 1),
        edge_recall=counter["true_edges"] / max(counter["target_edges"], 1),
        dangling_rate=counter["dangling_edges"] / max(counter["predicted_edges"], 1),
        type_accuracy=counter["type_correct"] / max(counter["type_rows"], 1),
    )
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("Production initial checkpoint SHA changed")
    # CPU execution is deliberate: this is a deterministic proposal ceiling,
    # not a GPU trial.  Restricting threads also makes replay reproducible.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config, payload = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    totals = Counter()
    profiles: dict[str, Counter] = defaultdict(Counter)
    for index, row in enumerate(_rows(), 1):
        result = _one(row, model, config, payload["provenance"])
        totals.update(result)
        profiles[row["profile_id"]].update(result)
        if index % 10 == 0:
            print(json.dumps({"images": index, "edges": totals["true_edges"]}), flush=True)
    report = {
        "schema_version": "chemvector-identity-aligned-direct-graph-proposal-ceiling-r9-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "proposal_contract": "frozen Atom candidates -> all unordered Atom pairs -> direct pair presence; no LSD/line-first; target-free proposal",
        "dataset": str(DATA),
        "split": "fresh_holdout",
        "images": 95,
        "thresholds": {"atom_logit": ATOM_THRESHOLD, "pair_probability": PAIR_PROBABILITY_THRESHOLD, "atom_match_radius_posthoc_px": ATOM_MATCH_RADIUS},
        "holdout": _finish(totals),
        "holdout_profiles": {key: _finish(value) for key, value in sorted(profiles.items())},
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "gate": {
            "checkpoint_unchanged": digest(INITIAL) == SHA,
            "sealed_test_read": False,
            "training_admissible": False,
            "gpu_go": False,
            "proposal_gate": {
                "edge_precision_min": _finish(totals)["edge_precision"] >= 0.965917,
                "edge_recall_min": _finish(totals)["edge_recall"] > 0.850031,
                "dangling_max": _finish(totals)["dangling_rate"] <= 0.05,
            },
            "reason": "direct graph proposal ceiling only; no scorer training before deterministic gate",
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "holdout": report["holdout"], "gate": report["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
