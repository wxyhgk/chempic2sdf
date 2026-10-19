#!/usr/bin/env python3
"""Bounded CPU residual scorer for the direct AtomPair graph route (r9).

The frozen proposal is exactly the r9 path: AtomObject candidates followed by
all unordered Atom pairs and the existing DirectAtomPair head.  This script
trains only a tiny residual presence scorer on a fixed train-only subset; the
proposal, OCR, initial checkpoint, and production graph are frozen.  The full
95-image fresh holdout is replayed after training, with profile metrics and the
unchanged baseline reported side by side.  No sealed artifact is opened.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
ATOM_THRESHOLD = -0.25
BASE_PAIR_THRESHOLD = 0.85
MATCH_RADIUS = 16.0
TRAIN_IMAGES_PER_PROFILE = 24
STEPS = 500


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DirectPairResidual(nn.Module):
    """Small residual over frozen DirectAtomPair hidden evidence."""

    def __init__(self, input_dim: int = 262, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Step zero is an exact baseline replay; the sidecar cannot silently
        # alter the frozen route before an optimizer update.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def rows() -> list[dict]:
    values = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    return sorted(values, key=lambda row: row["sample_key"])


def _target(key: str) -> tuple[torch.Tensor, set[tuple[int, int]], dict[tuple[int, int], str]]:
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda row: int(row["atom_id"]))
    centers = torch.tensor([row["center"] for row in atoms], dtype=torch.float32)
    edge_types = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        edge_types[edge] = str(bond["chemical_type"])
    return centers, set(edge_types), edge_types


def _map_atoms(predicted: torch.Tensor, target: torch.Tensor) -> dict[int, int]:
    if not len(predicted) or not len(target):
        return {}
    distances = torch.cdist(predicted.float() * IMAGE_SIZE, target.float())
    candidates = [
        (float(distances[pi, ti]), pi, ti)
        for pi in range(distances.size(0))
        for ti in range(distances.size(1))
        if float(distances[pi, ti]) <= MATCH_RADIUS
    ]
    output: dict[int, int] = {}
    used: set[int] = set()
    for _, pi, ti in sorted(candidates):
        if pi not in output and ti not in used:
            output[pi] = ti
            used.add(ti)
    return output


def _extract(row: dict, model, config, *, need_labels: bool) -> dict[str, object]:
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(base.with_suffix(".png")).convert("RGB")
    pixels = image_tensor(image)[None]
    with torch.inference_mode():
        outputs = structure_forward(model, pixels, config, ATOM_THRESHOLD)
        selected = outputs[6]
        slots = torch.where(selected["atom_mask"][0])[0]
        pairs, sample_indices = enumerate_unordered_atom_pairs(selected["atom_mask"])
        direct = model.pair(
            outputs[0], selected["atom_features"], selected["atom_geometry"][..., :2],
            selected["atom_scores"], pairs, sample_indices,
        )
    hidden = direct["hidden"].float()
    base_logits = direct["presence_logits"].float()
    points = selected["atom_geometry"][0, slots, :2].float()
    # Geometric scalars are image-only.  They provide no target identity or
    # graph information and make the sidecar inspectable independent of logits.
    # ``pairs`` indexes padded Atom slots; ``points`` is compact active
    # geometry used only for post-hoc target matching.
    all_points = selected["atom_geometry"][0, :, :2].float()
    pair_points = all_points[pairs]
    delta = pair_points[:, 1] - pair_points[:, 0] if len(pairs) else points.new_empty((0, 2))
    distance = torch.linalg.vector_norm(delta * IMAGE_SIZE, dim=-1) if len(pairs) else points.new_empty(0)
    midpoint = pair_points.mean(dim=1) if len(pairs) else points.new_empty((0, 2))
    scores = selected["atom_scores"][0, pairs].float() if len(pairs) else points.new_empty((0, 2))
    pair_features = torch.cat((hidden, base_logits[:, None], midpoint, distance[:, None], scores), dim=-1)
    labels = None
    mapping = {}
    target_edges: set[tuple[int, int]] = set()
    target_type_by_edge: dict[tuple[int, int], str] = {}
    if need_labels:
        target_points, target_edges, target_type_by_edge = _target(row["sample_key"])
        mapping = _map_atoms(points, target_points)
        labels = torch.zeros(len(pairs), dtype=torch.float32)
        for index, (left, right) in enumerate(pairs.tolist()):
            li = int((slots == left).nonzero(as_tuple=False)[0])
            ri = int((slots == right).nonzero(as_tuple=False)[0])
            if li in mapping and ri in mapping and tuple(sorted((mapping[li], mapping[ri]))) in target_edges:
                labels[index] = 1.0
    return {
        "features": pair_features,
        "base_logits": base_logits,
        "chemical_logits": direct["chemical_logits"].float(),
        "pairs": pairs,
        "slots": slots,
        "points": points,
        "labels": labels,
        "mapping": mapping,
        "target_edges": target_edges,
        "target_type_by_edge": target_type_by_edge,
        "target_atoms": int(len(_target(row["sample_key"])[0])),
        "row": row,
    }


def _score_rows(rows_data: list[dict[str, object]], scorer: DirectPairResidual | None, threshold: float) -> Counter:
    totals = Counter()
    for item in rows_data:
        pairs = item["pairs"]
        base_logits = item["base_logits"]
        logits = base_logits if scorer is None else base_logits + scorer(item["features"])
        chosen = torch.where(logits.sigmoid() >= threshold)[0]
        slots, mapping = item["slots"], item["mapping"]
        predicted = set()
        dangling = 0
        for index in chosen.tolist():
            left, right = [int(v) for v in pairs[index].tolist()]
            li = int((slots == left).nonzero(as_tuple=False)[0]); ri = int((slots == right).nonzero(as_tuple=False)[0])
            if li not in mapping or ri not in mapping:
                dangling += 1
            else:
                predicted.add(tuple(sorted((mapping[li], mapping[ri]))))
        target_edges = item["target_edges"]
        true_edges = predicted & target_edges
        type_correct = 0
        for edge in true_edges:
            # Find the chosen row that produced this mapped edge.  This stays
            # post-hoc; chemical type itself is never an input to scoring.
            for index in chosen.tolist():
                left, right = [int(v) for v in pairs[index].tolist()]
                li = int((slots == left).nonzero(as_tuple=False)[0]); ri = int((slots == right).nonzero(as_tuple=False)[0])
                if li not in mapping or ri not in mapping or tuple(sorted((mapping[li], mapping[ri]))) != edge:
                    continue
                pred_type = BOND_TYPES[int(item.get("chemical_logits", torch.zeros(len(pairs), len(BOND_TYPES)))[index].argmax())] if "chemical_logits" in item else None
                # Chemical logits are attached below by _extract only when
                # metrics are requested; absent logits simply omit type rows.
                if pred_type is not None:
                    totals["type_rows"] += 1
                    type_correct += int(pred_type == item["target_type_by_edge"].get(edge))
                break
        totals.update({
            "images": 1, "target_atoms": item["target_atoms"],
            "selected_atom_candidates": int(len(slots)), "matched_atoms": int(len(mapping)),
            "predicted_edges": int(len(chosen)), "mapped_edges": int(len(predicted)),
            "true_edges": int(len(true_edges)), "target_edges": int(len(target_edges)),
            "dangling_edges": int(dangling), "type_correct": int(type_correct),
        })
    return totals


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
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_rows = rows()
    train_rows = [row for row in all_rows if row.get("split") == "train"]
    holdout_rows = [row for row in all_rows if row.get("split") == "fresh_holdout"]
    by_profile = defaultdict(list)
    for row in sorted(train_rows, key=lambda value: value["sample_key"]):
        if len(by_profile[row["profile_id"]]) < TRAIN_IMAGES_PER_PROFILE:
            by_profile[row["profile_id"]].append(row)
    train_rows = [row for key in sorted(by_profile) for row in by_profile[key]]
    if len(train_rows) != 96 or len(holdout_rows) != 95:
        raise RuntimeError("Fixed train/holdout cardinality changed")
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in holdout_rows}:
        raise RuntimeError("Train and fresh holdout molecule identities overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    print(json.dumps({"phase": "cache_train", "images": len(train_rows)}), flush=True)
    train_data = [_extract(row, model, config, need_labels=True) for row in train_rows]
    features = torch.cat([item["features"] for item in train_data])
    labels = torch.cat([item["labels"] for item in train_data])
    if len(features) == 0 or not bool(labels.any()):
        raise RuntimeError("No positive direct pair rows in fixed train subset")
    scorer = DirectPairResidual(input_dim=features.size(1))
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1.0)).clamp(1.0, 32.0)
    best_state, best_loss = None, float("inf")
    events = []
    for step in range(1, STEPS + 1):
        scorer.train(); optimizer.zero_grad(set_to_none=True)
        residual = scorer(features)
        loss = F.binary_cross_entropy_with_logits(features.new_tensor(0.0) + (torch.cat([item["base_logits"] for item in train_data]) + residual), labels, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0); optimizer.step()
        if float(loss) < best_loss:
            best_loss = float(loss); best_state = copy.deepcopy(scorer.state_dict())
        if step in {1, 100, 250, 500}:
            events.append({"step": step, "train_loss": float(loss)})
    if best_state is None:
        raise RuntimeError("Scorer never produced a finite best state")
    scorer.load_state_dict(best_state); scorer.eval().requires_grad_(False)
    train_logits = torch.cat([item["base_logits"] + scorer(item["features"]) for item in train_data])
    # Threshold is selected from train rows only and then frozen for holdout.
    thresholds = [i / 100 for i in range(35, 96)]
    threshold_scores = []
    for threshold in thresholds:
        predicted = train_logits.sigmoid() >= threshold
        tp = int((predicted & (labels > 0.5)).sum()); fp = int((predicted & (labels <= 0.5)).sum()); fn = int((~predicted & (labels > 0.5)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        threshold_scores.append((f1, threshold, tp, fp, fn))
    _, selected_threshold, tp, fp, fn = max(threshold_scores, key=lambda value: (value[0], value[1]))
    print(json.dumps({"phase": "cache_holdout", "images": len(holdout_rows), "threshold": selected_threshold}), flush=True)
    holdout_data = [_extract(row, model, config, need_labels=True) for row in holdout_rows]
    # Keep a baseline replay and sidecar replay.  Both use identical candidates;
    # only the sidecar residual differs.
    baseline = _score_rows(holdout_data, None, BASE_PAIR_THRESHOLD)
    scored = _score_rows(holdout_data, scorer, selected_threshold)
    by_profile = {}
    for profile in sorted({row["profile_id"] for row in holdout_rows}):
        subset = [item for item in holdout_data if item["row"]["profile_id"] == profile]
        by_profile[profile] = {"baseline": _finish(_score_rows(subset, None, BASE_PAIR_THRESHOLD)), "scored": _finish(_score_rows(subset, scorer, selected_threshold))}
    report = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9-0.1",
        "complete": True, "read_only": True, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "frozen Atom candidates -> all unordered Atom pairs -> frozen DirectAtomPair hidden; residual scorer only; no LSD/line-first",
        "dataset": str(DATA), "train_split": "train", "holdout_split": "fresh_holdout",
        "train_images": len(train_rows), "holdout_images": len(holdout_rows),
        "train_profiles": {key: len(value) for key, value in sorted(by_profile.items())},
        "train_rows": {"pair_rows": int(len(labels)), "positive_rows": int(labels.sum()), "negative_rows": int((labels == 0).sum())},
        "scorer": {"class": "DirectPairResidual", "input_dim": int(features.size(1)), "hidden_dim": 32, "parameters": sum(p.numel() for p in scorer.parameters()), "steps": STEPS, "best_train_loss": best_loss, "events": events, "pos_weight": float(pos_weight), "threshold_selected_train_only": float(selected_threshold), "threshold_train_f1": max(threshold_scores)[0], "threshold_train_confusion": {"tp": tp, "fp": fp, "fn": fn}},
        "baseline_holdout": _finish(baseline), "scored_holdout": _finish(scored), "holdout_profiles": by_profile,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
        "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False, "gpu_go": False, "production_go": False, "cpu_scorer_complete": True, "scored_edge_gate": {"precision": _finish(scored)["edge_precision"] >= 0.965917, "recall": _finish(scored)["edge_recall"] > 0.850031, "dangling": _finish(scored)["dangling_rate"] <= 0.05}, "next": "compare scorer against baseline; no GPU unless scorer is a strict holdout improvement"},
    }
    OUT.mkdir(parents=True)
    torch.save({"schema_version": report["schema_version"], "state_dict": scorer.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "scorer-state.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "baseline": report["baseline_holdout"], "scored": report["scored_holdout"], "gate": report["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
