#!/usr/bin/env python3
"""Read-only CPU attribution of false residual proposals on the fixed r5 slice.

The residual-only shadow has high recall but poor precision.  This audit does
not retrain or sweep a deployment threshold.  It replays only the frozen
production baseline on the fixed 160/80 train/holdout slice and joins it in
memory with the inference-only residual candidate JSONL.  Ground-truth layer
geometry is used only after candidate construction to classify errors.

The report is deliberately aggregate: no target coordinates, atom ids, or
labels are emitted as a model-facing artifact.  It also records whether a
single hard-negative shadow is worth attempting under the existing <=800-step
budget.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-r5-residual-fp-attribution-cpu-v2/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0
TARGET_RADIUS = 6.0
DUPLICATE_RADIUS = 6.0
BASELINE_RING_RADIUS = 20.0
NEAR_TARGET_RADIUS = 12.0
STROKE_RADIUS = 4.0
TEXT_PAD = 4.0
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
SEED = 20261018


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_candidates() -> dict[str, list[dict[str, float]]]:
    """Validate the inference-only candidate contract before joining labels."""
    result: dict[str, list[dict[str, float]]] = {}
    forbidden = {
        "targets", "target_indices", "matched", "element", "visible_element",
        "atom_id", "graph", "bonds", "ground_truth", "label",
    }
    with CANDIDATES.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row.get("sample_key"), str) or row.get("source") != "shadow":
                raise ValueError(f"invalid candidate source at line {line_no}")
            if forbidden & set(row):
                raise ValueError(f"forbidden target field at line {line_no}")
            key = row["sample_key"]
            if key in result or not isinstance(row.get("candidates"), list):
                raise ValueError(f"duplicate or malformed candidate row at line {line_no}")
            values: list[dict[str, float]] = []
            for value in row["candidates"]:
                if set(value) != {"x", "y", "score"}:
                    raise ValueError(f"candidate feature contract failed at line {line_no}")
                clean = {k: float(value[k]) for k in ("x", "y", "score")}
                if not (0.0 <= clean["x"] <= 1.0 and 0.0 <= clean["y"] <= 1.0):
                    raise ValueError(f"candidate coordinate out of range at line {line_no}")
                if not all(torch.isfinite(torch.tensor(v)).item() for v in clean.values()):
                    raise ValueError(f"non-finite candidate at line {line_no}")
                values.append(clean)
            result[key] = values
    if len(result) != 1170:
        raise ValueError(f"expected 1,170 candidate rows, found {len(result)}")
    return result


def _point_segment_distance(point: torch.Tensor, begin: torch.Tensor,
                            end: torch.Tensor) -> float:
    """Distance in pixels from one point to a line segment."""
    vector = end - begin
    denom = float(torch.dot(vector, vector))
    if denom <= 1e-9:
        return float(torch.linalg.vector_norm(point - begin))
    t = float(torch.dot(point - begin, vector)) / denom
    t = max(0.0, min(1.0, t))
    projection = begin + t * vector
    return float(torch.linalg.vector_norm(point - projection))


def _text_contains(point: torch.Tensor, layer_atoms: list[dict], pad: float) -> bool:
    x, y = float(point[0]), float(point[1])
    for atom in layer_atoms:
        if not atom.get("visible_text") or not atom.get("bbox"):
            continue
        left, top, right, bottom = [float(v) for v in atom["bbox"]]
        if left - pad <= x <= right + pad and top - pad <= y <= bottom + pad:
            return True
    return False


def _classify_candidate(point: torch.Tensor, score: float, baseline: torch.Tensor,
                        target_points: torch.Tensor, target_elements: list[str],
                        target_visible: list[bool], bond_segments: list[tuple[torch.Tensor, torch.Tensor]],
                        layer_atoms: list[dict]) -> tuple[str, dict]:
    """Return a mutually exclusive primary error class and diagnostics."""
    baseline_distance = float(torch.cdist(point[None] * IMAGE, baseline * IMAGE).min()) if len(baseline) else float("inf")
    target_distances = torch.linalg.vector_norm(target_points * IMAGE - point[None] * IMAGE, dim=1)
    nearest_distance = float(target_distances.min()) if len(target_points) else float("inf")
    nearest = int(target_distances.argmin()) if len(target_points) else -1
    close = torch.where(target_distances <= TARGET_RADIUS)[0].tolist()
    # A unique hit is a true residual only if its target is outside baseline ownership.
    if len(close) == 1:
        index = int(close[0])
        if baseline_distance <= DUPLICATE_RADIUS:
            category = "protected_collision"
        else:
            category = "true_residual"
    elif len(close) > 1:
        category = "ambiguous_target"
    elif nearest_distance <= NEAR_TARGET_RADIUS:
        category = "near_target_miss"
    else:
        # These are the useful hard-negative strata.  Their order is fixed so
        # each false candidate contributes once to the primary histogram.
        if DUPLICATE_RADIUS < baseline_distance <= BASELINE_RING_RADIUS:
            category = "baseline_protected_ring"
        elif _text_contains(point * IMAGE, layer_atoms, TEXT_PAD):
            category = "visible_text_region"
        elif any(_point_segment_distance(point * IMAGE, begin, end) <= STROKE_RADIUS
                 for begin, end in bond_segments):
            category = "bond_stroke_region"
        else:
            category = "background"
    details = {
        "score": float(score),
        "nearest_target_distance_px": nearest_distance,
        "baseline_distance_px": baseline_distance,
        "nearest_target_element": target_elements[nearest] if nearest >= 0 else "none",
        "nearest_target_visible": bool(target_visible[nearest]) if nearest >= 0 else False,
    }
    return category, details


def _finish(counter: Counter) -> dict:
    out = dict(counter)
    predictions = max(counter["residual_candidates"], 1)
    targets = max(counter["residual_targets"], 1)
    out["residual_precision"] = counter["true_residual"] / predictions
    out["residual_recall_upper_from_candidates"] = counter["true_residual"] / targets
    out["hard_negative_candidates"] = sum(counter[k] for k in (
        "near_target_miss", "baseline_protected_ring", "visible_text_region",
        "bond_stroke_region", "background", "ambiguous_target",
        "duplicate_residual_hit", "protected_collision"))
    out["hard_negative_to_true_ratio"] = out["hard_negative_candidates"] / max(counter["true_residual"], 1)
    for key in ("true_residual", "near_target_miss", "baseline_protected_ring",
                "visible_text_region", "bond_stroke_region", "background",
                "ambiguous_target", "duplicate_residual_hit", "protected_collision"):
        out[f"{key}_fraction_of_residual_candidates"] = counter[key] / predictions
    return out


def _score_buckets(rows: list[dict]) -> dict[str, dict[str, int | float]]:
    buckets = [("lt_0.10", 0.0, 0.10), ("0.10_0.25", 0.10, 0.25),
               ("0.25_0.50", 0.25, 0.50), ("0.50_0.75", 0.50, 0.75),
               ("ge_0.75", 0.75, float("inf"))]
    out = {}
    for name, lo, hi in buckets:
        chosen = [row for row in rows if lo <= row["score"] < hi]
        c = Counter(row["category"] for row in chosen)
        out[name] = {"candidates": len(chosen), **dict(c),
                     "precision": c["true_residual"] / max(len(chosen), 1)}
    return out


def _run_split(rows: list[dict], candidates: dict[str, list[dict[str, float]]],
               model, config, payload, device: torch.device) -> dict:
    total = Counter()
    by_profile: dict[str, Counter] = defaultdict(Counter)
    by_element: dict[str, Counter] = defaultdict(Counter)
    candidate_rows: list[dict] = []
    for row_index, row in enumerate(rows, 1):
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        with torch.inference_mode():
            output = structure_forward(model, image_tensor(sample["image"])[None].to(device),
                                       config, payload["provenance"]["atom_threshold"])
        selected = output[6]
        mask = selected["atom_mask"][0].bool()
        baseline = selected["atom_geometry"][0, mask, :2].detach().cpu()
        atoms = sorted(sample["graph"]["atoms"], key=lambda x: x["atom_id"])
        layer_atoms = sorted(sample["layers"]["atoms"], key=lambda x: x["atom_id"])
        target_points = torch.tensor([x["center"] for x in layer_atoms], dtype=torch.float32) / IMAGE
        elements: list[str] = []
        visible: list[bool] = []
        for atom in atoms:
            element = str(atom.get("element", "other"))
            is_visible = bool(next(x for x in layer_atoms if x["atom_id"] == atom["atom_id"]).get("visible_text", False))
            if element == "C" and not is_visible:
                element = "implicit_C"
            elif element not in {"O", "N"}:
                element = "other"
            elements.append(element); visible.append(is_visible)
        bond_segments = []
        for bond in sample["layers"]["bonds"]:
            bond_segments.append((torch.tensor(bond["begin"], dtype=torch.float32) / IMAGE,
                                  torch.tensor(bond["end"], dtype=torch.float32) / IMAGE))
        profile_total = by_profile[row["profile_id"]]
        # Frozen baseline ownership is computed before candidate labels.  The
        # candidate artifact itself contains no target information.
        target_dist = torch.cdist(baseline * IMAGE, target_points * IMAGE) if len(baseline) else torch.empty((0, len(target_points)))
        baseline_owned = target_dist.min(dim=0).values <= DUPLICATE_RADIUS if len(baseline) else torch.zeros(len(target_points), dtype=torch.bool)
        pending: list[dict] = []
        for value in candidates[row["sample_key"]]:
            point = torch.tensor([value["x"], value["y"]], dtype=torch.float32)
            # The shadow's fixed residual guard is part of the observed contract.
            base_distance = float(torch.cdist(point[None] * IMAGE, baseline * IMAGE).min()) if len(baseline) else float("inf")
            if base_distance <= DUPLICATE_RADIUS:
                continue
            category, details = _classify_candidate(point, value["score"], baseline,
                                                    target_points, elements, visible,
                                                    bond_segments, layer_atoms)
            distances = torch.linalg.vector_norm(target_points * IMAGE - point[None] * IMAGE, dim=1)
            pending.append({"category": category, "details": details, "profile_id": row["profile_id"],
                            "distances": distances, "point": point})
        # Match candidate hits to targets one-to-one, using the same greedy
        # distance rule as the residual shadow.  Additional candidates on an
        # already claimed target are hard negatives, not true positives.
        pairs = []
        for ci, item in enumerate(pending):
            for ti in torch.where(item["distances"] <= TARGET_RADIUS)[0].tolist():
                pairs.append((float(item["distances"][ti]), ci, int(ti)))
        used_candidates: set[int] = set(); used_targets: set[int] = set()
        for distance, ci, ti in sorted(pairs):
            if ci in used_candidates or ti in used_targets:
                continue
            used_candidates.add(ci); used_targets.add(ti)
            pending[ci]["assigned_target"] = ti
        for ci, item in enumerate(pending):
            category = item["category"]
            close = torch.where(item["distances"] <= TARGET_RADIUS)[0].tolist()
            if ci in used_candidates:
                target_index = int(item["assigned_target"])
                category = "protected_collision" if bool(baseline_owned[target_index]) else "true_residual"
            elif close:
                # Every close target was already assigned to a better candidate.
                # Keep baseline-owned collisions distinct from residual duplicates.
                category = "protected_collision" if any(bool(baseline_owned[int(t)]) for t in close) else "duplicate_residual_hit"
            total["residual_candidates"] += 1; profile_total["residual_candidates"] += 1
            total[category] += 1; profile_total[category] += 1
            if category == "true_residual":
                target_index = int(item["assigned_target"])
                by_element[elements[target_index]]["true_residual"] += 1
                by_element[elements[target_index]]["predictions"] += 1
            candidate_rows.append({"category": category, **item["details"], "profile_id": item["profile_id"]})
        # Count residual target denominator after frozen baseline ownership.
        total["target_atoms"] += len(target_points); total["residual_targets"] += int((~baseline_owned).sum())
        profile_total["target_atoms"] += len(target_points); profile_total["residual_targets"] += int((~baseline_owned).sum())
        if row_index % 40 == 0:
            print(json.dumps({"split_progress": row_index, "split_size": len(rows)}), flush=True)
    report = {"images": len(rows), "all": _finish(total),
              "by_profile": {k: _finish(v) for k, v in sorted(by_profile.items())},
              "by_element": {k: dict(v) for k, v in sorted(by_element.items())},
              "score_buckets": _score_buckets(candidate_rows)}
    return report


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    candidates = _load_candidates()
    all_rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line]
    accepted = [row for row in all_rows if row.get("render_status") == "accepted"]
    train = sorted([row for row in accepted if row["split"] == "train"], key=lambda x: x["sample_key"])[:TRAIN_LIMIT]
    holdout = sorted([row for row in accepted if row["split"] == "holdout"], key=lambda x: x["sample_key"])[:HOLDOUT_LIMIT]
    if len(train) != TRAIN_LIMIT or len(holdout) != HOLDOUT_LIMIT:
        raise RuntimeError("fixed r5 split changed")
    device = torch.device("cpu")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    before = file_digest(INITIAL)
    train_report = _run_split(train, candidates, model, config, payload, device)
    holdout_report = _run_split(holdout, candidates, model, config, payload, device)
    after = file_digest(INITIAL)

    h = holdout_report["all"]
    false_count = h["residual_candidates"] - h["true_residual"]
    # With all 165 observed true hits retained, precision .85 allows at most
    # floor(165/.85 - 165) false candidates.  This is a planning bound, not a
    # holdout threshold or a claim that the next model will achieve it.
    allowable_false = int((h["true_residual"] / 0.85) - h["true_residual"])
    required_suppression = max(0, false_count - allowable_false)
    required_fraction = required_suppression / max(false_count, 1)
    hard = sum(h.get(k, 0) for k in ("near_target_miss", "baseline_protected_ring",
                                     "visible_text_region", "bond_stroke_region",
                                     "background", "ambiguous_target", "protected_collision"))
    # A candidate hard-negative run is worthwhile only if enough positives and
    # separable negatives exist on both splits.  This gate authorizes protocol
    # preparation only; it never authorizes GPU training or deployment.
    gate = {
        "checkpoint_unchanged": before == INITIAL_SHA == after,
        "sealed_test_read": False,
        "candidate_artifact_valid": True,
        "train_positive_candidates_ge_50": train_report["all"]["true_residual"] >= 50,
        "holdout_positive_candidates_ge_50": h["true_residual"] >= 50,
        "train_hard_negative_candidates_ge_100": train_report["all"]["hard_negative_candidates"] >= 100,
        "holdout_hard_negative_candidates_ge_100": hard >= 100,
        "all_profiles_present": len(holdout_report["by_profile"]) == 6 and len(train_report["by_profile"]) == 6,
        "required_false_positive_suppression_fraction_le_0_75": required_fraction <= 0.75,
    }
    gate["hard_negative_shadow_worth_attempting"] = all(gate.values())
    report = {
        "schema_version": "chemvector-identity-aligned-r5-residual-fp-attribution-cpu-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "initial_sha256_after": after,
        "candidate_artifact_sha256": digest(CANDIDATES),
        "fixed_slice": {"train_images": len(train), "holdout_images": len(holdout),
                        "target_radius_px": TARGET_RADIUS, "duplicate_radius_px": DUPLICATE_RADIUS,
                        "device": str(device)},
        "train": train_report, "holdout": holdout_report,
        "planning_bound": {
            "holdout_false_candidates": false_count,
            "holdout_allowable_false_at_0_85_if_all_true_hits_retained": allowable_false,
            "minimum_false_candidates_to_suppress": required_suppression,
            "required_false_suppression_fraction": required_fraction,
            "interpretation": "diagnostic arithmetic only; no holdout threshold was selected",
        },
        "hard_negative_plan": {
            "positive_source": "unique residual target hits from in-memory labels",
            "hard_negative_sources": ["near_target_miss", "baseline_protected_ring",
                                     "visible_text_region", "bond_stroke_region", "background"],
            "protected_positive_rule": "retain one positive per residual target; never train on baseline-owned coordinates as positives",
            "sampling": "stratify by profile and error category; cap background negatives at 2x hard negatives per batch",
            "loss": "positive-presence plus pairwise/ranking loss against category-stratified hard negatives; fixed decoder and baseline guard",
            "budget": "one fresh <=800-step shadow, no threshold sweep, no production graph merge",
            "promotion_gates": ["precision >= 0.85", "recall >= 0.80", "hard collision <= 0.10", "one-residual bond coverage >= 0.50", "independent edge audit passes"],
        },
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "holdout": h, "planning_bound": report["planning_bound"], "gate": gate}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
