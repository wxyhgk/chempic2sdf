#!/usr/bin/env python3
"""Read-only r5 vertex-to-frozen-bond integration shadow.

This tool never trains, writes a checkpoint, or uses target labels to create
candidate points.  Without an explicit vertex-state receipt it performs the
contract preflight and records ``NOT_READY``; this prevents accidentally
evaluating a randomly initialized head as the r5 shadow result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.components.bonds.objects.bond_object import BOND_TYPE_TO_ID
from chemvector.components.graph.proposals.vertex_bond_integration import (
    attach_vertex_candidates,
    attachment_geometry_metrics,
)
from chemvector.core.model import ChemVectorConfig, ChemVertexDetector
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.vertex import decode_vertices

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
SHADOW_REPORT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v2/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-bond-integration-20261017-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
ATTACH_RADIUS = 12.0
DEDUP_RADIUS = 6.0
TARGET_RADIUS = 6.0
AMBIGUITY_RADIUS = 20.0
BATCH_SIZE = 8
# The vertex head's role heatmap is a conditional annotation, not an
# existence/objectness field.  Candidate existence must come from the trained
# vertex heatmap used by the r5 quality receipt; role is read only after a peak
# has been selected.  Keeping this explicit prevents the old role-aware max()
# decoder from turning background role peaks into graph candidates.
DECODER_ROLE_AWARE = False


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _targets(sample: dict[str, Any]) -> tuple[torch.Tensor, list[str], list[dict[str, Any]]]:
    atoms = sorted(sample["graph"]["atoms"], key=lambda row: row["atom_id"])
    by_id = {row["atom_id"]: row for row in sample["layers"]["atoms"]}
    points = torch.tensor([by_id[row["atom_id"]]["center"] for row in atoms], dtype=torch.float32) / IMAGE_SIZE
    elements = [row["element"] for row in atoms]
    bonds = list(sample["graph"]["bonds"])
    return points, elements, bonds


def _candidate_labels(points: torch.Tensor, target_points: torch.Tensor) -> list[int | None]:
    """Label candidates after generation, with strict ownership only."""
    if not len(points):
        return []
    distances = torch.cdist(points * IMAGE_SIZE, target_points * IMAGE_SIZE)
    labels: list[int | None] = []
    for index in range(len(points)):
        close = torch.where(distances[index] <= TARGET_RADIUS)[0]
        near = torch.where(distances[index] <= AMBIGUITY_RADIUS)[0]
        labels.append(int(close[0]) if len(close) == 1 and len(near) == 1 else None)
    return labels


def _bond_target_map(
    endpoint: torch.Tensor, target_points: torch.Tensor, bonds: list[dict[str, Any]]
) -> int | None:
    if not bonds:
        return None
    scores: list[tuple[float, int]] = []
    for index, bond in enumerate(bonds):
        expected = target_points[[bond["begin_atom"], bond["end_atom"]]]
        forward = torch.linalg.vector_norm((endpoint - expected) * IMAGE_SIZE, dim=-1).amax()
        reverse = torch.linalg.vector_norm((endpoint - expected.flip(0)) * IMAGE_SIZE, dim=-1).amax()
        scores.append((float(min(forward, reverse)), index))
    scores.sort()
    if scores[0][0] > ATTACH_RADIUS or (len(scores) > 1 and scores[1][0] <= ATTACH_RADIUS):
        return None
    return scores[0][1]


def _empty_metrics() -> dict[str, Any]:
    return {
        "images": 0, "predictions": 0, "active_bonds": 0, "active_endpoints": 0,
        "attached_endpoints": 0, "endpoint_attachment_rate": 0.0,
        "two_endpoint_bonds": 0, "two_endpoint_bond_rate": 0.0,
        "correct_attached_endpoints": 0, "correct_attached_endpoint_precision": 0.0,
        "complete_predicted_bonds": 0, "correct_complete_bonds": 0,
        "correct_two_endpoint_bond_precision": 0.0, "target_bonds": 0,
        "recovered_target_bonds": 0, "bond_recovery_recall": 0.0,
        "bond_type_correct": 0, "bond_type_evaluated": 0,
        "bond_type_accuracy_conditional": 0.0, "candidate_reuse_conflicts": 0,
        "candidate_reuse_endpoints": 0, "endpoint_distance_sum": 0.0,
        "endpoint_distance_count": 0, "endpoint_distance_values": [],
    }


def _add_path_metrics(
    totals: dict[str, Any], points: torch.Tensor, scores: torch.Tensor,
    bond_endpoints: torch.Tensor, bond_types: torch.Tensor, bond_mask: torch.Tensor,
    target_points: torch.Tensor, elements: list[str], target_bonds: list[dict[str, Any]],
    profile: str,
) -> None:
    result = attach_vertex_candidates(points, bond_endpoints, bond_mask, scores=scores,
                                      radius_px=ATTACH_RADIUS, duplicate_radius_px=DEDUP_RADIUS)
    geom = attachment_geometry_metrics(result, bond_mask)
    totals["images"] += 1
    totals["predictions"] += int(len(result.candidate_kept))
    for key in ("active_bonds", "active_endpoints", "attached_endpoints", "two_endpoint_bonds",
                "candidate_reuse_conflicts", "candidate_reuse_endpoints"):
        totals[key] += int(geom[key])
    distances = result.endpoint_distance_px[result.endpoint_distance_px.isfinite()]
    totals["endpoint_distance_sum"] += float(distances.sum())
    totals["endpoint_distance_count"] += int(len(distances))
    totals["endpoint_distance_values"].extend(float(value) for value in distances.tolist())
    labels = _candidate_labels(points[result.candidate_kept], target_points)
    candidate_map = {index: labels[index] for index in range(len(labels))}
    recovered: set[int] = set()
    for bond_index in torch.where(bond_mask)[0].tolist():
        proposal_target = _bond_target_map(bond_endpoints[bond_index], target_points, target_bonds)
        left, right = result.endpoint_candidate[bond_index].tolist()
        complete = left >= 0 and right >= 0 and left != right
        if complete:
            totals["complete_predicted_bonds"] += 1
        expected_ids = None
        if proposal_target is not None:
            target_bond = target_bonds[proposal_target]
            expected_ids = {target_bond["begin_atom"], target_bond["end_atom"]}
        actual_ids = {candidate_map.get(left), candidate_map.get(right)} if complete else set()
        if complete and expected_ids is not None and actual_ids == expected_ids:
            totals["correct_complete_bonds"] += 1
            recovered.add(proposal_target)
            totals["bond_type_evaluated"] += 1
            expected_type = BOND_TYPE_TO_ID[target_bonds[proposal_target]["chemical_type"]]
            predicted_type = int(torch.argmax(bond_types[bond_index, 4:])) if bond_types.ndim == 2 else -1
            totals["bond_type_correct"] += int(predicted_type == expected_type)
        for endpoint_index, candidate in enumerate((left, right)):
            if candidate < 0:
                continue
            totals["attached_endpoints"] += 0  # explicit: geometry counted above
            if expected_ids is not None and candidate_map.get(candidate) in expected_ids:
                totals["correct_attached_endpoints"] += 1
    totals["target_bonds"] += len(target_bonds)
    totals["recovered_target_bonds"] += len(recovered)


def _finalize_metrics(total: dict[str, Any]) -> dict[str, Any]:
    out = dict(total)
    values = sorted(out.pop("endpoint_distance_values", []))
    out["endpoint_attachment_rate"] = total["attached_endpoints"] / max(total["active_endpoints"], 1)
    out["two_endpoint_bond_rate"] = total["two_endpoint_bonds"] / max(total["active_bonds"], 1)
    out["correct_attached_endpoint_precision"] = total["correct_attached_endpoints"] / max(total["attached_endpoints"], 1)
    out["correct_two_endpoint_bond_precision"] = total["correct_complete_bonds"] / max(total["complete_predicted_bonds"], 1)
    out["bond_recovery_recall"] = total["recovered_target_bonds"] / max(total["target_bonds"], 1)
    out["bond_type_accuracy_conditional"] = total["bond_type_correct"] / max(total["bond_type_evaluated"], 1)
    out["mean_endpoint_distance_px"] = total["endpoint_distance_sum"] / max(total["endpoint_distance_count"], 1)
    if values:
        out["p50_endpoint_distance_px"] = values[len(values) // 2]
        out["p90_endpoint_distance_px"] = values[min(len(values) - 1, int(len(values) * 0.90))]
    else:
        out["p50_endpoint_distance_px"] = None
        out["p90_endpoint_distance_px"] = None
    return out


def _load_vertex_state(path: Path, head: torch.nn.Module, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != "chemvector-r5-vertex-state-0.1":
        raise ValueError("vertex state schema is not the reviewed r5 state format")
    if payload.get("initial_sha256") != INITIAL_SHA or "head_state_dict" not in payload:
        raise ValueError("vertex state is not bound to the reviewed initial checkpoint")
    if "optimizer_state_dict" in payload or "sealed_test" in payload:
        raise ValueError("optimizer/sealed state is forbidden in integration artifact")
    head.load_state_dict(payload["head_state_dict"], strict=True)
    block = payload.get("vision_stride4_block_state_dict")
    if block is not None:
        model.vision.stages[0][-1].load_state_dict(block, strict=True)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint digest changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line]
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    if {r["molecule_id"] for r in train} & {r["molecule_id"] for r in holdout}:
        raise RuntimeError("train/holdout molecule identity overlap")
    report: dict[str, Any] = {
        "schema_version": "chemvector-identity-aligned-vertex-bond-integration-0.1",
        "complete": False, "read_only": True, "training_performed": False,
        "deployment_authorized": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
        "data_sha256": digest(DATA / "summary.json"), "train_images": len(train),
        "holdout_images": len(holdout), "decoder": {"attach_radius_px": ATTACH_RADIUS,
        "dedup_radius_px": DEDUP_RADIUS, "candidate_threshold": .05,
        "maximum_candidates": 128, "role_aware": DECODER_ROLE_AWARE,
        "existence_source": "vertex_heatmap_logits"},
    }
    state_path = Path(args.vertex_state) if args.vertex_state else None
    if state_path is None or not state_path.exists():
        report.update({"status": "NOT_READY", "reason": "reviewed r5 vertex state receipt is absent; no random head is evaluated",
                       "vertex_state": str(state_path) if state_path else None,
                       "gate": {"training_admissible": False, "deployment_authorized": False,
                                "production_unchanged": True, "next": "export reviewed r5 head state, then rerun this read-only audit"}})
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n"); return report

    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu only for a bounded contract smoke")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    head = ChemVertexDetector(ChemVectorConfig(vision_dims=(96, 192, 384, 512), vertex_feature_dim=64)).to(device)
    _load_vertex_state(state_path, head, model)
    head.eval(); model.eval()

    # First pass: old atom proposals and frozen bond proposals from the exact
    # production model.  No target tensors enter this pass.
    frozen: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        pixels = image_tensor(sample["image"])[None].to(device)
        with torch.inference_mode():
            _, _, _, atoms, bonds, _, _ = structure_forward(model, pixels, config, payload["provenance"]["atom_threshold"])
        frozen[row["sample_key"]] = {
            "old_points": atoms["atom_geometry"][0, :, :2].detach().cpu(),
            "old_scores": atoms["atom_scores"][0].detach().cpu(),
            "old_mask": atoms["atom_mask"][0].detach().cpu(),
            "bond_endpoints": bonds["bond_endpoints"][0].detach().cpu(),
            "bond_types": bonds["bond_types"][0].detach().cpu(),
            "bond_mask": bonds["bond_mask"][0].detach().cpu(),
        }

    # Second pass: only the supplied new vertex state observes the images.
    totals = {split: {path: _empty_metrics() for path in ("old", "new", "oracle")} for split in ("train", "holdout")}
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        target_points, elements, target_bonds = _targets(sample)
        cached = frozen[row["sample_key"]]
        old_points = cached["old_points"][cached["old_mask"]]
        old_scores = cached["old_scores"][cached["old_mask"]]
        oracle_scores = torch.ones(len(target_points))
        with torch.inference_mode():
            pixels = image_tensor(sample["image"])[None].to(device)
            features = model.vision.forward_stroke_resolution(pixels)
            decoded = decode_vertices(head(features), threshold=.05, maximum_vertices=128,
                                      role_aware=DECODER_ROLE_AWARE, nms_kernel_size=3)[0]
        new_points = torch.tensor([[p["x"], p["y"]] for p in decoded], dtype=torch.float32).cpu()
        new_scores = torch.tensor([p.get("score", 0.0) for p in decoded], dtype=torch.float32).cpu()
        endpoint, types, mask = cached["bond_endpoints"], cached["bond_types"], cached["bond_mask"]
        bucket = totals[row["split"]]
        _add_path_metrics(bucket["old"], old_points, old_scores, endpoint, types, mask, target_points, elements, target_bonds, row["profile_id"])
        _add_path_metrics(bucket["new"], new_points, new_scores, endpoint, types, mask, target_points, elements, target_bonds, row["profile_id"])
        _add_path_metrics(bucket["oracle"], target_points, oracle_scores, endpoint, types, mask, target_points, elements, target_bonds, row["profile_id"])
    report.update({"status": "COMPLETE", "complete": True,
                   "vertex_state": str(state_path),
                   "vertex_state_sha256": digest(state_path),
                   "by_split": {split: {path: _finalize_metrics(metrics) for path, metrics in paths.items()} for split, paths in totals.items()},
                   "gate": {"training_admissible": False, "deployment_authorized": False,
                            "production_unchanged": digest(INITIAL) == INITIAL_SHA,
                            "next": "independent review; no promotion"}})
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex-state", help="reviewed torch state receipt; absent means contract-only NOT_READY")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cpu", action="store_true", help="allow CPU forward-only smoke")
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({"status": result.get("status"), "gate": result.get("gate")}, indent=2))
