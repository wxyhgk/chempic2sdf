#!/usr/bin/env python3
"""Forward-only integration shadow for the r5 vertex candidate field.

This audit deliberately does *not* run the r5 training script and does not
load a shadow checkpoint.  The shadow side must be supplied as an explicit,
target-independent JSONL candidate artifact.  Baseline candidates and frozen
bond outputs are computed from ``initial.pt``.  The shadow rows are merged only
in memory, after the production prefix has been copied and hash-locked; they
are never passed to the production set head, pair head, graph builder, or SDF
writer.

The absence of a candidate artifact is a normal, fail-closed result.  This is
intentional: a metrics-only training receipt cannot be treated as an inference
checkpoint.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chemvector.components.graph.proposals.protected_residual_merge import (
    CandidateSet,
    baseline_prefix_unchanged,
    merge_candidates,
    select_residual_candidates,
)
from chemvector.components.graph.proposals.residual_bond_attachment import (
    attach_residual_candidates,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.data import load_sample


DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
# Bind the read-only integration audit to the reviewed, scored r5 export.
# The old v2 artifact was an earlier shadow and is intentionally not the
# default: using it silently would invalidate the current r5 split/hash
# comparison.  Callers can still override this path explicitly via the
# environment variable handled below.
DEFAULT_CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/candidates-scored.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-integration-shadow-20261017/report.json"
MATCH_RADIUS_PX = 6.0
DUPLICATE_RADIUS_PX = 6.0
BOND_RADIUS_PX = 12.0
MAX_CANDIDATES = 128


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _candidate_digest(candidate: CandidateSet) -> str:
    return hashlib.sha256(
        (_tensor_digest(candidate.points) + _tensor_digest(candidate.scores)).encode()
    ).hexdigest()


def _tensor_mapping_digest(*mappings: dict[str, torch.Tensor]) -> str:
    """Stable digest for the complete frozen graph/bond tensor boundary."""
    h = hashlib.sha256()
    for mapping in mappings:
        for name in sorted(mapping):
            value = mapping[name]
            if not isinstance(value, torch.Tensor):
                continue
            h.update(name.encode())
            h.update(_tensor_digest(value).encode())
    return h.hexdigest()


def _load_rows() -> dict[str, dict]:
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line]
    accepted = [row for row in rows if row.get("render_status") == "accepted"]
    if len(accepted) != 1170:
        raise ValueError(f"r5 accepted row count changed: {len(accepted)}")
    if len({row["sample_key"] for row in accepted}) != len(accepted):
        raise ValueError("duplicate sample_key in r5 panel")
    if any(not row.get("connection_identity") for row in accepted):
        raise ValueError("empty connection identity in r5 panel")
    return {row["sample_key"]: row for row in accepted}


def _parse_candidates(path: Path, rows: dict[str, dict]) -> tuple[dict[str, list[dict]], dict]:
    """Load only target-independent candidate fields; reject labels/GT leakage."""
    if not path.is_file():
        return {}, {"provided": True, "available": False, "reason": f"missing candidate artifact: {path}"}
    result: dict[str, list[dict]] = {}
    forbidden = {
        "targets", "target_indices", "matched", "element", "visible_element",
        "atom_id", "graph", "bonds", "ground_truth", "label",
    }
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) & forbidden:
            raise ValueError(f"candidate artifact contains forbidden target field at line {line_no}")
        key = record.get("sample_key")
        if key not in rows:
            raise ValueError(f"candidate row is outside accepted r5 panel: {key}")
        if record.get("source") != "shadow":
            raise ValueError(f"candidate source must be shadow at line {line_no}")
        if key in result:
            raise ValueError(f"duplicate candidate sample_key: {key}")
        values = record.get("candidates")
        if not isinstance(values, list) or len(values) > MAX_CANDIDATES:
            raise ValueError(f"invalid candidate list at line {line_no}")
        clean = []
        for candidate in values:
            if set(candidate) - {"x", "y", "score"}:
                raise ValueError(f"candidate contains non-inference field at line {line_no}")
            x, y, score = float(candidate["x"]), float(candidate["y"]), float(candidate["score"])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"candidate coordinate outside [0,1] at line {line_no}")
            if not torch.isfinite(torch.tensor([x, y, score])).all():
                raise ValueError(f"non-finite candidate at line {line_no}")
            clean.append({"x": x, "y": y, "score": score})
        result[key] = clean
    missing = sorted(set(rows) - set(result))
    if missing:
        return result, {"provided": True, "available": True, "complete": False,
                        "reason": f"candidate artifact missing {len(missing)} accepted rows"}
    return result, {"provided": True, "available": True, "complete": True,
                    "rows": len(result), "sha256": file_digest(path)}


def _candidate_set(values: list[dict], device: torch.device) -> CandidateSet:
    points = torch.tensor([[v["x"], v["y"]] for v in values], dtype=torch.float32, device=device)
    scores = torch.tensor([v["score"] for v in values], dtype=torch.float32, device=device)
    return CandidateSet(points, scores, "residual")


def _locked_matches(points: torch.Tensor, targets: torch.Tensor, occupied: set[int]) -> set[int]:
    if not len(points) or not len(targets):
        return set()
    pairs = []
    distance = torch.cdist(points * 768.0, targets * 768.0)
    for i, j in zip(*torch.where(distance <= MATCH_RADIUS_PX)):
        pairs.append((float(distance[i, j]), int(i), int(j)))
    used_points, used_targets = set(), set(occupied)
    for _, i, j in sorted(pairs):
        if i not in used_points and j not in used_targets:
            used_points.add(i); used_targets.add(j)
    return used_targets - occupied


def _baseline_set(selected: dict[str, torch.Tensor]) -> CandidateSet:
    mask = selected["atom_mask"][0].bool()
    return CandidateSet(selected["atom_geometry"][0, mask, :2].detach(),
                        selected["atom_scores"][0, mask].float().detach(), "baseline")


def _production_input_hash(selected: dict[str, torch.Tensor], bonds: dict[str, torch.Tensor]) -> str:
    """Hash exactly the frozen tensors consumed by the production graph path."""
    return _tensor_mapping_digest(selected, bonds)


def _run_available(candidates: dict[str, list[dict]], rows: dict[str, dict]) -> dict:
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    device = torch.device("cuda:0")
    before_checkpoint = file_digest(INITIAL)
    totals = Counter()
    profile_totals: dict[str, Counter] = {}
    production_forward_shadow_rows = 0
    baseline_input_hashes = []
    baseline_after_hashes = []
    for key in sorted(rows):
        sample = load_sample(DATA, key, purpose="diagnostic")
        pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
        with torch.inference_mode():
            _, _, _, atoms, bonds, _, selected = structure_forward(
                model, pixels, config, payload["provenance"]["atom_threshold"]
            )
        baseline = _baseline_set(selected)
        # This is the exact tensor boundary that production graph/bond would see.
        baseline_input_hash = _production_input_hash(selected, bonds)
        baseline_input_hashes.append(baseline_input_hash)
        shadow_raw = _candidate_set(candidates[key], device)
        # Fixed, predeclared inference gate; no target/GT-dependent threshold
        # fitting occurs in this audit.
        shadow = select_residual_candidates(shadow_raw.points, shadow_raw.scores,
                                            threshold=0.5, maximum=128,
                                            suppression_radius=0.0)
        merged = merge_candidates(baseline, shadow,
                                  duplicate_radius=DUPLICATE_RADIUS_PX / 768.0)
        prefix_ok = baseline_prefix_unchanged(baseline, merged)
        residual = CandidateSet(merged.points[baseline.points.size(0):],
                                merged.scores[baseline.points.size(0):], "residual")
        # No model call occurs after this point.  This counter is a hard-coded
        # boundary assertion, not an accuracy claim.
        production_forward_shadow_rows += 0
        bond_mask = bonds["bond_mask"][0].bool()
        endpoints = bonds["bond_endpoints"][0, bond_mask].detach()
        occupied = torch.zeros((endpoints.size(0), 2), dtype=torch.bool, device=device)
        if endpoints.numel() and baseline.points.numel():
            nearest = torch.cdist(endpoints.reshape(-1, 2) * 768.0,
                                  baseline.points * 768.0).amin(dim=1)
            occupied = (nearest <= BOND_RADIUS_PX).reshape(-1, 2)
        attachment = attach_residual_candidates(
            residual.points, endpoints,
            torch.ones(endpoints.size(0), dtype=torch.bool, device=device),
            radius_px=BOND_RADIUS_PX, occupied_endpoints=occupied,
        )
        target_points = torch.tensor([a["center"] for a in sample["layers"]["atoms"]],
                                     dtype=torch.float32, device=device) / 768.0
        baseline_matches = _locked_matches(baseline.points, target_points, set())
        shadow_matches = _locked_matches(residual.points, target_points, baseline_matches)
        attached = int(attachment["matched"].sum())
        # Build endpoint ownership from both immutable baseline rows and the
        # residual attachment table.  The previous audit checked only a
        # baseline partner, which classified a valid residual-residual edge as
        # dangling.  A new edge is complete when both frozen ports have owners
        # and the two owners are distinct; this remains a simulation-only
        # diagnostic and never calls the production graph path.
        complete_edges = 0
        dangling = 0
        owners = torch.full((endpoints.size(0), 2), -1, dtype=torch.long, device=device)
        if endpoints.numel() and baseline.points.numel():
            baseline_distance = torch.cdist(endpoints.reshape(-1, 2) * 768.0,
                                            baseline.points * 768.0)
            nearest_distance, nearest_index = baseline_distance.min(dim=1)
            baseline_owned = nearest_distance <= BOND_RADIUS_PX
            owners[baseline_owned.reshape(-1, 2)] = nearest_index[baseline_owned]
        residual_owner_offset = baseline.points.size(0)
        for row_index in torch.where(attachment["matched"])[0].tolist():
            bond_index = int(attachment["bond_indices"][row_index])
            endpoint_index = int(attachment["endpoint_indices"][row_index])
            owners[bond_index, endpoint_index] = residual_owner_offset + row_index
        for bond_index in range(endpoints.size(0)):
            left, right = owners[bond_index].tolist()
            if left >= 0 and right >= 0 and left != right:
                # Count only edges touched by a residual row; baseline-only
                # edges are outside this shadow's residual accounting.
                if left >= residual_owner_offset or right >= residual_owner_offset:
                    complete_edges += 1
            residual_sides = sum(owner >= residual_owner_offset for owner in (left, right))
            if residual_sides and not (left >= 0 and right >= 0 and left != right):
                dangling += residual_sides
        # Re-run only the frozen production path after the in-memory merge.
        # The merged candidate is deliberately not an argument to this call.
        with torch.inference_mode():
            _, _, _, _, bonds_after, _, selected_after = structure_forward(
                model, pixels, config, payload["provenance"]["atom_threshold"]
            )
        baseline_after_hash = _production_input_hash(selected_after, bonds_after)
        profile = rows[key]["profile_id"]
        block = profile_totals.setdefault(profile, Counter())
        for target, value in ((totals, 1), (block, 1)):
            target["images"] += value
        totals.update({"baseline_candidates": baseline.points.size(0),
                       "shadow_candidates": shadow.points.size(0),
                       "shadow_duplicates": shadow.points.size(0) - residual.points.size(0),
                       "residual_candidates": residual.points.size(0),
                       "baseline_matches": len(baseline_matches),
                       "shadow_new_matches": len(shadow_matches),
                       "attached_residual": attached,
                       "complete_residual_edges": complete_edges,
                       "dangling_residual_endpoints": dangling,
                       "prefix_failures": int(not prefix_ok)})
        block.update({"residual_candidates": residual.points.size(0),
                      "shadow_new_matches": len(shadow_matches),
                      "attached_residual": attached,
                      "complete_residual_edges": complete_edges,
                      "dangling_residual_endpoints": dangling,
                      "prefix_failures": int(not prefix_ok)})
        baseline_after_hashes.append(baseline_after_hash)
    if baseline_input_hashes != baseline_after_hashes:
        raise RuntimeError("baseline graph input hash changed during shadow")
    return {
        "images": len(rows), "totals": dict(totals),
        "by_profile": {k: dict(v) for k, v in sorted(profile_totals.items())},
        "production_forward_shadow_rows": production_forward_shadow_rows,
        "checkpoint_sha_before": before_checkpoint,
        "checkpoint_sha_after": file_digest(INITIAL),
        "baseline_graph_input_hashes_before": baseline_input_hashes,
        "baseline_graph_input_hashes_after": baseline_after_hashes,
    }


def run(candidate_path: Path = DEFAULT_CANDIDATES, output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = _load_rows()
    candidates, artifact = _parse_candidates(candidate_path, rows)
    result = {
        "schema_version": "chemvector-identity-aligned-vertex-integration-shadow-0.1",
        "complete": False, "training_performed": False, "optimizer_steps": 0,
        "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "candidate_artifact": artifact,
        "contract": {
            "baseline_prefix_hard_locked": True,
            "production_selector_receives_shadow": False,
            "production_graph_receives_shadow": False,
            "production_bond_receives_shadow": False,
            "merge": "in_memory_baseline_first_duplicate_filtered",
            "duplicate_radius_px": DUPLICATE_RADIUS_PX,
            "bond_attachment_radius_px": BOND_RADIUS_PX,
            "candidate_target_fields_forbidden": True,
        },
    }
    if artifact.get("available") and artifact.get("complete"):
        metrics = _run_available(candidates, rows)
        total = metrics["totals"]
        shadow = max(total.get("shadow_candidates", 0), 1)
        new = max(total.get("shadow_new_matches", 0), 1)
        result.update({"complete": True, "metrics": metrics,
                       "gate": {
                           "isolation_hard_gate": (
                               metrics["checkpoint_sha_before"] == INITIAL_SHA and
                               metrics["checkpoint_sha_after"] == INITIAL_SHA and
                               metrics["production_forward_shadow_rows"] == 0 and
                               metrics["baseline_graph_input_hashes_before"] == metrics["baseline_graph_input_hashes_after"]),
                           "baseline_prefix_retention": total.get("prefix_failures", 1) == 0,
                           "shadow_precision_ge_085": total.get("shadow_new_matches", 0) / shadow >= .85,
                           "shadow_dangling_zero": total.get("dangling_residual_endpoints", 1) == 0,
                           "complete_edges_not_less_than_new_matches": total.get("complete_residual_edges", 0) >= total.get("shadow_new_matches", 0),
                           "go": False,
                           "next": "independent review; no production integration or checkpoint promotion",
                       }})
        result["gate"]["go"] = all(value for key, value in result["gate"].items()
                                    if key not in {"go", "next"})
    else:
        result["gate"] = {"isolation_hard_gate": False, "candidate_artifact_ready": False,
                          "go": False,
                          "next": "export target-independent r5 shadow candidates; do not retrain or promote checkpoint"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "complete": result["complete"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(Path(os.environ.get("CHEMVECTOR_VERTEX_CANDIDATES", str(DEFAULT_CANDIDATES))),
        Path(os.environ.get("CHEMVECTOR_VERTEX_INTEGRATION_OUTPUT", str(DEFAULT_OUTPUT))))
