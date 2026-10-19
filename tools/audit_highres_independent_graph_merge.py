"""Forward-only audit of an isolated high-resolution graph merge.

The frozen production selector is evaluated first.  A separately trained,
in-memory 384px rescue branch contributes only graph candidates; no rescue row
is passed through the production set head.  Assignment is locked in two
phases: baseline candidates claim targets first, then residual candidates may
claim only unoccupied targets.  This distinguishes candidate capacity from
set-head ownership coupling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.graph.proposals.protected_residual_merge import (
    CandidateSet,
    baseline_prefix_unchanged,
    merge_candidates,
)
from chemvector.components.graph.proposals.residual_bond_attachment import (
    attach_residual_candidates,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.current_initial_cohort import split_group_disjoint
from tools.shadow_atom_highres_pixel_branch import PixelStride2Rescue, _highres_objective


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(__import__("os").environ.get(
    "CHEMVECTOR_HIGHRES_MERGE_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-highres-independent-graph-merge-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TRAIN_IMAGES = 120
STEPS = 200
RESCUE_K = 32
DEDUP_RADIUS_PX = 6.0
MATCH_RADIUS_PX = 6.0
BOND_SUPPORT_RADIUS_PX = 12.0
LABEL_MODE = os.environ.get("CHEMVECTOR_HIGHRES_LABEL_MODE", "all")
if LABEL_MODE not in {"all", "residual_text"}:
    raise ValueError("CHEMVECTOR_HIGHRES_LABEL_MODE must be all or residual_text")


def _positive(anchors: torch.Tensor, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    target = torch.zeros((anchors.size(0), 1, h, w), dtype=torch.bool, device=anchors.device)
    x = (anchors[..., 0] * w).clamp(0, w - 1e-4).long()
    y = (anchors[..., 1] * h).clamp(0, h - 1e-4).long()
    batch = torch.arange(anchors.size(0), device=anchors.device)[:, None].expand_as(mask)
    target[batch[mask], 0, y[mask], x[mask]] = True
    return target


def _rescue_candidates(logits: torch.Tensor) -> CandidateSet:
    score = logits.sigmoid()[0, 0]
    height, width = score.shape
    peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
    py, px = torch.where(peaks)
    values = score[py, px]
    order = torch.argsort(values, descending=True)[:RESCUE_K]
    points = torch.stack(
        ((px[order].float() + 0.5) / width, (py[order].float() + 0.5) / height),
        dim=-1,
    )
    return CandidateSet(points, values[order], "residual")


def _locked_match(
    baseline: CandidateSet,
    residual: CandidateSet,
    target_points: torch.Tensor,
    baseline_mapping: set[int],
) -> tuple[set[int], set[int], set[int]]:
    """Match baseline first, then residual only to remaining targets."""
    remaining = [i for i in range(target_points.size(0)) if i not in baseline_mapping]
    if not remaining or residual.points.size(0) == 0:
        return baseline_mapping, set(), set()
    cost = torch.cdist(residual.points * 768.0, target_points[remaining] * 768.0)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    matched = set()
    matched_rows = set()
    for row, col in zip(rows, cols):
        if float(cost[row, col]) <= MATCH_RADIUS_PX:
            matched.add(int(remaining[col]))
            matched_rows.add(int(row))
    return baseline_mapping, matched, matched_rows


def _edge_audit(
    baseline: CandidateSet,
    residual: CandidateSet,
    bond_endpoints: torch.Tensor,
    attachment: dict[str, torch.Tensor],
    new_rows: set[int],
    *,
    radius_px: float,
) -> dict[str, int]:
    """Check residual attachments for complete, non-duplicated bond edges."""
    bond_count = bond_endpoints.size(0)
    if bond_count == 0:
        return {"residual_edges": 0, "new_residual_edges": 0, "dangling_residual_endpoints": 0, "one_to_one_violations": 0}
    distance = torch.cdist(
        bond_endpoints.reshape(-1, 2) * 768.0,
        baseline.points * 768.0,
    ).reshape(bond_count, 2, -1)
    nearest_distance, nearest_index = distance.min(dim=-1)
    labels: dict[tuple[int, int], tuple[str, int]] = {}
    for bond_index in range(bond_count):
        for endpoint_index in range(2):
            if float(nearest_distance[bond_index, endpoint_index]) <= radius_px:
                labels[(bond_index, endpoint_index)] = ("baseline", int(nearest_index[bond_index, endpoint_index]))
    matched_rows = torch.where(attachment["matched"])[0].tolist()
    endpoint_pairs = []
    for row in matched_rows:
        bond_index = int(attachment["bond_indices"][row])
        endpoint_index = int(attachment["endpoint_indices"][row])
        labels[(bond_index, endpoint_index)] = ("residual", int(row))
        endpoint_pairs.append((bond_index, endpoint_index))
    residual_edges = new_edges = dangling = 0
    for bond_index in range(bond_count):
        left = labels.get((bond_index, 0))
        right = labels.get((bond_index, 1))
        residual_side = [item for item in (left, right) if item is not None and item[0] == "residual"]
        if not residual_side:
            continue
        if left is None or right is None or (left[0] == right[0] == "residual" and left[1] == right[1]):
            dangling += len(residual_side)
            continue
        residual_edges += 1
        if any(row in new_rows for _, row in residual_side):
            new_edges += 1
    return {
        "residual_edges": residual_edges,
        "new_residual_edges": new_edges,
        "dangling_residual_endpoints": dangling,
        "one_to_one_violations": len(endpoint_pairs) - len(set(endpoint_pairs)),
    }


def _partner_completion(
    baseline: CandidateSet,
    residual: CandidateSet,
    bond_endpoints: torch.Tensor,
    attachment: dict[str, torch.Tensor],
    baseline_ids: set[int],
    new_ids: set[int],
    new_rows: set[int],
    target_points: torch.Tensor,
) -> dict[str, int]:
    """Audit whether the opposite side of a residual bond port can add an atom."""
    rows = torch.where(attachment["matched"])[0].tolist()
    if not rows:
        return {"partner_candidates": 0, "partner_new_matches": 0, "new_row_partner_matches": 0}
    existing = torch.cat((baseline.points, residual.points), dim=0)
    partners = []
    owner_rows = []
    for row in rows:
        bond_index = int(attachment["bond_indices"][row])
        endpoint_index = int(attachment["endpoint_indices"][row])
        partner = bond_endpoints[bond_index, 1 - endpoint_index]
        if existing.numel() and bool(torch.linalg.vector_norm(existing - partner, dim=-1).min() <= MATCH_RADIUS_PX / 768.0):
            continue
        partners.append(partner)
        owner_rows.append(row)
    if not partners:
        return {"partner_candidates": 0, "partner_new_matches": 0, "new_row_partner_matches": 0}
    partner_points = torch.stack(partners)
    remaining = [i for i in range(target_points.size(0)) if i not in baseline_ids and i not in new_ids]
    if not remaining:
        return {"partner_candidates": len(partners), "partner_new_matches": 0, "new_row_partner_matches": 0}
    cost = torch.cdist(partner_points * 768.0, target_points[remaining] * 768.0)
    rows_i, cols_i = linear_sum_assignment(cost.detach().cpu().numpy())
    matched_rows = {
        int(row) for row, col in zip(rows_i, cols_i)
        if float(cost[row, col]) <= MATCH_RADIUS_PX
    }
    new_matches = len(matched_rows)
    new_owner_matches = sum(
        1 for row in matched_rows if owner_rows[row] in new_rows
    )
    return {"partner_candidates": len(partners), "partner_new_matches": new_matches, "new_row_partner_matches": new_owner_matches}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    device = torch.device("cuda:0")
    fixed = json.loads(COHORT.read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    manifest = source_checks()["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    train_rows, holdout_rows, split_meta = split_group_disjoint(rows, TRAIN_IMAGES)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )

    def load(row):
        sample = loader(manifest, row)
        batch = collator([sample])
        return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}, row

    train = [load(row) for row in train_rows]
    holdout = [load(row) for row in holdout_rows]
    branch = PixelStride2Rescue().to(device).train()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-3, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        tensors, row = train[(step - 1) % len(train)]
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
            base_logits = model.atom(features)["heatmap_logits"]
        logits = F.interpolate(base_logits, size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
        if LABEL_MODE == "all":
            positive = _positive(tensors["atom_anchors"], tensors["atom_mask"], 384, 384)
        else:
            with torch.no_grad():
                _, _, _, _, _, _, selected = structure_forward(
                    model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
                )
                mapping, _ = match_atom_targets(selected, tensors, config)
                baseline_ids = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
            text = tensors["atom_text_mask"][0].bool()
            valid = tensors["atom_mask"][0].bool()
            residual = valid & text
            for index in baseline_ids:
                if 0 <= index < residual.numel():
                    residual[index] = False
            positive = _positive(
                tensors["atom_anchors"], residual[None], 384, 384
            )
        loss, parts = _highres_objective(logits, positive)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()), "gradient_norm": float(norm), **parts})

    def evaluate(items):
        images = baseline_matched = fused_matched = residual_candidates = residual_positive = 0
        residual_bond_supported = residual_positive_bond_supported = 0
        residual_bond_attachable = residual_positive_bond_attachable = 0
        residual_new_text_matches = residual_new_implicit_matches = 0
        residual_edges = new_residual_edges = dangling_residual_endpoints = one_to_one_violations = 0
        partner_candidates = partner_new_matches = new_row_partner_matches = 0
        prefix_failures = 0
        per_image = []
        branch.eval()
        with torch.no_grad():
            for tensors, row in items:
                _, _, _, _, bonds, _, selected = structure_forward(
                    model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
                )
                mapping, _ = match_atom_targets(selected, tensors, config)
                baseline_ids = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
                target_points = tensors["atom_anchors"][0, : int(tensors["atom_mask"].sum()), :2]
                baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2]
                baseline_scores = selected["atom_scores"][0, selected["atom_mask"][0]].float()
                baseline = CandidateSet(baseline_points, baseline_scores, "baseline")
                base_logits = model.atom(model.vision.forward_stroke_resolution(tensors["pixel_values"]))["heatmap_logits"]
                logits = F.interpolate(base_logits, size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
                residual_raw = _rescue_candidates(logits)
                merged = merge_candidates(baseline, residual_raw, duplicate_radius=DEDUP_RADIUS_PX / 768.0)
                if not baseline_prefix_unchanged(baseline, merged):
                    prefix_failures += 1
                # Recompute residual rows after deterministic duplicate removal.
                residual = CandidateSet(merged.points[baseline.points.size(0):], merged.scores[baseline.points.size(0):], "residual")
                _, new_ids, new_rows = _locked_match(baseline, residual, target_points, baseline_ids)
                if new_ids:
                    text_mask = tensors["atom_text_mask"][0, list(new_ids)].bool()
                    residual_new_text_matches += int(text_mask.sum())
                    residual_new_implicit_matches += int((~text_mask).sum())
                bond_mask = bonds["bond_mask"][0].bool()
                bond_endpoints = bonds["bond_endpoints"][0, bond_mask]
                if residual.points.size(0) and bond_endpoints.size(0):
                    endpoint_distance = torch.cdist(
                        residual.points * 768.0,
                        bond_endpoints.reshape(-1, 2) * 768.0,
                    ).amin(dim=1)
                    supported = endpoint_distance <= BOND_SUPPORT_RADIUS_PX
                    residual_bond_supported += int(supported.sum())
                    if new_rows:
                        residual_positive_bond_supported += int(
                            supported[torch.as_tensor(sorted(new_rows), device=supported.device)].sum()
                        )
                if residual.points.size(0) and bond_endpoints.size(0):
                    occupied = torch.zeros(
                        (bond_endpoints.size(0), 2), dtype=torch.bool, device=device
                    )
                    baseline_endpoint_distance = torch.cdist(
                        bond_endpoints.reshape(-1, 2) * 768.0,
                        baseline.points * 768.0,
                    )
                    nearest = baseline_endpoint_distance.amin(dim=1)
                    occupied = (nearest <= BOND_SUPPORT_RADIUS_PX).reshape(-1, 2)
                    attachment = attach_residual_candidates(
                        residual.points,
                        bond_endpoints,
                        torch.ones(bond_endpoints.size(0), dtype=torch.bool, device=device),
                        radius_px=BOND_SUPPORT_RADIUS_PX,
                        occupied_endpoints=occupied,
                    )
                    residual_bond_attachable += int(attachment["matched"].sum())
                    if new_rows:
                        new_row_tensor = torch.as_tensor(sorted(new_rows), device=device)
                        residual_positive_bond_attachable += int(
                            attachment["matched"][new_row_tensor].sum()
                        )
                    edge_metrics = _edge_audit(
                        baseline, residual, bond_endpoints, attachment, new_rows,
                        radius_px=BOND_SUPPORT_RADIUS_PX,
                    )
                    residual_edges += edge_metrics["residual_edges"]
                    new_residual_edges += edge_metrics["new_residual_edges"]
                    dangling_residual_endpoints += edge_metrics["dangling_residual_endpoints"]
                    one_to_one_violations += edge_metrics["one_to_one_violations"]
                    partner_metrics = _partner_completion(
                        baseline, residual, bond_endpoints, attachment,
                        baseline_ids, new_ids, new_rows, target_points,
                    )
                    partner_candidates += partner_metrics["partner_candidates"]
                    partner_new_matches += partner_metrics["partner_new_matches"]
                    new_row_partner_matches += partner_metrics["new_row_partner_matches"]
                images += 1
                baseline_matched += len(baseline_ids)
                fused_matched += len(baseline_ids) + len(new_ids)
                residual_candidates += residual.points.size(0)
                residual_positive += len(new_ids)
                per_image.append({"sample_key": row["sample_key"], "baseline_matched": len(baseline_ids), "residual_candidates": int(residual.points.size(0)), "residual_new_matches": len(new_ids)})
        return {"images": images, "baseline_matched": baseline_matched, "fused_locked_matched": fused_matched, "residual_candidates": residual_candidates, "residual_new_matches": residual_positive, "residual_new_text_matches": residual_new_text_matches, "residual_new_implicit_matches": residual_new_implicit_matches, "residual_bond_supported": residual_bond_supported, "residual_new_bond_supported": residual_positive_bond_supported, "residual_bond_attachable": residual_bond_attachable, "residual_new_bond_attachable": residual_positive_bond_attachable, "residual_edges": residual_edges, "new_residual_edges": new_residual_edges, "dangling_residual_endpoints": dangling_residual_endpoints, "one_to_one_violations": one_to_one_violations, "partner_candidates": partner_candidates, "partner_new_matches": partner_new_matches, "new_row_partner_matches": new_row_partner_matches, "prefix_failures": prefix_failures, "per_image": per_image}

    train_metrics = evaluate(train)
    holdout_metrics = evaluate(holdout)
    result = {
        "schema_version": "chemvector-highres-independent-graph-merge-0.1",
        "complete": True,
        "training_performed": True,
        "in_memory_only": True,
        "optimizer_steps": STEPS,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "split": split_meta,
        "contract": {"baseline_selector_frozen": True, "residual_set_head_bypassed": True, "assignment": "baseline-first-locked", "dedup_radius_px": DEDUP_RADIUS_PX, "match_radius_px": MATCH_RADIUS_PX, "bond_support_radius_px": BOND_SUPPORT_RADIUS_PX},
        "label_mode": LABEL_MODE,
        "train": train_metrics,
        "holdout": holdout_metrics,
        "events": events,
        "gate": {"prefix_immutable": train_metrics["prefix_failures"] == 0 and holdout_metrics["prefix_failures"] == 0, "candidate_route_only": True, "next_step_requires_bond_graph_audit": True},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "split": split_meta, "train": {k: v for k, v in train_metrics.items() if k != "per_image"}, "holdout": {k: v for k, v in holdout_metrics.items() if k != "per_image"}}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
