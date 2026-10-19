"""Forward-only audit of residual endpoint candidates constrained by bond ports.

The frozen production selector is evaluated first.  A tiny high-resolution
branch is trained in memory only to make residual endpoint proposals.  Unlike
the previous rescue-scorer experiments, a residual proposal is considered
graph-compatible only when it can claim one *unoccupied* endpoint of a frozen
bond.  The old set head is never called with the residual rows.

This is a reachability/audit step, not a deployable decoder or a scorer
training run.  It intentionally reports deterministic feature rankings and
does not fit a threshold on the holdout split.
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
from chemvector.training.graph.relations.train_direct_atom_pair_bond import (
    match_atom_targets,
)
from tools.current_initial_cohort import split_group_disjoint
from tools.shadow_atom_highres_pixel_branch import PixelStride2Rescue, _highres_objective


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_BOND_PORT_AUDIT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-bond-port-candidate-reachability-20261008"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_BOND_PORT_TRAIN_IMAGES", "120"))
HOLDOUT_IMAGES = int(os.environ.get("CHEMVECTOR_BOND_PORT_HOLDOUT_IMAGES", "0"))
STEPS = int(os.environ.get("CHEMVECTOR_BOND_PORT_STEPS", "200"))
HIGHRES = 384
IMAGE_SIZE = 768.0
RESCUE_POOL = 64
DEDUP_RADIUS_PX = 6.0
MATCH_RADIUS_PX = 6.0
PORT_RADIUS_PX = 12.0
EVAL_LIMIT = int(os.environ.get("CHEMVECTOR_BOND_PORT_EVAL_LIMIT", "0"))


def _positive(anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(
        (anchors.size(0), 1, HIGHRES, HIGHRES), dtype=torch.bool, device=anchors.device
    )
    x = (anchors[..., 0] * HIGHRES).clamp(0, HIGHRES - 1e-4).long()
    y = (anchors[..., 1] * HIGHRES).clamp(0, HIGHRES - 1e-4).long()
    batch = torch.arange(anchors.size(0), device=anchors.device)[:, None].expand_as(mask)
    target[batch[mask], 0, y[mask], x[mask]] = True
    return target


def _rescue_candidates(logits: torch.Tensor) -> CandidateSet:
    score = logits.sigmoid()[0, 0]
    peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
    py, px = torch.where(peaks)
    values = score[py, px]
    order = torch.argsort(values, descending=True, stable=True)[:RESCUE_POOL]
    points = torch.stack(
        ((px[order].float() + 0.5) / HIGHRES,
         (py[order].float() + 0.5) / HIGHRES), dim=-1
    )
    return CandidateSet(points, values[order], "residual")


def _locked_target_ids(
    baseline_ids: set[int], residual: CandidateSet, target_points: torch.Tensor,
) -> torch.Tensor:
    """Return one-to-one IDs for remaining targets (``-1`` means no match)."""
    labels = torch.full((residual.points.size(0),), -1, dtype=torch.long, device=residual.points.device)
    remaining = [i for i in range(target_points.size(0)) if i not in baseline_ids]
    if not remaining or not residual.points.numel():
        return labels
    cost = torch.cdist(residual.points * IMAGE_SIZE, target_points[remaining] * IMAGE_SIZE)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    for row, col in zip(rows, cols):
        if float(cost[row, col]) <= MATCH_RADIUS_PX:
            labels[int(row)] = int(remaining[col])
    return labels


def _match_bond_ports(
    predicted: torch.Tensor,
    predicted_mask: torch.Tensor,
    target_ports: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[dict[int, tuple[int, bool]], torch.Tensor]:
    """Match frozen bond proposals to target ports for an oracle audit only.

    The matching is used solely to label the audit.  It never influences
    candidate generation or attachment.  The returned map records whether
    predicted endpoint side ``s`` maps to target side ``s`` or ``1-s``.
    """
    result: dict[int, tuple[int, bool]] = {}
    if not bool(predicted_mask.any()) or not bool(target_mask.any()):
        return result, torch.empty((0, 2), dtype=torch.long, device=predicted.device)
    pred_ids = torch.where(predicted_mask)[0]
    target_ids = torch.where(target_mask)[0]
    pred = predicted[pred_ids].float()
    target = target_ports[target_ids].float()
    forward = torch.linalg.vector_norm((pred[:, None] - target[None]) * IMAGE_SIZE, dim=-1).amax(-1)
    reverse = torch.linalg.vector_norm((pred[:, None] - target[None].flip(2)) * IMAGE_SIZE, dim=-1).amax(-1)
    cost = torch.minimum(forward, reverse)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    for row, col in zip(rows, cols):
        if float(cost[row, col]) <= PORT_RADIUS_PX:
            result[int(pred_ids[row])] = (int(target_ids[col]), bool(reverse[row, col] < forward[row, col]))
    return result, target_ids


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    if scores.numel() == 0 or not bool(labels.any()):
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order].float()
    cumulative = sorted_labels.cumsum(0)
    ranks = torch.arange(1, len(sorted_labels) + 1, device=scores.device, dtype=torch.float32)
    return float((cumulative / ranks * sorted_labels).sum() / sorted_labels.sum())


def _rank_features(scores: torch.Tensor, port_distance: torch.Tensor) -> dict[str, torch.Tensor]:
    scores = scores.reshape(-1)
    port_distance = port_distance.reshape(-1)
    finite = torch.isfinite(port_distance)
    port = torch.exp(-port_distance.clamp_min(0.0) / PORT_RADIUS_PX)
    # These rankings are fixed before looking at holdout labels.  The first
    # asks whether the branch score survives the port gate; the second adds a
    # monotone distance prior and is not a learned scorer.
    return {
        "branch_score": scores,
        "port_proximity": port.masked_fill(~finite, 0.0),
        "score_plus_port": scores + 0.5 * port.masked_fill(~finite, 0.0),
    }


def _summarize(items: list[dict], *, include_only_attached: bool = False) -> dict:
    total = positives = attached = attached_positives = 0
    score_values: dict[str, list[torch.Tensor]] = {k: [] for k in ("branch_score", "port_proximity", "score_plus_port")}
    label_values: list[torch.Tensor] = []
    for item in items:
        labels = item["labels"].reshape(-1)
        attached_mask = item["attached"].reshape(-1).bool()
        keep = attached_mask.bool() if include_only_attached else torch.ones_like(labels, dtype=torch.bool)
        total += int(keep.sum())
        positives += int(((labels >= 0) & keep).sum())
        attached += int(attached_mask.sum())
        attached_positives += int(((labels >= 0) & attached_mask).sum())
        ranks = _rank_features(item["scores"].reshape(-1), item["port_distance"].reshape(-1))
        for name, values in ranks.items():
            if values.numel() != keep.numel():
                raise ValueError(
                    f"candidate feature length mismatch sample={item['sample_key']} "
                    f"name={name} scores={item['scores'].shape} "
                    f"port={item['port_distance'].shape} labels={labels.shape} "
                    f"attached={attached_mask.shape} values={values.shape}"
                )
            score_values[name].append(values[keep])
        label_values.append((labels >= 0)[keep])
    result = {
        "candidates": total,
        "positives": positives,
        "positive_rate": positives / max(total, 1),
        "attached_candidates": attached,
        "attached_positives": attached_positives,
        "attached_precision": attached_positives / max(attached, 1),
    }
    if not label_values:
        return result
    labels = torch.cat(label_values)
    for name, values in score_values.items():
        scores = torch.cat(values) if values else labels.new_empty(0, dtype=torch.float32)
        result[f"{name}_auprc"] = _average_precision(scores, labels)
        for k in (8, 16):
            selected = []
            start = 0
            for item in items:
                ranks = _rank_features(item["scores"], item["port_distance"])[name]
                keep = item["attached"].bool() if include_only_attached else torch.ones_like(item["labels"], dtype=torch.bool)
                indices = torch.where(keep)[0]
                if len(indices):
                    take = indices[torch.argsort(ranks[indices], descending=True, stable=True)[: min(k, len(indices))]]
                    selected.append(item["labels"][take] >= 0)
            matched = int(torch.cat(selected).sum()) if selected else 0
            count = int(sum(len(x) for x in selected))
            result[f"{name}_top{k}_matched"] = matched
            result[f"{name}_top{k}_selected"] = count
            result[f"{name}_top{k}_precision"] = matched / max(count, 1)
    return result


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
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 fixed cohort rows, found {len(rows)}")
    train_rows, holdout_rows, split_meta = split_group_disjoint(rows, TRAIN_IMAGES)
    if HOLDOUT_IMAGES:
        if HOLDOUT_IMAGES < 1 or HOLDOUT_IMAGES > len(holdout_rows):
            raise ValueError("CHEMVECTOR_BOND_PORT_HOLDOUT_IMAGES exceeds split")
        holdout_rows = holdout_rows[:HOLDOUT_IMAGES]
        split_meta = {**split_meta, "evaluated_holdout_images": HOLDOUT_IMAGES}
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
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
        logits = F.interpolate(base_logits, size=(HIGHRES, HIGHRES), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
        loss, parts = _highres_objective(logits, _positive(tensors["atom_anchors"], tensors["atom_mask"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()), "gradient_norm": float(norm), **parts})

    def collect(items):
        if EVAL_LIMIT:
            items = items[:EVAL_LIMIT]
        records: list[dict] = []
        prefix_failures = 0
        target_unmatched = target_port_reachable = 0
        attached_true = 0
        with torch.no_grad():
            branch.eval()
            for tensors, row in items:
                _, atom_output, _, raw, bonds, _, selected = structure_forward(
                    model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
                )
                baseline_ids = {int(v) for v in match_atom_targets(selected, tensors, config)[0][0].tolist() if int(v) >= 0}
                baseline = CandidateSet(
                    selected["atom_geometry"][0, selected["atom_mask"][0], :2],
                    selected["atom_scores"][0, selected["atom_mask"][0]].float(),
                    "baseline",
                )
                base_logits = model.atom(model.vision.forward_stroke_resolution(tensors["pixel_values"]))["heatmap_logits"]
                logits = F.interpolate(base_logits, size=(HIGHRES, HIGHRES), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
                residual_raw = _rescue_candidates(logits)
                merged = merge_candidates(baseline, residual_raw, duplicate_radius=DEDUP_RADIUS_PX / IMAGE_SIZE)
                prefix_failures += int(not baseline_prefix_unchanged(baseline, merged))
                residual = CandidateSet(
                    merged.points[baseline.points.size(0):],
                    merged.scores[baseline.points.size(0):],
                    "residual",
                )
                bond_mask = bonds["bond_mask"][0].bool()
                bond_endpoints = bonds["bond_endpoints"][0]
                active_bond_endpoints = bond_endpoints[bond_mask]
                active_bond_mask = torch.ones(
                    active_bond_endpoints.size(0), dtype=torch.bool, device=device
                )
                valid_bond_endpoints = active_bond_mask[:, None].expand(-1, 2)
                if active_bond_endpoints.numel():
                    nearest_baseline = torch.cdist(
                        active_bond_endpoints.reshape(-1, 2) * IMAGE_SIZE,
                        baseline.points * IMAGE_SIZE,
                    ).amin(dim=1) if len(baseline.points) else torch.full((active_bond_endpoints.reshape(-1, 2).size(0),), float("inf"), device=device)
                    occupied = (nearest_baseline <= PORT_RADIUS_PX).reshape(-1, 2)
                else:
                    occupied = torch.zeros((0, 2), dtype=torch.bool, device=device)
                attachment = attach_residual_candidates(
                    residual.points,
                    active_bond_endpoints,
                    active_bond_mask,
                    radius_px=PORT_RADIUS_PX,
                    occupied_endpoints=occupied,
                )
                target_points = tensors["atom_anchors"][0, : int(tensors["atom_mask"].sum()), :2]
                labels = _locked_target_ids(baseline_ids, residual, target_points).detach().cpu()
                unmatched = [i for i in range(len(target_points)) if i not in baseline_ids]
                target_unmatched += len(unmatched)
                if unmatched and len(active_bond_endpoints):
                    free_points = active_bond_endpoints[~occupied & valid_bond_endpoints]
                    if len(free_points):
                        distances = torch.cdist(target_points[unmatched] * IMAGE_SIZE, free_points * IMAGE_SIZE).amin(dim=1)
                        target_port_reachable += int((distances <= PORT_RADIUS_PX).sum())
                # Oracle port compatibility is label-only: map the frozen
                # predicted port to a GT bond and check that the attached
                # endpoint is that bond's GT atom.  This does not affect
                # candidate generation or assignment.
                pred_to_target, _ = _match_bond_ports(
                    active_bond_endpoints,
                    active_bond_mask,
                    tensors["bond_ports"][0], tensors["bond_mask"][0].bool(),
                )
                attachment_matched = attachment["matched"].detach().cpu()
                attachment_bonds = attachment["bond_indices"].detach().cpu()
                attachment_endpoints = attachment["endpoint_indices"].detach().cpu()
                attachment_distances = attachment["distances_px"].detach().cpu()
                oracle_attached = torch.zeros_like(attachment_matched)
                for candidate_index in torch.where(attachment_matched)[0].tolist():
                    bond_index = int(attachment_bonds[candidate_index])
                    endpoint_index = int(attachment_endpoints[candidate_index])
                    mapped = pred_to_target.get(bond_index)
                    target_id = int(labels[candidate_index])
                    if mapped is None or target_id < 0:
                        continue
                    target_bond, reversed_order = mapped
                    side = 1 - endpoint_index if reversed_order else endpoint_index
                    refs = tensors["bond_atom_refs"][0].detach().cpu()
                    if target_bond < 0 or target_bond >= refs.size(0):
                        continue
                    target_ref = int(refs[target_bond, side])
                    oracle_attached[candidate_index] = target_ref == target_id
                attached_true += int(oracle_attached.sum())
                records.append({
                    "sample_key": row["sample_key"],
                    # Move audit-only records off CUDA before the next sample;
                    # this also prevents a deferred device assertion from
                    # being reported at aggregate ranking rather than at the
                    # operation that produced the malformed record.
                    "scores": residual.scores.detach().cpu().reshape(-1),
                    "labels": labels.detach().cpu().reshape(-1),
                    "attached": attachment_matched.reshape(-1),
                    "oracle_attached": oracle_attached.detach().cpu().reshape(-1),
                    "port_distance": attachment_distances.reshape(-1),
                })
        summary = _summarize(records)
        summary["attached_only"] = _summarize(records, include_only_attached=True)
        summary["oracle_attached_only"] = _summarize(
            [{**item, "attached": item["oracle_attached"]} for item in records],
            include_only_attached=True,
        )
        summary.update({
            "images": len(items),
            "prefix_failures": prefix_failures,
            "unmatched_targets": target_unmatched,
            "unmatched_targets_near_free_port": target_port_reachable,
            "target_port_reachability": target_port_reachable / max(target_unmatched, 1),
                "attached_true": attached_true,
        })
        return summary, records

    train_summary, train_records = collect(train)
    holdout_summary, holdout_records = collect(holdout)
    result = {
        "schema_version": "chemvector-bond-port-candidate-reachability-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": True,
        "in_memory_only": True,
        "optimizer_steps": STEPS,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "split": split_meta,
        "contract": {
            "baseline_selector_frozen": True,
            "residual_set_head_bypassed": True,
            "baseline_first_locked_assignment": True,
            "one_to_one_unoccupied_ports": True,
            "port_radius_px": PORT_RADIUS_PX,
            "candidate_dedup_radius_px": DEDUP_RADIUS_PX,
            "candidate_match_radius_px": MATCH_RADIUS_PX,
        },
        "train": train_summary,
        "holdout": holdout_summary,
        "gate": {
            "prefix_immutable": train_summary["prefix_failures"] == 0 and holdout_summary["prefix_failures"] == 0,
            "candidate_route_only": True,
            "scorer_training_admissible": False,
            "reason": "reachability audit only; threshold/scorer must be preregistered after review",
        },
        "events": events,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "split": split_meta, "train": train_summary, "holdout": holdout_summary}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
