"""Read-only K=16 high-resolution sidecar retention shadow.

The production primary candidate set (128) stays byte-for-byte intact.  For
audit only, 16 novel 384px rescue candidates are appended (144 total) and
passed through the existing set head.  No checkpoint is written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import replace

from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.graph.joint.predicted_incidence import decode_predicted_atom_proposals
from chemvector.components.atoms.objects.spatial_atom_object import atom_local_peak_margin_loss
from chemvector.components.vision.vertex import focal_heatmap_loss
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.shadow_atom_highres_pixel_branch import (
    PixelStride2Rescue,
    _baseline,
    _highres_ids,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_HIGHRES_SELECTOR_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-highres-selector-retention-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TRAIN_IMAGES = 120
STEPS = 200
RESCUE_K = 16
# A rescue point must be farther than twice the atom matching radius from
# every primary proposal.  A 6 px-only exclusion still lets a rescue point
# steal the Hungarian assignment of an old selected target (the two points
# can be 6--12 px apart while both sit within 6 px of the target).  The
# 12 px guard is therefore the minimum contract-safe geometric separation;
# this remains an audit-only policy and does not alter production decoding.
RESCUE_MIN_DIST_PX = 12.0
OBJECTIVE = os.environ.get("CHEMVECTOR_HIGHRES_SELECTOR_OBJECTIVE", "local_margin")


def _branch_loss(logits, positive):
    if OBJECTIVE == "local_margin":
        return atom_local_peak_margin_loss(logits, positive, margin=0.75, kernel_size=3)
    if OBJECTIVE == "focal":
        return focal_heatmap_loss(logits, positive.float())
    if OBJECTIVE == "focal_hard":
        base = focal_heatmap_loss(logits, positive.float())
        local_max = logits == F.max_pool2d(logits, 3, 1, 1)
        protected = F.max_pool2d(positive.float(), 3, 1, 1).bool()
        values = logits[local_max & ~protected]
        if not len(values):
            return base
        values = values.topk(min(64, len(values))).values
        return base + 0.25 * F.softplus(values - logits[positive].median() + 0.5).mean()
    raise ValueError(f"Unknown highres selector objective: {OBJECTIVE}")


def _fused_atoms(base_output, proposal_config, side_points_px, side_scores, protected_points=None):
    primary = decode_predicted_atom_proposals(
        base_output, replace(proposal_config, maximum_atoms=128)
    )
    device = primary["atom_features"].device
    _, _, height, width = base_output["heatmap_logits"].shape
    # Keep rescue candidates geometrically novel; primary candidates are never
    # removed or re-ranked.
    # Only immutable baseline-selected points are protected here.  Filtering
    # against all 128 raw proposals is needlessly conservative: an unselected
    # raw proposal cannot steal a baseline assignment.  The caller may pass
    # the selected geometry; fallback keeps the old all-primary behavior for
    # standalone callers.
    primary_points = protected_points if protected_points is not None else primary["atom_geometry"][0, :, :2]
    rescue = side_points_px.to(device) / 768.0
    rescue_scores = side_scores.to(device)
    if len(rescue):
        distance = torch.cdist(rescue, primary_points * 1.0)
        keep = distance.amin(dim=1) > (RESCUE_MIN_DIST_PX / 768.0)
        # Keep score/geometry rows paired.  Previously only ``rescue`` was
        # filtered, then ``side_scores[:len(rescue)]`` was used below; when a
        # near-baseline peak was removed, scores silently shifted to another
        # geometry and the set-head retention audit no longer described the
        # candidate that was actually appended.
        rescue = rescue[keep]
        rescue_scores = rescue_scores[keep]
    if len(rescue) > RESCUE_K:
        rescue = rescue[:RESCUE_K]
    cells_x = (rescue[:, 0] * width).clamp(0, width - 1e-4).long() if len(rescue) else torch.empty(0, dtype=torch.long, device=device)
    cells_y = (rescue[:, 1] * height).clamp(0, height - 1e-4).long() if len(rescue) else torch.empty(0, dtype=torch.long, device=device)
    if len(rescue):
        batch = torch.zeros(len(rescue), dtype=torch.long, device=device)
        features = base_output["object_features"].permute(0, 2, 3, 1)[batch, cells_y, cells_x][None]
        roles = base_output["role_logits"].permute(0, 2, 3, 1)[batch, cells_y, cells_x].softmax(-1)[None]
        centers = base_output["text_centers"].permute(0, 2, 3, 1)[batch, cells_y, cells_x][None]
        extents = base_output["text_extents"].permute(0, 2, 3, 1)[batch, cells_y, cells_x][None]
        geometry = torch.cat((rescue[None], roles, centers, extents), dim=-1)
        scores = rescue_scores[: len(rescue)][None]
        indices = (cells_y * width + cells_x)[None]
        mask = torch.ones((1, len(rescue)), dtype=torch.bool, device=device)
    else:
        features = primary["atom_features"][:, :0]
        geometry = primary["atom_geometry"][:, :0]
        scores = primary["atom_scores"][:, :0]
        indices = primary["atom_indices"][:, :0]
        mask = primary["atom_mask"][:, :0]
    return {
        "atom_features": torch.cat((primary["atom_features"], features), dim=1),
        "atom_geometry": torch.cat((primary["atom_geometry"], geometry), dim=1),
        "atom_scores": torch.cat((primary["atom_scores"], scores), dim=1),
        "atom_indices": torch.cat((primary["atom_indices"], indices), dim=1),
        "atom_mask": torch.cat((primary["atom_mask"], mask), dim=1),
    }


def _selected_ids(selected, tensors, proposal_config):
    mapping, _ = match_atom_targets(selected, tensors, proposal_config)
    return {int(value) for value in mapping[0].tolist() if int(value) >= 0}


def _locked_assignment(policy, baseline_selected, tensors, proposal_config):
    """Evaluate sidecar matches without allowing it to steal baseline targets.

    This is an evaluation-only assignment.  The baseline selected prefix is
    matched first with the production Hungarian rule and then removed from
    the target pool.  Sidecar candidates are matched only to the remaining
    targets.  Candidate selection never reads this result.
    """
    baseline_mapping, _ = match_atom_targets(baseline_selected, tensors, proposal_config)
    baseline_target_ids = {int(x) for x in baseline_mapping[0].tolist() if int(x) >= 0}
    base_n = int(baseline_selected["atom_mask"].sum())
    side_n = int(policy["atom_mask"].sum()) - base_n
    if side_n <= 0:
        return baseline_target_ids, set()
    target_count = int(tensors["atom_mask"].sum())
    remaining = [i for i in range(target_count) if i not in baseline_target_ids]
    if not remaining:
        return baseline_target_ids, set()
    side_points = policy["atom_geometry"][0, base_n:base_n + side_n, :2]
    expected = tensors["atom_anchors"][0, remaining, :2]
    cost = torch.cdist(side_points.float(), expected.float()) * proposal_config.image_size
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    side_target_ids = {
        int(remaining[column]) for row, column in zip(rows, cols)
        if float(cost[row, column]) <= proposal_config.atom_match_radius_px
    }
    return baseline_target_ids, side_target_ids


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    device = torch.device("cuda:0")
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    model, proposal_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=proposal_config.maximum_atoms, bond_queries=proposal_config.maximum_bonds),
        purpose="training",
    )
    cache = [_baseline(model, proposal_config, loader, manifest, row, collator, device) for row in rows]
    gap_rows = [item for item in cache if item[3]["raw_matched"] < item[3]["targets"]]
    train_items = gap_rows[:TRAIN_IMAGES]
    train_keys = {item[3]["sample_key"] for item in train_items}
    if len(train_items) != TRAIN_IMAGES:
        raise ValueError("Insufficient training gap rows")
    branch = PixelStride2Rescue().to(device).train()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-3, weight_decay=0.0)
    for step in range(STEPS):
        _, _, tensors, _ = train_items[step % len(train_items)]
        pixels = tensors["pixel_values"]
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(pixels)
            base = model.atom(features)["heatmap_logits"]
        logits = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False) + branch(pixels)
        # Use the same fixed local-margin objective as the capacity trial.
        anchors, mask = tensors["atom_anchors"], tensors["atom_mask"]
        positive = torch.zeros((1, 1, 384, 384), dtype=torch.bool, device=device)
        x = (anchors[..., 0] * 384).clamp(0, 383.999).long()
        y = (anchors[..., 1] * 384).clamp(0, 383.999).long()
        batch = torch.zeros_like(x)
        positive[batch[mask], 0, y[mask], x[mask]] = True
        loss = _branch_loss(logits, positive)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
        optimizer.step()
    results = []
    with torch.no_grad():
        for sample, batch, tensors, old in cache:
            features, base_output, bond_output, atoms, bonds, base_set, base_selected = structure_forward(
                model, tensors["pixel_values"], proposal_config, payload["provenance"]["atom_threshold"]
            )
            base_ids = _selected_ids(base_selected, tensors, proposal_config)
            logits = F.interpolate(base_output["heatmap_logits"], size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
            score = logits.sigmoid()[0, 0]
            peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
            py, px = torch.where(peaks)
            values = score[py, px]
            order = torch.argsort(values, descending=True)[:RESCUE_K]
            side_points = torch.stack(((px[order].float() + 0.5) * 768.0 / 384.0, (py[order].float() + 0.5) * 768.0 / 384.0), dim=1)
            side_scores = values[order]
            protected = base_selected["atom_geometry"][0, base_selected["atom_mask"][0], :2]
            fused = _fused_atoms(base_output, proposal_config, side_points, side_scores, protected_points=protected)
            fused_set = model.set_head(
                fused["atom_features"], fused["atom_geometry"], fused["atom_scores"], fused["atom_mask"],
                bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"],
            )
            fused_selected = {**fused, "atom_mask": fused["atom_mask"] & (fused_set["atom_presence_logits"] >= payload["provenance"]["atom_threshold"])}
            fused_ids = _selected_ids(fused_selected, tensors, proposal_config)
            baseline_mapping, _ = match_atom_targets(base_selected, tensors, proposal_config)
            occupied = {int(value) for value in baseline_mapping[0].tolist() if int(value) >= 0}
            side_geometry = fused["atom_geometry"][:, 128:, :2][0]
            unmatched_targets = sorted(set(range(old["targets"])) - occupied)
            safe_new = set()
            safe_fp = 0
            if len(side_geometry) and unmatched_targets:
                safe_cost = torch.cdist(side_geometry, tensors["atom_anchors"][0, unmatched_targets]) * 768.0
                safe_rows, safe_cols = linear_sum_assignment(safe_cost.detach().cpu().numpy())
                for side_row, target_col in zip(safe_rows, safe_cols):
                    if float(safe_cost[side_row, target_col]) <= 6.0:
                        safe_new.add(unmatched_targets[int(target_col)])
                    else:
                        safe_fp += 1
            safe_fp += max(0, len(side_geometry) - len(safe_new) - safe_fp)
            # Independent fill policy: baseline selected candidates are copied
            # unchanged; only unused capacity may be filled by sidecar peaks.
            baseline_mask = base_selected["atom_mask"]
            baseline_count = int(baseline_mask.sum())
            capacity = max(0, 128 - baseline_count)
            side_count = min(RESCUE_K, capacity, int(fused["atom_mask"][:, 128:].sum()))
            policy = {}
            for key in ("atom_features", "atom_geometry", "atom_scores", "atom_indices"):
                primary_values = base_selected[key][baseline_mask].reshape(1, -1, base_selected[key].shape[-1]) if key in ("atom_features", "atom_geometry") else base_selected[key][baseline_mask].reshape(1, -1)
                side_values = fused[key][:, 128:128 + side_count]
                policy[key] = torch.cat((primary_values, side_values), dim=1)
            policy["atom_mask"] = torch.ones((1, baseline_count + side_count), dtype=torch.bool, device=tensors["pixel_values"].device)
            policy_ids = _selected_ids(policy, tensors, proposal_config)
            locked_base_ids, locked_side_ids = _locked_assignment(policy, base_selected, tensors, proposal_config)
            results.append({"sample_key": old["sample_key"], "profile_id": old["profile_id"],
                            "baseline_selected_ids": sorted(base_ids), "fused_selected_ids": sorted(fused_ids),
                            "baseline_selected_matched": len(base_ids), "fused_selected_matched": len(fused_ids),
                            "lost_selected_ids": sorted(base_ids - fused_ids), "new_selected_ids": sorted(fused_ids - base_ids),
                            "policy_selected_ids": sorted(policy_ids),
                            "policy_lost_selected_ids": sorted(base_ids - policy_ids),
                            "policy_new_selected_ids": sorted(policy_ids - base_ids),
                            "policy_selected_matched": len(policy_ids),
                            "locked_baseline_matched": len(locked_base_ids),
                            "locked_sidecar_matched": len(locked_side_ids),
                            "locked_total_matched": len(locked_base_ids) + len(locked_side_ids),
                            "policy_sidecar_count": side_count,
                            "assignment_safe_baseline_ids": sorted(occupied),
                            "assignment_safe_new_ids": sorted(safe_new),
                            "assignment_safe_sidecar_count": int(len(side_geometry)),
                            "assignment_safe_false_positive_count": int(safe_fp),
                            "holdout": old["sample_key"] not in train_keys})
    holdout = [r for r in results if r["holdout"]]
    seen = [r for r in results if not r["holdout"]]
    summary = lambda xs: {"images": len(xs), "baseline_matched": sum(x["baseline_selected_matched"] for x in xs),
                          "fused_matched": sum(x["fused_selected_matched"] for x in xs),
                          "lost_selected_ids": sum(len(x["lost_selected_ids"]) for x in xs),
                          "new_selected_ids": sum(len(x["new_selected_ids"]) for x in xs)}
    policy_summary = lambda xs: {"images": len(xs), "baseline_matched": sum(x["baseline_selected_matched"] for x in xs),
                                 "policy_matched": sum(x["policy_selected_matched"] for x in xs),
                                 "lost_selected_ids": sum(len(x["policy_lost_selected_ids"]) for x in xs),
                                 "new_selected_ids": sum(len(x["policy_new_selected_ids"]) for x in xs),
                                 "sidecar_candidates": sum(x["policy_sidecar_count"] for x in xs)}
    locked_summary = lambda xs: {"images": len(xs),
                                 "baseline_matched": sum(x["baseline_selected_matched"] for x in xs),
                                 "locked_baseline_matched": sum(x["locked_baseline_matched"] for x in xs),
                                 "locked_sidecar_matched": sum(x["locked_sidecar_matched"] for x in xs),
                                 "locked_total_matched": sum(x["locked_total_matched"] for x in xs),
                                 "sidecar_candidates": sum(x["policy_sidecar_count"] for x in xs)}
    safe_summary = lambda xs: {"images": len(xs),
                               "baseline_matched": sum(len(x["assignment_safe_baseline_ids"]) for x in xs),
                               "new_matched": sum(len(x["assignment_safe_new_ids"]) for x in xs),
                               "false_positives": sum(x["assignment_safe_false_positive_count"] for x in xs),
                               "sidecar_candidates": sum(x["assignment_safe_sidecar_count"] for x in xs)}
    result = {"schema_version": "chemvector-highres-selector-retention-shadow-0.1", "complete": True,
              "training_performed": False, "virtual_steps": STEPS, "train_images": TRAIN_IMAGES,
              "sealed_test_read": False, "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
              "candidate_contract": "primary128_plus_rescue16_audit_only", "objective": OBJECTIVE, "rescue_min_dist_px": RESCUE_MIN_DIST_PX,
              "seen": summary(seen), "holdout": summary(holdout),
              "policy_seen": policy_summary(seen), "policy_holdout": policy_summary(holdout),
              "locked_seen": locked_summary(seen), "locked_holdout": locked_summary(holdout),
              "assignment_safe_seen": safe_summary(seen), "assignment_safe_holdout": safe_summary(holdout),
              "results": results,
              "gate": {"holdout_old_selected_retained": summary(holdout)["lost_selected_ids"] == 0,
                       "holdout_selected_not_worse": summary(holdout)["fused_matched"] >= summary(holdout)["baseline_matched"],
                       "independent_fill_old_selected_retained": policy_summary(holdout)["lost_selected_ids"] == 0,
                       "independent_fill_not_worse": policy_summary(holdout)["policy_matched"] >= policy_summary(holdout)["baseline_matched"],
                       "assignment_safe_new_at_least_5": safe_summary(holdout)["new_matched"] >= 5,
                       "pass": summary(holdout)["lost_selected_ids"] == 0 and summary(holdout)["fused_matched"] >= summary(holdout)["baseline_matched"]}}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"], "holdout": result["holdout"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
