"""Forward-only audit of the high-resolution residual candidate contract.

This audit makes the ownership boundary explicit.  The frozen baseline is
assigned first and its target IDs are immutable.  High-resolution peaks are
then counted before/after the baseline-distance guard, and only the remaining
targets are available to the sidecar matcher.  It intentionally does not
train a branch, write a checkpoint, or read sealed identities.

The trace is useful because the field shadow reports a global (joint)
Hungarian union while the readiness/scorer path reports a two-stage residual
assignment.  Those numbers are not interchangeable.  Each row records both
the raw and selected baseline views so that the contract can be audited
without guessing which proposal prefix was used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.graph.joint.predicted_incidence import PredictedProposalConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
FIXED_REPORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-highres-candidate-parity-20261006"
BASELINE_GUARD_PX = 12.0
MATCH_RADIUS_PX = 6.0
PEAK_POOL_K = 64
PEAK_K = 16


def _ids(atoms: dict, tensors: dict, config: PredictedProposalConfig) -> set[int]:
    mapping, _ = match_atom_targets(atoms, tensors, config)
    return {int(v) for v in mapping[0].tolist() if int(v) >= 0}


def _peaks(logits: torch.Tensor, *, pool_k: int = PEAK_POOL_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Return points in 768px image units and their scores, before guards."""
    score = logits.sigmoid()[0, 0]
    local = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
    py, px = torch.where(local)
    values = score[py, px]
    order = torch.argsort(values, descending=True)[:pool_k]
    # The branch is 384x384, while source images and target anchors are 768px.
    points = torch.stack(
        ((px[order].float() + 0.5) * 768.0 / score.shape[1],
         (py[order].float() + 0.5) * 768.0 / score.shape[0]),
        dim=1,
    )
    return points, values[order]


def _locked_sidecar(
    points_px: torch.Tensor,
    anchors_px: torch.Tensor,
    baseline_ids: set[int],
    target_count: int,
) -> tuple[set[int], int, list[int]]:
    """Two-stage assignment: baseline IDs are immutable, sidecar owns rest."""
    remaining = [i for i in range(target_count) if i not in baseline_ids]
    if not len(points_px) or not remaining:
        return set(), len(points_px), []
    cost = torch.cdist(points_px.float(), anchors_px[remaining].float())
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    matched: set[int] = set()
    distances: list[float] = []
    for row, col in zip(rows, cols):
        value = float(cost[row, col])
        distances.append(value)
        if value <= MATCH_RADIUS_PX:
            matched.add(int(remaining[col]))
    return matched, len(points_px), [int(round(v * 100)) for v in distances]


def _trace_row(
    *,
    row: dict,
    sample,
    tensors: dict,
    raw: dict,
    selected: dict,
    logits: torch.Tensor,
    config: PredictedProposalConfig,
) -> dict:
    target_count = int(tensors["atom_mask"].sum())
    anchors_px = tensors["atom_anchors"][0, :target_count, :2] * 768.0
    raw_ids = _ids(raw, tensors, config)
    selected_ids = _ids(selected, tensors, config)

    raw_points = raw["atom_geometry"][0, raw["atom_mask"][0], :2] * 768.0
    selected_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2] * 768.0
    points, scores = _peaks(logits)
    if len(selected_points):
        baseline_distance = torch.cdist(points, selected_points).amin(dim=1)
        keep = baseline_distance > BASELINE_GUARD_PX
    else:
        baseline_distance = torch.full((len(points),), float("inf"), device=points.device)
        keep = torch.ones((len(points),), dtype=torch.bool, device=points.device)
    guarded_points = points[keep]
    guarded_scores = scores[keep]
    # Readiness uses top PEAK_K after the ownership guard; keep this separate
    # from the raw field pool to expose where candidate loss happens.
    if len(guarded_scores) > PEAK_K:
        order = torch.argsort(guarded_scores, descending=True)[:PEAK_K]
        final_points = guarded_points[order]
        final_scores = guarded_scores[order]
    else:
        final_points, final_scores = guarded_points, guarded_scores

    side_ids, side_count, side_distances = _locked_sidecar(
        final_points, anchors_px, selected_ids, target_count
    )
    raw_side_ids, _, _ = _locked_sidecar(
        final_points, anchors_px, raw_ids, target_count
    )
    return {
        "sample_key": row["sample_key"],
        "profile_id": sample["meta"]["profile_id"],
        "targets": target_count,
        "raw_baseline_candidate_count": int(raw["atom_mask"].sum()),
        "selected_baseline_candidate_count": int(selected["atom_mask"].sum()),
        "raw_baseline_ids": sorted(raw_ids),
        "selected_baseline_ids": sorted(selected_ids),
        "raw_baseline_matched": len(raw_ids),
        "selected_baseline_matched": len(selected_ids),
        "raw_residual_target_count": target_count - len(raw_ids),
        "selected_residual_target_count": target_count - len(selected_ids),
        "raw_peak_count": int(len(points)),
        "guarded_peak_count": int(len(guarded_points)),
        "final_candidate_count": int(len(final_points)),
        "guard_rejected_peak_count": int((~keep).sum()),
        "guard_min_distance_px": float(baseline_distance.min()) if len(baseline_distance) else None,
        "final_score_range": [float(final_scores.min()), float(final_scores.max())] if len(final_scores) else [],
        "selected_locked_sidecar_matched": sorted(side_ids),
        "raw_locked_sidecar_matched": sorted(raw_side_ids),
        "selected_locked_sidecar_count": int(len(side_ids)),
        "raw_locked_sidecar_count": int(len(raw_side_ids)),
        "selected_sidecar_false_positive_count": int(len(final_points) - len(side_ids)),
        "raw_sidecar_false_positive_count": int(len(final_points) - len(raw_side_ids)),
        "selected_locked_assignment_distances_centipx": side_distances,
    }


def _summary(rows: list[dict]) -> dict:
    def total(key: str) -> int:
        return sum(int(item.get(key, 0)) for item in rows)

    return {
        "images": len(rows),
        "targets": total("targets"),
        "raw_baseline_matched": total("raw_baseline_matched"),
        "selected_baseline_matched": total("selected_baseline_matched"),
        "raw_peak_count": total("raw_peak_count"),
        "guarded_peak_count": total("guarded_peak_count"),
        "final_candidate_count": total("final_candidate_count"),
        "guard_rejected_peak_count": total("guard_rejected_peak_count"),
        "selected_locked_sidecar_matched": total("selected_locked_sidecar_count"),
        "raw_locked_sidecar_matched": total("raw_locked_sidecar_count"),
        "selected_sidecar_false_positives": total("selected_sidecar_false_positive_count"),
        "raw_sidecar_false_positives": total("raw_sidecar_false_positive_count"),
        "images_with_selected_sidecar_positive": sum(bool(item["selected_locked_sidecar_matched"]) for item in rows),
        "images_with_raw_sidecar_positive": sum(bool(item["raw_locked_sidecar_matched"]) for item in rows),
    }


def _profile_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["profile_id"]), []).append(row)
    return {profile: _summary(items) for profile, items in sorted(grouped.items())}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    fixed = json.loads(FIXED_REPORT.read_text())
    fixed_keys = {item["sample_key"] for item in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = [r for r in manifest["samples"] if r["sample_key"] in fixed_keys]
    rows.sort(key=lambda r: r["sample_key"])
    if args.max_images:
        rows = rows[: int(args.max_images)]
    model, model_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to("cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = PredictedProposalConfig(
        maximum_atoms=128,
        maximum_bonds=model_config.maximum_bonds,
        atom_threshold=0.01,
        bond_threshold=0.01,
        atom_match_radius_px=6.0,
        bond_match_radius_px=12.0,
        image_size=768.0,
    )
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    trace = []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {k: v.to("cuda:0") for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, atom_output, _, raw, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, float(payload["provenance"]["atom_threshold"])
            )
            # This is intentionally a frozen-field forward pass; no branch is
            # trained here.  Zero logits represent the current sidecar field
            # before a branch is admitted, which lets us audit interface order
            # and ownership independent of a new model's quality.
            base = F.interpolate(atom_output["heatmap_logits"], size=(384, 384), mode="bilinear", align_corners=False)
            trace.append(_trace_row(row=row, sample=sample, tensors=tensors, raw=raw, selected=selected, logits=base, config=config))
            if index % 100 == 0:
                print(json.dumps({"processed": index, "total": len(rows)}), flush=True)
    result = {
        "schema_version": "chemvector-highres-candidate-parity-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "fixed_cohort_report": str(FIXED_REPORT),
        "images_requested": len(rows),
        "contract": {
            "baseline_assignment": "immutable selected baseline first",
            "residual_ownership": "sidecar may match only targets not in selected baseline IDs",
            "peak_pool_k": PEAK_POOL_K,
            "post_guard_k": PEAK_K,
            "baseline_guard_px": BASELINE_GUARD_PX,
            "target_match_radius_px": MATCH_RADIUS_PX,
            "field_branch": "frozen 192px heatmap upsampled to 384px; no sidecar training",
        },
        "summary": _summary(trace),
        "by_profile": _profile_summary(trace),
        "rows": trace,
        "gate": {
            "baseline_ids_recorded": True,
            "guard_order_recorded": True,
            "residual_ids_recorded": True,
            "positive_sidecar_observed": _summary(trace)["selected_locked_sidecar_matched"] > 0,
            "ready_for_training": False,
            "reason": "parity trace is diagnostic only; candidate field was not trained",
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "summary": result["summary"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0, help="0 means the complete fixed cohort")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
