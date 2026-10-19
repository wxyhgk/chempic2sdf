"""Forward-only protected replacement contract audit.

The production selected proposal prefix is copied byte-for-byte and is never
re-ranked, replaced, or passed through a changed selector.  This audit asks
whether the current AtomObject output can append a small number of text-role
residual slots in unused capacity (``128 - len(selected)``).  Candidate cells
are chosen without targets; labels are consulted only afterward for locked
residual accounting.  It reads only the fixed public 1,200-image cohort.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

from chemvector.atom_bond_scene import AtomBondSceneConfig
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
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_PROTECTED_TEXT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-protected-text-slots-20261006"),
))
CAPACITY = 128
POOL = 4096
SLOT_K = (1, 2, 4, 8, 16)
BASELINE_GUARD_PX = 12.0
SIDE_SUPPRESSION_PX = 6.0
MATCH_PX = 6.0


def _ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(value) for value in mapping[0].tolist() if int(value) >= 0}


def _append(base: dict, output: dict, indices: torch.Tensor, config) -> dict:
    """Append dense cells to a selected prefix without touching prefix rows."""
    _, _, height, width = output["heatmap_logits"].shape
    if not len(indices):
        return {key: value[:, :int(base["atom_mask"].sum())].clone()
                for key, value in base.items() if key in (
                    "atom_features", "atom_geometry", "atom_scores", "atom_indices", "atom_mask")}
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices.remainder(width)
    batch = torch.zeros(len(indices), dtype=torch.long, device=indices.device)
    fmap = output["object_features"].permute(0, 2, 3, 1)
    features = fmap[batch, y, x][None]
    anchors = output["anchors"][0].permute(1, 2, 0)[y, x]
    roles = output["role_logits"][0].permute(1, 2, 0)[y, x].softmax(-1)
    centers = output["text_centers"][0].permute(1, 2, 0)[y, x]
    extents = output["text_extents"][0].permute(1, 2, 0)[y, x]
    geometry = torch.cat((anchors, roles, centers, extents), dim=-1)[None]
    scores = output["heatmap_logits"][0, 0].flatten()[indices][None].sigmoid()
    side = {
        "atom_features": features,
        "atom_geometry": geometry,
        "atom_scores": scores,
        "atom_indices": indices[None],
        "atom_mask": torch.ones((1, len(indices)), dtype=torch.bool, device=indices.device),
    }
    n = int(base["atom_mask"].sum())
    return {key: torch.cat((base[key][:, :n], side[key]), dim=1)
            for key in side}


def _select(output: dict, baseline_points: torch.Tensor, k: int, config) -> torch.Tensor:
    heat = output["heatmap_logits"][0, 0].sigmoid()
    role_text = output["role_logits"][0, 1].sigmoid()
    score = heat * role_text
    anchors = output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    order = torch.argsort(score.flatten(), descending=True)[:POOL]
    guard = BASELINE_GUARD_PX / config.image_size
    suppression = SIDE_SUPPRESSION_PX / config.image_size
    chosen: list[int] = []
    for index in order.tolist():
        point = anchors[index]
        if len(baseline_points) and bool(torch.linalg.vector_norm(
                baseline_points - point, dim=1).min() <= guard):
            continue
        if chosen:
            old = anchors[torch.as_tensor(chosen, device=anchors.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= suppression):
                continue
        chosen.append(int(index))
        if len(chosen) >= k:
            break
    return torch.as_tensor(chosen, dtype=torch.long, device=anchors.device)


def _locked(side: dict, tensors: dict, baseline_ids: set[int], config) -> set[int]:
    n_targets = int(tensors["atom_mask"].sum())
    remaining = [i for i in range(n_targets) if i not in baseline_ids]
    if not remaining or not int(side["atom_mask"].sum()):
        return set()
    points = side["atom_geometry"][0, :, :2] * config.image_size
    expected = tensors["atom_anchors"][0, remaining] * config.image_size
    cost = torch.cdist(points.float(), expected.float())
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(remaining[col]) for row, col in zip(rows, cols)
            if float(cost[row, col]) <= MATCH_PX}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or current initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys),
                  key=lambda row: row["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    totals = {str(k): {"images": 0, "eligible_images": 0, "candidates": 0,
                       "positives": 0, "false_positives": 0, "residual_targets": 0,
                       "prefix_exact": 0, "set_head_shape_ok": 0}
              for k in SLOT_K}
    by_profile = defaultdict(lambda: defaultdict(int))
    max_prefix_delta = 0.0
    set_prefix_max_delta = 0.0
    set_prefix_mean_delta = 0.0
    set_prefix_flip_count = 0
    set_prefix_rows = 0
    selected_count_sum = 0
    selected_count_min = CAPACITY
    selected_count_max = 0
    selected_full_capacity_images = 0
    set_head_finite_ok = 0
    with torch.no_grad():
        for image_index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items()
                       if isinstance(value, torch.Tensor)}
            _, output, bond_output, _, bonds, base_set, selected = structure_forward(
                model, tensors["pixel_values"], config,
                payload["provenance"]["atom_threshold"],
            )
            n = int(selected["atom_mask"].sum())
            selected_count_sum += n
            selected_count_min = min(selected_count_min, n)
            selected_count_max = max(selected_count_max, n)
            selected_full_capacity_images += int(n >= CAPACITY)
            baseline_ids = _ids(selected, tensors, config)
            baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2]
            available = max(0, CAPACITY - n)
            profile = str(sample["meta"]["profile_id"])
            for k in SLOT_K:
                block = totals[str(k)]
                block["images"] += 1
                if available <= 0:
                    continue
                block["eligible_images"] += 1
                take = min(k, available)
                indices = _select(output, baseline_points, take, config)
                fused = _append(selected, output, indices, config)
                prefix_n = min(n, int(fused["atom_mask"].sum()))
                prefix_exact = all(torch.equal(fused[key][:, :prefix_n], selected[key][:, :prefix_n])
                                   for key in ("atom_features", "atom_geometry", "atom_scores", "atom_indices", "atom_mask"))
                block["prefix_exact"] += int(prefix_exact)
                max_prefix_delta = max(max_prefix_delta, float((
                    fused["atom_geometry"][:, :prefix_n] - selected["atom_geometry"][:, :prefix_n]
                ).abs().max()))
                try:
                    set_output = model.set_head(
                        fused["atom_features"], fused["atom_geometry"], fused["atom_scores"], fused["atom_mask"],
                        bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"],
                    )
                    block["set_head_shape_ok"] += 1
                    if all(bool(torch.isfinite(value).all()) for value in set_output.values()
                           if isinstance(value, torch.Tensor)):
                        set_head_finite_ok += 1
                    base_logits = base_set["atom_presence_logits"][0, :n]
                    fused_logits = set_output["atom_presence_logits"][0, :n]
                    delta = (fused_logits - base_logits).abs()
                    set_prefix_max_delta = max(set_prefix_max_delta, float(delta.max()))
                    set_prefix_mean_delta += float(delta.sum())
                    set_prefix_rows += int(n)
                    set_prefix_flip_count += int((
                        (base_logits >= payload["provenance"]["atom_threshold"]) !=
                        (fused_logits >= payload["provenance"]["atom_threshold"])
                    ).sum())
                except Exception:
                    pass
                positives = _locked(
                    {key: fused[key][:, n:] for key in ("atom_features", "atom_geometry", "atom_scores", "atom_indices", "atom_mask")},
                    tensors, baseline_ids, config,
                )
                count = int(indices.numel())
                block["candidates"] += count
                block["positives"] += len(positives)
                block["false_positives"] += count - len(positives)
                block["residual_targets"] += int(tensors["atom_mask"].sum()) - len(baseline_ids)
                by_profile[f"{profile}/k{k}"]["candidates"] += count
                by_profile[f"{profile}/k{k}"]["positives"] += len(positives)
            if image_index % 200 == 0:
                print(json.dumps({"images": image_index}), flush=True)

    for values in totals.values():
        values["precision"] = values["positives"] / max(values["candidates"], 1)
        values["residual_recall"] = values["positives"] / max(values["residual_targets"], 1)
    result = {
        "schema_version": "chemvector-protected-text-slots-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "contract": {
            "baseline": "selected prefix copied exactly; no replacement or reranking",
            "capacity": CAPACITY,
            "candidate_role": "dense heatmap probability times TEXT role probability",
            "pool": POOL,
            "baseline_guard_px": BASELINE_GUARD_PX,
            "sidecar_suppression_px": SIDE_SUPPRESSION_PX,
            "match_radius_px": MATCH_PX,
            "target_independent_selection": True,
            "residual_ownership": "baseline IDs assigned first; sidecar claims only remaining targets",
        },
        "totals": totals,
        "by_profile": {key: dict(value) for key, value in sorted(by_profile.items())},
        "invariants": {
            "prefix_exact_all": all(v["prefix_exact"] == v["eligible_images"] for v in totals.values()),
            "max_prefix_geometry_delta": max_prefix_delta,
            "set_head_shape_ok_all": all(v["set_head_shape_ok"] == v["eligible_images"] for v in totals.values()),
            "set_head_finite_ok": set_head_finite_ok == sum(v["eligible_images"] for v in totals.values()),
            "set_prefix_presence_max_abs_delta": set_prefix_max_delta,
            "set_prefix_presence_mean_abs_delta": set_prefix_mean_delta / max(set_prefix_rows, 1),
            "set_prefix_presence_threshold_flips": set_prefix_flip_count,
            "production_capacity_unchanged": True,
        },
        "selected_capacity": {
            "images": len(rows),
            "min_selected": selected_count_min,
            "max_selected": selected_count_max,
            "mean_selected": selected_count_sum / max(len(rows), 1),
            "full_capacity_images": selected_full_capacity_images,
            "available_slot_sum": len(rows) * CAPACITY - selected_count_sum,
        },
        "gate": {
            "prefix_retention": True,
            "minimum_train_positives": 50,
            "minimum_train_precision": 0.85,
            "minimum_holdout_precision": 0.85,
            "minimum_holdout_new_matches": 14,
            "bounded_training_admissible": False,
            "reason": "contract support only; current untrained text sidecar must pass precision and holdout gates before any scorer trial",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "totals": totals, "invariants": result["invariants"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
