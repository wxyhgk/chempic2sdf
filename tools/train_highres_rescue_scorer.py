"""Bounded in-memory RescueScorer trial for the frozen high-resolution sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

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
from tools.shadow_atom_highres_pixel_branch import PixelStride2Rescue


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_RESCUE_SCORER_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-rescue-scorer-trial-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BRANCH_STEPS = 200
SCORER_STEPS = 400
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_RESCUE_SCORER_TRAIN_IMAGES", "120"))
PEAK_K = int(os.environ.get("CHEMVECTOR_RESCUE_SCORER_PEAK_K", "16"))
PEAK_POOL_K = int(os.environ.get("CHEMVECTOR_RESCUE_SCORER_PEAK_POOL_K", "64"))
RESCUE_MIN_DIST_PX = float(os.environ.get("CHEMVECTOR_RESCUE_SCORER_MIN_DIST_PX", "12.0"))
CAPACITY = 128
READINESS = os.environ.get("CHEMVECTOR_RESCUE_SCORER_READINESS", "0") == "1"
# Pre-registered train-only readiness grid; holdout never selects a threshold.
READINESS_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
THRESHOLD_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
PRECISION_FLOOR = float(os.environ.get("CHEMVECTOR_RESCUE_SCORER_PRECISION_FLOOR", "0.85"))
MIN_TRAIN_POSITIVES = 50
MIN_TRAIN_AUPRC = 0.50
OBJECTIVE = os.environ.get("CHEMVECTOR_RESCUE_SCORER_OBJECTIVE", "local_margin")
TARGET_POOL_MANIFEST = os.environ.get("CHEMVECTOR_RESCUE_SCORER_TARGET_POOL", "")
RESIDUAL_ONLY = os.environ.get("CHEMVECTOR_RESCUE_SCORER_RESIDUAL_ONLY", "0") == "1"
# Opt-in diagnostic contract: ``raw`` matches the field shadow's baseline
# proposal IDs; ``selected`` preserves the historical readiness behavior.
BASELINE_CONTRACT = os.environ.get("CHEMVECTOR_RESCUE_SCORER_BASELINE_CONTRACT", "selected")
if BASELINE_CONTRACT not in {"selected", "raw"}:
    raise ValueError("CHEMVECTOR_RESCUE_SCORER_BASELINE_CONTRACT must be selected or raw")
CAPACITY_MODE = os.environ.get("CHEMVECTOR_RESCUE_SCORER_CAPACITY_MODE", "fixed128")
if CAPACITY_MODE not in {"fixed128", "predicted_count"}:
    raise ValueError("CHEMVECTOR_RESCUE_SCORER_CAPACITY_MODE must be fixed128 or predicted_count")


class RescueScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _branch_loss(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    if OBJECTIVE == "local_margin":
        return atom_local_peak_margin_loss(logits, positive, margin=0.75, kernel_size=3)
    if OBJECTIVE == "focal":
        return focal_heatmap_loss(logits, positive.float())
    if OBJECTIVE == "focal_hard":
        base = focal_heatmap_loss(logits, positive.float())
        # Hard negatives are local maxima outside a one-cell protection ring.
        local_max = logits == F.max_pool2d(logits, 3, 1, 1)
        protected = F.max_pool2d(positive.float(), 3, 1, 1).bool()
        negative = local_max & ~protected
        values = logits[negative]
        if not len(values):
            return base
        values = values.topk(min(64, len(values))).values
        target_level = logits[positive].median()
        hard = F.softplus(values - target_level + 0.5).mean()
        return base + 0.25 * hard
    raise ValueError(f"Unknown high-resolution branch objective: {OBJECTIVE}")


def _baseline_ids(atoms, tensors, config):
    mapping, _ = match_atom_targets(atoms, tensors, config)
    return {int(value) for value in mapping[0].tolist() if int(value) >= 0}


def _candidate_rows(model, branch, proposal_config, atom_threshold, loader, manifest, row, collator, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        features, atom_output, _, raw, _, _, selected = structure_forward(
            model, tensors["pixel_values"], proposal_config, atom_threshold
        )
        baseline_view = selected if BASELINE_CONTRACT == "selected" else raw
        baseline_ids = _baseline_ids(baseline_view, tensors, proposal_config)
        selected_points = baseline_view["atom_geometry"][0, baseline_view["atom_mask"][0], :2]
        predicted_count = int(atom_output["count_logits"].argmax(-1)[0])
        logits = F.interpolate(atom_output["heatmap_logits"], size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
        score = logits.sigmoid()[0, 0]
        score_patches = F.unfold(score[None, None], kernel_size=5, padding=2)[0]
        score_patches[12] = float("-inf")
        prominence_map = score - score_patches.amax(0).view_as(score)
        peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
        py, px = torch.where(peaks)
        values = score[py, px]
        order = torch.argsort(values, descending=True)[:PEAK_POOL_K]
        points_px = torch.stack(((px[order].float() + 0.5) * 768.0 / 384.0, (py[order].float() + 0.5) * 768.0 / 384.0), dim=1)
        values = values[order]
        candidate_px, candidate_py = px[order], py[order]
        if len(selected_points):
            distances_selected = torch.cdist(points_px / 768.0, selected_points)
            keep = distances_selected.amin(dim=1) > (RESCUE_MIN_DIST_PX / 768.0)
            points_px, values = points_px[keep], values[keep]
            candidate_px, candidate_py = candidate_px[keep], candidate_py[keep]
        if len(values) > PEAK_K:
            keep_order = torch.argsort(values, descending=True)[:PEAK_K]
            points_px, values = points_px[keep_order], values[keep_order]
            candidate_px, candidate_py = candidate_px[keep_order], candidate_py[keep_order]
        target_count = int(tensors["atom_mask"].sum())
        unmatched = sorted(set(range(target_count)) - baseline_ids)
        labels = torch.zeros(len(points_px), dtype=torch.float32, device=device)
        if len(points_px) and unmatched:
            distances = torch.cdist(points_px, tensors["atom_anchors"][0, unmatched] * 768.0)
            rows_i, cols_i = linear_sum_assignment(distances.detach().cpu().numpy())
            for i, j in zip(rows_i, cols_i):
                if float(distances[i, j]) <= 6.0:
                    labels[int(i)] = 1.0
        gray = tensors["pixel_values"][0].mean(0)
        darkness, gradients, fractional_offsets = [], [], []
        for x, y in points_px.tolist():
            cx, cy = int(round(x)), int(round(y))
            crop = gray[max(0, cy - 2):min(768, cy + 3), max(0, cx - 2):min(768, cx + 3)]
            darkness.append(float(-crop.mean()))
            gx = gray[cy, min(767, cx + 1)] - gray[cy, max(0, cx - 1)]
            gy = gray[min(767, cy + 1), cx] - gray[max(0, cy - 1), cx]
            gradients.append(float(torch.sqrt(gx.square() + gy.square())))
            coarse_x, coarse_y = (x + 0.5) / 2.0, (y + 0.5) / 2.0
            fractional_offsets.append(float((
                (coarse_x - round(coarse_x)) ** 2 + (coarse_y - round(coarse_y)) ** 2
            ) ** 0.5))
        if len(points_px):
            min_distance = torch.cdist(points_px, selected_points * 768.0).amin(dim=1) if len(selected_points) else torch.full((len(points_px),), 768.0, device=device)
            prominence = prominence_map[candidate_py, candidate_px]
            base_h, base_w = atom_output["anchor_offset_logits"].shape[-2:]
            base_x = (candidate_px.float() * base_w / 384.0).clamp(0, base_w - 1e-4).long()
            base_y = (candidate_py.float() * base_h / 384.0).clamp(0, base_h - 1e-4).long()
            base_offsets = atom_output["anchor_offset_logits"][0, :, base_y, base_x].sigmoid().T
            predicted_px = torch.stack(((base_x.float() + base_offsets[:, 0]) * 768.0 / base_w,
                                        (base_y.float() + base_offsets[:, 1]) * 768.0 / base_h), dim=1)
            consistency = -torch.linalg.vector_norm(predicted_px - points_px, dim=1) / 768.0
            feats = torch.stack((values, min_distance / 768.0, torch.tensor(darkness, device=device), prominence,
                                 consistency, torch.tensor(fractional_offsets, device=device),
                                 torch.tensor(gradients, device=device)), dim=1)
        else:
            feats = torch.zeros((0, 7), device=device)
    return {
        "sample_key": row["sample_key"], "profile_id": sample["meta"]["profile_id"],
        "features": feats, "labels": labels, "baseline_ids": sorted(baseline_ids),
        "targets": target_count, "baseline_selected_count": len(baseline_ids),
        "predicted_count": predicted_count,
        "sidecar_capacity": (max(0, predicted_count - len(baseline_ids))
                              if CAPACITY_MODE == "predicted_count" else CAPACITY),
    }


def _average_precision(items) -> dict:
    """Average precision from frozen highres scores; no optimizer/scorer."""
    pairs = []
    for item in items:
        for index, (score, label) in enumerate(zip(
                item["features"][:, 0].detach().cpu().tolist(),
                item["labels"].detach().cpu().tolist())):
            pairs.append((float(score), index, int(label)))
    pairs.sort(key=lambda value: (-value[0], value[1]))
    positives = sum(value[2] for value in pairs)
    if not positives:
        return {"candidates": len(pairs), "positives": 0, "auprc": None}
    seen = 0
    area = 0.0
    for rank, (_, _, label) in enumerate(pairs, 1):
        if label:
            seen += 1
            area += seen / rank
    return {"candidates": len(pairs), "positives": positives, "auprc": area / positives}


def _score_separability(items) -> dict:
    positives, negatives = [], []
    for item in items:
        scores = item["features"][:, 0].detach().cpu().tolist()
        labels = item["labels"].detach().cpu().tolist()
        positives.extend(float(score) for score, label in zip(scores, labels) if label)
        negatives.extend(float(score) for score, label in zip(scores, labels) if not label)
    def summary(values):
        if not values:
            return {"count": 0, "mean": None, "median": None, "max": None}
        values = sorted(values)
        return {"count": len(values), "mean": sum(values) / len(values),
                "median": values[len(values) // 2], "max": values[-1]}
    return {"positive": summary(positives), "negative": summary(negatives)}


def _threshold_metrics(items, threshold: float) -> dict:
    selected = positives = 0
    for item in items:
        capacity = item.get("sidecar_capacity", max(0, CAPACITY - item["baseline_selected_count"]))
        if capacity <= 0 or not len(item["features"]):
            continue
        scores = item["features"][:, 0]
        chosen = torch.where(scores >= threshold)[0]
        if len(chosen) > capacity:
            chosen = chosen[torch.argsort(scores[chosen], descending=True)[:capacity]]
        selected += int(len(chosen))
        positives += int(item["labels"][chosen].sum())
    total_positive = sum(int(item["labels"].sum()) for item in items)
    return {"threshold": threshold, "selected": selected, "matched": positives,
            "false_positives": selected - positives,
            "precision": positives / max(selected, 1),
            "recall": positives / max(total_positive, 1),
            "available_positives": total_positive}


def _feature_ap(items, feature_index: int, *, descending: bool = True) -> dict:
    pairs = []
    for item in items:
        values = item["features"][:, feature_index].detach().cpu().tolist()
        labels = item["labels"].detach().cpu().tolist()
        pairs.extend((float(value), index, int(label)) for index, (value, label) in enumerate(zip(values, labels)))
    pairs.sort(key=lambda value: ((-value[0] if descending else value[0]), value[1]))
    positives = sum(value[2] for value in pairs)
    area = seen = 0.0
    for rank, (_, _, label) in enumerate(pairs, 1):
        if label:
            seen += 1.0
            area += seen / rank
    return {"candidates": len(pairs), "positives": positives,
            "auprc": area / positives if positives else None}


def _fixed_feature_ranking(items, train_items) -> dict:
    """Untrained, deterministic feature ablation/ranking audit.

    Feature standardization uses only train-sidecar unlabeled moments.  The
    reported top-k precision is diagnostic; no scorer weights are learned.
    """
    matrix = torch.cat([item["features"] for item in train_items if len(item["features"])], dim=0)
    mean = matrix.mean(0)
    std = matrix.std(0).clamp_min(1e-6)
    specs = {"score": (1, 0, 0, 0, 0, 0, 0), "distance": (0, 1, 0, 0, 0, 0, 0),
             "darkness": (0, 0, 1, 0, 0, 0, 0), "prominence": (0, 0, 0, 1, 0, 0, 0),
             "consistency": (0, 0, 0, 0, 1, 0, 0), "fractional_offset": (0, 0, 0, 0, 0, -1, 0),
             "gradient": (0, 0, 0, 0, 0, 0, 1), "score+distance": (1, .5, 0, 0, 0, 0, 0),
             "score+distance+prominence": (1, .5, 0, .0, 0, 0, 0.0),
             "score+distance+consistency+gradient": (1, .5, 0, 0, .5, 0, .5)}
    result = {}
    for name, weights in specs.items():
        w = torch.tensor(weights, device=mean.device)
        scores = []
        for item in items:
            z = (item["features"] - mean) / std
            scores.append((z * w).sum(-1))
        rows = []
        for k in (1, 2, 4, 8, 16):
            selected = matched = available = 0
            for item, ranking in zip(items, scores):
                available += int(item["labels"].sum())
                if not len(ranking):
                    continue
                take = min(k, len(ranking))
                chosen = torch.argsort(ranking, descending=True)[:take]
                selected += int(len(chosen)); matched += int(item["labels"][chosen].sum())
            rows.append({"k": k, "selected": selected, "matched": matched,
                         "false_positives": selected - matched,
                         "precision": matched / max(selected, 1),
                         "recall": matched / max(available, 1)})
        result[name] = {"topk": rows}
    result["univariate_auprc"] = {
        "score": _feature_ap(items, 0), "distance": _feature_ap(items, 1),
        "darkness": _feature_ap(items, 2), "prominence": _feature_ap(items, 3),
        "consistency": _feature_ap(items, 4), "fractional_offset": _feature_ap(items, 5, descending=False),
        "gradient": _feature_ap(items, 6)}
    return result


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    device = torch.device("cuda:0")
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {item["sample_key"] for item in fixed["per_sample"]}
    sources = source_checks(); manifest = sources["manifest"]
    pool_keys = None
    if TARGET_POOL_MANIFEST:
        pool = json.loads(Path(TARGET_POOL_MANIFEST).read_text())
        pool_keys = {row["sample_key"] for row in pool["rows"]}
        pool_rows = [row for row in manifest["samples"] if row["sample_key"] in pool_keys]
        fixed_rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
        if len(pool_rows) != int(pool["size"]) or pool_keys & fixed_keys:
            raise ValueError("Target pool identity/overlap contract failed")
        rows = fixed_rows + pool_rows
    else:
        rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    model, proposal_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    atom_threshold = float(payload["provenance"]["atom_threshold"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=proposal_config.maximum_atoms, bond_queries=proposal_config.maximum_bonds), purpose="training")
    # Select the same deterministic 120 raw-gap images as the high-resolution
    # capacity trial; all other fixed-cohort images are holdout.
    baseline_cache = []
    for row in rows:
        sample = loader(manifest, row); batch = collator([sample]); tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            _, _, _, raw, _, _, _ = structure_forward(model, tensors["pixel_values"], proposal_config, atom_threshold)
        mapping, _ = match_atom_targets(raw, tensors, proposal_config)
        baseline_cache.append((row, int(tensors["atom_mask"].sum()), int((mapping >= 0).sum())))
    train_keys = (pool_keys if pool_keys is not None else
                  {row["sample_key"] for row, target, matched in baseline_cache if matched < target})
    train_keys = set(sorted(train_keys)[:TRAIN_IMAGES])
    if len(train_keys) != TRAIN_IMAGES:
        raise ValueError("Insufficient fixed raw-gap training images")
    branch = PixelStride2Rescue().to(device).train()
    branch_optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-3, weight_decay=0.0)
    train_rows = [row for row in rows if row["sample_key"] in train_keys]
    for step in range(BRANCH_STEPS):
        row = train_rows[step % len(train_rows)]; sample = loader(manifest, row); batch = collator([sample]); tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"]); base = model.atom(features)["heatmap_logits"]
        logits = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
        target_mask = tensors["atom_mask"]
        if RESIDUAL_ONLY:
            with torch.no_grad():
                _, _, _, raw_atoms, _, _, _ = structure_forward(model, tensors["pixel_values"], proposal_config, atom_threshold)
                mapping, _ = match_atom_targets(raw_atoms, tensors, proposal_config)
            target_mask = target_mask.clone()
            for target_id in mapping[0].tolist():
                if int(target_id) >= 0 and int(target_id) < target_mask.shape[1]:
                    target_mask[0, int(target_id)] = False
        positive = torch.zeros((1, 1, 384, 384), dtype=torch.bool, device=device); x = (tensors["atom_anchors"][..., 0] * 384).clamp(0, 383.999).long(); y = (tensors["atom_anchors"][..., 1] * 384).clamp(0, 383.999).long(); batch_index = torch.zeros_like(x); positive[batch_index[target_mask], 0, y[target_mask], x[target_mask]] = True
        loss = _branch_loss(logits, positive); branch_optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0); branch_optimizer.step()
    records = [_candidate_rows(model, branch, proposal_config, atom_threshold, loader, manifest, row, collator, device) for row in rows]
    train_records = [r for r in records if r["sample_key"] in train_keys]; holdout_records = [r for r in records if r["sample_key"] not in train_keys]
    train_feature_rows = [r for r in train_records if len(r["features"])]
    if not train_feature_rows:
        if not READINESS:
            raise ValueError("No train-sidecar candidates for scorer trial")
        result = {"schema_version": "chemvector-rescue-scorer-readiness-0.2", "complete": True,
                  "readiness_only": True, "training_performed": False, "branch_virtual_steps": BRANCH_STEPS,
                  "scorer_virtual_steps": 0, "scorer_parameter_count": sum(p.numel() for p in RescueScorer().parameters()),
                  "train_images": TRAIN_IMAGES, "train_record_count": len(train_records), "records_with_features": sum(bool(len(r["features"])) for r in train_records), "objective": OBJECTIVE, "residual_only": RESIDUAL_ONLY, "baseline_contract": BASELINE_CONTRACT, "capacity_mode": CAPACITY_MODE,
                  "target_pool_manifest": TARGET_POOL_MANIFEST or None, "sealed_test_read": False,
                  "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
                  "train_auprc": {"candidates": 0, "positives": 0, "auprc": None},
                  "holdout_auprc": _average_precision(holdout_records),
                  "recommended_train_threshold": None, "scorer_events": [],
                  "gate": {"train_precision_floor": PRECISION_FLOOR, "minimum_train_positives": MIN_TRAIN_POSITIVES,
                           "minimum_train_auprc": MIN_TRAIN_AUPRC, "train_coverage_pass": False,
                           "train_precision_floor_pass": False, "bounded_training_admissible": False,
                           "reason": "no_train_sidecar_candidates"}}
        OUTPUT.mkdir(parents=True); (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2)); return result
    x_train = torch.cat([r["features"] for r in train_feature_rows], dim=0); y_train = torch.cat([r["labels"] for r in train_feature_rows], dim=0)
    if not bool(y_train.any()):
        if not READINESS:
            raise ValueError("No positive rescue labels in scorer training slice")
        train_ap = _average_precision(train_records)
        result = {"schema_version": "chemvector-rescue-scorer-readiness-0.2", "complete": True,
                  "readiness_only": True, "training_performed": False,
                  "branch_virtual_steps": BRANCH_STEPS, "scorer_virtual_steps": 0,
                  "scorer_parameter_count": sum(p.numel() for p in RescueScorer().parameters()),
                  "train_images": TRAIN_IMAGES, "objective": OBJECTIVE,
                  "residual_only": RESIDUAL_ONLY, "baseline_contract": BASELINE_CONTRACT, "capacity_mode": CAPACITY_MODE,
                  "target_pool_manifest": TARGET_POOL_MANIFEST or None,
                  "sealed_test_read": False, "deployment_authorized": False,
                  "initial_sha256": INITIAL_SHA, "train_auprc": train_ap,
                  "holdout_auprc": _average_precision(holdout_records),
                  "recommended_train_threshold": None, "scorer_events": [],
                  "gate": {"train_precision_floor": PRECISION_FLOOR,
                           "minimum_train_positives": MIN_TRAIN_POSITIVES,
                           "minimum_train_auprc": MIN_TRAIN_AUPRC,
                           "train_coverage_pass": False,
                           "train_precision_floor_pass": False,
                           "bounded_training_admissible": False,
                           "reason": "no_positive_residual_labels"}}
        OUTPUT.mkdir(parents=True)
        (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2))
        return result
    scorer_events = []
    scorer = None
    if not READINESS:
        scorer = RescueScorer().to(device).train(); scorer_optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3, weight_decay=0.0)
        pos_weight = ((y_train == 0).sum() / y_train.sum()).clamp(1.0, 20.0)
        for step in range(SCORER_STEPS):
            logits = scorer(x_train); loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight); scorer_optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0); scorer_optimizer.step();
            if step in (0, SCORER_STEPS - 1): scorer_events.append({"step": step + 1, "loss": float(loss.detach())})
    def choose_threshold():
        """Calibrate only on train-sidecar labels; holdout is never searched."""
        probability = scorer(x_train).sigmoid().detach()
        best = None
        for threshold in THRESHOLD_GRID:
            chosen = torch.where(probability >= threshold)[0]
            positives = int(y_train[chosen].sum()) if len(chosen) else 0
            precision = positives / max(len(chosen), 1)
            recall = positives / max(int(y_train.sum()), 1)
            candidate = (precision >= PRECISION_FLOOR, recall, precision, threshold, len(chosen))
            if best is None or candidate > best:
                best = candidate
        passed, recall, precision, threshold, selected = best
        return {
            "threshold": float(threshold), "precision": float(precision),
            "recall": float(recall), "selected": int(selected),
            "precision_floor": PRECISION_FLOOR, "pass": bool(passed),
        }

    calibration = None if READINESS else choose_threshold()
    threshold = None if READINESS else calibration["threshold"]

    def evaluate(items):
        baseline = new = fp = selected = 0
        by_style = {}
        for r in items:
            baseline += len(r["baseline_ids"]); cap = r.get("sidecar_capacity", max(0, CAPACITY - r["baseline_selected_count"]))
            if not len(r["features"]) or cap == 0: chosen = torch.empty(0, dtype=torch.long, device=device)
            else:
                probability = scorer(r["features"]).sigmoid(); chosen = torch.where(probability >= threshold)[0]; chosen = chosen[torch.argsort(probability[chosen], descending=True)[:cap]]
            positives = int(r["labels"][chosen].sum()) if len(chosen) else 0
            new += positives; fp += int(len(chosen) - positives); selected += int(len(chosen))
            value = by_style.setdefault(r["profile_id"], [0, 0, 0, 0]); value[0] += len(r["baseline_ids"]); value[1] += positives; value[2] += int(len(chosen) - positives); value[3] += int(len(chosen))
        return {"images": len(items), "baseline_matched": baseline, "new_matched": new, "false_positives": fp, "selected_sidecar": selected, "by_style": by_style}
    if READINESS:
        train_grid = [_threshold_metrics(train_records, t) for t in READINESS_THRESHOLDS]
        holdout_grid = [_threshold_metrics(holdout_records, t) for t in READINESS_THRESHOLDS]
        eligible = [row for row in train_grid if row["precision"] >= PRECISION_FLOOR and row["matched"] > 0]
        recommendation = max(eligible, key=lambda row: (row["recall"], -row["threshold"])) if eligible else None
        train_ap = _average_precision(train_records)
        train_coverage_pass = int(train_ap["positives"]) >= MIN_TRAIN_POSITIVES and float(train_ap["auprc"] or 0.0) >= MIN_TRAIN_AUPRC
        result = {"schema_version": "chemvector-rescue-scorer-readiness-0.2", "complete": True,
                  "readiness_only": True, "training_performed": False, "branch_virtual_steps": BRANCH_STEPS,
                  "scorer_virtual_steps": 0, "scorer_parameter_count": sum(p.numel() for p in RescueScorer().parameters()),
                  "train_images": TRAIN_IMAGES, "objective": OBJECTIVE, "residual_only": RESIDUAL_ONLY, "baseline_contract": BASELINE_CONTRACT, "capacity_mode": CAPACITY_MODE, "target_pool_manifest": TARGET_POOL_MANIFEST or None, "peak_k": PEAK_K, "rescue_min_dist_px": RESCUE_MIN_DIST_PX, "atom_threshold": atom_threshold, "sealed_test_read": False, "deployment_authorized": False,
                  "initial_sha256": INITIAL_SHA, "train_auprc": train_ap,
                  "holdout_auprc": _average_precision(holdout_records),
                  "train_score_separability": _score_separability(train_records),
                  "holdout_score_separability": _score_separability(holdout_records),
                  "train_fixed_feature_ranking": _fixed_feature_ranking(train_records, train_records),
                  "holdout_fixed_feature_ranking": _fixed_feature_ranking(holdout_records, train_records),
                  "train_thresholds": train_grid,
                  "holdout_thresholds": holdout_grid, "recommended_train_threshold": recommendation,
                  "scorer_events": [], "gate": {"train_precision_floor": PRECISION_FLOOR,
                  "minimum_train_positives": MIN_TRAIN_POSITIVES, "minimum_train_auprc": MIN_TRAIN_AUPRC,
                  "train_coverage_pass": train_coverage_pass,
                  "train_precision_floor_pass": recommendation is not None and train_coverage_pass,
                  "bounded_training_admissible": recommendation is not None and train_coverage_pass}}
    else:
        result = {"schema_version": "chemvector-rescue-scorer-trial-0.2", "complete": True, "training_performed": True, "branch_virtual_steps": BRANCH_STEPS, "scorer_virtual_steps": SCORER_STEPS, "scorer_parameter_count": sum(p.numel() for p in scorer.parameters()), "train_images": TRAIN_IMAGES, "objective": OBJECTIVE, "baseline_contract": BASELINE_CONTRACT, "capacity_mode": CAPACITY_MODE, "target_pool_manifest": TARGET_POOL_MANIFEST or None, "peak_k": PEAK_K, "rescue_min_dist_px": RESCUE_MIN_DIST_PX, "atom_threshold": atom_threshold, "sealed_test_read": False, "deployment_authorized": False, "calibration": calibration, "train": evaluate(train_records), "holdout": evaluate(holdout_records), "scorer_events": scorer_events}
        holdout_precision = result["holdout"]["new_matched"] / max(result["holdout"]["selected_sidecar"], 1)
        result["gate"] = {"train_calibration_pass": calibration["pass"], "holdout_new_positive": result["holdout"]["new_matched"] > 0, "holdout_precision_at_least_floor": holdout_precision >= PRECISION_FLOOR, "holdout_precision": holdout_precision, "pass": calibration["pass"] and result["holdout"]["new_matched"] > 0 and holdout_precision >= PRECISION_FLOOR}
    OUTPUT.mkdir(parents=True); (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {"output": str(OUTPUT), "gate": result["gate"]}
    if READINESS:
        summary.update({"train_auprc": result["train_auprc"], "holdout_auprc": result["holdout_auprc"], "recommended_train_threshold": result["recommended_train_threshold"]})
    else:
        summary.update({"train": result["train"], "holdout": result["holdout"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return result


if __name__ == "__main__":
    run()
