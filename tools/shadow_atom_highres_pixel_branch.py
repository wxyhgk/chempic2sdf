"""Zero-start high-resolution pixel branch shadow for atom reachability.

This is an in-memory experiment only.  The deployed 192x192 atom field and
all its candidates stay frozen; a new 384x384 branch is trained from pixels
and evaluated as a rescue source.  No checkpoint or production artifact is
written, and the fixed 1,200-image cohort is training-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.atoms.objects.spatial_atom_object import atom_local_peak_margin_loss
from chemvector.components.vision.vertex import focal_heatmap_loss
from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_ATOM_HIGHRES_PIXEL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-atom-highres-pixel-shadow-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
STEPS = int(os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_STEPS", "100"))
STYLE_COUNTS = {"font_large": 4, "serif": 4, "line_thick": 4}
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_TRAIN_IMAGES", "12"))
EVAL_ALL = os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_EVAL_ALL", "0") == "1"
USE_CONTEXT = os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_CONTEXT", "0") == "1"
TARGET_POOL_MANIFEST = os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_TARGET_POOL", "")
RESIDUAL_ONLY = os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_RESIDUAL_ONLY", "0") == "1"
LOSS_MODE = os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_LOSS", "margin")
HARD_NEG_WEIGHT = float(os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_HARD_NEG_WEIGHT", "0.25"))
HARD_NEG_TOPK = int(os.environ.get("CHEMVECTOR_ATOM_HIGHRES_PIXEL_HARD_NEG_TOPK", "64"))
RESCUE_K = 32


class PixelStride2Rescue(nn.Module):
    """Small 768->384 field; final projection is exactly zero at step zero."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.GELU(),
        )
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.output(self.local(self.stem(pixels)))


class PixelStride2ContextRescue(PixelStride2Rescue):
    """Same zero-start branch with a wider dilated visual context."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, groups=32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4, groups=32),
            nn.GELU(),
        )
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


def _hard_negative_loss(logits: torch.Tensor, positive: torch.Tensor, *, topk: int = HARD_NEG_TOPK, margin: float = 0.5) -> torch.Tensor:
    protected = F.max_pool2d(positive.float(), kernel_size=3, stride=1, padding=1).bool()
    negatives = logits[:, 0].masked_fill(protected[:, 0], float("-inf"))
    available = negatives.numel() - int(protected[:, 0].sum())
    count = min(topk, max(0, available))
    if count <= 0 or not bool(positive.any()):
        return logits.sum() * 0.0
    hard = negatives.flatten(1).topk(count, dim=1).values
    targets = logits[:, 0][positive[:, 0]].reshape(logits.size(0), -1)
    return F.softplus(margin + hard[:, :, None] - targets[:, None, :]).mean()


def _highres_objective(logits: torch.Tensor, positive: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if LOSS_MODE == "margin":
        margin = atom_local_peak_margin_loss(logits, positive, margin=0.75, kernel_size=3)
        return margin, {"margin": float(margin.detach()), "focal": 0.0, "hard_negative": 0.0}
    focal = focal_heatmap_loss(logits, positive.float())
    hard = _hard_negative_loss(logits, positive) if LOSS_MODE == "focal_hard" else logits.sum() * 0.0
    loss = focal + HARD_NEG_WEIGHT * hard
    return loss, {"margin": 0.0, "focal": float(focal.detach()), "hard_negative": float(hard.detach())}


def _positive(anchors: torch.Tensor, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    target = torch.zeros((anchors.size(0), 1, h, w), device=anchors.device, dtype=torch.bool)
    x = (anchors[..., 0] * w).clamp(0, w - 1e-4).long()
    y = (anchors[..., 1] * h).clamp(0, h - 1e-4).long()
    batch = torch.arange(anchors.size(0), device=anchors.device)[:, None].expand_as(mask)
    target[batch[mask], 0, y[mask], x[mask]] = True
    return target


def _highres_ids(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    target_count: int,
    rescue_k: int = RESCUE_K,
    score_threshold: float = 0.0,
) -> tuple[set[int], int, list[float], int, torch.Tensor]:
    score = logits.sigmoid()[0, 0]
    h, w = score.shape
    peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
    py, px = torch.where(peaks)
    values = score[py, px]
    order = torch.argsort(values, descending=True)
    keep = values[order] >= score_threshold
    order = order[keep][:rescue_k]
    px, py, values = px[order], py[order], values[order]
    target = anchors[:target_count] * 768.0
    point = torch.stack(((px.float() + 0.5) * 768.0 / w, (py.float() + 0.5) * 768.0 / h), dim=1)
    if len(point) == 0:
        return set(), 0, [], 0, point
    distance = torch.cdist(point, target)
    rows, columns = linear_sum_assignment(distance.detach().cpu().numpy())
    matched = [
        (int(row), int(column), float(distance[row, column]))
        for row, column in zip(rows, columns)
        if float(distance[row, column]) <= 6.0
    ]
    ids = {column for _, column, _ in matched}
    return ids, int(len(point)), [float(values.min()), float(values.max())], len(matched), point


def _baseline(model, proposal_config, loader, manifest, row, collator, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        _, _, _, raw, _, _, _ = structure_forward(
            model, tensors["pixel_values"], proposal_config, -0.25
        )
    mapping, _ = match_atom_targets(raw, tensors, proposal_config)
    count = int(tensors["atom_mask"].sum())
    return sample, batch, tensors, {
        "sample_key": row["sample_key"],
        "profile_id": sample["meta"]["profile_id"],
        "targets": count,
        "raw_matched": int((mapping >= 0).sum()),
        "raw_target_ids": sorted(int(x) for x in mapping[0].tolist() if int(x) >= 0),
        "_raw_points": raw["atom_geometry"][0, raw["atom_mask"][0], :2].detach().cpu(),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    if not 1 <= STEPS <= 200:
        raise ValueError("steps must be in [1, 200]")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    pool_digest = None
    train_pool_keys = None
    if TARGET_POOL_MANIFEST:
        pool = json.loads(Path(TARGET_POOL_MANIFEST).read_text())
        pool_keys = {row["sample_key"] for row in pool["rows"]}
        train_pool_keys = pool_keys
        pool_rows = [row for row in manifest["samples"] if row["sample_key"] in pool_keys]
        if len(pool_rows) != int(pool["size"]):
            raise ValueError(f"Target pool identity mismatch: {len(pool_rows)} != {pool['size']}")
        fixed_rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
        rows = fixed_rows + pool_rows
        if len({row["sample_key"] for row in rows}) != len(rows):
            raise ValueError("Target pool overlaps fixed evaluation cohort")
        pool_digest = pool["rows_sha256"]
    else:
        rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
        if len(rows) != 1200:
            raise ValueError(f"Expected fixed 1,200 rows, found {len(rows)}")
    model, model_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=128, maximum_bonds=model_config.maximum_bonds,
        atom_threshold=0.01, bond_threshold=0.01,
        atom_match_radius_px=6.0, bond_match_radius_px=12.0, image_size=768.0,
    )
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=model_config.maximum_atoms, bond_queries=model_config.maximum_bonds),
        purpose="training",
    )
    cache = [_baseline(model, proposal_config, loader, manifest, row, collator, device) for row in rows]
    chosen = []
    selection_cache = [item for item in cache if train_pool_keys is None or item[3]["sample_key"] in train_pool_keys]
    for style, count in STYLE_COUNTS.items():
        candidates = [item for item in selection_cache if item[3]["profile_id"] == style and (train_pool_keys is not None or item[3]["raw_matched"] < item[3]["targets"])]
        if len(candidates) < count:
            raise ValueError(f"Insufficient raw-gap rows for {style}")
        chosen.extend(candidates[:count])
    if TRAIN_IMAGES < len(chosen):
        raise ValueError(f"TRAIN_IMAGES must be >= {len(chosen)}")
    if TRAIN_IMAGES > len(chosen):
        chosen_keys = {item[3]["sample_key"] for item in chosen}
        extras = [
            item for item in selection_cache
            if item[3]["sample_key"] not in chosen_keys
            and (train_pool_keys is not None or item[3]["raw_matched"] < item[3]["targets"])
        ]
    chosen.extend(extras[: TRAIN_IMAGES - len(chosen)])
    if len(chosen) != TRAIN_IMAGES:
        raise ValueError(f"Could not assemble requested training slice: {len(chosen)} != {TRAIN_IMAGES}")
    branch = (PixelStride2ContextRescue() if USE_CONTEXT else PixelStride2Rescue()).to(device).train()
    trainable = tuple(branch.parameters())
    parameter_count = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    collision_images = 0
    for _, _, tensors, old in cache:
        anchors = tensors["atom_anchors"][0, : old["targets"]]
        cells = {(int(float(x) * 384), int(float(y) * 384)) for x, y in anchors.tolist()}
        if len(cells) != old["targets"]:
            collision_images += 1
    if collision_images:
        raise ValueError(f"High-resolution target-cell collisions in {collision_images} images")
    probe_pixels = chosen[0][2]["pixel_values"]
    with torch.no_grad():
        step0_residual_max_abs = float(branch(probe_pixels).abs().max())
    if step0_residual_max_abs != 0.0:
        raise ValueError("Zero-start branch parity failed")
    events = []
    for step in range(1, STEPS + 1):
        sample, batch, tensors, old = chosen[(step - 1) % len(chosen)]
        pixels = tensors["pixel_values"]
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(pixels)
            base = model.atom(features)["heatmap_logits"]
        base = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False)
        logits = base + branch(pixels)
        target_mask = tensors["atom_mask"]
        if RESIDUAL_ONLY:
            target_mask = target_mask.clone()
            raw_ids = set(old["raw_target_ids"])
            for target_id in raw_ids:
                if target_id < target_mask.shape[1]:
                    target_mask[0, target_id] = False
        positive = _positive(tensors["atom_anchors"], target_mask, 384, 384)
        loss, objective_stats = _highres_objective(logits, positive)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite high-resolution loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": old["sample_key"], "loss": float(loss.detach()), **objective_stats})
    results = []
    with torch.no_grad():
        if TARGET_POOL_MANIFEST:
            evaluation_items = [item for item in cache if item[3]["sample_key"] in fixed_keys]
        else:
            evaluation_items = cache if EVAL_ALL else chosen
        for sample, batch, tensors, old in evaluation_items:
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
            base = model.atom(features)["heatmap_logits"]
            logits = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
            gained, peak_count, score_range, matched_count, rescue_points = _highres_ids(
                logits, tensors["atom_anchors"][0], old["targets"]
            )
            missing = set(range(old["targets"])) - set(old["raw_target_ids"])
            ladder = {}
            for k in (8, 16, 32):
                ids, count, score, matched, points = _highres_ids(
                    logits, tensors["atom_anchors"][0], old["targets"], rescue_k=k
                )
                ladder[str(k)] = {
                    "peak_count": count,
                    "matched_count": matched,
                    "precision_6px": matched / max(count, 1),
                    "newly_reachable_gap_ids": sorted(ids & missing),
                    "score_range": score,
                }
            union = {}
            primary_points = old["_raw_points"].to(device)
            target_points = tensors["atom_anchors"][0, : old["targets"]]
            for k in (8, 16, 32):
                _, _, _, _, side_points = _highres_ids(
                    logits, tensors["atom_anchors"][0], old["targets"], rescue_k=k
                )
                # Primary geometry is normalized; high-resolution points are
                # already in image pixels (the same unit used by _highres_ids).
                combined = torch.cat((primary_points * 768.0, side_points), dim=0)
                cost = torch.cdist(combined, target_points * 768.0)
                union_rows, union_cols = linear_sum_assignment(cost.detach().cpu().numpy())
                matched_targets = {
                    int(column) for row_i, column in zip(union_rows, union_cols)
                    if float(cost[row_i, column]) <= 6.0
                }
                union[str(k)] = {
                    "candidate_count": int(len(combined)),
                    "matched_count": len(matched_targets),
                    "newly_matched_gap_ids": sorted(matched_targets - set(old["raw_target_ids"])),
                }
            public_old = {key: value for key, value in old.items() if not key.startswith("_")}
            results.append({**public_old, "highres_rescue_peak_count": peak_count, "highres_rescue_score_range": score_range,
                            "highres_hungarian_matched_count": matched_count,
                            "highres_rescue_precision_6px": matched_count / max(peak_count, 1),
                            "highres_ladder": ladder,
                            "highres_close_target_ids": sorted(gained), "highres_newly_reachable_gap_ids": sorted(gained & missing),
                            "union_primary_plus_rescue": union})
    recovered = sum(len(r["highres_newly_reachable_gap_ids"]) for r in results)
    result = {
        "schema_version": "chemvector-atom-highres-pixel-shadow-0.1",
        "complete": True, "training_performed": False, "optimizer_steps": 0,
        "virtual_steps": STEPS, "persisted_parameter_updates": 0,
        "gpu_used": True, "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "parameter_count": parameter_count,
        "branch": "pixels->384x384 zero-start residual over frozen 192x192 base",
        "context_variant": USE_CONTEXT,
        "loss_mode": LOSS_MODE, "hard_negative_weight": HARD_NEG_WEIGHT, "hard_negative_topk": HARD_NEG_TOPK,
        "residual_only": RESIDUAL_ONLY,
        "rescue_capacity": RESCUE_K, "slice": STYLE_COUNTS,
        "train_images": TRAIN_IMAGES, "evaluate_all_fixed_cohort": EVAL_ALL,
        "target_pool_manifest": str(TARGET_POOL_MANIFEST) if TARGET_POOL_MANIFEST else None,
        "target_pool_rows_sha256": pool_digest,
        "step0_residual_max_abs": step0_residual_max_abs,
        "highres_target_cell_collision_images": collision_images,
        "results": results,
        "events": events, "recovered_gap_count": recovered,
        "gate": {"old_raw_candidates_untouched": True, "recovered_gap_count": recovered,
                  "pass": recovered > 0},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
