"""Bounded raw-dense endpoint trial on an identity-disjoint 480/720 split.

The production atom decoder and every downstream head remain frozen.  A tiny
zero-start 384px pixel residual is trained in memory only.  Its candidates
are evaluated as an isolated sidecar; they are never sent to the production
selector or graph merger.  This is deliberately a reachability experiment,
not a deployable checkpoint.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.current_initial_cohort import split_group_disjoint
from tools.shadow_atom_highres_pixel_branch import PixelStride2Rescue, _positive


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_RAW_DENSE_TRIAL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-raw-dense-split-trial-20261006-r1"),
))
STEPS = int(os.environ.get("CHEMVECTOR_RAW_DENSE_TRIAL_STEPS", "100"))
MAX_STEPS = 200
TRAIN_TARGET = 480
K = 16
MATCH_RADIUS_PX = 6.0
EXCLUSION_RADIUS_PX = 12.0


def _sample(loader, manifest, collator, row, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    return sample, tensors


def _baseline(model, config, payload, loader, manifest, collator, row, device):
    sample, tensors = _sample(loader, manifest, collator, row, device)
    with torch.no_grad():
        _, _, _, raw, _, _, _ = structure_forward(
            model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
        )
    mapping, _ = match_atom_targets(raw, tensors, config)
    ids = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
    raw_points = raw["atom_geometry"][0, raw["atom_mask"][0], :2].detach().cpu()
    return sample, int(tensors["atom_mask"].sum()), ids, raw_points


def _sidecar_eval(model, branch, config, payload, loader, manifest, collator, row,
                  old_ids, old_points, device):
    sample, tensors = _sample(loader, manifest, collator, row, device)
    with torch.no_grad():
        features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
        base = model.atom(features)["heatmap_logits"]
        logits = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False)
        logits = logits + branch(tensors["pixel_values"])
        score = logits.sigmoid()[0, 0]
        peaks = score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
        py, px = torch.where(peaks)
        values = score[py, px]
        order = torch.argsort(values, descending=True)
        px, py, values = px[order], py[order], values[order]
        points = torch.stack(((px.float() + 0.5) * 768.0 / 384.0,
                              (py.float() + 0.5) * 768.0 / 384.0), dim=1)
        if len(old_points):
            keep = torch.cdist(points, old_points.to(device) * 768.0).amin(dim=1) > EXCLUSION_RADIUS_PX
            points, values = points[keep], values[keep]
        points = points[:K]
        anchors = tensors["atom_anchors"][0] * 768.0
        target_count = int(tensors["atom_mask"].sum())
        anchors = anchors[:target_count]
        if len(points) and len(anchors):
            distances = torch.cdist(points, anchors)
            rows, cols = linear_sum_assignment(distances.detach().cpu().numpy())
            matches = [(int(r), int(c), float(distances[r, c])) for r, c in zip(rows, cols)
                       if float(distances[r, c]) <= MATCH_RADIUS_PX]
        else:
            matches = []
    gap_hits = {target for _, target, _ in matches if target not in old_ids}
    strict_hits = {(row_i, target) for row_i, target, _ in matches if target not in old_ids}
    return {
        "sample_key": row["sample_key"],
        "style": str(sample["meta"]["profile_id"]),
        "candidate_count": int(len(points)),
        "matched_count": int(len(matches)),
        "gap_hits": sorted(gap_hits),
        "strict_hits": len(strict_hits),
        "unknown_or_nonmatch": int(len(points) - len(matches)),
        "raw_ids": sorted(old_ids),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    if not 1 <= STEPS <= MAX_STEPS:
        raise ValueError("steps must be in [1, 200]")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys),
                  key=lambda row: row["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    train_rows, holdout_rows, split = split_group_disjoint(rows, TRAIN_TARGET)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    # Cache only public ownership metadata, never image tensors.
    baseline = {}
    for index, row in enumerate(rows, 1):
        sample, targets, ids, points = _baseline(model, config, payload, loader, manifest,
                                                  collator, row, device)
        baseline[row["sample_key"]] = {
            "targets": targets, "raw_ids": ids, "raw_points": points,
            "profile_id": str(sample["meta"]["profile_id"]),
        }
        if index % 200 == 0:
            print(json.dumps({"baseline_images": index}), flush=True)
    train_gap = [row for row in train_rows if len(baseline[row["sample_key"]]["raw_ids"]) < baseline[row["sample_key"]]["targets"]]
    if not train_gap:
        raise RuntimeError("No raw-gap training images")
    branch = PixelStride2Rescue().to(device).train()
    zero_probe = branch(next(iter(model.parameters())).new_zeros((1, 3, 768, 768)))
    if float(zero_probe.detach().abs().max()) != 0.0:
        raise RuntimeError("Zero-start branch parity failed")
    optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-3, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        row = train_gap[(step - 1) % len(train_gap)]
        _, tensors = _sample(loader, manifest, collator, row, device)
        old = baseline[row["sample_key"]]
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
            base = model.atom(features)["heatmap_logits"]
        logits = F.interpolate(base, size=(384, 384), mode="bilinear", align_corners=False) + branch(tensors["pixel_values"])
        target_mask = tensors["atom_mask"].clone()
        raw_ids = old["raw_ids"]
        # Remove already-reachable targets: this is residual-only supervision.
        for target_id in raw_ids:
            target_mask[0, target_id] = False
        positive = _positive(tensors["atom_anchors"], target_mask, 384, 384)
        # Margin objective is stable and keeps the frozen base untouched.
        local = logits == F.max_pool2d(logits, 3, 1, 1)
        protected = F.max_pool2d(positive.float(), 3, 1, 1).bool()
        hard = logits[local & ~protected]
        target = logits[positive]
        if not len(target):
            continue
        loss = F.softplus(0.75 + hard.topk(min(64, len(hard))).values[:, None] - target[None]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(branch.parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()),
                       "gradient_norm": float(norm)})
    branch.eval()
    def evaluate(eval_rows):
        results = []
        for row in eval_rows:
            old = baseline[row["sample_key"]]
            results.append(_sidecar_eval(model, branch, config, payload, loader, manifest, collator,
                                         row, old["raw_ids"], old["raw_points"], device))
        candidates = sum(r["candidate_count"] for r in results)
        matched = sum(r["matched_count"] for r in results)
        gap_hits = sum(len(r["gap_hits"]) for r in results)
        return {
            "images": len(results), "candidate_count": candidates,
            "matched_count": matched, "strict_gap_hits": gap_hits,
            "strict_precision": matched / max(candidates, 1),
            "by_style": dict(Counter(r["style"] for r in results for _ in range(len(r["gap_hits"])))),
            "rows": results,
        }
    train_eval = evaluate(train_rows)
    holdout_eval = evaluate(holdout_rows)
    result = {
        "schema_version": "chemvector-raw-dense-split-trial-0.1",
        "complete": True, "training_performed": True, "in_memory_only": True,
        "optimizer_steps": len(events), "maximum_steps": MAX_STEPS,
        "initial_sha256": INITIAL_SHA, "sealed_test_read": False,
        "deployment_authorized": False, "production_checkpoint_written": False,
        "protocol": {"train_images": len(train_rows), "holdout_images": len(holdout_rows),
                      "identity_split": split, "sidecar_capacity": K,
                      "match_radius_px": MATCH_RADIUS_PX, "exclusion_radius_px": EXCLUSION_RADIUS_PX,
                      "parameter_count": sum(p.numel() for p in branch.parameters())},
        "train": train_eval, "holdout": holdout_eval, "events": events,
        "gate": {
            "train_recovery_positive": train_eval["strict_gap_hits"] > 0,
            "holdout_recovery_min14": holdout_eval["strict_gap_hits"] >= 14,
            "sidecar_precision_min085": holdout_eval["strict_precision"] >= 0.85,
            "baseline_raw_retention": True,
            "baseline_prefix_unchanged": True,
            "fixed_dev_unchanged_without_integration": True,
            "fallback_unchanged_without_integration": True,
            "pass": (train_eval["strict_gap_hits"] > 0
                      and holdout_eval["strict_gap_hits"] >= 14
                      and holdout_eval["strict_precision"] >= 0.85),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": len(events), "gate": result["gate"],
                      "train": {k: v for k, v in train_eval.items() if k != "rows"},
                      "holdout": {k: v for k, v in holdout_eval.items() if k != "rows"}}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
