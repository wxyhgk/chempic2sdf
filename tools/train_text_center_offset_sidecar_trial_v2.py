"""Corrected text-center sidecar probe.

This version fixes the earlier diagnostic protocol: group/molecule-disjoint
split, residual-text-only labels, candidate-specific AP, masked negatives that
protect implicit atom cells, and explicit protected-prefix/set-head context
checks. The production model remains frozen and no checkpoint is written.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
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
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-text-center-offset-sidecar-v2-20261006"
STEPS = 100
TRAIN_TARGET = 120
POOL = 4096
SIDE_K = (16, 32)
BASELINE_GUARD_PX = 12.0
SUPPRESSION_PX = 6.0
MATCH_PX = 6.0
OFFSET_RADIUS_PX = 32.0


class Head(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, -2.0)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        return self.score(x), self.offset(x)


def _grid(h, w, device):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack(((x + .5) / w, (y + .5) / h), dim=-1)


def masked_focal(logits, positive, protected):
    p = logits.float().sigmoid().clamp(1e-5, 1 - 1e-5)
    pos = positive.bool()
    neg = (~protected.bool()) & ~pos
    loss = (-torch.log(p) * (1 - p) ** 2 * pos).sum()
    loss = loss + (-torch.log(1 - p) * p ** 2 * neg).sum()
    return loss / pos.sum().clamp_min(1)


def masked_rank(logits, positive, protected, margin=.5):
    values = logits[:, 0].float()
    peaks = values == F.max_pool2d(values[:, None], 3, 1, 1)[:, 0]
    false = peaks & ~protected[:, 0].bool()
    losses = []
    for b in range(logits.size(0)):
        pos = torch.where(positive[b, 0].flatten())[0]
        if not len(pos):
            continue
        candidates = torch.where(false[b].flatten())[0]
        if not len(candidates):
            continue
        scores = values[b].flatten()[candidates]
        hard = candidates[torch.argsort(scores, descending=True)[: min(32, len(candidates))]]
        losses.append(F.softplus(margin + values[b].flatten()[hard][None] - values[b].flatten()[pos][:, None]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def candidates(score, delta, baseline_points, k):
    h, w = score.shape[-2:]
    base = _grid(h, w, score.device)
    points = (base[None] + delta.tanh().permute(0, 2, 3, 1) * (OFFSET_RADIUS_PX / 768.0)).clamp(0, 1)
    # The score ranking is over flattened H*W cells; keep the coordinate
    # tensor in the same flattened layout before indexing.  The previous
    # diagnostic version indexed [H, W, 2] with a flattened index, which
    # produced out-of-bounds failures as soon as a ranked cell crossed row 0.
    points_flat = points.reshape(points.size(0), h * w, 2)
    values = score[0, 0].sigmoid().flatten()
    order = torch.argsort(values, descending=True)[: min(POOL, len(values))]
    chosen = []
    for idx in order.tolist():
        point = points_flat[0, idx]
        if len(baseline_points) and bool(torch.linalg.vector_norm(baseline_points - point, dim=1).min() <= BASELINE_GUARD_PX / 768.0):
            continue
        if chosen:
            old = points_flat[0, torch.as_tensor(chosen, device=points.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= SUPPRESSION_PX / 768.0):
                continue
        chosen.append(idx)
        if len(chosen) >= k:
            break
    indices = torch.as_tensor(chosen, dtype=torch.long, device=score.device)
    return points_flat[0, indices], values[indices], indices


def match_rows(points, targets, allowed_ids):
    allowed = sorted(allowed_ids)
    if not allowed or not len(points):
        return set()
    cost = torch.cdist(points * 768.0, targets[allowed] * 768.0)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(row) for row, col in zip(rows, cols) if float(cost[row, col]) <= MATCH_PX}


def average_precision(scores, labels):
    if not labels or not any(labels):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = total = 0
    ap = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            hits += 1
            ap += hits / rank
        total += labels[i]
    return ap / max(total, 1)


def _record(model, config, payload, sample, row, collator, device):
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        features, _, _, _, bonds, _, selected = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        mapping, _ = match_atom_targets(selected, tensors, config)
    count = int(tensors["atom_mask"].sum())
    baseline_ids = {int(v) for v in mapping[0].tolist() if int(v) >= 0}
    text_ids = set(torch.where(tensors["atom_text_mask"][0, :count])[0].tolist())
    return {
        "row": row,
        "sample": sample,
        "features": features.detach().cpu(),
        "bonds": {k: v.detach().cpu() for k, v in bonds.items()},
        "target_anchors": tensors["atom_anchors"][0, :count].detach().cpu(),
        "target_centers": tensors["atom_text_center"][0, :count].detach().cpu(),
        "baseline_ids": sorted(baseline_ids),
        "text_ids": sorted(text_ids),
        "residual_text_ids": sorted(text_ids - baseline_ids),
        "baseline_points": selected["atom_geometry"][0, selected["atom_mask"][0], :2].detach().cpu(),
        "target_count": count,
        "atom_geometry": selected["atom_geometry"].detach().cpu(),
        "atom_features": selected["atom_features"].detach().cpu(),
        "atom_scores": selected["atom_scores"].detach().cpu(),
        "atom_mask": selected["atom_mask"].detach().cpu(),
    }


def evaluate(head, records, model, payload, device):
    result = {}
    ownership = defaultdict(int)
    for k in SIDE_K:
        block = defaultdict(int)
        aps = []
        for record in records:
            with torch.no_grad():
                score, delta = head(record["features"].to(device))
                points, values, indices = candidates(score, delta, record["baseline_points"].to(device), k)
            positive_rows = match_rows(points, record["target_anchors"].to(device), record["residual_text_ids"])
            labels = [int(i in positive_rows) for i in range(len(points))]
            aps.append(average_precision(values.detach().cpu().tolist(), labels))
            block["images"] += 1
            block["candidates"] += len(points)
            block["positives"] += sum(labels)
            block["false_positives"] += len(points) - sum(labels)
            # Explicit protected-prefix check: append rows to set head and
            # verify whether its incidence context changes old prefix logits.
            if len(points):
                _, _, height, width = score.shape
                y, x = torch.div(indices, width, rounding_mode="floor"), indices.remainder(width)
                # The sidecar lives outside the production Atom proposal
                # feature space (backbone channels are 96 while set-head atom
                # tokens are d_model=128).  For the ownership probe we use a
                # neutral, correctly shaped token; this isolates whether the
                # mere presence of appended rows changes the frozen selector
                # context, without pretending an untrained projection exists.
                side_features = record["atom_features"].new_zeros(
                    (1, len(indices), record["atom_features"].size(-1))
                ).to(device)
                base = {k0: v.to(device) for k0, v in record.items() if False}  # explicit no-op
                atom = {k0: record[k0].to(device) for k0 in ("atom_features", "atom_geometry", "atom_scores", "atom_mask")}
                atom["atom_features"] = torch.cat((atom["atom_features"][:, :int(atom["atom_mask"].sum())], side_features), 1)
                side_geom = torch.cat((points[None], torch.zeros((1, len(points), atom["atom_geometry"].size(-1) - 2), device=device)), -1)
                atom["atom_geometry"] = torch.cat((atom["atom_geometry"][:, :int(record["atom_mask"].sum())], side_geom), 1)
                atom["atom_scores"] = torch.cat((atom["atom_scores"][:, :int(record["atom_mask"].sum())], values[None]), 1)
                atom["atom_mask"] = torch.cat((atom["atom_mask"][:, :int(record["atom_mask"].sum())], torch.ones((1, len(points)), dtype=torch.bool, device=device)), 1)
                bonds = {key: value.to(device) for key, value in record["bonds"].items()}
                with torch.no_grad():
                    base_set = model.set_head(record["atom_features"].to(device), record["atom_geometry"].to(device), record["atom_scores"].to(device), record["atom_mask"].to(device), bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])
                    fused_set = model.set_head(atom["atom_features"], atom["atom_geometry"], atom["atom_scores"], atom["atom_mask"], bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])
                n = int(record["atom_mask"].sum())
                delta_prefix = (base_set["atom_presence_logits"][:, :n] - fused_set["atom_presence_logits"][:, :n]).abs()
                ownership["prefix_checks"] += 1
                ownership["prefix_context_changed"] += int(bool((delta_prefix > 1e-6).any()))
                ownership["prefix_max_abs_delta"] = max(ownership["prefix_max_abs_delta"], float(delta_prefix.max()))
        block["precision"] = block["positives"] / max(block["candidates"], 1)
        block["mean_candidate_ap"] = sum(aps) / max(len(aps), 1)
        result[str(k)] = dict(block)
    result["ownership"] = dict(ownership)
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    device = torch.device("cuda:0")
    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    manifest = source_checks()["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in keys]
    train_rows, holdout_rows, split_meta = split_group_disjoint(rows, TRAIN_TARGET)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    train = [_record(model, config, payload, loader(manifest, row), row, collator, device) for row in train_rows]
    holdout = [_record(model, config, payload, loader(manifest, row), row, collator, device) for row in holdout_rows]
    focus = [r for r in train if r["residual_text_ids"]]
    head = Head(int(train[0]["features"].shape[1])).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        record = focus[(step - 1) % len(focus)]
        logits, delta = head(record["features"].to(device))
        h, w = logits.shape[-2:]
        positive = torch.zeros_like(logits, dtype=torch.bool)
        protected = torch.zeros_like(logits, dtype=torch.bool)
        offset_targets = torch.zeros((1, 2, h, w), device=device)
        grid = _grid(h, w, logits.device)
        for target_id in record["text_ids"]:
            cx, cy = int(float(record["target_centers"][target_id, 0]) * w), int(float(record["target_centers"][target_id, 1]) * h)
            cx, cy = min(w - 1, max(0, cx)), min(h - 1, max(0, cy))
            protected[0, 0, cy, cx] = True
            if target_id in record["residual_text_ids"]:
                positive[0, 0, cy, cx] = True
                offset_targets[0, :, cy, cx] = ((record["target_anchors"][target_id].to(device) - grid[cy, cx]) / (OFFSET_RADIUS_PX / 768.0)).clamp(-1, 1)
        # Protect implicit atom cells as well.
        for target_id in range(record["target_count"]):
            cx, cy = int(float(record["target_centers"][target_id, 0]) * w), int(float(record["target_centers"][target_id, 1]) * h)
            protected[0, 0, min(h - 1, max(0, cy)), min(w - 1, max(0, cx))] = True
        focal = masked_focal(logits, positive, protected)
        rank = masked_rank(logits, positive, protected)
        mask = positive.expand(-1, 2, -1, -1)
        offset = F.smooth_l1_loss(delta.tanh()[mask], offset_targets[mask]) if bool(mask.any()) else logits.sum() * 0
        loss = focal + rank + 4 * offset
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": record["row"]["sample_key"], "loss": float(loss.detach()), "focal": float(focal.detach()), "rank": float(rank.detach()), "offset": float(offset.detach()), "gradient_norm": float(norm)})
    train_metrics = evaluate(head, train, model, payload, device)
    holdout_metrics = evaluate(head, holdout, model, payload, device)
    result = {"schema_version": "chemvector-text-center-offset-sidecar-v2-0.1", "complete": True, "training_performed": True, "in_memory_only": True, "optimizer_steps": STEPS, "sealed_test_read": False, "deployment_authorized": False, "initial_sha256": INITIAL_SHA, "cohort_report": str(COHORT.relative_to(ROOT)), "split": split_meta, "focus_images": len(focus), "focus_targets": sum(len(r["residual_text_ids"]) for r in focus), "train": train_metrics, "holdout": holdout_metrics, "events": events, "gate": {"precision_floor": .85, "train_positive_min": 50, "bounded_training_admissible": False, "reason": "corrected protocol requires precision/AUPRC and immutable prefix; this is a diagnostic trial only"}}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "split": split_meta, "train": train_metrics, "holdout": holdout_metrics}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
