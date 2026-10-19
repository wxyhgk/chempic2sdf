"""Bounded text-center objectness + center-to-anchor offset sidecar trial."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.atoms.objects.g3_atom_center_field import atom_center_global_peak_rank_loss
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from chemvector.vertex import focal_heatmap_loss


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TEXT_CENTER_OFFSET_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-text-center-offset-sidecar-20261006"),
))
STEPS = int(os.environ.get("CHEMVECTOR_TEXT_CENTER_OFFSET_STEPS", "100"))
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_TEXT_CENTER_OFFSET_TRAIN_IMAGES", "120"))
LR = float(os.environ.get("CHEMVECTOR_TEXT_CENTER_OFFSET_LR", "1e-3"))
POOL = 4096
SIDE_K = (16, 32)
SUPPRESSION_PX = 6.0
BASELINE_GUARD_PX = 12.0
MATCH_RADIUS_PX = 6.0
OFFSET_RADIUS_PX = 32.0


class TextCenterOffsetHead(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, -2.0)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score(features), self.offset(features)


def _grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32), indexing="ij"
    )
    return torch.stack(((x + 0.5) / w, (y + 0.5) / h), dim=-1)


def _decode(score: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = score.shape[-2:]
    base = _grid(h, w, score.device)
    points = (base[None] + delta.tanh().permute(0, 2, 3, 1) * (OFFSET_RADIUS_PX / 768.0)).clamp(0.0, 1.0)
    return score.sigmoid(), points.reshape(score.size(0), -1, 2)


def _candidates(score: torch.Tensor, points: torch.Tensor, baseline: torch.Tensor, k: int) -> torch.Tensor:
    values = score[0, 0].flatten()
    order = torch.argsort(values, descending=True)[: min(POOL, len(values))]
    selected: list[int] = []
    guard = BASELINE_GUARD_PX / 768.0
    radius = SUPPRESSION_PX / 768.0
    for index in order.tolist():
        point = points[0, index]
        if len(baseline) and bool(torch.linalg.vector_norm(baseline - point, dim=1).min() <= guard):
            continue
        if selected:
            old = points[0, torch.as_tensor(selected, device=points.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= radius):
                continue
        selected.append(index)
        if len(selected) >= k:
            break
    return points[0, torch.as_tensor(selected, device=points.device)] if selected else points[0, :0]


def _residual(points: torch.Tensor, targets: torch.Tensor, baseline_ids: set[int], count: int) -> set[int]:
    remaining = [i for i in range(count) if i not in baseline_ids]
    if not remaining or not len(points):
        return set()
    cost = torch.cdist(points * 768.0, targets[remaining] * 768.0)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(remaining[col]) for row, col in zip(rows, cols) if float(cost[row, col]) <= MATCH_RADIUS_PX}


def _evaluate(head, records, device) -> dict:
    metrics = {}
    for k in SIDE_K:
        block = defaultdict(int)
        for record in records:
            with torch.no_grad():
                score, delta = head(record["features"].to(device))
                probabilities, points = _decode(score, delta)
                candidates = _candidates(probabilities, points, record["baseline_points"].to(device), k)
            positives = _residual(candidates, record["target_anchors"].to(device), set(record["baseline_ids"]), record["target_count"])
            block["images"] += 1
            block["candidates"] += len(candidates)
            block["positives"] += len(positives)
            block["false_positives"] += len(candidates) - len(positives)
            block["residual_text_targets"] += len(record["residual_text_ids"])
        item = dict(block)
        item["precision"] = item["positives"] / max(item["candidates"], 1)
        metrics[str(k)] = item
    return metrics


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    if not 1 <= STEPS <= 100 or not 1 <= TRAIN_IMAGES < 1200:
        raise ValueError("Trial limits are invalid")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    device = torch.device("cuda:0")
    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys), key=lambda r: r["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    records = []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            features, _, _, _, _, _, selected = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            mapping, _ = match_atom_targets(selected, tensors, config)
            baseline_ids = {int(v) for v in mapping[0].tolist() if int(v) >= 0}
            count = int(tensors["atom_mask"].sum())
            text_mask = tensors["atom_text_mask"][0, :count].bool()
            text_ids = set(torch.where(text_mask)[0].tolist())
            records.append({
                "sample_key": row["sample_key"],
                "features": features.detach().cpu(),
                "target_anchors": tensors["atom_anchors"][0, :count].detach().cpu(),
                "target_centers": tensors["atom_text_center"][0, :count].detach().cpu(),
                "text_ids": sorted(text_ids),
                "residual_text_ids": sorted(text_ids - baseline_ids),
                "baseline_ids": sorted(baseline_ids),
                "baseline_points": selected["atom_geometry"][0, selected["atom_mask"][0], :2].detach().cpu(),
                "target_count": count,
            })
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)
    train, holdout = records[:TRAIN_IMAGES], records[TRAIN_IMAGES:]
    focus = [record for record in train if record["residual_text_ids"]]
    if len(focus) < 4:
        raise ValueError("Insufficient residual-text train images")
    feature_channels = int(train[0]["features"].shape[1])
    head = TextCenterOffsetHead(feature_channels).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        record = focus[(step - 1) % len(focus)]
        logits, delta = head(record["features"].to(device))
        h, w = logits.shape[-2:]
        positive = torch.zeros_like(logits, dtype=torch.bool)
        offset_targets = torch.zeros((1, 2, h, w), device=device)
        anchors = record["target_anchors"].to(device)
        centers = record["target_centers"].to(device)
        grid = _grid(h, w, device)
        for target_id in record["text_ids"]:
            x = min(w - 1, max(0, int(float(centers[target_id, 0]) * w)))
            y = min(h - 1, max(0, int(float(centers[target_id, 1]) * h)))
            positive[0, 0, y, x] = True
            target_delta = ((anchors[target_id] - grid[y, x]) / (OFFSET_RADIUS_PX / 768.0)).clamp(-1.0, 1.0)
            offset_targets[0, :, y, x] = target_delta
        focal = focal_heatmap_loss(logits, positive.float())
        rank = atom_center_global_peak_rank_loss(logits, positive, candidate_budget=128, boundary_window=16, margin=0.5)
        offset_loss = F.smooth_l1_loss(delta.tanh()[positive.expand(-1, 2, -1, -1)], offset_targets[positive.expand(-1, 2, -1, -1)]) if bool(positive.any()) else logits.sum() * 0.0
        loss = focal + rank + 4.0 * offset_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": record["sample_key"], "focal": float(focal.detach()), "rank": float(rank.detach()), "offset": float(offset_loss.detach()), "loss": float(loss.detach()), "gradient_norm": float(norm)})
    train_metrics = _evaluate(head, train, device)
    holdout_metrics = _evaluate(head, holdout, device)
    result = {
        "schema_version": "chemvector-text-center-offset-sidecar-0.1",
        "complete": True,
        "training_performed": True,
        "in_memory_only": True,
        "optimizer_steps": STEPS,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "protocol": {"train_images": TRAIN_IMAGES, "holdout_images": len(holdout), "focus_images": len(focus), "pool": POOL, "side_k": SIDE_K, "baseline_guard_px": BASELINE_GUARD_PX, "suppression_px": SUPPRESSION_PX, "offset_radius_px": OFFSET_RADIUS_PX, "parameter_count": sum(p.numel() for p in head.parameters())},
        "train": train_metrics,
        "holdout": holdout_metrics,
        "events": events,
        "gate": {"precision_floor": 0.85, "train_positive_min": 50, "bounded_training_admissible": False, "reason": "sidecar candidate precision must be audited with AUPRC and downstream ownership before integration"},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_metrics, "holdout": holdout_metrics}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
