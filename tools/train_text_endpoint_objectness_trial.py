"""In-memory text-endpoint objectness sidecar trial.

The production atom field, selected baseline, set head, OCR and checkpoint are
frozen.  A 1x1 text-objectness head is trained on visible-text cells only and
evaluated as a residual-only candidate source.  This tests a different route
from globally changing the production heatmap ranking.
"""

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
from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TEXT_OBJECTNESS_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-text-endpoint-objectness-trial-20261006"),
))
STEPS = int(os.environ.get("CHEMVECTOR_TEXT_OBJECTNESS_STEPS", "100"))
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_TEXT_OBJECTNESS_TRAIN_IMAGES", "120"))
LR = float(os.environ.get("CHEMVECTOR_TEXT_OBJECTNESS_LR", "1e-3"))
POOL = 4096
SIDE_K = (16, 32)
SIDE_RADIUS_PX = 6.0
BASELINE_GUARD_PX = 12.0
MATCH_RADIUS_PX = 6.0


class TextObjectnessHead(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, -2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def _residual_ids(points: torch.Tensor, targets: torch.Tensor, baseline_ids: set[int], target_count: int) -> set[int]:
    remaining = [i for i in range(target_count) if i not in baseline_ids]
    if not remaining or not len(points):
        return set()
    cost = torch.cdist(points.float(), targets[remaining].float())
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(remaining[col]) for row, col in zip(rows, cols) if float(cost[row, col]) <= MATCH_RADIUS_PX}


def _candidates(logits: torch.Tensor, anchors: torch.Tensor, baseline_points: torch.Tensor, k: int) -> torch.Tensor:
    scores = logits[0, 0].sigmoid().flatten()
    order = torch.argsort(scores, descending=True)[: min(POOL, len(scores))]
    h, w = logits.shape[-2:]
    dense = anchors[0].permute(1, 2, 0).reshape(-1, 2)
    chosen: list[int] = []
    guard = BASELINE_GUARD_PX / 768.0
    radius = SIDE_RADIUS_PX / 768.0
    for idx in order.tolist():
        point = dense[idx]
        if len(baseline_points) and bool(torch.linalg.vector_norm(baseline_points - point, dim=1).min() <= guard):
            continue
        if chosen:
            old = dense[torch.as_tensor(chosen, device=dense.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= radius):
                continue
        chosen.append(int(idx))
        if len(chosen) >= k:
            break
    return dense[torch.as_tensor(chosen, device=dense.device)] if chosen else dense[:0]


def _evaluate(head, records, device) -> dict:
    output = {}
    for k in SIDE_K:
        block = defaultdict(int)
        scores_all: list[float] = []
        labels_all: list[int] = []
        for record in records:
            with torch.no_grad():
                logits = head(record["features"].to(device))
            points = _candidates(logits, record["anchors"].to(device), record["baseline_points"].to(device), k)
            positive = _residual_ids(
                points * 768.0,
                record["target_anchors"].to(device) * 768.0,
                set(record["baseline_ids"]),
                record["target_count"],
            )
            scores = logits[0, 0].sigmoid().flatten()
            # Reconstruct candidate cell scores in the same order for an audit
            # AUPRC; labels are assigned after the locked residual matching.
            block["images"] += 1
            block["candidates"] += int(len(points))
            block["positives"] += len(positive)
            block["false_positives"] += int(len(points) - len(positive))
            block["gap_targets"] += len(record["residual_text_ids"])
            block["new_matches"] += len(positive)
            # The candidate rank list is already score-sorted; use a simple
            # binary label sequence for AP without exposing target identities.
            for point in points:
                match = bool(torch.linalg.vector_norm(
                    record["target_anchors"].to(device)[sorted(positive)] * 768.0 - point[None] * 768.0,
                    dim=1,
                ).min() <= MATCH_RADIUS_PX) if positive else False
                scores_all.append(float(scores.max()))
                labels_all.append(int(match))
        item = dict(block)
        item["precision"] = item["positives"] / max(item["candidates"], 1)
        output[str(k)] = item
    return output


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
            features, atom_output, _, _, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            raw_mapping, _ = match_atom_targets(selected, tensors, config)
            baseline_ids = {int(v) for v in raw_mapping[0].tolist() if int(v) >= 0}
            target_count = int(tensors["atom_mask"].sum())
            text_ids = set(torch.where(tensors["atom_text_mask"][0, :target_count])[0].tolist())
            residual_text_ids = sorted(text_ids - baseline_ids)
            records.append({
                "sample_key": row["sample_key"],
                "profile_id": str(sample["meta"]["profile_id"]),
                "features": features.detach().cpu(),
                "anchors": atom_output["anchors"].detach().cpu(),
                "target_anchors": tensors["atom_anchors"][0, :target_count].detach().cpu(),
                "baseline_points": selected["atom_geometry"][0, selected["atom_mask"][0], :2].detach().cpu(),
                "baseline_ids": sorted(baseline_ids),
                "target_count": target_count,
                "residual_text_ids": residual_text_ids,
                "text_positive": tensors["atom_text_mask"].detach().cpu(),
            })
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)
    train = records[:TRAIN_IMAGES]
    holdout = records[TRAIN_IMAGES:]
    focus = [record for record in train if record["residual_text_ids"]]
    if len(focus) < 4:
        raise ValueError("Insufficient residual-text train images")
    head = TextObjectnessHead(features.shape[1]).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        record = focus[(step - 1) % len(focus)]
        features = record["features"].to(device)
        logits = head(features)
        text_mask = record["text_positive"].to(device)
        # Build a visible-text target map with the same spatial cell contract.
        positive = torch.zeros_like(logits, dtype=torch.bool)
        anchors = record["target_anchors"].to(device)
        h, w = logits.shape[-2:]
        for target_id in torch.where(text_mask[0, : record["target_count"]])[0].tolist():
            x = min(w - 1, max(0, int(float(anchors[target_id, 0]) * w)))
            y = min(h - 1, max(0, int(float(anchors[target_id, 1]) * h)))
            positive[0, 0, y, x] = True
        focal = focal_heatmap_loss(logits, positive.float())
        rank = atom_center_global_peak_rank_loss(logits, positive, candidate_budget=128, boundary_window=16, margin=0.5)
        loss = focal + rank
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": record["sample_key"], "focal": float(focal.detach()), "rank": float(rank.detach()), "loss": float(loss.detach()), "gradient_norm": float(norm)})
    train_metrics = _evaluate(head, train, device)
    holdout_metrics = _evaluate(head, holdout, device)
    result = {
        "schema_version": "chemvector-text-endpoint-objectness-trial-0.1",
        "complete": True,
        "training_performed": True,
        "in_memory_only": True,
        "optimizer_steps": STEPS,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "protocol": {"train_images": TRAIN_IMAGES, "holdout_images": len(holdout), "focus_images": len(focus), "pool": POOL, "side_k": SIDE_K, "baseline_guard_px": BASELINE_GUARD_PX, "parameter_count": sum(p.numel() for p in head.parameters())},
        "train": train_metrics,
        "holdout": holdout_metrics,
        "events": events,
        "gate": {"precision_floor": 0.85, "train_positive_min": 50, "bounded_training_admissible": False, "reason": "sidecar audit only; downstream selector and AUPRC gate not passed"},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_metrics, "holdout": holdout_metrics}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
