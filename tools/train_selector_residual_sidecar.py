"""Bounded independent selector sidecar for raw-matched selector drops.

Only a small scorer over frozen raw-candidate features is trained in memory.
The production set-head, selected prefix, atom decoder, vision, pair/OCR and
chemistry heads remain frozen.  Unknown candidate rows are excluded from the
loss and reported separately.  No sidecar candidate is sent to production
graph inference.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import torch
from torch import nn
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


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_SELECTOR_SIDECAR_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-selector-residual-sidecar-20261006-r1"),
))
MAX_STEPS = 200
MAX_NEW_PER_IMAGE = 16
MATCH_RADIUS_PX = 6.0
PRECISION_FLOOR = 0.85
THRESHOLDS = tuple(i / 100 for i in range(20, 100, 5))


class ResidualSelector(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _row_features(model, config, payload, loader, manifest, collator, row, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        _, _, _, raw, _, set_output, selected = structure_forward(
            model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
        )
        raw_mapping, _ = match_atom_targets(raw, tensors, config)
        selected_mapping, _ = match_atom_targets(selected, tensors, config)
        raw_values = raw_mapping[0].detach().cpu().tolist()
        selected_targets = {int(value) for value in selected_mapping[0].detach().cpu().tolist() if int(value) >= 0}
        selected_mask = selected["atom_mask"][0].detach().cpu().bool()
        raw_mask = raw["atom_mask"][0].detach().cpu().bool()
        target_count = int(tensors["atom_mask"].sum())
        anchors = tensors["atom_anchors"][0, :target_count, :2].detach().cpu() * 768.0
        atom_features = raw["atom_features"][0].detach().cpu()
        xy = raw["atom_geometry"][0, :, :2].detach().cpu()
        raw_scores = raw["atom_scores"][0].detach().cpu()
        presence = set_output["atom_presence_logits"][0].detach().cpu()
        pre_presence = set_output["atom_pre_presence_logits"][0].detach().cpu()
    rows = []
    for slot, target in enumerate(raw_values):
        if not bool(raw_mask[slot]) or bool(selected_mask[slot]):
            continue  # Existing selected prefix is immutable and not rescored.
        target = int(target)
        if target >= 0 and target not in selected_targets:
            label = 1  # strict raw-matched selector drop
        elif target >= 0:
            continue
        else:
            point = xy[slot] * 768.0
            nearest = float(torch.linalg.vector_norm(anchors - point, dim=1).min()) if len(anchors) else float("inf")
            label = 0 if nearest > MATCH_RADIUS_PX else -1  # ambiguous duplicate = unknown
        feature = torch.cat((
            atom_features[slot].float(),
            raw_scores[slot].reshape(1).float(), xy[slot].float(),
            presence[slot].reshape(1).float(), pre_presence[slot].reshape(1).float(),
        ))
        rows.append({"feature": feature, "label": label, "sample_key": row["sample_key"],
                     "style": str(sample["meta"]["profile_id"]), "target": target})
    return rows


def _metrics(items, scorer, mean, std, threshold, device):
    selected = positives = negatives = unknown = available = 0
    by_style = Counter()
    for item in items:
        if not item:
            continue
        features = torch.stack([r["feature"] for r in item]).to(device)
        labels = torch.tensor([r["label"] for r in item], device=device)
        probabilities = scorer((features - mean) / std).sigmoid()
        chosen = torch.where(probabilities >= threshold)[0]
        if len(chosen) > MAX_NEW_PER_IMAGE:
            chosen = chosen[torch.argsort(probabilities[chosen], descending=True)[:MAX_NEW_PER_IMAGE]]
        positives += int((labels[chosen] == 1).sum())
        negatives += int((labels[chosen] == 0).sum())
        unknown += int((labels[chosen] < 0).sum())
        available += int((labels == 1).sum())
        selected += int(len(chosen))
        if len(chosen):
            by_style[item[0]["style"]] += int((labels[chosen] == 1).sum())
    return {"selected": selected, "positives": positives, "confirmed_negatives": negatives,
            "unknown_selected": unknown, "available_positives": available,
            "known_precision": positives / max(positives + negatives, 1),
            "conservative_precision": positives / max(selected, 1),
            "recall": positives / max(available, 1), "new_by_style": dict(by_style)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks(); manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys), key=lambda r: r["sample_key"])
    train_rows, holdout_rows, split = split_group_disjoint(rows, 480)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    train_items, holdout_items = [], []
    for index, row in enumerate(train_rows + holdout_rows, 1):
        values = _row_features(model, config, payload, loader, manifest, collator, row, device)
        (train_items if index <= len(train_rows) else holdout_items).append(values)
        if index % 200 == 0:
            print(json.dumps({"images": index}), flush=True)
    train_flat = [value for item in train_items for value in item]
    holdout_flat = [value for item in holdout_items for value in item]
    known_train = [value for value in train_flat if value["label"] >= 0]
    if not known_train or sum(value["label"] == 1 for value in known_train) < 50:
        raise RuntimeError("Selector sidecar lacks 50 known train positives")
    matrix = torch.stack([value["feature"] for value in known_train]).to(device)
    mean = matrix.mean(0); std = matrix.std(0).clamp_min(1e-5)
    x = (matrix - mean) / std
    y = torch.tensor([value["label"] for value in known_train], device=device).float()
    scorer = ResidualSelector(x.shape[1]).to(device).train()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3, weight_decay=0.0)
    pos_weight = ((y == 0).sum() / y.sum()).clamp(1.0, 20.0)
    events = []
    for step in range(1, MAX_STEPS + 1):
        logits = scorer(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite selector sidecar loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(scorer.parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (1, MAX_STEPS):
            events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    scorer.eval()
    grid = []
    for threshold in THRESHOLDS:
        metric = _metrics(train_items, scorer, mean, std, threshold, device)
        metric["threshold"] = threshold
        grid.append(metric)
    eligible = [row for row in grid if row["conservative_precision"] >= PRECISION_FLOOR and row["positives"] > 0]
    calibration = max(eligible, key=lambda row: (row["recall"], -row["threshold"])) if eligible else None
    holdout = _metrics(holdout_items, scorer, mean, std, calibration["threshold"], device) if calibration else None
    train_metric = calibration
    result = {
        "schema_version": "chemvector-selector-residual-sidecar-0.1",
        "complete": True, "training_performed": True, "in_memory_only": True,
        "optimizer_steps": MAX_STEPS, "parameter_count": sum(p.numel() for p in scorer.parameters()),
        "initial_sha256": INITIAL_SHA, "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "protocol": {"train_images": len(train_rows), "holdout_images": len(holdout_rows),
                      "identity_split": split, "max_new_per_image": MAX_NEW_PER_IMAGE,
                      "unknown_excluded_from_loss": True, "precision_floor": PRECISION_FLOOR},
        "candidate_counts": {"train": len(train_flat), "holdout": len(holdout_flat),
                             "train_known": len(known_train),
                             "train_positive": sum(v["label"] == 1 for v in train_flat),
                             "holdout_positive": sum(v["label"] == 1 for v in holdout_flat),
                             "train_unknown": sum(v["label"] < 0 for v in train_flat),
                             "holdout_unknown": sum(v["label"] < 0 for v in holdout_flat)},
        "calibration": calibration, "train": train_metric, "holdout": holdout,
        "threshold_grid": grid, "events": events,
        "gate": {
            "train_precision_floor": calibration is not None,
            "holdout_new_positive": bool(holdout and holdout["positives"] >= 5),
            "holdout_conservative_precision": bool(holdout and holdout["conservative_precision"] >= PRECISION_FLOOR),
            "old_selected_retention": True,
            "fixed_dev_unchanged_without_integration": True,
            "pass": bool(calibration and holdout and holdout["positives"] >= 5
                         and holdout["conservative_precision"] >= PRECISION_FLOOR),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "candidate_counts": result["candidate_counts"],
                      "calibration": calibration, "holdout": holdout, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
