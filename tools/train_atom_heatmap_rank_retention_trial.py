"""Bounded current-initial heatmap ranking trial with baseline retention.

Only the 129-parameter atom heatmap readout is updated in memory.  The trial
uses the existing dense target cells plus a global top-128 ranking loss, while
replaying initial raw peak logits so old endpoint ownership is protected.  It
never writes a production checkpoint or reads sealed identities.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

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
    "CHEMVECTOR_HEATMAP_RANK_TRIAL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-heatmap-rank-retention-trial-20261006"),
))
STEPS = int(os.environ.get("CHEMVECTOR_HEATMAP_RANK_STEPS", "200"))
TRAIN_IMAGES = int(os.environ.get("CHEMVECTOR_HEATMAP_RANK_TRAIN_IMAGES", "120"))
EVAL_EVERY = int(os.environ.get("CHEMVECTOR_HEATMAP_RANK_EVAL_EVERY", "100"))
LR = float(os.environ.get("CHEMVECTOR_HEATMAP_RANK_LR", "1e-4"))
FOCAL_WEIGHT = 1.0
RANK_WEIGHT = 1.0
RETENTION_WEIGHT = 2.0
RANK_MARGIN = 0.5
RETENTION_MARGIN = 0.1


def _ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(v) for v in mapping[0].tolist() if int(v) >= 0}


def _forward_record(model, config, payload, loader, manifest, row, collator, device) -> dict:
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        _, atom_output, _, raw, _, _, selected = structure_forward(
            model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
        )
    raw_ids = _ids(raw, tensors, config)
    selected_ids = _ids(selected, tensors, config)
    raw_cells = raw["atom_indices"][0, raw["atom_mask"][0]].detach().cpu()
    old_logits = atom_output["heatmap_logits"][0, 0].flatten()[raw_cells].detach().cpu()
    return {
        "row": row,
        "sample": sample,
        "tensors": {key: value.detach().cpu() for key, value in tensors.items()},
        "targets": int(tensors["atom_mask"].sum()),
        "raw_ids": sorted(raw_ids),
        "selected_ids": sorted(selected_ids),
        "raw_cells": raw_cells,
        "old_logits": old_logits,
    }


def _evaluate(model, config, payload, records, collator, device) -> dict:
    raw_matched = selected_matched = targets = 0
    raw_recovered = selected_recovered = 0
    raw_lost = selected_lost = 0
    by_profile: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        row = record["row"]
        sample = record["sample"]
        batch = collator([sample])
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.no_grad():
            _, _, _, raw, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
        raw_ids = _ids(raw, tensors, config)
        selected_ids = _ids(selected, tensors, config)
        old_raw, old_selected = set(record["raw_ids"]), set(record["selected_ids"])
        profile = str(sample["meta"]["profile_id"])
        targets += record["targets"]
        raw_matched += len(raw_ids)
        selected_matched += len(selected_ids)
        raw_recovered += len(raw_ids - old_raw)
        selected_recovered += len(selected_ids - old_selected)
        raw_lost += len(old_raw - raw_ids)
        selected_lost += len(old_selected - selected_ids)
        bucket = by_profile[profile]
        bucket["images"] += 1
        bucket["raw_recovered"] += len(raw_ids - old_raw)
        bucket["selected_recovered"] += len(selected_ids - old_selected)
        bucket["raw_lost"] += len(old_raw - raw_ids)
        bucket["selected_lost"] += len(old_selected - selected_ids)
    return {
        "images": len(records),
        "targets": targets,
        "raw_matched": raw_matched,
        "selected_matched": selected_matched,
        "raw_recovered_vs_initial": raw_recovered,
        "selected_recovered_vs_initial": selected_recovered,
        "raw_lost_vs_initial": raw_lost,
        "selected_lost_vs_initial": selected_lost,
        "by_profile": {key: dict(value) for key, value in sorted(by_profile.items())},
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    if not 1 <= STEPS <= 200 or not 1 <= TRAIN_IMAGES < 1200:
        raise ValueError("Trial limits are invalid")
    if EVAL_EVERY < 1 or EVAL_EVERY > STEPS:
        raise ValueError("Invalid evaluation interval")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    trainable = tuple(model.atom.heatmap.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    if sum(parameter.numel() for parameter in trainable) != 129:
        raise ValueError("Unexpected heatmap parameter count")
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )

    # Capture immutable initial ownership and raw-peak retention anchors before
    # any update.  Records store only CPU tensors and public cohort metadata.
    records = [_forward_record(model, config, payload, loader, manifest, row, collator, device) for row in rows]
    train_records = records[:TRAIN_IMAGES]
    holdout_records = records[TRAIN_IMAGES:]
    baseline_train = _evaluate(model, config, payload, train_records, collator, device)
    baseline_holdout = _evaluate(model, config, payload, holdout_records, collator, device)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    events: list[dict] = []
    checkpoints: list[dict] = []
    stop_reason = None
    for step in range(1, STEPS + 1):
        record = train_records[(step - 1) % len(train_records)]
        tensors = {key: value.to(device) for key, value in record["tensors"].items()}
        output = model.atom(model.vision.forward_stroke_resolution(tensors["pixel_values"]))
        positive = build_vertex_heatmap(
            tensors["atom_anchors"], tensors["atom_mask"],
            output["heatmap_logits"].shape[-2], output["heatmap_logits"].shape[-1], radius_cells=0,
        ).bool()
        focal = focal_heatmap_loss(output["heatmap_logits"], positive.float())
        rank = atom_center_global_peak_rank_loss(
            output["heatmap_logits"], positive, candidate_budget=128,
            boundary_window=16, margin=RANK_MARGIN,
        )
        cells = record["raw_cells"].to(device)
        old_logits = record["old_logits"].to(device)
        current_flat = output["heatmap_logits"][0, 0].flatten()
        if len(cells):
            retention = F.softplus(RETENTION_MARGIN + old_logits - current_flat[cells]).mean()
        else:
            retention = focal * 0.0
        loss = FOCAL_WEIGHT * focal + RANK_WEIGHT * rank + RETENTION_WEIGHT * retention
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite trial loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
            raise RuntimeError("Missing/nonfinite heatmap gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({
            "step": step,
            "sample_key": record["row"]["sample_key"],
            "loss": float(loss.detach()),
            "focal": float(focal.detach()),
            "rank": float(rank.detach()),
            "retention": float(retention.detach()),
            "gradient_norm": float(norm),
        })
        if step % EVAL_EVERY != 0 and step != STEPS:
            continue
        train_now = _evaluate(model, config, payload, train_records, collator, device)
        holdout_now = _evaluate(model, config, payload, holdout_records, collator, device)
        checkpoint = {"step": step, "train": train_now, "holdout": holdout_now}
        checkpoints.append(checkpoint)
        if train_now["selected_lost_vs_initial"] or holdout_now["selected_lost_vs_initial"]:
            stop_reason = "selected_baseline_id_lost"
            break

    final = checkpoints[-1] if checkpoints else {"step": 0, "train": baseline_train, "holdout": baseline_holdout}
    train = final["train"]
    holdout = final["holdout"]
    result = {
        "schema_version": "chemvector-heatmap-rank-retention-trial-0.1",
        "complete": True,
        "training_performed": True,
        "in_memory_only": True,
        "optimizer_steps": int(final["step"]),
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "protocol": {
            "train_images": TRAIN_IMAGES,
            "holdout_images": len(holdout_records),
            "maximum_steps": STEPS,
            "parameter_scope": ["atom.heatmap.weight", "atom.heatmap.bias"],
            "parameter_count": 129,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "focal_weight": FOCAL_WEIGHT,
            "rank_weight": RANK_WEIGHT,
            "retention_weight": RETENTION_WEIGHT,
            "rank_margin": RANK_MARGIN,
            "retention_margin": RETENTION_MARGIN,
        },
        "baseline": {"train": baseline_train, "holdout": baseline_holdout},
        "checkpoints": checkpoints,
        "events": events,
        "stop_reason": stop_reason,
        "gate": {
            "train_recovery_positive": train["raw_recovered_vs_initial"] > 0,
            "holdout_recovery_min14": holdout["raw_recovered_vs_initial"] >= 14,
            "train_selected_retention": train["selected_lost_vs_initial"] == 0,
            "holdout_selected_retention": holdout["selected_lost_vs_initial"] == 0,
            "bounded_trial_pass": (
                train["raw_recovered_vs_initial"] > 0
                and holdout["raw_recovered_vs_initial"] >= 14
                and train["selected_lost_vs_initial"] == 0
                and holdout["selected_lost_vs_initial"] == 0
            ),
            "deployment_allowed": False,
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": result["optimizer_steps"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
