#!/usr/bin/env python3
"""Bounded in-memory dense proposal-head shadow on the r5 quarantine panel.

The existing production atom/bond/set/OCR modules remain frozen and are never
merged with this head.  The run writes only a metrics receipt, not a model
checkpoint or deployment artifact.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from chemvector.core.model import ChemVectorConfig, ChemVertexDetector
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.vertex import vertex_detection_loss
from tools.identity_vertex_decoder import decode_dense


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/report.json"
BASELINE = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-baseline-20261017-r5-v3/report.json"
STATE_OUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/vertex-state.pt"
CANDIDATE_OUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/candidates.jsonl"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
STEPS = 800
BATCH_SIZE = 8
EVAL_EVERY = 100
TOLERANCE_PX = 6.0
SEED = 20261017


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_model_digest(model, trainable_prefixes: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def collate(samples: list[dict]) -> dict[str, torch.Tensor | list]:
    max_atoms = max(len(sample["graph"]["atoms"]) for sample in samples)
    pixels = torch.stack([image_tensor(sample["image"]) for sample in samples])
    anchors = torch.zeros((len(samples), max_atoms, 2), dtype=torch.float32)
    mask = torch.zeros((len(samples), max_atoms), dtype=torch.bool)
    aromatic = torch.zeros((len(samples), max_atoms), dtype=torch.float32)
    roles = torch.zeros((len(samples), max_atoms), dtype=torch.long)
    elements: list[list[str]] = []
    visible_elements: list[list[str | None]] = []
    keys = []
    for batch_index, sample in enumerate(samples):
        by_id = {a["atom_id"]: a for a in sample["layers"]["atoms"]}
        row_elements: list[str] = []
        row_visible: list[str | None] = []
        for atom in sorted(sample["graph"]["atoms"], key=lambda a: a["atom_id"]):
            atom_id = atom["atom_id"]
            visual = by_id[atom_id]
            anchors[batch_index, atom_id] = torch.tensor(visual["center"], dtype=torch.float32) / 768.0
            mask[batch_index, atom_id] = True
            aromatic[batch_index, atom_id] = float(atom["aromatic"])
            roles[batch_index, atom_id] = int(bool(visual["visible_text"]))
            row_elements.append(atom["element"])
            row_visible.append(atom["element"] if visual["visible_text"] else None)
        elements.append(row_elements)
        visible_elements.append(row_visible)
        keys.append(sample["key"])
    return {"pixel_values": pixels, "anchor_points": anchors, "anchor_mask": mask,
            "aromatic_targets": aromatic, "role_targets": roles, "elements": elements,
            "visible_elements": visible_elements, "keys": keys}


def rank_loss(logits: torch.Tensor, anchors: torch.Tensor, mask: torch.Tensor, margin: float = .5) -> torch.Tensor:
    """Global positive-vs-negative heatmap margin, with cell collisions masked."""
    b, _, h, w = logits.shape
    x = (anchors[..., 0] * w).clamp(0, w - 1e-4).long()
    y = (anchors[..., 1] * h).clamp(0, h - 1e-4).long()
    target = torch.zeros((b, h, w), device=logits.device, dtype=torch.bool)
    batch = torch.arange(b, device=logits.device)[:, None].expand_as(mask)
    target[batch[mask], y[mask], x[mask]] = True
    flat = logits[:, 0]
    losses = []
    for i in range(b):
        pos = flat[i][target[i]]
        neg = flat[i][~target[i]]
        if len(pos) and len(neg):
            hard = neg.topk(min(128, len(neg))).values
            losses.append(F.relu(margin - pos[:, None] + hard[None, :]).mean())
    return torch.stack(losses).mean() if losses else flat.sum() * 0


@torch.no_grad()
def evaluate(head, vision, rows: list[dict], device: torch.device, threshold: float = .05) -> dict:
    head.eval(); vision.eval()
    totals = Counter()
    strata = defaultdict(Counter)
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        batch = collate([load_sample(DATA, r["sample_key"], purpose="diagnostic") for r in batch_rows])
        pixels = batch["pixel_values"].to(device)
        with torch.inference_mode():
            features = vision.forward_stroke_resolution(pixels)
            output = head(features)
            decoded = decode_dense(output, threshold=threshold, maximum_vertices=128, nms_kernel_size=3)
        for bi, predictions in enumerate(decoded):
            targets = batch["anchor_points"][bi][batch["anchor_mask"][bi]].tolist()
            elements = batch["elements"][bi]
            visible_elements = batch["visible_elements"][bi]
            used_pred, used_target = set(), set()
            pairs = []
            for pi, pred in enumerate(predictions):
                for ti, target in enumerate(targets):
                    distance = ((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2) ** .5 * 768.0
                    if distance <= TOLERANCE_PX:
                        pairs.append((distance, pi, ti))
            for distance, pi, ti in sorted(pairs):
                if pi in used_pred or ti in used_target:
                    continue
                used_pred.add(pi); used_target.add(ti)
            target_neighbors = Counter(ti for _, _, ti in pairs)
            pred_neighbors = Counter(pi for _, pi, _ in pairs)
            strict_targets = {ti for ti, count in target_neighbors.items() if count == 1
                              and sum(1 for _, pi, target in pairs if target == ti and pred_neighbors[pi] == 1) == 1}
            strict_predictions = {pi for pi, count in pred_neighbors.items() if count == 1
                                  and sum(1 for _, pred, ti in pairs if pred == pi and target_neighbors[ti] == 1) == 1}
            totals["strict_matches"] += len(strict_targets)
            totals["strict_predictions"] += len(strict_predictions)
            totals["strict_targets"] += len(strict_targets)
            totals["predictions"] += len(predictions); totals["targets"] += len(targets); totals["matches"] += len(used_target)
            for element in ("O", "N"):
                target_indices = {i for i, value in enumerate(visible_elements) if value == element}
                matched = len(target_indices & used_target)
                strata[element]["targets"] += len(target_indices); strata[element]["matches"] += matched
                strata[element]["strict_matches"] += len(target_indices & strict_targets)
            target_indices = {i for i, value in enumerate(elements)
                              if value == "C" and visible_elements[i] is None}
            strata["implicit_C"]["targets"] += len(target_indices)
            strata["implicit_C"]["matches"] += len(target_indices & used_target)
            strata["implicit_C"]["strict_matches"] += len(target_indices & strict_targets)
            strata[batch_rows[bi]["profile_id"]]["targets"] += len(targets)
            strata[batch_rows[bi]["profile_id"]]["matches"] += len(used_target)
            strata[batch_rows[bi]["profile_id"]]["strict_matches"] += len(strict_targets)
    result = {"predictions": totals["predictions"], "targets": totals["targets"], "matches": totals["matches"],
              "precision": totals["matches"] / max(totals["predictions"], 1),
              "recall": totals["matches"] / max(totals["targets"], 1),
              "strict_predictions": totals["strict_predictions"], "strict_targets": totals["strict_targets"],
              "strict_precision": totals["strict_matches"] / max(totals["strict_predictions"], 1),
              "strict_recall": totals["strict_matches"] / max(totals["targets"], 1),
              "by_stratum": {k: {**dict(v), "recall": v["matches"] / max(v["targets"], 1),
                                  "strict_recall": v["strict_matches"] / max(v["targets"], 1)} for k, v in sorted(strata.items())}}
    return result


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("gate", {}).get("training_admissible", True) or baseline.get("gate", {}).get("checkpoint_unchanged") is not True:
        raise ValueError("baseline receipt is not a frozen read-only comparator")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    train = [r for r in rows if r["split"] == "train"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    train_ids = {r["molecule_id"] for r in train}; holdout_ids = {r["molecule_id"] for r in holdout}
    if train_ids & holdout_ids or not train or not holdout:
        raise ValueError("invalid identity split")
    for row in rows:
        if not row.get("connection_identity"):
            raise ValueError("empty connection identity")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    production_model, proposal_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available():
        raise RuntimeError("initial/CUDA binding failed")
    device = torch.device("cuda:0")
    production_model.to(device)
    production_model.eval().requires_grad_(False)
    production_before = state_digest(production_model)
    model = deepcopy(production_model)
    model.to(device)
    model.eval().requires_grad_(False)
    trainable_prefixes = ("vision.stages.0.1.",)
    before = frozen_model_digest(model, trainable_prefixes)
    vertex_config = ChemVectorConfig(vision_dims=(96, 192, 384, 512), vertex_feature_dim=64)
    head = ChemVertexDetector(vertex_config).to(device)
    for parameter in model.vision.parameters():
        parameter.requires_grad_(False)
    for parameter in model.vision.stages[0][-1].parameters():
        parameter.requires_grad_(True)
    head_params = list(head.parameters())
    block_params = [p for p in model.vision.stages[0][-1].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": 2e-4}, {"params": block_params, "lr": 1e-6}], weight_decay=0.0)
    collision_images = 0
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        cells = [(int(a["center"][0] / 4), int(a["center"][1] / 4)) for a in sample["layers"]["atoms"]]
        collision_images += int(len(cells) != len(set(cells)))
    if collision_images:
        raise ValueError(f"grid collision images must be excluded before shadow: {collision_images}")
    rng = random.Random(SEED)
    events = []
    started = time.perf_counter()
    cursor = 0
    for step in range(1, STEPS + 1):
        if cursor + BATCH_SIZE > len(train):
            rng.shuffle(train); cursor = 0
        batch_rows = train[cursor:cursor + BATCH_SIZE]; cursor += BATCH_SIZE
        batch = collate([load_sample(DATA, r["sample_key"], purpose="diagnostic") for r in batch_rows])
        pixels = batch["pixel_values"].to(device); anchors = batch["anchor_points"].to(device)
        mask = batch["anchor_mask"].to(device); aromatic = batch["aromatic_targets"].to(device); roles = batch["role_targets"].to(device)
        model.vision.train(); head.train()
        features = model.vision.forward_stroke_resolution(pixels)
        output = head(features)
        losses = vertex_detection_loss(output, anchors, mask, aromatic, roles, role_loss_weight=.5)
        losses["rank_loss"] = rank_loss(output["vertex_heatmap_logits"], anchors, mask)
        losses["loss"] = losses["loss"] + .5 * losses["rank_loss"]
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head_params + block_params, 1.0); optimizer.step()
        if step % EVAL_EVERY == 0:
            metrics = evaluate(head, model.vision, holdout, device)
            events.append({"step": step, "seconds": time.perf_counter() - started, "loss": float(losses["loss"].detach()), "holdout": metrics})
            print(json.dumps({"step": step, "loss": events[-1]["loss"], "holdout_recall": metrics["recall"], "holdout_precision": metrics["precision"]}), flush=True)
    model.eval().requires_grad_(False); model.zero_grad(set_to_none=True)
    after = frozen_model_digest(model, trainable_prefixes)
    checkpoint_after = digest(INITIAL)
    production_after = state_digest(production_model)
    if production_before != production_after or checkpoint_after != INITIAL_SHA:
        raise RuntimeError("frozen production state or checkpoint changed")
    final = events[-1]["holdout"] if events else {}
    train_final = evaluate(head, model.vision, train, device)
    holdout_final = final
    gates = {
        "holdout_visible_O_recall_ge_050": holdout_final.get("by_stratum", {}).get("O", {}).get("recall", 0.0) >= 0.50,
        "holdout_visible_N_recall_ge_050": holdout_final.get("by_stratum", {}).get("N", {}).get("recall", 0.0) >= 0.50,
        "holdout_all_atom_recall_ge_080": holdout_final.get("recall", 0.0) >= 0.80,
        "holdout_implicit_C_recall_ge_070": holdout_final.get("by_stratum", {}).get("implicit_C", {}).get("recall", 0.0) >= 0.70,
        "holdout_precision_ge_030": holdout_final.get("precision", 0.0) >= 0.30,
        "train_recall_ge_frozen_baseline": train_final.get("recall", 0.0) >= baseline["by_split"]["train"]["recall"],
        "holdout_recall_ge_frozen_baseline": holdout_final.get("recall", 0.0) >= baseline["by_split"]["holdout"]["recall"],
        "holdout_precision_ge_frozen_baseline": holdout_final.get("precision", 0.0) >= baseline["by_split"]["holdout"]["precision"],
    }
    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    if STATE_OUT.exists() or CANDIDATE_OUT.exists():
        raise FileExistsError("reviewed shadow artifacts already exist")
    torch.save({"schema_version": "chemvector-r5-vertex-state-0.1", "initial_sha256": INITIAL_SHA,
                "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "vision_stride4_block_state_dict": {k: v.detach().cpu() for k, v in model.vision.stages[0][-1].state_dict().items()}}, STATE_OUT)
    with CANDIDATE_OUT.open("x", encoding="utf-8") as handle:
        for row in rows:
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            pixels = image_tensor(sample["image"])[None].to(device)
            with torch.inference_mode():
                output = head(model.vision.forward_stroke_resolution(pixels))
                points = decode_dense(output, threshold=.05, maximum_vertices=128, nms_kernel_size=3, with_scores=True)[0]
            record = {"sample_key": row["sample_key"], "source": "shadow",
                      "candidates": [{"x": float(x), "y": float(y), "score": float(score)}
                                     for x, y, score in points]}
            json.dump(record, handle, ensure_ascii=False, sort_keys=True); handle.write("\n")
    report = {"schema_version": "chemvector-identity-aligned-vertex-head-shadow-0.1", "complete": True,
              "training_performed": True, "training_scope": "in_memory_new_vertex_head_only", "gpu_used": True,
              "optimizer_steps": STEPS, "sealed_test_read": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": checkpoint_after,
              "production_state_before": production_before, "production_state_after": production_after,
              "branch_frozen_state_before": before, "branch_frozen_state_after": after,
              "data_sha256": digest(DATA / "summary.json"), "train_images": len(train), "holdout_images": len(holdout),
              "baseline_sha256": digest(BASELINE),
              "vertex_state_sha256": digest(STATE_OUT), "candidate_artifact_sha256": digest(CANDIDATE_OUT),
              "head_parameters": sum(p.numel() for p in head_params), "stride4_block_parameters": sum(p.numel() for p in block_params),
              "collision_images": collision_images, "events": events, "final_train": train_final,
              "final_holdout": holdout_final, "gate": {"production_unchanged": production_before == production_after and checkpoint_after == INITIAL_SHA,
                       "collision_images_zero": collision_images == 0, **gates,
                       "all_shadow_metrics_pass": all(gates.values()), "training_admissible": False,
                       "next": "independent review; no integration or checkpoint promotion"}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "final_holdout": final, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
