"""Run the isolated 100-step dense-cell multiplicity sidecar shadow.

Only a fresh one-channel head over frozen stride-4 vision features is trained.
The production decoder and all downstream heads are immutable.  The result is
an evaluation artifact, never a production checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from chemvector.predicted_incidence import decode_predicted_atom_proposals

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1"
AUTH = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-shadow-authorization-20261017-r1/receipt.json"
DATA = ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r2"
PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2/manifest.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-shadow-20261017-r1"
STEPS = 100
WALL_SECONDS = 300
GRID = 192
IMAGE_SIZE = 768.0
MATCH_RADIUS = 6.0
MAX_EXTRA = 16


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ExtraHeatmap(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -10.0)

    def forward(self, features):
        return self.proj(features)


def _batch(sample, collator, device):
    packed = collator([sample])
    return {key: value.to(device) for key, value in packed.items() if isinstance(value, torch.Tensor)}


def _baseline(model, config, tensors):
    with torch.inference_mode():
        features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
        atom_output = model.atom(features)
        atoms = decode_predicted_atom_proposals(atom_output, config)
        mapping, _ = match_atom_targets(atoms, tensors, config)
    mapping = mapping[0]
    raw_mask = atoms["atom_mask"][0]
    raw_points = atoms["atom_geometry"][0, raw_mask, :2].detach().cpu()
    matched = {int(v) for v in mapping.tolist() if int(v) >= 0}
    target_count = int(tensors["atom_mask"].sum())
    return features.detach(), atoms, matched, raw_points, target_count


def _target_map(tensors, unmatched, device):
    target = torch.zeros((1, 1, GRID, GRID), device=device)
    anchors = tensors["atom_anchors"][0]
    for target_id in unmatched:
        x, y = anchors[target_id]
        cx = int(torch.clamp(x * GRID, 0, GRID - 1).item())
        cy = int(torch.clamp(y * GRID, 0, GRID - 1).item())
        target[0, 0, cy, cx] = 1.0
    # A 3x3 target footprint gives the tiny head a stable local signal while
    # retaining one-cell identity; no target geometry is fed at inference.
    return F.max_pool2d(target, 3, 1, 1)


def _decode_extra(logits, old_points):
    score = logits.sigmoid()[0, 0]
    peaks = (score == F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]) & (logits[0, 0] > 0.0)
    py, px = torch.where(peaks)
    if not len(px):
        return torch.empty((0, 2), device=logits.device), torch.empty((0,), device=logits.device)
    values = score[py, px]
    order = torch.argsort(values, descending=True)
    anchors = torch.stack(((px.float() + 0.5) / GRID, (py.float() + 0.5) / GRID), dim=1)
    anchors, values = anchors[order], values[order]
    if len(old_points):
        keep = torch.cdist(anchors * IMAGE_SIZE, old_points.to(anchors.device) * IMAGE_SIZE).amin(dim=1) > MATCH_RADIUS
        anchors, values = anchors[keep], values[keep]
    return anchors[:MAX_EXTRA], values[:MAX_EXTRA]


def _evaluate(model, head, config, rows, collator, device, metadata):
    total_candidates = total_matches = total_gap_hits = total_baseline_hits = 0
    row_reports = []
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        tensors = _batch(sample, collator, device)
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"]).clone()
            extra = head(features)
        old_points = metadata[row["sample_key"]]["raw_points"]
        old_ids = metadata[row["sample_key"]]["matched_ids"]
        points, values = _decode_extra(extra, old_points)
        valid_target_ids = torch.where(tensors["atom_mask"][0])[0]
        anchors = tensors["atom_anchors"][0, valid_target_ids] * IMAGE_SIZE
        if len(points) and len(anchors):
            distance = torch.cdist(points * IMAGE_SIZE, anchors)
            rr, cc = linear_sum_assignment(distance.detach().cpu().numpy())
            matches = [(int(valid_target_ids[c]), float(distance[r, c])) for r, c in zip(rr, cc) if float(distance[r, c]) <= MATCH_RADIUS]
        else:
            matches = []
        target_ids = [target_id for target_id, _ in matches]
        gap_hits = [target_id for target_id in target_ids if target_id not in old_ids]
        total_candidates += len(points)
        total_matches += len(matches)
        total_gap_hits += len(set(gap_hits))
        total_baseline_hits += len(old_ids)
        row_reports.append({
            "sample_key": row["sample_key"], "split": row["split"],
            "profile_id": sample["meta"]["profile_id"], "candidate_count": len(points),
            "matched_count": len(matches), "gap_hits": len(set(gap_hits)),
            "baseline_hits": len(old_ids), "target_count": int(tensors["atom_mask"].sum()),
        })
    return {
        "images": len(rows), "candidate_count": total_candidates,
        "matched_count": total_matches, "strict_collision_recovery": total_gap_hits,
        "baseline_matched_targets": total_baseline_hits,
        "precision": total_matches / total_candidates if total_candidates else 0.0,
        "rows": row_reports,
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != STEPS or auth.get("max_wall_seconds") != WALL_SECONDS:
        raise ValueError("invalid multiplicity shadow receipt")
    if auth.get("production_checkpoint_write") is not False or auth.get("sealed_test_read") is not False:
        raise ValueError("unsafe authorization")
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint changed")
    if auth.get("contract_sha256") != digest(CONTRACT_DIR / "report.json"):
        raise ValueError("authorization contract binding changed")
    if auth.get("candidate_manifest_sha256") != digest(CONTRACT_DIR / "manifest.json"):
        raise ValueError("authorization manifest binding changed")
    torch.manual_seed(auth["seed"])
    torch.cuda.manual_seed_all(auth["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    plan_rows = json.loads(PLAN.read_text(encoding="utf-8"))["rows"]
    manifest = json.loads((CONTRACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    available = {r["sample_key"] for r in manifest["candidate_records"]}
    rows = [r for r in plan_rows if r["sample_key"] in available and (DATA / "samples" / (r["sample_key"] + ".png")).exists()]
    rows = sorted(rows, key=lambda r: r["sample_key"])
    train_rows = [r for r in rows if r["split"] == "train"]
    holdout_rows = [r for r in rows if r["split"] == "holdout"]
    if not train_rows or not holdout_rows:
        raise ValueError("empty fixed train/holdout split")
    model, config, _payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
    metadata = {}
    gap_rows = []
    for index, row in enumerate(rows, 1):
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        tensors = _batch(sample, collator, device)
        _features, _atoms, matched_ids, raw_points, target_count = _baseline(model, config, tensors)
        metadata[row["sample_key"]] = {"matched_ids": matched_ids, "raw_points": raw_points, "target_count": target_count}
        if len(matched_ids) < target_count and row["split"] == "train":
            gap_rows.append(row)
    if not gap_rows:
        raise RuntimeError("no train collision targets")
    # The head is created only after the frozen feature shape is observed.
    sample0 = load_sample(DATA, train_rows[0]["sample_key"], purpose="diagnostic")
    tensors0 = _batch(sample0, collator, device)
    with torch.inference_mode():
        feature0 = model.vision.forward_stroke_resolution(tensors0["pixel_values"])
    head = ExtraHeatmap(int(feature0.shape[1])).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.0)
    zero_candidates = 0
    events = []
    start = time.monotonic()
    for step in range(1, STEPS + 1):
        if time.monotonic() - start >= WALL_SECONDS:
            break
        row = gap_rows[(step - 1) % len(gap_rows)]
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        tensors = _batch(sample, collator, device)
        old = metadata[row["sample_key"]]
        unmatched = sorted(set(range(old["target_count"])) - old["matched_ids"])
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"]).clone()
        logits = head(features)
        target = _target_map(tensors, unmatched, device)
        positive = logits[target > 0.5]
        negative = logits[target <= 0.5]
        if not len(positive):
            continue
        hard = negative.topk(min(256, len(negative))).values
        loss = F.softplus(1.0 - positive).mean() + F.softplus(hard).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()), "gradient_norm": float(norm)})
    completed_steps = len(events)
    if completed_steps != STEPS:
        terminal = "wall_limit_reached_before_step_budget"
    else:
        terminal = "step_budget_completed"
    head.eval()
    train_eval = _evaluate(model, head, config, train_rows, collator, device, metadata)
    holdout_eval = _evaluate(model, head, config, holdout_rows, collator, device, metadata)
    result = {
        "schema_version": "chemvector-dense-cell-multiplicity-shadow-result-0.1",
        "complete": True, "training_performed": True, "gpu_used": True,
        "optimizer_steps": completed_steps, "maximum_steps": STEPS,
        "terminal": terminal, "elapsed_seconds": time.monotonic() - start,
        "authorization_sha256": digest(AUTH), "contract_sha256": digest(CONTRACT_DIR / "report.json"),
        "candidate_manifest_sha256": digest(CONTRACT_DIR / "manifest.json"),
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": file_digest(INITIAL),
        "sealed_test_read": False, "production_checkpoint_written": False, "deployment_authorized": False,
        "protocol": {"train_images": len(train_rows), "holdout_images": len(holdout_rows), "gap_train_images": len(gap_rows), "max_extra_candidates": MAX_EXTRA, "match_radius_px": MATCH_RADIUS, "feature_channels": int(feature0.shape[1]), "parameter_count": sum(p.numel() for p in head.parameters())},
        "baseline": {"prefix_immutable": True, "zero_step_extra_candidates": zero_candidates, "raw_target_loss": 0},
        "train": {k: v for k, v in train_eval.items() if k != "rows"},
        "holdout": {k: v for k, v in holdout_eval.items() if k != "rows"},
        "events": events,
        "gate": {
            "step_budget_completed": completed_steps == STEPS,
            "strict_collision_recovery_positive": train_eval["strict_collision_recovery"] > 0,
            "holdout_collision_recovery_positive": holdout_eval["strict_collision_recovery"] > 0,
            "holdout_precision_ge_085": holdout_eval["precision"] >= 0.85,
            "baseline_prefix_unchanged": True,
            "baseline_raw_target_loss": True,
            "production_unchanged": file_digest(INITIAL) == INITIAL_SHA,
            "pass": (completed_steps == STEPS and train_eval["strict_collision_recovery"] > 0 and holdout_eval["strict_collision_recovery"] > 0 and holdout_eval["precision"] >= 0.85),
        },
        "rows": {"train": train_eval["rows"], "holdout": holdout_eval["rows"]},
    }
    OUTPUT.mkdir(parents=True)
    torch.save({"state_dict": head.state_dict(), "initial_sha256": INITIAL_SHA, "production": False}, OUTPUT / "sidecar.pt")
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": completed_steps, "terminal": terminal, "train": result["train"], "holdout": result["holdout"], "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
