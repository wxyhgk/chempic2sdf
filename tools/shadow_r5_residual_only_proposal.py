#!/usr/bin/env python3
"""One bounded residual-only proposal shadow for the reviewed r5 slice.

The production checkpoint, selector, bond/OCR heads, and graph path remain
frozen.  A fresh dense vertex head is trained only against atoms not owned by
the frozen baseline.  Baseline ownership and all target labels are used only
in memory for supervision/audit; the emitted candidate artifact has no target
fields.  This script writes only sidecar artifacts under a new audit folder.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.core.model import ChemVectorConfig, ChemVertexDetector
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.vertex import vertex_detection_loss
from tools.identity_vertex_decoder import decode_dense

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow"
STATE_OUT = OUT / "sidecar-state.pt"
CANDIDATE_OUT = OUT / "candidates.jsonl"
REPORT_OUT = OUT / "report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
STEPS = 800
BATCH_SIZE = 8
EVAL_EVERY = 100
TARGET_RADIUS_PX = 6.0
DUPLICATE_RADIUS_PX = 6.0
SEED = 20261018


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_model_digest(model, trainable_prefixes: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _target(sample: dict) -> tuple[list[dict], torch.Tensor, torch.Tensor, list[str], list[bool]]:
    atoms = sorted(sample["graph"]["atoms"], key=lambda x: x["atom_id"])
    layer = {x["atom_id"]: x for x in sample["layers"]["atoms"]}
    points = torch.tensor([layer[x["atom_id"]]["center"] for x in atoms], dtype=torch.float32) / 768.0
    bonds = torch.tensor([[int(x["begin_atom"]), int(x["end_atom"])] for x in sample["graph"]["bonds"]], dtype=torch.long)
    elements = [str(x.get("element", "other")) for x in atoms]
    visible = [bool(layer[x["atom_id"]].get("visible_text", False)) for x in atoms]
    return atoms, points, bonds, elements, visible


def _baseline_points(selected: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = selected["atom_mask"][0].bool()
    return selected["atom_geometry"][0, mask, :2].detach().cpu()


def _cache_baseline(model, config, payload, rows: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    cache = {}
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            pixels = image_tensor(sample["image"])[None].to(device)
            out = structure_forward(model, pixels, config, payload["provenance"]["atom_threshold"])
            cache[row["sample_key"]] = _baseline_points(out[6])
            if index % 100 == 0:
                print(json.dumps({"baseline_cached": index}), flush=True)
    return cache


def _records(rows: list[dict], baseline: dict[str, torch.Tensor]) -> list[dict]:
    result = []
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        atoms, points, bonds, elements, visible = _target(sample)
        base = baseline[row["sample_key"]]
        distance = torch.cdist(points * 768.0, base * 768.0) if len(base) else torch.full((len(points), 0), float("inf"))
        owned = distance.min(dim=1).values <= TARGET_RADIUS_PX if len(base) else torch.zeros(len(points), dtype=torch.bool)
        residual = ~owned
        max_atoms = len(atoms)
        anchors = torch.zeros((max_atoms, 2), dtype=torch.float32)
        mask = torch.zeros(max_atoms, dtype=torch.bool)
        aromatic = torch.zeros(max_atoms, dtype=torch.float32)
        roles = torch.zeros(max_atoms, dtype=torch.long)
        for index, atom in enumerate(atoms):
            anchors[index] = points[index]
            if residual[index]:
                mask[index] = True
            aromatic[index] = float(atom.get("aromatic", False))
            roles[index] = int(visible[index])
        # Baseline points/collision cells are retained in memory only.
        result.append({"row": row, "sample": sample, "anchors": anchors, "mask": mask,
                       "aromatic": aromatic, "roles": roles, "points": points, "bonds": bonds,
                       "elements": elements, "visible": visible, "owned": owned,
                       "baseline": base})
    return result


def _collate(items: list[dict]) -> dict[str, torch.Tensor | list]:
    n = len(items); max_atoms = max(len(x["anchors"]) for x in items)
    pixels = torch.stack([image_tensor(x["sample"]["image"]) for x in items])
    anchors = torch.zeros((n, max_atoms, 2), dtype=torch.float32)
    mask = torch.zeros((n, max_atoms), dtype=torch.bool)
    aromatic = torch.zeros((n, max_atoms), dtype=torch.float32)
    roles = torch.zeros((n, max_atoms), dtype=torch.long)
    baselines: list[torch.Tensor] = []
    for bi, item in enumerate(items):
        m = len(item["anchors"]); anchors[bi, :m] = item["anchors"]; mask[bi, :m] = item["mask"]
        aromatic[bi, :m] = item["aromatic"]; roles[bi, :m] = item["roles"]; baselines.append(item["baseline"])
    return {"pixel_values": pixels, "anchor_points": anchors, "anchor_mask": mask,
            "aromatic_targets": aromatic, "role_targets": roles, "baselines": baselines,
            "items": items}


def _rank_loss(logits: torch.Tensor, anchors: torch.Tensor, mask: torch.Tensor, margin: float = .5) -> torch.Tensor:
    b, _, h, w = logits.shape
    x = (anchors[..., 0] * w).clamp(0, w - 1e-4).long(); y = (anchors[..., 1] * h).clamp(0, h - 1e-4).long()
    target = torch.zeros((b, h, w), device=logits.device, dtype=torch.bool)
    batch = torch.arange(b, device=logits.device)[:, None].expand_as(mask)
    if mask.any(): target[batch[mask], y[mask], x[mask]] = True
    flat = logits[:, 0]; losses = []
    for i in range(b):
        pos, neg = flat[i][target[i]], flat[i][~target[i]]
        if len(pos) and len(neg):
            hard = neg.topk(min(128, len(neg))).values
            losses.append(F.relu(margin - pos[:, None] + hard[None, :]).mean())
    return torch.stack(losses).mean() if losses else flat.sum() * 0


def _collision_loss(logits: torch.Tensor, baselines: list[torch.Tensor]) -> torch.Tensor:
    # Penalize high residual heat at protected baseline cells.  This uses only
    # frozen baseline geometry and is deliberately detached from GT labels.
    values = []
    _, _, h, w = logits.shape
    for bi, points in enumerate(baselines):
        if not len(points): continue
        x = (points[:, 0] * w).clamp(0, w - 1e-4).long(); y = (points[:, 1] * h).clamp(0, h - 1e-4).long()
        values.append(F.softplus(logits[bi, 0, y, x] + 2.0).mean())
    return torch.stack(values).mean() if values else logits.sum() * 0


def _decode(head, vision, item, device: torch.device) -> list[tuple[float, float, float]]:
    with torch.inference_mode():
        output = head(vision.forward_stroke_resolution(image_tensor(item["sample"]["image"])[None].to(device)))
        return decode_dense(output, threshold=.05, maximum_vertices=128, nms_kernel_size=3, with_scores=True)[0]


def _metrics(head, vision, records: list[dict], device: torch.device) -> dict:
    total = Counter(); strata: dict[str, Counter] = defaultdict(Counter); profiles: dict[str, Counter] = defaultdict(Counter)
    for item in records:
        predictions = _decode(head, vision, item, device)
        pred = torch.tensor([[x[0], x[1]] for x in predictions], dtype=torch.float32) if predictions else torch.empty((0, 2))
        scores = [x[2] for x in predictions]
        baseline = item["baseline"]
        # Residual output cannot claim a protected baseline location.
        if len(pred) and len(baseline):
            residual_pred = torch.cdist(pred * 768.0, baseline * 768.0).amin(dim=1) > DUPLICATE_RADIUS_PX
        else:
            residual_pred = torch.ones(len(pred), dtype=torch.bool)
        pred_res = pred[residual_pred]; score_res = [s for s, keep in zip(scores, residual_pred.tolist()) if keep]
        targets = item["points"]; residual_target = ~item["owned"]
        d = torch.cdist(pred_res * 768.0, targets * 768.0) if len(pred_res) else torch.full((0, len(targets)), float("inf"))
        pairs = []
        for pi, ti in zip(*torch.where(d <= TARGET_RADIUS_PX)):
            if residual_target[int(ti)]: pairs.append((float(d[pi, ti]), int(pi), int(ti)))
        used_p, used_t = set(), set()
        for _, pi, ti in sorted(pairs):
            if pi not in used_p and ti not in used_t: used_p.add(pi); used_t.add(ti)
        # Candidate collision metric is measured on decoded residual rows.
        collision = 0
        if len(pred_res) and len(baseline):
            collision = int((torch.cdist(pred_res * 768.0, baseline * 768.0).amin(dim=1) <= DUPLICATE_RADIUS_PX).sum())
        # Residual atom status per target, and bond upper-bound with the
        # residual predictions after strict matching.
        pred_owned_targets = set(used_t)
        baseline_owned = item["owned"].tolist()
        combined = [baseline_owned[i] or i in pred_owned_targets for i in range(len(targets))]
        residual_hit = [i in pred_owned_targets for i in range(len(targets))]
        bond_rows = [(int(a), int(b)) for a, b in item["bonds"].tolist()] if len(item["bonds"]) else []
        block = Counter({"target_atoms": len(targets), "residual_targets": int(residual_target.sum()), "residual_predictions": len(pred_res), "residual_matches": len(used_t), "residual_collision": collision, "target_bonds": len(bond_rows), "combined_bonds": sum(combined[a] and combined[b] for a, b in bond_rows), "one_residual_bonds": sum(residual_hit[a] or residual_hit[b] for a, b in bond_rows), "both_residual_bonds": sum(residual_hit[a] and residual_hit[b] for a, b in bond_rows)})
        for key, value in block.items(): total[key] += value
        profile = profiles[item["row"]["profile_id"]]; profile.update(block)
        for i, element in enumerate(item["elements"]):
            key = element if element in {"O", "N"} else ("implicit_C" if element == "C" and not item["visible"][i] else "other")
            strata[key]["target_atoms"] += 1; strata[key]["residual_targets"] += int(residual_target[i]); strata[key]["residual_matches"] += int(i in pred_owned_targets)
    def finish(c: Counter) -> dict:
        out = dict(c); rt=max(c["residual_targets"],1); rp=max(c["residual_predictions"],1); tb=max(c["target_bonds"],1)
        out.update({"residual_precision": c["residual_matches"]/rp, "residual_recall": c["residual_matches"]/rt, "hard_collision_rate": c["residual_collision"]/rp, "combined_bond_coverage": c["combined_bonds"]/tb, "one_residual_bond_coverage": c["one_residual_bonds"]/tb, "both_residual_bond_coverage": c["both_residual_bonds"]/tb})
        return out
    return {"aggregate": finish(total), "by_profile": {k: finish(v) for k,v in sorted(profiles.items())}, "by_element": {k: {**dict(v), "residual_recall": v["residual_matches"]/max(v["residual_targets"],1)} for k,v in sorted(strata.items())}}


def run() -> dict:
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA: raise RuntimeError("initial checkpoint digest changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    rows = [x for x in rows if x.get("render_status") == "accepted"]
    train_rows = sorted([x for x in rows if x["split"] == "train"], key=lambda x: x["sample_key"])[:160]
    hold_rows = sorted([x for x in rows if x["split"] == "holdout"], key=lambda x: x["sample_key"])[:80]
    if len(train_rows) != 160 or len(hold_rows) != 80: raise RuntimeError("fixed r5 slice changed")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED)
    device = torch.device("cuda:0")
    production, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA); production.to(device).eval().requires_grad_(False)
    production_before = state_digest(production)
    all_rows = train_rows + hold_rows
    baseline = _cache_baseline(production, config, payload, all_rows, device)
    train = _records(train_rows, baseline); hold = _records(hold_rows, baseline)
    model = deepcopy(production).to(device).eval().requires_grad_(False)
    trainable_prefixes = ("vision.stages.0.1.",); frozen_before = frozen_model_digest(model, trainable_prefixes)
    vconfig = ChemVectorConfig(vision_dims=(96, 192, 384, 512), vertex_feature_dim=64)
    head = ChemVertexDetector(vconfig).to(device)
    for p_ in model.vision.stages[0][-1].parameters(): p_.requires_grad_(True)
    head_params = list(head.parameters()); block_params = [p_ for p_ in model.vision.stages[0][-1].parameters() if p_.requires_grad]
    optim = torch.optim.AdamW([{"params": head_params, "lr": 2e-4}, {"params": block_params, "lr": 1e-6}], weight_decay=0.0)
    rng = random.Random(SEED); cursor=0; events=[]; started=time.perf_counter()
    for step in range(1, STEPS+1):
        if cursor+BATCH_SIZE > len(train): rng.shuffle(train); cursor=0
        batch_items=train[cursor:cursor+BATCH_SIZE]; cursor+=BATCH_SIZE; batch=_collate(batch_items)
        pixels=batch["pixel_values"].to(device); anchors=batch["anchor_points"].to(device); mask=batch["anchor_mask"].to(device); aromatic=batch["aromatic_targets"].to(device); roles=batch["role_targets"].to(device)
        model.vision.train(); head.train(); output=head(model.vision.forward_stroke_resolution(pixels))
        losses=vertex_detection_loss(output,anchors,mask,aromatic,roles,role_loss_weight=.5); losses["rank_loss"]=_rank_loss(output["vertex_heatmap_logits"],anchors,mask); losses["collision_loss"]=_collision_loss(output["vertex_heatmap_logits"],batch["baselines"])
        losses["loss"]=losses["loss"] + .5*losses["rank_loss"] + .1*losses["collision_loss"]
        if not torch.isfinite(losses["loss"]): raise RuntimeError(f"nonfinite loss at step {step}")
        optim.zero_grad(set_to_none=True); losses["loss"].backward(); torch.nn.utils.clip_grad_norm_(head_params+block_params,1.0); optim.step()
        if step % EVAL_EVERY == 0:
            ev={"step":step,"seconds":time.perf_counter()-started,"loss":float(losses["loss"].detach()),"holdout":_metrics(head,model.vision,hold,device)};events.append(ev);print(json.dumps({"step":step,"loss":ev["loss"],"residual_recall":ev["holdout"]["aggregate"]["residual_recall"],"residual_precision":ev["holdout"]["aggregate"]["residual_precision"]}),flush=True)
    model.eval().requires_grad_(False); model.zero_grad(set_to_none=True); frozen_after=frozen_model_digest(model,trainable_prefixes); checkpoint_after=digest(INITIAL); production_after=state_digest(production)
    if checkpoint_after != INITIAL_SHA or production_before != production_after: raise RuntimeError("production state/checkpoint changed")
    metrics_train=_metrics(head,model.vision,train,device); metrics_hold=_metrics(head,model.vision,hold,device)
    OUT.mkdir(parents=True,exist_ok=False)
    torch.save({"schema_version":"chemvector-r5-residual-only-state-0.1","initial_sha256":INITIAL_SHA,"head_state_dict":{k:v.detach().cpu() for k,v in head.state_dict().items()},"vision_stride4_block_state_dict":{k:v.detach().cpu() for k,v in model.vision.stages[0][-1].state_dict().items()}},STATE_OUT)
    with CANDIDATE_OUT.open("x") as handle:
        for row in rows:
            sample=load_sample(DATA,row["sample_key"],purpose="diagnostic")
            with torch.inference_mode(): points=decode_dense(head(model.vision.forward_stroke_resolution(image_tensor(sample["image"])[None].to(device))),threshold=.05,maximum_vertices=128,nms_kernel_size=3,with_scores=True)[0]
            json.dump({"sample_key":row["sample_key"],"source":"shadow","candidates":[{"x":float(x),"y":float(y),"score":float(score)} for x,y,score in points]},handle,ensure_ascii=False,sort_keys=True);handle.write("\n")
    gate={"production_unchanged":checkpoint_after==INITIAL_SHA and production_before==production_after,"frozen_prefix_unchanged":frozen_before==frozen_after,"holdout_residual_precision_ge_085":metrics_hold["aggregate"]["residual_precision"]>=.85,"holdout_residual_recall_ge_080":metrics_hold["aggregate"]["residual_recall"]>=.80,"holdout_collision_le_010":metrics_hold["aggregate"]["hard_collision_rate"]<=.10,"holdout_combined_bond_coverage_ge_095":metrics_hold["aggregate"]["combined_bond_coverage"]>=.95,"holdout_one_residual_bond_coverage_ge_050":metrics_hold["aggregate"]["one_residual_bond_coverage"]>=.50,"training_admissible":False,"go":False,"next":"run independent residual edge integration audit; never promote sidecar directly"}
    report={"schema_version":"chemvector-identity-aligned-r5-residual-only-shadow-0.1","complete":True,"training_performed":True,"training_scope":"in_memory_residual_vertex_head_only","gpu_used":True,"optimizer_steps":STEPS,"sealed_test_read":False,"deployment_authorized":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"initial_checkpoint_sha256_after":checkpoint_after,"production_state_before":production_before,"production_state_after":production_after,"frozen_prefix_before":frozen_before,"frozen_prefix_after":frozen_after,"train_images":len(train),"holdout_images":len(hold),"events":events,"train":metrics_train,"holdout":metrics_hold,"state_sha256":digest(STATE_OUT),"candidate_artifact_sha256":digest(CANDIDATE_OUT),"gate":gate}
    REPORT_OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");print(json.dumps({"output":str(REPORT_OUT),"holdout":metrics_hold,"gate":gate},indent=2,ensure_ascii=False));return report


if __name__ == "__main__": run()
