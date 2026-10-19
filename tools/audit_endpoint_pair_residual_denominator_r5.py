#!/usr/bin/env python3
"""Read-only residual-denominator audit for the isolated r5 pair sidecar.

The old edge audit divided sidecar recovery by *all* target bonds.  That is
not the contract of a residual sidecar: bonds whose two endpoints are already
owned by the frozen baseline are outside its denominator.  This audit keeps
the exact frozen inference, candidate artifact, sidecar state and train-only
threshold, then reports baseline-owned, residual-owned, sidecar-recovered and
union coverage without changing any model or checkpoint.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.preflight_endpoint_pair_compatibility_r5 import (
    DATA, CANDIDATES, INITIAL_SHA, TRAIN_LIMIT, HOLDOUT_LIMIT,
    _candidates, _candidate_feature_rows, _target, _proposal_target,
    PairMLP, digest,
)

BASE = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow-v1"
STATE = BASE / "sidecar-state.pt"
SHADOW_REPORT = BASE / "report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-residual-denominator-r5/report.json"
IMAGE = 768.0
TARGET_RADIUS = 6.0
DUPLICATE_RADIUS = 6.0


def baseline_owned(points: torch.Tensor, baseline: torch.Tensor, a: int, b: int) -> bool:
    """Both target endpoints have one unambiguous frozen baseline owner."""
    if not len(baseline):
        return False
    for idx in (a, b):
        d = torch.linalg.vector_norm((baseline - points[idx]) * IMAGE, dim=1)
        close = torch.where(d <= TARGET_RADIUS)[0]
        near = torch.where(d <= 20.0)[0]
        if len(close) != 1 or len(near) != 1:
            return False
    return True


def owner_pool(values, selected):
    cp = torch.tensor([[x["x"], x["y"]] for x in values], dtype=torch.float32)
    baseline = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2].detach().cpu()
    residual = (torch.cdist(cp * IMAGE, baseline * IMAGE).amin(dim=1) > DUPLICATE_RADIUS
                if len(baseline) else torch.ones(len(cp), dtype=torch.bool))
    return torch.cat((cp, baseline), dim=0), torch.cat((residual, torch.zeros(len(baseline), dtype=torch.bool)))


def run(full: bool = False) -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    report = json.loads(SHADOW_REPORT.read_text())
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    if report.get("initial_sha256") != INITIAL_SHA or report.get("mode") != "shadow":
        raise RuntimeError("sidecar receipt mismatch")
    if state.get("initial_sha256") != INITIAL_SHA:
        raise RuntimeError("sidecar state mismatch")
    cand = _candidates()
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    rows = [x for x in rows if x.get("render_status") == "accepted"]
    all_train = sorted([x for x in rows if x["split"] == "train"], key=lambda x: x["sample_key"])
    all_hold = sorted([x for x in rows if x["split"] == "holdout"], key=lambda x: x["sample_key"])
    # The sidecar was fitted on the fixed 160/80 slice.  ``--full`` only
    # expands the read-only evaluation denominator; it never refits or
    # recalibrates the sidecar on the extra rows.
    fit_train = all_train[:TRAIN_LIMIT]
    train = all_train if full else fit_train
    hold = all_hold if full else all_hold[:HOLDOUT_LIMIT]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this bounded replay requires CUDA")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    # Reconstruct target-free train normalization exactly as the integration audit.
    train_features = []
    for row in fit_train:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        with torch.inference_mode():
            out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
        bonds = {k: v.detach().cpu() for k, v in out[4].items()}
        selected = {k: v.detach().cpu() for k, v in out[6].items()}
        f, _, _ = _candidate_feature_rows(sample, cand[row["sample_key"]], bonds, selected)
        train_features.extend(f)
    x = np.asarray(train_features, np.float32)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    side = PairMLP(x.shape[1]).to(device)
    side.load_state_dict(state["state_dict"], strict=True)
    side.eval()
    threshold = float(report["metrics"]["threshold"])
    result = {}
    for split, chosen in (("train", train), ("holdout", hold)):
        t = Counter()
        for row in chosen:
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            points, target_bonds = _target(sample)
            with torch.inference_mode():
                out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
            bonds = {k: v.detach().cpu() for k, v in out[4].items()}
            selected = {k: v.detach().cpu() for k, v in out[6].items()}
            baseline = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2]
            feats, meta, _ = _candidate_feature_rows(sample, cand[row["sample_key"]], bonds, selected)
            if not feats:
                continue
            with torch.inference_mode():
                scores = side(torch.tensor((np.asarray(feats, np.float32) - mean) / std, device=device)).sigmoid().cpu().numpy()
            owner_points, is_residual = owner_pool(cand[row["sample_key"]], selected)
            chosen_by_q = {}
            for score, m in zip(scores, meta):
                qi, i, j, _ = m
                if not (is_residual[i] or is_residual[j]) or score < threshold:
                    continue
                if qi not in chosen_by_q or score > chosen_by_q[qi][0]:
                    chosen_by_q[qi] = (float(score), i, j)
            baseline_ids = set()
            mapped_query_ids = set()
            for qi in torch.where(bonds["bond_mask"][0].bool())[0].tolist():
                hit = _proposal_target(bonds["bond_endpoints"][0, qi], points, target_bonds)
                if hit is not None:
                    mapped_query_ids.add(hit[0])
            residual_ids = set()
            for bi, (a, b) in enumerate(target_bonds):
                t["target_bonds"] += 1
                if baseline_owned(points, baseline, a, b):
                    baseline_ids.add(bi); t["baseline_owned"] += 1
                else:
                    residual_ids.add(bi); t["residual_target_bonds"] += 1
                if bi in mapped_query_ids:
                    t["frozen_bond_query_available"] += 1
            recovered = set()
            for qi, (_, i, j) in chosen_by_q.items():
                hit = _proposal_target(bonds["bond_endpoints"][0, qi], points, target_bonds)
                if hit is None:
                    continue
                bi, (a, b) = hit
                di = torch.linalg.vector_norm((owner_points[i] - points[a]) * IMAGE)
                dj = torch.linalg.vector_norm((owner_points[j] - points[b]) * IMAGE)
                if di <= TARGET_RADIUS and dj <= TARGET_RADIUS and bi in residual_ids:
                    recovered.add(bi)
            t["sidecar_selected_edges"] += len(chosen_by_q)
            t["sidecar_recovered_residual"] += len(recovered)
            t["union_recovered"] += len(baseline_ids | recovered)
        out = dict(t)
        out["baseline_prefix_recall"] = t["baseline_owned"] / max(t["target_bonds"], 1)
        out["residual_target_denominator"] = t["residual_target_bonds"]
        out["sidecar_residual_recall"] = t["sidecar_recovered_residual"] / max(t["residual_target_bonds"], 1)
        out["overall_union_recall"] = t["union_recovered"] / max(t["target_bonds"], 1)
        result[split] = out
    output = {"schema_version": "chemvector-identity-aligned-endpoint-pair-residual-denominator-r5-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "sealed_test_read": False, "deployment_authorized": False,
              "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
              "candidate_artifact_sha256": digest(CANDIDATES), "sidecar_state_sha256": digest(STATE),
              "threshold": threshold, "split": {"fit_train_images": len(fit_train), "train_images": len(train), "holdout_images": len(hold), "full_evaluation": full},
              "by_split": result,
              "gate": {"baseline_prefix_preserved": True,
                       "residual_denominator_defined": True,
                       "production_unchanged": digest(INITIAL) == INITIAL_SHA,
                       "go": False,
                       "next": "review residual recall and union recall; no production integration from this audit"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="evaluate all 696/474 rows with the fixed 160-row fit normalization")
    run(full=parser.parse_args().full)
