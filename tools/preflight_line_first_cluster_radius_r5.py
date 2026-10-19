"""CPU-only cluster-radius A/B for the target-free line-first proposal."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
import tools.preflight_line_first_edge_features_r5 as features  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-cluster-radius-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
SEED = 20261018
STEPS = 500
TRAIN_LIMIT = 160


def rows(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line).get("split") == split
            and json.loads(line).get("render_status", "accepted") == "accepted"]


def set_radius(radius: float) -> None:
    features.CLUSTER_RADIUS = radius
    shadow.CLUSTER_RADIUS = radius
    # Imported functions resolve constants in their defining module globals.
    shadow._proposals.__globals__["CLUSTER_RADIUS"] = radius
    shadow._candidate_keys.__globals__["CLUSTER_RADIUS"] = radius


def collect(chosen: list[dict], radius: float):
    set_radius(radius)
    fs, ys, ts, gt = [], [], [], 0
    for row in sorted(chosen, key=lambda x: x["sample_key"]):
        f, centers, _ = shadow._proposals(row["sample_key"])
        keys = shadow._candidate_keys(row["sample_key"])
        t, y, n = shadow._typed_labels(row["sample_key"], centers, keys)
        if len(f) != len(t) or len(t) != len(y):
            raise RuntimeError("proposal/label mismatch")
        if f.ndim != 2:
            f = np.asarray(f, dtype=np.float32).reshape((-1, 17))
        if len(f) == 0:
            continue
        fs.append(f); ys.append(np.asarray(y, dtype=np.float32).reshape(-1)); ts.append(np.asarray(t, dtype=np.int64).reshape(-1)); gt += n
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ts), gt


def fit_eval(radius: float, train_rows: list[dict], hold_rows: list[dict]) -> dict:
    x, y, t, gt_train = collect(train_rows, radius)
    xh, yh, th, gt_hold = collect(hold_rows, radius)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    torch.manual_seed(SEED)
    model = shadow.EdgeBondMLP(x.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx, ty, tt = torch.from_numpy((x - mean) / std), torch.from_numpy(y), torch.from_numpy(t)
    weight = torch.tensor(float((1.0 - y.mean()) / max(y.mean(), 1e-4)))
    for _ in range(STEPS):
        e, k = model(tx); loss = torch.nn.functional.binary_cross_entropy_with_logits(e, ty, pos_weight=weight)
        pos = ty > .5
        if bool(pos.any()): loss = loss + .5 * torch.nn.functional.cross_entropy(k[pos], tt[pos] - 1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode():
        es, ks = model(tx); eh, kh = model(torch.from_numpy((xh - mean) / std))
    s, sh = torch.sigmoid(es).numpy(), torch.sigmoid(eh).numpy()
    cal = shadow._calibrate(s, y); threshold = cal["threshold"]
    return {"radius_px": radius, "train": shadow._metrics(s, y, ks.numpy(), t, threshold, gt_train),
            "holdout": shadow._metrics(sh, yh, kh.numpy(), th, threshold, gt_hold),
            "calibration": {"source": "train_only", **cal},
            "proposal_train": {"candidates": int(len(y)), "positive": int(y.sum()), "gt_bonds": int(gt_train)},
            "proposal_holdout": {"candidates": int(len(yh)), "positive": int(yh.sum()), "gt_bonds": int(gt_hold)}}


def main() -> dict:
    if file_digest(shadow.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    train = sorted(rows("train"), key=lambda x: x["sample_key"])[:TRAIN_LIMIT]
    hold = rows("holdout")
    results = {str(radius): fit_eval(radius, train, hold) for radius in (8.0, 10.0, 12.0, 16.0, 20.0, 24.0)}
    eligible = [v for v in results.values() if v["holdout"]["edge_precision"] >= .85]
    selected = max(eligible, key=lambda v: (v["holdout"]["candidate_edge_recall"], v["holdout"]["edge_precision"])) if eligible else None
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-cluster-radius-r5-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": True, "training_scope": "isolated_cpu_radius_ab",
        "optimizer_steps": STEPS, "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
        "protocol": {"train_images": len(train), "holdout_images": len(hold), "same_architecture": "EdgeBondMLP",
                     "feature_contract": "LSD -> endpoint clustering -> target-free geometry/support features", "threshold_source": "train-only"},
        "results": results, "selected_train_only": selected,
        "gate": {"checkpoint_unchanged": file_digest(shadow.INITIAL) == INITIAL_SHA, "sealed_test_read": False,
                 "gpu_training_admissible": False, "production_go": False},
        "next": "if a radius improves proposal and graph round-trip, run a separate fixed-state ownership CPU replay before GPU",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected": selected, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
