"""CPU preflight for adding the identity-disjoint valence extension to sidecar training."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum import chemical_head_probe as probe  # noqa: E402
from tools import preflight_line_first_edge_features_r5 as base  # noqa: E402

FIXED = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
EXT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-edge-r5-cpu/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def rows(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
            if line.strip() and (json.loads(line).get("split") == split)
            and json.loads(line).get("render_status", "accepted") == "accepted"]


def collect(root: Path, selected: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    base.DATA = root
    fs, ys = [], []
    for row in sorted(selected, key=lambda x: x["sample_key"]):
        f, centers, _ = base._proposals(row["sample_key"])
        y = base._labels(row["sample_key"], centers, f)
        if len(f) != len(y):
            raise RuntimeError("proposal/label cardinality mismatch")
        fs.append(f); ys.append(y)
    return (np.concatenate(fs) if fs else np.empty((0, 17), np.float32),
            np.concatenate(ys) if ys else np.empty(0, np.float32))


def main() -> dict:
    if probe.file_digest(probe.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    fixed_train, fixed_holdout = rows(FIXED, "train"), rows(FIXED, "holdout")
    extension_train = rows(EXT, "train")
    if len(fixed_train) != 696 or len(fixed_holdout) != 474 or len(extension_train) != 216:
        raise ValueError("unexpected fixed/extension accounting")
    fixed_ids = {r["molecule_id"] for r in rows(FIXED, "train") + fixed_holdout}
    if fixed_ids & {r["molecule_id"] for r in extension_train}:
        raise ValueError("extension overlaps fixed identities")
    # Proposal features remain target-free; labels are joined only after proposal construction.
    x_old, y_old = collect(FIXED, fixed_train)
    x_ext, y_ext = collect(EXT, extension_train)
    x_hold, y_hold = collect(FIXED, fixed_holdout)
    x_train, y_train = np.concatenate([x_old, x_ext]), np.concatenate([y_old, y_ext])
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(20261018)
    model = base.EdgeMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx, ty = torch.from_numpy((x_train - mean) / std), torch.from_numpy(y_train)
    pos_weight = torch.tensor(float((1.0 - y_train.mean()) / max(y_train.mean(), 1e-4)))
    for _ in range(500):
        logits = model(tx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        train_scores = torch.sigmoid(model(tx)).numpy()
        hold_scores = torch.sigmoid(model(torch.from_numpy((x_hold - mean) / std))).numpy()
    calibration = base._calibrate(train_scores, y_train)
    threshold = calibration["threshold"]
    train_metrics = base._metrics(train_scores, y_train, threshold)
    hold_metrics = base._metrics(hold_scores, y_hold, threshold)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-edge-r5-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": True,
        "training_scope": "isolated_cpu_edge_feature_preflight", "optimizer_steps": 500,
        "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False,
        "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
        "data": {"fixed_train_images": len(fixed_train), "extension_train_images": len(extension_train),
                 "holdout_images": len(fixed_holdout), "extension_identity_disjoint": True},
        "calibration": {"source": "combined_train_only", **calibration},
        "train": train_metrics, "holdout": hold_metrics,
        "feature_contract": "LSD raster -> endpoint clusters -> unordered edge; geometry/support only",
        "gate": {
            "checkpoint_unchanged": probe.file_digest(probe.INITIAL) == INITIAL_SHA,
            "sealed_test_read": False,
            "holdout_precision_ge_0_85": hold_metrics["precision"] >= .85,
            "holdout_recall_ge_0_50": hold_metrics["recall"] >= .50,
            "holdout_auprc_ge_0_85": hold_metrics["auprc"] >= .85,
            "gpu_training_admissible": False,
        },
        "next": "compare against frozen r5 shadow; GPU shadow remains unauthorized until explicit comparison passes",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
