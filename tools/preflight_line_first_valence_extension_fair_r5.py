"""Fair CPU A/B for the line-first valence extension.

This preflight deliberately uses the same sidecar architecture, feature
contract, seed, optimizer steps, train slice and calibration rule as the
frozen r5 shadow.  The only variable is whether the identity-disjoint
valence/topology extension is appended to the 160-image train slice.  It
never reads sealed test data and writes only a diagnostic report (no model
checkpoint and no production artifact).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402

FIXED = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
EXT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-fair-r5/report.json"
SEED = 20261018
STEPS = 800
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80


def _rows(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
            if line.strip() and (json.loads(line).get("split") == split)
            and json.loads(line).get("render_status", "accepted") == "accepted"]


def _collect(root: Path, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
    """Collect target-free proposal features and post-proposal labels."""
    shadow.DATA = root
    # ``_proposals`` is imported from the CPU preflight module and resolves
    # its DATA global in that module, not in ``shadow``.  Set both explicitly
    # so extension rows cannot accidentally be opened from the fixed source.
    shadow._proposals.__globals__["DATA"] = root
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    types: list[np.ndarray] = []
    groups: list[tuple[int, int]] = []
    gt_bonds = 0
    offset = 0
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        f, centers, _ = shadow._proposals(row["sample_key"])
        keys = shadow._candidate_keys(row["sample_key"])
        t, y, n = shadow._typed_labels(row["sample_key"], centers, keys)
        if len(f) != len(t) or len(t) != len(y):
            raise RuntimeError(f"proposal/label mismatch: {row['sample_key']}")
        features.append(f)
        labels.append(y.astype(np.float32))
        types.append(t)
        groups.append((offset, offset + len(y)))
        offset += len(y)
        gt_bonds += n
    if not features:
        return (np.empty((0, 17), np.float32), np.empty(0, np.float32),
                np.empty(0, np.int64), 0, [])
    return np.concatenate(features), np.concatenate(labels), np.concatenate(types), gt_bonds, groups


def _profile_metrics(scores: np.ndarray, labels: np.ndarray, types: np.ndarray,
                     threshold: float, gt_bonds: int,
                     rows: list[dict], groups: list[tuple[int, int]]) -> dict:
    # Aggregate profile counts and use graph JSON for the true bond denominator.
    profile = defaultdict(lambda: {"candidate_edges": 0, "candidate_positive_edges": 0,
                                   "gt_bonds": 0, "tp": 0, "fp": 0, "fn": 0,
                                   "typed_total": 0, "typed_correct": 0})
    # ``gt_bonds`` is only available as an aggregate from _collect; use the
    # fixed graph files to derive profile totals without touching sealed data.
    for row, (lo, hi) in zip(sorted(rows, key=lambda item: item["sample_key"]), groups):
        root = shadow.DATA
        graph = json.loads((root / "samples" / row["sample_key"] ).with_suffix(".graph.json").read_text())
        p = profile[row["profile_id"]]
        s = scores[lo:hi]; y = labels[lo:hi]; keep = s >= threshold; positive = y.astype(bool)
        p["candidate_edges"] += int(len(y)); p["candidate_positive_edges"] += int(positive.sum())
        p["gt_bonds"] += int(len(graph["bonds"]))
        p["tp"] += int((keep & positive).sum()); p["fp"] += int((keep & ~positive).sum()); p["fn"] += int((~keep & positive).sum())
    for name, p in profile.items():
        p["edge_precision"] = p["tp"] / max(p["tp"] + p["fp"], 1)
        p["graph_bond_recall"] = p["tp"] / max(p["gt_bonds"], 1)
    return dict(sorted(profile.items()))


def _fit_and_eval(name: str, train_rows: list[dict], hold_rows: list[dict],
                  full_hold_rows: list[dict]) -> dict:
    x_train, y_train, t_train, gt_train, _ = _collect(FIXED if name == "old_only" else TRAIN_ROOT, train_rows)
    x_hold, y_hold, t_hold, gt_hold, groups_hold = _collect(FIXED, hold_rows)
    x_full, y_full, t_full, gt_full, groups_full = _collect(FIXED, full_hold_rows)
    mean = x_train.mean(0)
    std = np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED)
    model = shadow.EdgeBondMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx = torch.from_numpy((x_train - mean) / std)
    ty = torch.from_numpy(y_train)
    tt = torch.from_numpy(t_train)
    pos_weight = torch.tensor(float((1.0 - y_train.mean()) / max(y_train.mean(), 1e-4)))
    started = time.time()
    for _ in range(STEPS):
        edge_logits, kind_logits = model(tx)
        edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, ty, pos_weight=pos_weight)
        positive = ty > 0.5
        kind_loss = torch.nn.functional.cross_entropy(kind_logits[positive], tt[positive] - 1) if bool(positive.any()) else edge_logits.sum() * 0.0
        loss = edge_loss + 0.5 * kind_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        tr_e, tr_k = model(tx)
        ho_e, ho_k = model(torch.from_numpy((x_hold - mean) / std))
        fu_e, fu_k = model(torch.from_numpy((x_full - mean) / std))
    tr_s = torch.sigmoid(tr_e).numpy(); ho_s = torch.sigmoid(ho_e).numpy(); fu_s = torch.sigmoid(fu_e).numpy()
    tr_k = tr_k.numpy(); ho_k = ho_k.numpy(); fu_k = fu_k.numpy()
    calibration = shadow._calibrate(tr_s, y_train)
    threshold = calibration["threshold"]
    return {
        "optimizer_steps": STEPS, "elapsed_seconds": time.time() - started,
        "train": shadow._metrics(tr_s, y_train, tr_k, t_train, threshold, gt_train),
        "holdout_slice": shadow._metrics(ho_s, y_hold, ho_k, t_hold, threshold, gt_hold),
        "holdout_full": shadow._metrics(fu_s, y_full, fu_k, t_full, threshold, gt_full),
        "profiles_full": _profile_metrics(fu_s, y_full, t_full, threshold, gt_full, full_hold_rows, groups_full),
        "calibration": {"source": "train_only", **calibration},
    }


def main() -> dict:
    if file_digest(shadow.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    fixed_train = _rows(FIXED, "train")
    fixed_hold = _rows(FIXED, "holdout")
    extension = _rows(EXT, "train")
    if len(fixed_train) != 696 or len(fixed_hold) != 474 or len(extension) != 216:
        raise RuntimeError("fixed split or extension accounting changed")
    fixed_ids = {r["molecule_id"] for r in fixed_train + fixed_hold}
    if fixed_ids & {r["molecule_id"] for r in extension}:
        raise RuntimeError("extension identity overlap")
    # Keep the sidecar module's DATA global explicit for every collection.
    global TRAIN_ROOT
    TRAIN_ROOT = FIXED
    old_slice = sorted(fixed_train, key=lambda r: r["sample_key"])[:TRAIN_LIMIT]
    ext_rows = sorted(extension, key=lambda r: r["sample_key"])
    augmented = old_slice + ext_rows
    hold_slice = sorted(fixed_hold, key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    started = time.time()
    old = _fit_and_eval("old_only", old_slice, hold_slice, fixed_hold)
    # For the augmented run, _collect must use the extension root for the
    # appended rows.  The feature/label collection helper receives one root;
    # temporarily materialize a root-aware collection by collecting separately
    # below and fitting in a small wrapper.
    x_old, y_old, t_old, gt_old, _ = _collect(FIXED, old_slice)
    x_ext, y_ext, t_ext, gt_ext, _ = _collect(EXT, ext_rows)
    x_h, y_h, t_h, gt_h, groups_h = _collect(FIXED, hold_slice)
    x_f, y_f, t_f, gt_f, groups_f = _collect(FIXED, fixed_hold)
    x = np.concatenate([x_old, x_ext]); y = np.concatenate([y_old, y_ext]); t = np.concatenate([t_old, t_ext])
    mean = x.mean(0); std = np.maximum(x.std(0), 1e-5)
    torch.manual_seed(SEED); model = shadow.EdgeBondMLP(x.shape[1]); optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx = torch.from_numpy((x - mean) / std); ty = torch.from_numpy(y); tt = torch.from_numpy(t)
    pos_weight = torch.tensor(float((1.0 - y.mean()) / max(y.mean(), 1e-4)))
    for _ in range(STEPS):
        e, k = model(tx); loss = torch.nn.functional.binary_cross_entropy_with_logits(e, ty, pos_weight=pos_weight)
        positive = ty > 0.5
        if bool(positive.any()): loss = loss + 0.5 * torch.nn.functional.cross_entropy(k[positive], tt[positive] - 1)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        tr_e, tr_k = model(tx); hs, hk = model(torch.from_numpy((x_h - mean) / std)); fs, fk = model(torch.from_numpy((x_f - mean) / std))
    tr_s = torch.sigmoid(tr_e).numpy(); h_s = torch.sigmoid(hs).numpy(); f_s = torch.sigmoid(fs).numpy()
    tr_k = tr_k.numpy(); h_k = hk.numpy(); f_k = fk.numpy()
    cal = shadow._calibrate(tr_s, y); threshold = cal["threshold"]
    augmented_result = {
        "optimizer_steps": STEPS,
        "train": shadow._metrics(tr_s, y, tr_k, t, threshold, gt_old + gt_ext),
        "holdout_slice": shadow._metrics(h_s, y_h, h_k, t_h, threshold, gt_h),
        "holdout_full": shadow._metrics(f_s, y_f, f_k, t_f, threshold, gt_f),
        "profiles_full": _profile_metrics(f_s, y_f, t_f, threshold, gt_f, fixed_hold, groups_f),
        "calibration": {"source": "train_only", **cal},
    }
    old_full = old["holdout_full"]; new_full = augmented_result["holdout_full"]
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-fair-r5-0.1",
        "complete": True, "read_only": True, "training_performed": True,
        "training_scope": "isolated_cpu_fair_ab_only", "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(shadow.INITIAL),
        "protocol": {"architecture": "EdgeBondMLP", "feature_contract": "LSD -> endpoint clusters -> target-free edge features",
                     "seed": SEED, "optimizer_steps": STEPS, "old_train_images": len(old_slice),
                     "extension_train_images": len(ext_rows), "holdout_slice_images": len(hold_slice),
                     "full_holdout_images": len(fixed_hold), "threshold_source": "train-only", "sealed_test_read": False},
        "old_only": old, "old_plus_extension": augmented_result,
        "delta_full_holdout": {k: float(new_full[k] - old_full[k]) for k in ("edge_precision", "candidate_edge_recall", "conditional_bond_type_accuracy")},
        "gate": {
            "checkpoint_unchanged": file_digest(shadow.INITIAL) == INITIAL_SHA,
            "sealed_test_read": False,
            "same_architecture_and_steps": True,
            "full_precision_strictly_improves": new_full["edge_precision"] > old_full["edge_precision"],
            "full_graph_recall_strictly_improves": new_full["candidate_edge_recall"] > old_full["candidate_edge_recall"],
            "full_type_accuracy_not_lower": new_full["conditional_bond_type_accuracy"] >= old_full["conditional_bond_type_accuracy"],
            "gpu_training_admissible": False,
            "production_go": False,
        },
        "next": "GPU shadow remains inadmissible unless both full precision and graph recall strictly improve with no profile floor regression; proposal recall remains a separate ceiling.",
        "elapsed_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "delta_full_holdout": report["delta_full_holdout"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
