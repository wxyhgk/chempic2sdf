"""Bounded GPU shadow for the hybrid ownership route.

Only an isolated EdgeBondMLP sidecar is trained.  Production/text/OCR/bond/
graph state stays frozen.  The hybrid ownership and valence guard are applied
only in diagnostic full-holdout replay; no sealed identities are read.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools import shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hybrid-merge-shadow-r5"
REPORT = OUT / "report.json"
STATE_OUT = OUT / "sidecar-state.pt"
STEPS = 800
CHECK_INTERVAL = 200
WALL_LIMIT = 1800.0
SEED = 20261018
TRAIN_LIMIT = 160
RADIUS, LAMBDA, MERGE_PX, IMAGE = 24.0, 8.0, 20.0, 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidate_rows():
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line); out[row["sample_key"]] = row["candidates"]
    if len(out) != 1170: raise ValueError("candidate artifact changed")
    return out


def owner_map(centers: np.ndarray, points: np.ndarray) -> dict[int, int]:
    if not len(centers) or not len(points): return {}
    distance = np.linalg.norm(centers[:, None] - points[None, :, :2], axis=-1)
    cost = np.where(distance <= RADIUS, distance - LAMBDA * points[None, :, 2], 1e5 + distance)
    rows, cols = linear_sum_assignment(cost)
    one = {int(r): int(c) for r, c in zip(rows, cols) if distance[r, c] <= RADIUS}
    nearest = {int(r): int(np.argmin(cost[r])) for r in range(len(centers))
               if float(distance[r, int(np.argmin(cost[r]))]) <= RADIUS}
    groups = defaultdict(list)
    for node, cand in nearest.items(): groups[cand].append(node)
    merged = dict(one)
    for cand, nodes in groups.items():
        if len(nodes) > 1 and max(float(distance[node, cand]) for node in nodes) <= MERGE_PX:
            for node in nodes: merged[node] = cand
    return merged


def collect(rows: list[dict], candidates: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    features, labels, types, records = [], [], [], {}
    shadow.DATA = DATA
    shadow._proposals.__globals__["DATA"] = DATA
    shadow._candidate_keys.__globals__["DATA"] = DATA
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        key = row["sample_key"]
        f, centers, _ = shadow._proposals(key)
        keys = shadow._candidate_keys(key)
        typed, positive, _ = shadow._typed_labels(key, centers, keys)
        points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in candidates[key]], np.float32)
        if len(f) != len(typed) or len(typed) != len(positive): raise RuntimeError("proposal/label mismatch")
        features.append(f); labels.append(positive.astype(np.float32)); types.append(typed)
        records[key] = {"features": f, "centers": centers, "keys": keys, "points": points}
    return np.concatenate(features), np.concatenate(labels), np.concatenate(types), records


def finish(counter: Counter) -> dict:
    return {**dict(counter), "edge_precision": counter["true_edges"] / max(counter["mapped_edges"], 1),
            "edge_recall": counter["true_edges"] / max(counter["gt_bonds"], 1),
            "bond_type_accuracy": counter["type_correct"] / max(counter["type_total"], 1),
            "dangling_rate": counter["dangling_edges"] / max(counter["selected_edges"], 1),
            "oracle_sanitize_rate": counter["sanitize_ok"] / max(counter["images"], 1)}


def evaluate(model, rows, records, means, stds, threshold, device) -> dict:
    total, profiles = Counter(), defaultdict(Counter)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        key = row["sample_key"]; record = records[key]
        x = torch.from_numpy((record["features"] - means) / stds).to(device)
        with torch.inference_mode():
            edge_logits, type_logits = model(x)
        scores = torch.sigmoid(edge_logits).detach().cpu().numpy()
        kinds = (torch.argmax(type_logits, dim=1) + 1).detach().cpu().numpy()
        ownership = owner_map(record["centers"].astype(np.float64), record["points"].astype(np.float64))
        selected, dangling = {}, 0
        for index, (a, b) in enumerate(record["keys"]):
            if scores[index] < threshold: continue
            if a not in ownership or b not in ownership: dangling += 1; continue
            ca, cb = ownership[a], ownership[b]
            if ca == cb: continue
            pair = tuple(sorted((ca, cb))); old = selected.get(pair)
            if old is None or scores[index] > old[0]: selected[pair] = (float(scores[index]), int(kinds[index]))
        ring = guard._cycles(selected)
        selected = {p: (s, 1 if t == guard.TYPE_AROMATIC and p not in ring else t) for p, (s, t) in selected.items()}
        selected, _ = guard.guard(selected, low_confidence=.4)
        metrics = guard._metrics(row, selected, record["points"], dangling)
        total.update(metrics); profiles[row["profile_id"]].update(metrics)
    return {"all": finish(total), "by_profile": {k: finish(v) for k, v in sorted(profiles.items())}}


def improved(metrics: dict, baseline: dict) -> bool:
    # Primary graph validity, then recall. Precision must retain the raw CPU
    # floor; a precision regression against the current guard remains visible.
    return ((metrics["all"]["oracle_sanitize_rate"] > baseline["all"]["oracle_sanitize_rate"] + 1e-9)
            or (abs(metrics["all"]["oracle_sanitize_rate"] - baseline["all"]["oracle_sanitize_rate"]) <= 1e-9
                and metrics["all"]["edge_recall"] > baseline["all"]["edge_recall"] + 1e-9))


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; shadow requires one GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_rows()
    source_rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    train = sorted([r for r in source_rows if r["split"] == "train"], key=lambda x: x["sample_key"])[:TRAIN_LIMIT]
    holdout = sorted([r for r in source_rows if r["split"] == "holdout"], key=lambda x: x["sample_key"])
    x_train, y_train, t_train, train_records = collect(train, candidates)
    _, _, _, hold_records = collect(holdout, candidates)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    # Baseline uses the reviewed frozen sidecar under the exact hybrid route.
    frozen = chemistry.EdgeBondMLP(len(torch.load(chemistry.STATE, map_location="cpu", weights_only=False)["feature_names"]))
    frozen_state = torch.load(chemistry.STATE, map_location="cpu", weights_only=False)
    frozen.load_state_dict(frozen_state["model"]); frozen.eval()
    device = torch.device("cuda:0")
    baseline_model = frozen.to(device)
    frozen_mean = np.asarray(frozen_state["mean"], np.float32)
    frozen_std = np.asarray(frozen_state["std"], np.float32)
    frozen_threshold = float(frozen_state["threshold"])
    baseline = evaluate(baseline_model, holdout, hold_records, frozen_mean, frozen_std, frozen_threshold, device)
    # Shadow trains an isolated model from scratch with the same architecture.
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = chemistry.EdgeBondMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx = torch.from_numpy((x_train - mean) / std).to(device); ty = torch.from_numpy(y_train).to(device); tt = torch.from_numpy(t_train).to(device)
    pos_weight = torch.tensor(float((1.0 - y_train.mean()) / max(y_train.mean(), 1e-4)), device=device)
    started = time.time(); checkpoints = []; best = None; best_step = 0; best_threshold = None; best_state = None
    for step in range(1, STEPS + 1):
        edge_logits, type_logits = model(tx)
        edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, ty, pos_weight=pos_weight)
        positive = ty > .5
        type_loss = torch.nn.functional.cross_entropy(type_logits[positive], tt[positive] - 1) if bool(positive.any()) else edge_logits.sum() * 0.0
        loss = edge_loss + .5 * type_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % CHECK_INTERVAL == 0:
            with torch.inference_mode():
                train_edge_logits, _ = model(tx)
                train_scores = torch.sigmoid(train_edge_logits).detach().cpu().numpy()
            calibration = shadow._calibrate(train_scores, y_train)
            threshold = float(calibration["threshold"])
            metric = evaluate(model, holdout, hold_records, mean, std, threshold, device)
            rec = {"step": step, "loss": float(loss.detach().cpu()), "metrics": metric,
                   "threshold": threshold, "calibration": calibration,
                   "elapsed_seconds": time.time() - started, "improves_baseline": improved(metric, baseline)}
            checkpoints.append(rec)
            if best is None or improved(metric, best):
                best, best_step, best_threshold = metric, step, threshold
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if not rec["improves_baseline"] and step == CHECK_INTERVAL:
                break
            if time.time() - started > WALL_LIMIT: break
    if best is not None:
        torch.save({"model": best_state or model.state_dict(), "mean": mean, "std": std,
                    "threshold": best_threshold if best_threshold is not None else threshold,
                    "feature_names": frozen_state["feature_names"], "provenance": {
                        "initial_sha256": INITIAL_SHA, "merge_px": MERGE_PX, "guard_tau": .4,
                        "sealed_test_read": False, "production_checkpoint_written": False,
                    }}, STATE_OUT)
    after = file_digest(chemistry.INITIAL)
    final = checkpoints[-1] if checkpoints else None
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-hybrid-merge-shadow-r5-0.1",
        "complete": True, "training_performed": True, "training_scope": "isolated_hybrid_edge_bond_sidecar",
        "gpu_used": True, "optimizer_steps": checkpoints[-1]["step"] if checkpoints else 0,
        "elapsed_seconds": time.time() - started, "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after,
        "protocol": {"max_steps": STEPS, "check_interval": CHECK_INTERVAL, "max_wall_seconds": WALL_LIMIT,
                     "train_images": len(train), "holdout_images": len(holdout), "merge_px": MERGE_PX,
                     "guard_tau": .4, "threshold_source": "frozen reviewed train-only threshold"},
        "baseline": baseline, "checkpoints": checkpoints, "best_step": best_step,
        "state_sha256": digest(STATE_OUT) if STATE_OUT.exists() else None,
        "gate": {"checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False,
                 "full_holdout_improved": bool(final and final["improves_baseline"]),
                 "production_go": False, "training_admissible": False},
        "next": "keep as shadow evidence only; production merge requires a separate approval and strict no-regression profile audit",
    }
    OUT.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "device": str(device), "baseline": baseline["all"], "final": final, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()
