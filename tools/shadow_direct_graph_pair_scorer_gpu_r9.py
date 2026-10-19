#!/usr/bin/env python3
"""One isolated GPU shadow for the r9 direct AtomPair residual scorer.

The frozen production bundle is used only to cache target-free pair evidence;
the only trainable module is ``DirectPairResidual`` from the reviewed CPU
preflight.  Ground-truth graph files are read after proposal extraction for
in-memory metrics and are never passed to the proposal or optimizer.  The run
is deliberately bounded to 800 steps/30 minutes, checks the complete fresh
holdout every 200 steps, and writes only a quarantined sidecar receipt/state.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    ATOM_THRESHOLD,
    BASE_PAIR_THRESHOLD,
    DATA,
    SHA,
    STEPS as CPU_STEPS,
    DirectPairResidual,
    _extract,
    _finish,
    _score_rows,
    digest,
    rows,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9"
STATE = OUT / "sidecar-state.pt"
REPORT = OUT / "report.json"
MAX_STEPS = 800
CHECKPOINT_STEPS = (200, 400, 600, 800)
WALL_LIMIT_SECONDS = 1800.0
TRAIN_IMAGES_PER_PROFILE = 24
PROFILE_RECALL_FLOOR = 0.75
PROFILE_DANGLING_MAX = 0.05
PROFILE_TYPE_FLOOR = 0.90
EDGE_PRECISION_FLOOR = 0.965917
EDGE_RECALL_FLOOR = 0.850031
DANGLING_MAX = 0.05
TYPE_FLOOR = 0.90


def _threshold(train_data: list[dict[str, object]], scorer: DirectPairResidual) -> tuple[float, dict[str, float | int]]:
    """Select a threshold using train rows only (never holdout labels)."""
    scorer.eval()
    with torch.inference_mode():
        logits = torch.cat([
            item["base_logits"].float().cpu() + scorer(item["features"].float().cpu())
            for item in train_data
        ])
    labels = torch.cat([item["labels"].float().cpu() for item in train_data])
    candidates = [i / 100 for i in range(35, 96)]
    choices = []
    for threshold in candidates:
        selected = logits.sigmoid() >= threshold
        positive = labels > 0.5
        tp = int((selected & positive).sum())
        fp = int((selected & ~positive).sum())
        fn = int((~selected & positive).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        choices.append((f1, threshold, tp, fp, fn))
    f1, threshold, tp, fp, fn = max(choices, key=lambda value: (value[0], value[1]))
    return float(threshold), {"f1": float(f1), "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn}


def _cpu_snapshot(
    scorer_gpu: DirectPairResidual,
    train_data: list[dict[str, object]],
    holdout_data: list[dict[str, object]],
    baseline: dict[str, object],
) -> tuple[dict[str, object], float, dict[str, object]]:
    """Evaluate a GPU state on CPU without mutating the training module."""
    scorer = DirectPairResidual(input_dim=262)
    scorer.load_state_dict({key: value.detach().cpu() for key, value in scorer_gpu.state_dict().items()})
    scorer.eval()
    threshold, calibration = _threshold(train_data, scorer)
    scored = _finish(_score_rows(holdout_data, scorer, threshold))
    profiles: dict[str, dict[str, object]] = {}
    for profile in sorted({item["row"]["profile_id"] for item in holdout_data}):
        subset = [item for item in holdout_data if item["row"]["profile_id"] == profile]
        profiles[profile] = _finish(_score_rows(subset, scorer, threshold))
    strict_recall = float(scored["edge_recall"]) > float(baseline["edge_recall"])
    gate = {
        "strict_holdout_edge_recall_improvement": strict_recall,
        "holdout_edge_precision": float(scored["edge_precision"]) >= EDGE_PRECISION_FLOOR,
        "holdout_edge_recall": float(scored["edge_recall"]) > EDGE_RECALL_FLOOR,
        "holdout_dangling": float(scored["dangling_rate"]) <= DANGLING_MAX,
        "holdout_type_accuracy": float(scored["type_accuracy"]) >= TYPE_FLOOR,
        "all_profile_recall": bool(profiles) and all(float(v["edge_recall"]) >= PROFILE_RECALL_FLOOR for v in profiles.values()),
        "all_profile_dangling": bool(profiles) and all(float(v["dangling_rate"]) <= PROFILE_DANGLING_MAX for v in profiles.values()),
        "all_profile_type_accuracy": bool(profiles) and all(float(v["type_accuracy"]) >= PROFILE_TYPE_FLOOR for v in profiles.values()),
    }
    return scored, threshold, {"calibration_train_only": calibration, "profiles": profiles, "gate": gate}


def _profile_rows(all_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train_by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in sorted((r for r in all_rows if r.get("split") == "train"), key=lambda r: r["sample_key"]):
        if len(train_by_profile[row["profile_id"]]) < TRAIN_IMAGES_PER_PROFILE:
            train_by_profile[row["profile_id"]].append(row)
    train = [row for key in sorted(train_by_profile) for row in train_by_profile[key]]
    holdout = sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    if len(train) != 96 or len(holdout) != 95:
        raise RuntimeError(f"fixed train/holdout cardinality changed: {len(train)}/{len(holdout)}")
    train_ids = {row["molecule_id"] for row in train}
    holdout_ids = {row["molecule_id"] for row in holdout}
    if train_ids & holdout_ids:
        raise RuntimeError("train/fresh-holdout molecule identities overlap")
    return train, holdout


def run() -> dict[str, object]:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("r9 GPU shadow requires CUDA; no CPU fallback is allowed")
    if digest(INITIAL) != SHA:
        raise RuntimeError("production initial checkpoint SHA changed before shadow")
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    all_rows = rows()
    train_rows, holdout_rows = _profile_rows(all_rows)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    # Cache frozen model evidence on CPU.  This keeps proposal construction
    # byte-for-byte comparable with the reviewed deterministic CPU report;
    # only the sidecar optimizer below runs on CUDA.
    model.eval().requires_grad_(False)
    train_data = [_extract(row, model, config, need_labels=True) for row in train_rows]
    holdout_data = [_extract(row, model, config, need_labels=True) for row in holdout_rows]
    if time.monotonic() - started >= WALL_LIMIT_SECONDS:
        raise TimeoutError("frozen evidence cache exceeded 30-minute shadow bound")
    features = torch.cat([item["features"].float() for item in train_data])
    labels = torch.cat([item["labels"].float() for item in train_data])
    if len(features) == 0 or not bool(labels.any()):
        raise RuntimeError("No positive direct pair rows in fixed train subset")
    device = torch.device("cuda:0")
    scorer = DirectPairResidual(input_dim=features.size(1)).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    tx = features.to(device)
    ty = labels.to(device)
    base_logits = torch.cat([item["base_logits"].float() for item in train_data]).to(device)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1.0)).clamp(1.0, 32.0).to(device)
    baseline = _finish(_score_rows(holdout_data, None, BASE_PAIR_THRESHOLD))
    events: list[dict[str, object]] = []
    best_state = None
    best_event = None
    stopped_reason = "max_steps"
    for step in range(1, MAX_STEPS + 1):
        if time.monotonic() - started >= WALL_LIMIT_SECONDS:
            stopped_reason = "wall_limit"
            break
        scorer.train()
        residual = scorer(tx)
        loss = F.binary_cross_entropy_with_logits(base_logits + residual, ty, pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite scorer loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()
        if step not in CHECKPOINT_STEPS:
            continue
        scored, threshold, snapshot = _cpu_snapshot(scorer, train_data, holdout_data, baseline)
        strict = bool(snapshot["gate"]["strict_holdout_edge_recall_improvement"])
        event = {"step": step, "train_loss": float(loss.detach().cpu()), "threshold": threshold, "holdout": scored, **snapshot}
        events.append(event)
        if strict and all(bool(v) for v in snapshot["gate"].values()):
            best_state = copy.deepcopy(scorer.state_dict())
            best_event = event
        if step == 200 and not strict:
            stopped_reason = "no_strict_holdout_improvement_at_200"
            break
    if best_state is None:
        # A failed shadow is still a useful bounded receipt; save the final
        # state for audit only, never as a production checkpoint.
        best_state = copy.deepcopy(scorer.state_dict())
    scorer.load_state_dict(best_state)
    after = digest(INITIAL)
    OUT.mkdir(parents=True)
    state_payload = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9-0.1",
        "state_dict": {key: value.detach().cpu() for key, value in scorer.state_dict().items()},
        "initial_checkpoint_sha256": SHA,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "optimizer_steps": events[-1]["step"] if events else 0,
    }
    torch.save(state_payload, STATE)
    final_scored = best_event["holdout"] if best_event is not None else events[-1]["holdout"] if events else baseline
    final_snapshot = best_event if best_event is not None else events[-1] if events else {"gate": {}}
    report: dict[str, object] = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9-0.1",
        "complete": True,
        "read_only": False,
        "training_performed": True,
        "gpu_used": True,
        "device": str(device),
        "max_steps": MAX_STEPS,
        "optimizer_steps": events[-1]["step"] if events else 0,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "elapsed_seconds": time.monotonic() - started,
        "stopped_reason": stopped_reason,
        "checkpoints": events,
        "best_checkpoint": {"step": best_event["step"], "holdout": final_scored, "gate": final_snapshot.get("gate", {})} if best_event else None,
        "baseline_holdout": baseline,
        "final_scored_holdout": final_scored,
        "dataset": str(DATA),
        "train_split": "train",
        "holdout_split": "fresh_holdout",
        "train_images": len(train_rows),
        "holdout_images": len(holdout_rows),
        "proposal_contract": "frozen Atom candidates -> all unordered Atom pairs -> frozen DirectAtomPair hidden; GPU sidecar presence residual only; target labels post-hoc",
        "threshold_source": "train-only at each checkpoint",
        "sidecar": {"class": "DirectPairResidual", "parameters": sum(p.numel() for p in scorer.parameters()), "input_dim": features.size(1), "state_path": str(STATE)},
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "gate": {
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "sidecar_isolated": True,
            "gpu_used": True,
            "strict_holdout_graph_improvement": bool(final_snapshot.get("gate", {}).get("strict_holdout_edge_recall_improvement", False)),
            "holdout_profile_dangling_type_gates": all(bool(final_snapshot.get("gate", {}).get(k, False)) for k in ("all_profile_recall", "all_profile_dangling", "all_profile_type_accuracy")),
            "production_go": False,
        },
        "next": "shadow-only; do not promote sidecar or alter production checkpoint",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(REPORT), "device": str(device), "stopped_reason": stopped_reason, "best": report["best_checkpoint"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
