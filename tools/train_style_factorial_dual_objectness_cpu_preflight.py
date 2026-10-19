"""Dual CPU preflight: raw atomness and strict-drop rescue.

Head A (atomness) treats ``protected`` + ``strict_positive`` as positives and
``hard_negative`` as negatives. Head B (rescue) treats only ``strict_positive``
as positive; protected candidates are masked because they are already retained.
For both heads, near-unknown and selected-unmatched are masked. Training uses
natural candidate prevalence (no class balancing or threshold search) and a
fixed zero-logit gate, then reports conservative per-style precision/recall.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-dual-objectness-cpu-preflight-20261010-r1/report.json"
SEED = 20261015
STEPS = 300
BATCH = 2048
GATE = 0.0
CORE = ("coordgen", "original", "rdkit")
HARD = "hard_negative"
MASKED = {"near_unknown", "selected_unmatched"}


class DualHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        self.atom = nn.Linear(32, 1)
        self.rescue = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.atom(h).squeeze(-1), self.rescue(h).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature(row: dict) -> list[float]:
    return [float(row["raw_score"]), float(row["presence_logit"]), float(row["x_norm"]), float(row["y_norm"])]


def load_rows() -> tuple[list[dict], Counter]:
    rows, counts = [], Counter()
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
            counts[row["label"]] += 1
    return rows, counts


def metrics(logits: np.ndarray, rows: list[dict], positive: set[str], style: str | None = None) -> dict:
    mask = np.ones(len(rows), dtype=bool) if style is None else np.asarray([r["variant"] == style for r in rows])
    labels = np.asarray([r["label"] for r in rows], dtype=object)[mask]
    scores = logits[mask]
    pos = np.isin(labels, list(positive))
    known = pos | (labels == HARD)
    selected = scores >= GATE
    true = selected & pos
    masked = np.isin(labels, list(MASKED)) | ((positive == {"strict_positive"}) & (labels == "protected"))
    return {"candidates": len(labels), "known_candidates": int(known.sum()), "positive": int(pos.sum()),
            "selected": int(selected.sum()), "true_selected": int(true.sum()), "masked": int(masked.sum()),
            "masked_selected": int((selected & masked).sum()), "hard_negative_selected": int((selected & (labels == HARD)).sum()),
            "conservative_precision": int(true.sum()) / max(int(selected.sum()), 1),
            "known_precision": int(true.sum()) / max(int((selected & known).sum()), 1),
            "recall": int(true.sum()) / max(int(pos.sum()), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.manual_seed(SEED)
    rows, cache_counts = load_rows()
    train = [r for r in rows if r["replay_split"] == "train_core"]
    x = np.asarray([feature(r) for r in train], np.float32)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    x = (x - mean) / std
    labels = np.asarray([r["label"] for r in train], dtype=object)
    atom_valid = np.isin(labels, ["protected", "strict_positive", HARD])
    atom_y = np.isin(labels, ["protected", "strict_positive"]).astype(np.float32)
    rescue_valid = np.isin(labels, ["strict_positive", HARD])
    rescue_y = (labels == "strict_positive").astype(np.float32)
    head = DualHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, at, av = torch.tensor(x), torch.tensor(atom_y), torch.tensor(atom_valid)
    rt, rv = torch.tensor(rescue_y), torch.tensor(rescue_valid)
    gen = torch.Generator().manual_seed(SEED)
    events = []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(xt), (BATCH,), generator=gen)
        atom_logits, rescue_logits = head(xt[idx])
        losses = []
        if bool(av[idx].any()):
            losses.append(F.binary_cross_entropy_with_logits(atom_logits[av[idx]], at[idx][av[idx]]))
        if bool(rv[idx].any()):
            losses.append(F.binary_cross_entropy_with_logits(rescue_logits[rv[idx]], rt[idx][rv[idx]]))
        loss = sum(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                       "atom_rows": int(av[idx].sum()), "rescue_rows": int(rv[idx].sum())})
    head.eval()
    eval_rows = [r for r in rows if r["replay_split"] in {"dev_core", "dev_stress"}]
    ex = (np.asarray([feature(r) for r in eval_rows], np.float32) - mean) / std
    with torch.no_grad():
        atom_logits, rescue_logits = [a.numpy() for a in head(torch.tensor(ex))]
    report_metrics = {}
    for split in ("dev_core", "dev_stress"):
        subset_mask = np.asarray([r["replay_split"] == split for r in eval_rows])
        subset = [r for r in eval_rows if r["replay_split"] == split]
        report_metrics[split] = {
            "atomness": {style: metrics(atom_logits[subset_mask], subset, {"protected", "strict_positive"}, style)
                         for style in sorted({r["variant"] for r in subset})},
            "rescue": {style: metrics(rescue_logits[subset_mask], subset, {"strict_positive"}, style)
                       for style in sorted({r["variant"] for r in subset})},
        }
    result = {
        "schema_version": "chemvector-style-factorial-dual-objectness-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "cpu_shared_dual_scalar_heads",
        "gpu_used": False, "optimizer_steps": STEPS, "batch_size": BATCH,
        "checkpoint_written": False, "sealed_test_read": False, "seed": SEED,
        "fixed_logit_gate": GATE, "cache_sha256": digest(CACHE),
        "feature_names": ["raw_score", "presence_logit", "x_norm", "y_norm"],
        "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"],
        "label_protocol": {"atomness_positive": ["protected", "strict_positive"],
                           "rescue_positive": ["strict_positive"], "negative": [HARD],
                           "masked": ["near_unknown", "selected_unmatched", "protected_for_rescue"]},
        "cache_counts": dict(cache_counts), "train_rows": len(train),
        "train_atom_valid": int(atom_valid.sum()), "train_rescue_valid": int(rescue_valid.sum()),
        "metrics": report_metrics, "events": events,
        "gate": {
            "per_core_atom_precision_ge_085": all(report_metrics["dev_core"]["atomness"][s]["conservative_precision"] >= .85 for s in CORE),
            "per_core_atom_recall_ge_050": all(report_metrics["dev_core"]["atomness"][s]["recall"] >= .50 for s in CORE),
            "per_core_rescue_precision_ge_085": all(report_metrics["dev_core"]["rescue"][s]["conservative_precision"] >= .85 for s in CORE),
            "per_core_rescue_recall_ge_050": all(report_metrics["dev_core"]["rescue"][s]["recall"] >= .50 for s in CORE),
            "stress_separate": True, "training_ready": False,
            "reason": "CPU probe only; no GPU or production authorization",
        },
    }
    result["gate"]["strict_training_ready"] = all(result["gate"][k] for k in (
        "per_core_atom_precision_ge_085", "per_core_atom_recall_ge_050",
        "per_core_rescue_precision_ge_085", "per_core_rescue_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "metrics": report_metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
