"""CPU preflight for raw atom-vs-line candidate objectness.

This objective is deliberately different from selector-drop recovery:
``protected`` and ``strict_positive`` are both positive because the frozen raw
proposal matched a rendered atom; ``hard_negative`` is the only confirmed
negative.  ``near_unknown`` and ``selected_unmatched`` are masked from loss.
The only inputs are frozen candidate scalars (raw score, set presence logit,
and normalized coordinates); no nearest-distance/GT field is consumed as a
feature.  A single zero-logit gate is evaluated per style, with conservative
precision counting masked candidates selected by the probe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-cpu-preflight-20261010-r1/report.json"
SEED = 20261014
STEPS = 300
BATCH = 2048
FIXED_GATE = 0.0
CORE = ("coordgen", "original", "rdkit")
POSITIVE = {"protected", "strict_positive"}
NEGATIVE = {"hard_negative"}
MASKED = {"near_unknown", "selected_unmatched"}


class ScalarHead(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature(row: dict) -> np.ndarray:
    # nearest_distance_px is GT-derived and intentionally forbidden.
    return np.asarray([row["raw_score"], row["presence_logit"], row["x_norm"], row["y_norm"]], dtype=np.float32)


def load() -> tuple[dict[str, list], Counter]:
    groups: dict[str, list] = defaultdict(list)
    counts = Counter()
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row["label"]
            counts[label] += 1
            groups[row["replay_split"]].append(row)
    return groups, counts


def fit_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [r for r in rows if r["label"] in POSITIVE | NEGATIVE]
    x = np.stack([feature(r) for r in selected], axis=0)
    y = np.asarray([float(r["label"] in POSITIVE) for r in selected], dtype=np.float32)
    return x, y, np.asarray(selected, dtype=object)


def evaluate(logits: np.ndarray, rows: list[dict], style: str | None = None) -> dict:
    selected_rows = rows if style is None else [r for r in rows if r["variant"] == style]
    values = logits if style is None else logits[np.asarray([r["variant"] == style for r in rows], dtype=bool)]
    labels = np.asarray([r["label"] for r in selected_rows], dtype=object)
    positive = np.isin(labels, list(POSITIVE))
    known = np.isin(labels, list(POSITIVE | NEGATIVE))
    chosen = values >= FIXED_GATE
    true = chosen & positive
    masked = np.isin(labels, list(MASKED))
    return {
        "candidates": len(labels), "known_candidates": int(known.sum()),
        "positive_raw_atoms": int(positive.sum()), "hard_negative": int((labels == "hard_negative").sum()),
        "masked": int(masked.sum()), "selected": int(chosen.sum()), "true_selected": int(true.sum()),
        "masked_selected": int((chosen & masked).sum()), "hard_negative_selected": int((chosen & (labels == "hard_negative")).sum()),
        "conservative_precision": int(true.sum()) / max(int(chosen.sum()), 1),
        "known_precision": int(true.sum()) / max(int((chosen & known).sum()), 1),
        "raw_recall": int(true.sum()) / max(int(positive.sum()), 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(min(torch.get_num_threads(), 8)); torch.manual_seed(SEED)
    groups, cache_counts = load()
    train_rows = groups["train_core"]
    x, y, _ = fit_rows(train_rows)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    x = (x - mean) / std
    head = ScalarHead(x.shape[1]); optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, yt = torch.tensor(x), torch.tensor(y); gen = torch.Generator().manual_seed(SEED)
    events = []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(yt), (BATCH,), generator=gen)
        logits = head(xt[idx]); loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True); optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    head.eval(); metrics = {}
    with torch.no_grad():
        for split in ("dev_core", "dev_stress"):
            rows = groups[split]; raw_x = np.stack([feature(r) for r in rows]); raw_x = (raw_x - mean) / std
            logits = head(torch.tensor(raw_x)).numpy()
            metrics[split] = {style: evaluate(logits, rows, style) for style in sorted({r["variant"] for r in rows})}
            metrics[split]["all"] = evaluate(logits, rows)
    result = {
        "schema_version": "chemvector-style-factorial-raw-objectness-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "cpu_scalar_raw_atom_objectness",
        "gpu_used": False, "optimizer_steps": STEPS, "batch_size": BATCH, "checkpoint_written": False,
        "sealed_test_read": False, "seed": SEED, "fixed_logit_gate": FIXED_GATE,
        "cache_sha256": digest(CACHE), "feature_names": ["raw_score", "presence_logit", "x_norm", "y_norm"],
        "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"],
        "label_protocol": {"positive": sorted(POSITIVE), "negative": sorted(NEGATIVE), "masked": sorted(MASKED),
                           "near_unknown_loss_weight": 0.0, "selected_unmatched_loss_weight": 0.0},
        "cache_counts": dict(cache_counts), "train_rows": len(y), "train_positive": int(y.sum()),
        "train_negative": int((y == 0).sum()), "metrics": metrics, "events": events,
        "gate": {"per_core_conservative_precision_ge_085": all(metrics["dev_core"][s]["conservative_precision"] >= .85 for s in CORE),
                 "per_core_raw_recall_ge_050": all(metrics["dev_core"][s]["raw_recall"] >= .50 for s in CORE),
                 "stress_separate": True, "training_ready": False,
                 "reason": "CPU preflight only; raw objectness does not authorize GPU or production"},
    }
    result["gate"]["strict_training_ready"] = result["gate"]["per_core_conservative_precision_ge_085"] and result["gate"]["per_core_raw_recall_ge_050"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "train": {"rows": len(y), "positive": int(y.sum()), "negative": int((y == 0).sum())}, "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
