"""Run the one authorized, bounded raw atom-objectness GPU shadow.

Only a new in-memory sidecar is optimized.  The production checkpoint and all
existing model components remain frozen; no checkpoint is written.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-shadow-authorization-20261010-r1/receipt.json"
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/report.json"
FEATURES = ("raw_score", "presence_logit", "x_norm", "y_norm")
POSITIVE = {"protected", "strict_positive"}
NEGATIVE = {"hard_negative"}
MASKED = {"near_unknown", "selected_unmatched"}


class ScalarHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> tuple[list[dict], dict[str, list[dict]]]:
    train, eval_rows = [], defaultdict(list)
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["replay_split"] == "train_core" and row["label"] in POSITIVE | NEGATIVE:
                train.append(row)
            elif row["replay_split"] in {"dev_core", "dev_stress"}:
                eval_rows[row["replay_split"]].append(row)
    return train, eval_rows


def arrays(rows: list[dict], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    values = np.asarray([[float(r[name]) for name in FEATURES] for r in rows], dtype=np.float32)
    return torch.tensor((values - mean) / std, dtype=torch.float32)


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    labels = np.asarray([r["label"] for r in rows], dtype=object)
    positive = np.isin(labels, list(POSITIVE))
    known = np.isin(labels, list(POSITIVE | NEGATIVE))
    selected = logits >= 0.0
    true = selected & positive
    return {"candidates": len(rows), "known_candidates": int(known.sum()),
            "positive_raw_atoms": int(positive.sum()), "hard_negative": int((labels == "hard_negative").sum()),
            "masked": int(np.isin(labels, list(MASKED)).sum()), "selected": int(selected.sum()),
            "true_selected": int(true.sum()), "masked_selected": int((selected & ~known).sum()),
            "hard_negative_selected": int((selected & (labels == "hard_negative")).sum()),
            "conservative_precision": int(true.sum()) / max(int(selected.sum()), 1),
            "raw_recall": int(true.sum()) / max(int(positive.sum()), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != 100:
        raise ValueError("authorization does not match this shadow")
    if auth.get("initial_sha256") != INITIAL_SHA or digest(INITIAL) != INITIAL_SHA:
        raise ValueError("immutable initial checkpoint changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authorized shadow")
    torch.manual_seed(int(auth["seed"]))
    train, eval_rows = load()
    raw = np.asarray([[float(r[name]) for name in FEATURES] for r in train], dtype=np.float32)
    labels = np.asarray([float(r["label"] in POSITIVE) for r in train], dtype=np.float32)
    mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-5)
    device = torch.device("cuda:0")
    x, y = arrays(train, mean, std).to(device), torch.tensor(labels, device=device)
    model = ScalarHead().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    generator = torch.Generator(device=device).manual_seed(int(auth["seed"]))
    events = []
    for step in range(1, int(auth["max_steps"]) + 1):
        index = torch.randint(0, len(y), (int(auth["batch_size"]),), generator=generator, device=device)
        logits = model(x[index])
        loss = F.binary_cross_entropy_with_logits(logits, y[index])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    model.eval(); metrics = {}
    with torch.no_grad():
        for split, rows in eval_rows.items():
            x_eval = arrays(rows, mean, std).to(device)
            logits = model(x_eval).detach().cpu().numpy()
            by_variant = {}
            for variant in sorted({r["variant"] for r in rows}):
                mask = np.asarray([r["variant"] == variant for r in rows], dtype=bool)
                by_variant[variant] = evaluate([r for r in rows if r["variant"] == variant], logits[mask])
            metrics[split] = by_variant
    result = {"schema_version": "chemvector-style-factorial-raw-objectness-gpu-shadow-0.1",
              "complete": True, "training_performed": True, "in_memory_only": True,
              "training_scope": "raw_atom_objectness_sidecar", "optimizer_steps": 100,
              "gpu_used": True, "device": str(device), "checkpoint_written": False,
              "production_checkpoint_write": False, "deployment_authorized": False,
              "sealed_test_read": False, "initial_sha256": INITIAL_SHA,
              "initial_sha256_after": digest(INITIAL), "authorization_sha256": digest(AUTH),
              "cache_sha256": digest(CACHE), "train_rows": len(train), "train_positive": int(labels.sum()),
              "feature_names": list(FEATURES), "fixed_logit_gate": 0.0,
              "metrics": metrics, "events": events}
    core = [metrics["dev_core"][v] for v in ("coordgen", "original", "rdkit")]
    result["gate"] = {"core_precision_ge_085": all(r["conservative_precision"] >= .85 for r in core),
                       "core_recall_ge_050": all(r["raw_recall"] >= .50 for r in core),
                       "stress_separate": "dev_stress" in metrics,
                       "initial_unchanged": digest(INITIAL) == INITIAL_SHA,
                       "pass": False}
    result["gate"]["pass"] = all(result["gate"].values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "metrics": metrics, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
