"""CPU-only strict selector probe using candidate-centred raster crops.

This deliberately tests local visual evidence rather than the 31 scalar cache
features.  The loss contains only explicit strict positives and confirmed far
negatives; near-unknown rows are neutral/masked.  It is a diagnostic probe,
not a deployment model or GPU authorization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_coordinate_assignment import collect

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
SOURCE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261007-r2/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-strict-patch-cpu-preflight-20261007-r1/report.json"
SEED = 20261010
STEPS = 300
BATCH = 512
PATCH = 32
RESIZED = 16


class PatchHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(RESIZED * RESIZED, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_vector(image: Image.Image, x_norm: float, y_norm: float) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    cx, cy = round(float(x_norm) * w), round(float(y_norm) * h)
    half = PATCH // 2
    crop = np.full((PATCH, PATCH), 255, dtype=np.uint8)
    left, top = cx - half, cy - half
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left + PATCH), min(h, top + PATCH)
    crop[y0 - top:y1 - top, x0 - left:x1 - left] = gray[y0:y1, x0:x1]
    small = Image.fromarray(crop, mode="L").resize((RESIZED, RESIZED), Image.Resampling.BILINEAR)
    return (255.0 - np.asarray(small, dtype=np.float32).reshape(-1)) / 255.0


def make_arrays(rows: list[dict], images: dict[str, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for row in rows:
        x.append(patch_vector(images[row["sample_key"]], row["vector"][3], row["vector"][4]))
        y.append(float(row["strict"]))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    strict = np.asarray([r["strict"] for r in rows], dtype=bool)
    near = np.asarray([r["near_unknown"] for r in rows], dtype=bool)
    hard = np.asarray([r["hard_negative"] for r in rows], dtype=bool)
    selected = logits >= 0.0
    true = int((selected & strict).sum())
    return {"candidates": len(rows), "strict_positive": int(strict.sum()),
            "near_unknown": int(near.sum()), "hard_negative": int(hard.sum()),
            "selected": int(selected.sum()), "strict_selected": true,
            "near_unknown_selected": int((selected & near).sum()),
            "hard_negative_selected": int((selected & hard).sum()),
            "strict_precision_conservative": true / max(int(selected.sum()), 1),
            "strict_recall": true / max(int(strict.sum()), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Release audit is not ready")
    torch.manual_seed(SEED)
    train_all, holdout = collect("train"), collect("holdout")
    train = [r for r in train_all if r["strict"] or r["hard_negative"]]
    keys = {r["sample_key"] for r in train_all + holdout}
    images = {key: Image.open(SOURCE / "samples" / (key + ".png")).convert("L") for key in keys}
    x_train, y_train = make_arrays(train, images)
    x_train_t = torch.tensor(x_train)
    y_train_t = torch.tensor(y_train)
    head = PatchHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(SEED)
    events = []
    for step in range(1, STEPS + 1):
        # Natural-prior minibatches keep the fixed zero-logit gate calibrated;
        # near-unknown examples never enter this loss.
        index = torch.randint(0, len(x_train_t), (BATCH,), generator=generator)
        logits = head(x_train_t[index])
        loss = F.binary_cross_entropy_with_logits(logits, y_train_t[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    head.eval()
    with torch.no_grad():
        train_logits = head(torch.tensor(make_arrays(train_all, images)[0])).numpy()
        holdout_logits = head(torch.tensor(make_arrays(holdout, images)[0])).numpy()
    train_result, holdout_result = evaluate(train_all, train_logits), evaluate(holdout, holdout_logits)
    result = {"schema_version": "chemvector-targeted-selector-strict-patch-cpu-preflight-0.1",
              "complete": True, "training_performed": True, "training_scope": "local_raster_patch_strict_cpu_only",
              "gpu_used": False, "optimizer_steps": STEPS, "batch_size": BATCH,
              "checkpoint_written": False, "sealed_test_read": False,
              "release_manifest_sha256": digest(RELEASE / "manifest.json"),
              "source_manifest_sha256": digest(SOURCE / "manifest.json"), "seed": SEED,
              "patch_px": PATCH, "resized_px": RESIZED, "fixed_logit_gate": 0.0,
              "train_loss_rows": len(train), "near_unknown_training_rows": 0,
              "train": train_result, "holdout": holdout_result, "events": events,
              "gate": {"train_precision_ge_085": train_result["strict_precision_conservative"] >= 0.85,
                       "train_recall_ge_050": train_result["strict_recall"] >= 0.50,
                       "holdout_precision_ge_085": holdout_result["strict_precision_conservative"] >= 0.85,
                       "holdout_recall_ge_050": holdout_result["strict_recall"] >= 0.50,
                       "strict_training_ready": False,
                       "reason": "CPU visual probe only; no GPU or production authorization"}}
    result["gate"]["strict_training_ready"] = all(result["gate"][key] for key in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
