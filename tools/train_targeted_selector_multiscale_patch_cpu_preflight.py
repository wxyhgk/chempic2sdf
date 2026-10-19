"""Fixed CPU-only preflight for a multi-scale local visual selector.

The previous 32 px patch probe compressed every candidate into one 16 px
linear vector.  This probe keeps three fixed candidate-centred views (24, 48,
and 96 source pixels), encodes each with a small convolutional branch, and
joins the branches with a classifier.  It is deliberately a *preflight*:
there is no checkpoint, no GPU, no threshold sweep, and no production use.

Only explicit strict positives and confirmed far negatives are used in the
loss.  ``near_unknown`` members and unknown members of a strict group are
masked from both the loss and positive/negative fitting labels.  The fixed
logit gate is part of this experiment's protocol and is evaluated exactly
once on train and holdout.
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
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-multiscale-patch-cpu-preflight-20261007-r1/report.json"

SEED = 20261011
STEPS = 400
BATCH = 512
SCALES = (24, 48, 96)
PATCH_OUTPUT = 24
LOGIT_GATE = -2.75


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiScalePatchHead(nn.Module):
    """Small independent branch per spatial scale; no pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, 3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 48, 3, stride=2, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                for _ in SCALES
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(48 * len(SCALES)),
            nn.Linear(48 * len(SCALES), 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim == 4 and patches.shape[1] == len(SCALES):
            patches = patches.unsqueeze(2)
        if patches.ndim != 5 or patches.shape[1] != len(SCALES) or patches.shape[2] != 1:
            raise ValueError("Expected [batch, three scales, 1, H, W] patches")
        encoded = torch.cat(
            [branch(patches[:, index]) for index, branch in enumerate(self.branches)],
            dim=1,
        )
        return self.classifier(encoded).squeeze(-1)


def patch_vector(image: Image.Image, x_norm: float, y_norm: float, source_size: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    cx, cy = round(float(x_norm) * width), round(float(y_norm) * height)
    half = source_size // 2
    crop = np.full((source_size, source_size), 255, dtype=np.uint8)
    left, top = cx - half, cy - half
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + source_size), min(height, top + source_size)
    crop[y0 - top : y1 - top, x0 - left : x1 - left] = gray[y0:y1, x0:x1]
    resized = Image.fromarray(crop, mode="L").resize(
        (PATCH_OUTPUT, PATCH_OUTPUT), Image.Resampling.BILINEAR
    )
    return (255.0 - np.asarray(resized, dtype=np.float32)) / 255.0


def make_arrays(rows: list[dict], images: dict[str, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((len(rows), len(SCALES), PATCH_OUTPUT, PATCH_OUTPUT), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        x_norm, y_norm = row["vector"][3], row["vector"][4]
        for scale_index, source_size in enumerate(SCALES):
            values[index, scale_index] = patch_vector(
                images[row["sample_key"]], x_norm, y_norm, source_size
            )
        labels[index] = float(row["strict"])
    return values, labels


def evaluate(rows: list[dict], logits: np.ndarray) -> dict[str, float | int]:
    strict = np.asarray([row["strict"] for row in rows], dtype=bool)
    near = np.asarray([row["near_unknown"] for row in rows], dtype=bool)
    hard = np.asarray([row["hard_negative"] for row in rows], dtype=bool)
    selected = logits >= LOGIT_GATE
    true = int((selected & strict).sum())
    return {
        "candidates": len(rows),
        "strict_positive": int(strict.sum()),
        "near_unknown": int(near.sum()),
        "hard_negative": int(hard.sum()),
        "selected": int(selected.sum()),
        "strict_selected": true,
        "near_unknown_selected": int((selected & near).sum()),
        "hard_negative_selected": int((selected & hard).sum()),
        "strict_precision_conservative": true / max(int(selected.sum()), 1),
        "strict_recall": true / max(int(strict.sum()), 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Coordinate release audit is not ready")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_all, holdout = collect("train"), collect("holdout")
    # Explicitly exclude unknown members and near-unknown members from loss.
    train_loss = [row for row in train_all if row["strict"] or row["hard_negative"]]
    keys = {row["sample_key"] for row in train_all + holdout}
    images = {
        key: Image.open(SOURCE / "samples" / f"{key}.png").convert("L") for key in keys
    }
    x_train, y_train = make_arrays(train_loss, images)
    x_all, _ = make_arrays(train_all, images)
    x_holdout, _ = make_arrays(holdout, images)
    x_train_t, y_train_t = torch.tensor(x_train), torch.tensor(y_train)

    positive = np.flatnonzero(y_train > 0.0)
    negative = np.flatnonzero(y_train == 0.0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both strict positives and confirmed negatives are required")
    generator = np.random.default_rng(SEED)
    model = MultiScalePatchHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        # Fixed 1:1 class-balanced sampling isolates visual ranking from the
        # extreme candidate prior; LOGIT_GATE is fixed independently above.
        indices = np.concatenate(
            [generator.choice(positive, BATCH // 2, replace=True),
             generator.choice(negative, BATCH // 2, replace=True)]
        )
        generator.shuffle(indices)
        batch_index = torch.tensor(indices, dtype=torch.long)
        logits = model(x_train_t[batch_index])
        loss = F.binary_cross_entropy_with_logits(logits, y_train_t[batch_index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})

    model.eval()
    with torch.no_grad():
        train_logits = model(torch.tensor(x_all)).numpy()
        holdout_logits = model(torch.tensor(x_holdout)).numpy()
    train_result = evaluate(train_all, train_logits)
    holdout_result = evaluate(holdout, holdout_logits)
    result = {
        "schema_version": "chemvector-targeted-selector-multiscale-patch-cpu-preflight-0.1",
        "complete": True,
        "training_performed": True,
        "training_scope": "local_multiscale_raster_patch_strict_cpu_only",
        "gpu_used": False,
        "optimizer_steps": STEPS,
        "batch_size": BATCH,
        "checkpoint_written": False,
        "sealed_test_read": False,
        "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "seed": SEED,
        "scales_source_px": list(SCALES),
        "patch_output_px": PATCH_OUTPUT,
        "fixed_logit_gate": LOGIT_GATE,
        "train_loss_rows": len(train_loss),
        "near_unknown_training_rows": 0,
        "architecture": "three independent 3-layer Conv/GELU branches + LayerNorm/MLP",
        "train": train_result,
        "holdout": holdout_result,
        "events": events,
    }
    result["gate"] = {
        "train_precision_ge_085": train_result["strict_precision_conservative"] >= 0.85,
        "train_recall_ge_050": train_result["strict_recall"] >= 0.50,
        "holdout_precision_ge_085": holdout_result["strict_precision_conservative"] >= 0.85,
        "holdout_recall_ge_050": holdout_result["strict_recall"] >= 0.50,
        "strict_training_ready": False,
        "reason": "CPU visual probe only; no GPU or production authorization",
    }
    result["gate"]["strict_training_ready"] = all(
        result["gate"][key]
        for key in (
            "train_precision_ge_085",
            "train_recall_ge_050",
            "holdout_precision_ge_085",
            "holdout_recall_ge_050",
        )
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
