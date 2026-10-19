"""Fixed CPU group-abstention preflight for the style-factorial replay.

This probe asks a narrower question than a pointwise selector: can a model
decide whether an entire candidate group deserves another visual/chemical
pass?  It uses no rendered-atom distance as an input.  Group features are
computed from frozen candidate scores, candidate-to-candidate geometry, and
small image-only crops around the highest-presence candidates.  Protected
rows are context only, never negative labels.  Strict group labels are used
only as the supervised target (a group is positive iff it contains at least
one strict-positive row).

The protocol is fixed: train_core groups only; evaluate each core variant and
stress separately; balanced group sampling; one model and one prevalence
derived logit gate; no threshold sweep, GPU, checkpoint, or sealed data.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.data import ChemVectorIterableDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k/shards"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-group-abstention-cpu-preflight-20261007-r1/report.json"

SEED = 20261013
STEPS = 300
BATCH = 256
TOP_CANDIDATES = 8
PATCH_SIZE = 32
PIXEL_FEATURES_PER_CANDIDATE = 9


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_features(image: Image.Image, x_norm: float, y_norm: float) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    cx, cy = round(float(x_norm) * width), round(float(y_norm) * height)
    half = PATCH_SIZE // 2
    crop = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
    left, top = cx - half, cy - half
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, left + PATCH_SIZE), min(height, top + PATCH_SIZE)
    crop[y0 - top:y1 - top, x0 - left:x1 - left] = gray[y0:y1, x0:x1]
    ink = (255.0 - crop.astype(np.float32)) / 255.0
    mid = half
    q = [ink[:mid, :mid].mean(), ink[:mid, mid:].mean(),
         ink[mid:, :mid].mean(), ink[mid:, mid:].mean()]
    return np.asarray([
        ink.mean(), ink[:16, :16].mean(), ink[mid - 1:mid + 1].mean(),
        ink[:, mid - 1:mid + 1].mean(), *q,
        ink.max(), float((ink > 0.25).mean()),
    ], dtype=np.float32)


def group_features(rows: list[dict], image: Image.Image) -> tuple[np.ndarray, int]:
    """Build one image-only/geometry feature vector; no distance field read."""
    if not rows:
        raise ValueError("empty candidate group")
    score = np.asarray([float(row["raw_score"]) for row in rows], dtype=np.float32)
    presence = np.asarray([float(row["presence_logit"]) for row in rows], dtype=np.float32)
    points = np.asarray([[float(row["x_px"]), float(row["y_px"])] for row in rows], dtype=np.float32)
    order = np.argsort(-presence, kind="stable")
    top = order[:TOP_CANDIDATES]
    top_score, top_presence = score[top], presence[top]
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances += np.eye(len(points), dtype=np.float32) * 1e6
        nearest = distances.min(axis=1)
    else:
        nearest = np.asarray([768.0], dtype=np.float32)
    pixels = np.concatenate([patch_features(image, rows[index]["x_norm"], rows[index]["y_norm"]) for index in top])
    # Pixel summaries are on the highest-presence candidates only.  Candidate
    # geometry (not GT geometry) supplies competition/density cues.
    pixel_summary = np.asarray([
        pixels.mean(axis=0), pixels.max(axis=0), pixels[0],
    ], dtype=np.float32).reshape(-1)
    scalar = np.asarray([
        math.log1p(len(rows)), float((presence > 0.0).mean()),
        float(score.max()), float(score.mean()), float(score.std()),
        float(top_score.mean()), float(top_presence.max()), float(top_presence.mean()),
        float(top_presence[0] - top_presence[min(1, len(top_presence) - 1)]),
        float(nearest.min() / 768.0), float(np.median(nearest) / 768.0),
        float((nearest < 24.0).mean()), float((nearest < 48.0).mean()),
    ], dtype=np.float32)
    label = int(any(row["label"] == "strict_positive" for row in rows))
    return np.concatenate([scalar, pixel_summary]), label


class GroupHead(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dimensions), nn.Linear(dimensions, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate(rows: list[dict], logits: np.ndarray, groups: list[dict]) -> dict:
    selected = logits >= 0.0
    labels = np.asarray([row["label"] for row in groups], dtype=bool)
    true_groups = int((selected & labels).sum())
    selected_keys = {row["group"] for row, keep in zip(groups, selected) if keep}
    candidate_count = sum(len(row) for row in rows if row["sample_key"] in selected_keys)
    strict_candidates = sum(
        item["label"] == "strict_positive" for item in rows if item["sample_key"] in selected_keys
    )
    all_strict = sum(item["label"] == "strict_positive" for item in rows)
    return {
        "groups": len(groups), "positive_groups": int(labels.sum()),
        "selected_groups": int(selected.sum()), "selected_positive_groups": true_groups,
        "group_precision": true_groups / max(int(selected.sum()), 1),
        "group_recall": true_groups / max(int(labels.sum()), 1),
        "retained_candidates": int(candidate_count),
        "retained_strict_candidates": int(strict_candidates),
        "strict_candidate_recall_after_group_gate": strict_candidates / max(all_strict, 1),
    }


def load_rows() -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[(row["replay_split"], row["variant"], row["sample_key"])].append(row)
    return grouped


def load_manifests() -> dict[str, dict]:
    result = {}
    for split, name in (("train_core", "train-core-manifest.json"), ("dev_core", "dev-core-manifest.json"), ("dev_stress", "dev-stress-manifest.json")):
        payload = json.loads((CORE / name).read_text(encoding="utf-8"))
        for row in payload["rows"]:
            result[row["sample_key"]] = {**row, "replay_split": split}
    return result


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"]["jsonl_complete_and_valid"] or not audit["gate"]["matches_forward_aggregate"]:
        raise ValueError("candidate cache audit is not ready")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    grouped = load_rows()
    manifests = load_manifests()
    wanted = set(manifests)
    # Coordinate-collision samples are intentionally absent from the replay
    # cache.  They are excluded fail-closed, rather than reconstructed from
    # aggregate counts or silently treated as negative groups.
    skipped = sorted(wanted - {key for _, _, key in grouped})
    manifests = {key: value for key, value in manifests.items() if key not in set(skipped)}
    wanted = set(manifests)
    features: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    rows_for_eval: dict[str, list[dict]] = {}
    shards = sorted(str(path) for path in DATA.glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError(f"expected 20 train shards, got {len(shards)}")
    seen = set()
    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        key = sample["key"]
        if key not in wanted:
            continue
        split = manifests[key]["replay_split"]
        variant = manifests[key]["variant"]
        group = grouped.get((split, variant, key))
        if not group:
            raise ValueError(f"missing candidate group for non-skipped sample {key}")
        vector, label = group_features(group, sample["image"])
        features[key] = vector
        labels[key] = label
        rows_for_eval[key] = group
        seen.add(key)
    if seen != wanted:
        raise ValueError(f"missing images for {len(wanted - seen)} manifest samples")

    train_keys = [key for key in sorted(features) if manifests[key]["replay_split"] == "train_core"]
    train_labels = np.asarray([labels[key] for key in train_keys], dtype=np.float32)
    positive = np.flatnonzero(train_labels > 0.0)
    negative = np.flatnonzero(train_labels == 0.0)
    if not len(positive) or not len(negative):
        raise ValueError("both positive and negative training groups are required")
    x_train = np.asarray([features[key] for key in train_keys], dtype=np.float32)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    model = GroupHead(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    generator = np.random.default_rng(SEED)
    x_tensor, y_tensor = torch.tensor((x_train - mean) / std), torch.tensor(train_labels)
    events = []
    for step in range(1, STEPS + 1):
        index = np.concatenate([generator.choice(positive, BATCH // 2, replace=True), generator.choice(negative, BATCH // 2, replace=True)])
        generator.shuffle(index)
        logits = model(x_tensor[torch.tensor(index)])
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor[torch.tensor(index)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    model.eval()
    with torch.no_grad():
        all_keys = sorted(features)
        all_logits = model(torch.tensor(np.asarray([(features[key] - mean) / std for key in all_keys], dtype=np.float32))).numpy()
    # The gate is fixed by the train group prevalence, not fitted by searching
    # thresholds: balanced sampling requires this prior correction.
    prior = float(train_labels.mean())
    gate = math.log(prior / (1.0 - prior))
    # Correct balanced-sampler logits back to the train-group prior.
    all_logits = all_logits + gate
    by_key = dict(zip(all_keys, all_logits))
    train_groups = [{"group": key, "label": labels[key]} for key in train_keys]
    train_eval_rows = [rows_for_eval[key] for key in train_keys]
    train_flat = [row for rows in train_eval_rows for row in rows]
    by_variant = {}
    for split, variants in (("dev_core", ("coordgen", "original", "rdkit")), ("dev_stress", ("stress",))):
        for variant in variants:
            keys = [key for key in sorted(features) if manifests[key]["replay_split"] == split and manifests[key]["variant"] == variant]
            groups = [{"group": key, "label": labels[key]} for key in keys]
            flat = [row for key in keys for row in rows_for_eval[key]]
            by_variant[f"{split}/{variant}"] = evaluate(flat, np.asarray([by_key[key] for key in keys]), groups)
    train_result = evaluate(train_flat, np.asarray([by_key[key] for key in train_keys]), train_groups)
    result = {
        "schema_version": "chemvector-style-factorial-group-abstention-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "group_abstention_cpu_only",
        "gpu_used": False, "optimizer_steps": STEPS, "batch_size": BATCH,
        "checkpoint_written": False, "sealed_test_read": False,
        "cache_sha256": digest(CACHE), "cache_audit_sha256": digest(CACHE_AUDIT),
        "seed": SEED, "train_groups": len(train_keys), "train_positive_groups": int(train_labels.sum()),
        "replay_skipped_samples": skipped,
        "feature_dim": int(x_train.shape[1]), "top_candidates_for_pixels": TOP_CANDIDATES,
        "fixed_logit_gate": gate, "gate_definition": "logit(train_core positive-group prevalence), no threshold sweep",
        "train": train_result, "dev_by_variant": by_variant, "events": events,
        "input_contract": {"gt_distance_used_as_input": False, "protected_as_negative": False,
                           "near_unknown_as_negative": False, "sealed_test_read": False},
    }
    core = [by_variant[f"dev_core/{variant}"] for variant in ("coordgen", "original", "rdkit")]
    result["gate"] = {
        "each_core_group_precision_ge_085": all(x["group_precision"] >= .85 for x in core),
        "each_core_group_recall_ge_050": all(x["group_recall"] >= .50 for x in core),
        "stress_reported_separately": True, "training_ready": False,
        "reason": "group abstention CPU preflight only; no GPU authorization",
    }
    result["gate"]["training_ready"] = result["gate"]["each_core_group_precision_ge_085"] and result["gate"]["each_core_group_recall_ge_050"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "dev_by_variant": by_variant, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
