"""Read-only frozen replay for the collision-only offset geometry gate.

The exported cache contains only target-independent model/image features.  GT
ownership is used transiently to assign audit labels and is not written to the
cache.  No optimizer, checkpoint, selector, or graph path is touched.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.continuation import structure_forward

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r2"
PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2/manifest.json"
CANDIDATE = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
OUTPUT = ROOT / "outputs/audits/chemvector-offset-geometry-feature-cache-20261017-r1/candidates.jsonl"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
GRID = 192
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ink_fraction(pixels: torch.Tensor, x: float, y: float, radius: int = 16) -> float:
    gray = pixels[0].mean(0)
    cx, cy = int(round(x)), int(round(y))
    crop = gray[max(0, cy - radius):min(gray.shape[0], cy + radius + 1),
                max(0, cx - radius):min(gray.shape[1], cx + radius + 1)]
    return float((crop < 0.90).float().mean()) if crop.numel() else 0.0


def _prominence(heat: torch.Tensor, cy: int, cx: int) -> float:
    value = float(heat[cy, cx])
    y0, y1 = max(0, cy - 2), min(heat.shape[0], cy + 3)
    x0, x1 = max(0, cx - 2), min(heat.shape[1], cx + 3)
    patch = heat[y0:y1, x0:x1].clone()
    patch[cy - y0, cx - x0] = -float("inf")
    return value - float(patch.max()) if patch.numel() else 0.0


def _label(row: dict) -> str:
    if bool(row["selected_baseline"]):
        return "protected"
    distance = float(row["nearest_target_distance_px"])
    # Ownership is deliberately strict: a candidate that is close but lost the
    # one-to-one assignment remains unknown, never a positive training label.
    if bool(row.get("positive")) and row.get("target_id") is not None and distance <= 6.0:
        return "strict_positive"
    if distance <= 16.0:
        return "unknown"
    return "hard_negative"


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint binding failed")
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if candidate.get("training_performed") is not False or candidate.get("sealed_test_read") is not False:
        raise ValueError("candidate manifest is not read-only")
    if candidate.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("candidate manifest initial binding failed")
    rows = json.loads(PLAN.read_text(encoding="utf-8"))["rows"]
    raw_by_sample: dict[str, list[dict]] = {}
    for row in candidate["candidate_records"]:
        if row.get("source") == "raw_proposal":
            raw_by_sample.setdefault(row["sample_key"], []).append(row)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device).eval().requires_grad_(False)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    total = 0
    labels = {"protected": 0, "strict_positive": 0, "unknown": 0, "hard_negative": 0}
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for index, plan_row in enumerate(sorted(rows, key=lambda r: r["sample_key"]), 1):
            key = plan_row["sample_key"]
            source_rows = raw_by_sample.get(key, [])
            image_path = DATA / "samples" / (key + ".png")
            if not source_rows or not image_path.exists():
                continue
            sample = load_sample(DATA, key, purpose="diagnostic")
            batch = collator([sample])
            tensors = {name: value.to(device) for name, value in batch.items() if isinstance(value, torch.Tensor)}
            with torch.inference_mode():
                features, atom_output, _bond, raw, _bonds, _set, _selected = structure_forward(
                    model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
                )
            heat = atom_output["heatmap_logits"][0, 0].sigmoid()
            role = atom_output["role_logits"][0].softmax(0)[1]
            extent = atom_output["text_extents"][0]
            centers = atom_output["text_centers"][0]
            anchors = atom_output["anchors"][0]
            for row in source_rows:
                cell = int(row["cell_id"])
                cy, cx = divmod(cell, GRID)
                ax, ay = (float(raw["atom_geometry"][0, int(row["slot"]), 0]) * IMAGE_SIZE,
                          float(raw["atom_geometry"][0, int(row["slot"]), 1]) * IMAGE_SIZE)
                center_x, center_y = float(centers[0, cy, cx] * IMAGE_SIZE), float(centers[1, cy, cx] * IMAGE_SIZE)
                extent_x, extent_y = float(extent[0, cy, cx] * IMAGE_SIZE), float(extent[1, cy, cx] * IMAGE_SIZE)
                offset_norm = math.hypot(center_x - float(anchors[0, cy, cx] * IMAGE_SIZE),
                                         center_y - float(anchors[1, cy, cx] * IMAGE_SIZE)) / max(math.hypot(extent_x, extent_y), 1.0)
                out = {
                    "candidate_id": f"{key}:{int(row['slot'])}",
                    "sample_key": key, "split": row["split"], "profile_id": plan_row["profile_id"],
                    "protected": bool(row["selected_baseline"]),
                    "role_text_prob": float(role[cy, cx]),
                    "peak_prominence": _prominence(heat, cy, cx),
                    "ink_fraction": _ink_fraction(tensors["pixel_values"], ax, ay),
                    "text_extent_x_px": extent_x, "text_extent_y_px": extent_y,
                    "offset_norm": offset_norm, "heat_score": float(heat[cy, cx]),
                    "anchor_x_px": ax, "anchor_y_px": ay,
                    "label": _label(row),
                }
                labels[out["label"]] += 1
                total += 1
                handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 100 == 0:
                print(json.dumps({"images": index, "candidate_rows": total}), flush=True)
    after = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise RuntimeError("frozen model changed during replay")
    report = {
        "schema_version": "chemvector-offset-geometry-feature-cache-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": use_cuda, "sealed_test_read": False, "checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": file_digest(INITIAL),
        "candidate_manifest_sha256": digest(CANDIDATE), "feature_cache_sha256": digest(OUTPUT),
        "candidate_rows": total, "label_counts": labels,
        "forbidden_written": ["target_id", "nearest_distance_px", "nearest_target_distance_px", "layers", "graph"],
    }
    (OUTPUT.parent / "replay-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
