"""Read-only pixel/geometry diagnostic for the cap/stroke panel.

The frozen initial model is replayed on the rendered quarantine panel.  For
visible O/N atoms we compare raw-matched targets with raw gaps and record
same-source glyph geometry plus dense atom heatmap evidence.  Target geometry
is used only to label/aggregate the audit; it is never supplied to the model
or decoder.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_DIAG_PLAN", str(ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2")))
DATA = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_DIAG_DATA", str(ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r2")))
OUT = Path(os.environ.get("CHEMVECTOR_CAP_STROKE_DIAG_OUTPUT", str(ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-pixel-geometry-diagnostic-20261017-r1/report.json")))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True)
    result = torch.empty_like(order)
    result[order] = torch.arange(order.numel(), device=order.device)
    return result + 1


def _ink_stats(image, box):
    """Simple same-source raster stats; no OCR or connected components."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        return {"bbox_width_px": 0, "bbox_height_px": 0, "ink_fraction": None,
                "dark_pixel_count": 0, "border_ink_fraction": None}
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    values = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(y1 - y0, x1 - x0)
    dark = values < 220
    border = torch.zeros_like(dark)
    border[0] = True; border[-1] = True; border[:, 0] = True; border[:, -1] = True
    return {"bbox_width_px": x1 - x0, "bbox_height_px": y1 - y0,
            "ink_fraction": float(dark.float().mean()), "dark_pixel_count": int(dark.sum()),
            "border_ink_fraction": float(dark[border].float().mean())}


def _one(model, config, payload, collator, row):
    sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
    batch = collator([sample])
    tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.inference_mode():
        _, atom_output, _, raw, _, _, _ = structure_forward(
            model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
    mapping, _ = match_atom_targets(raw, tensors, config)
    matched = {int(v) for v in mapping[0].tolist() if int(v) >= 0}
    heat = atom_output["heatmap_logits"][0, 0].sigmoid()
    h, w = heat.shape
    peaks = heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    py, px = torch.where(peaks)
    ps = heat[py, px]
    order = torch.argsort(ps, descending=True)
    py, px, ps = py[order], px[order], ps[order]
    peak_anchors = atom_output["anchors"][0].permute(1, 2, 0)[py, px]
    all_anchors = atom_output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    all_scores = heat.reshape(-1)
    all_ranks = _rank(all_scores)
    rows = []
    atoms = sample["layers"]["atoms"]
    for target_id, atom in enumerate(atoms):
        if atom.get("element") not in {"O", "N"} or not atom.get("visible_text"):
            continue
        anchor = tensors["atom_anchors"][0, target_id]
        center = tensors["atom_text_center"][0, target_id]
        cell_x = min(w - 1, max(0, int(float(anchor[0]) * w)))
        cell_y = min(h - 1, max(0, int(float(anchor[1]) * h)))
        flat = cell_y * w + cell_x
        distances = torch.linalg.vector_norm((peak_anchors - anchor[None]) * IMAGE_SIZE, dim=1)
        near = int(torch.argmin(distances))
        near_all_dist = torch.linalg.vector_norm((all_anchors - anchor[None]) * IMAGE_SIZE, dim=1)
        near_all = int(torch.argmin(near_all_dist))
        box = atom.get("bbox")
        center_px = (center * IMAGE_SIZE).tolist()
        anchor_px = (anchor * IMAGE_SIZE).tolist()
        geometry = {"text_length": len(str(atom.get("text", ""))),
                    "text_center_anchor_distance_px": float(torch.linalg.vector_norm(center - anchor) * IMAGE_SIZE),
                    "bbox_center_x_px": (box[0] + box[2]) / 2 if box else None,
                    "bbox_center_y_px": (box[1] + box[3]) / 2 if box else None}
        geometry.update(_ink_stats(sample["image"], box) if box else {})
        rows.append({
            "sample_key": row["sample_key"], "profile_id": row["profile_id"], "split": row["split"],
            "target_id": target_id, "element": atom["element"], "text": atom.get("text", ""),
            "raw_matched": target_id in matched, "target_anchor_px": anchor_px,
            "target_text_center_px": center_px, "glyph_bbox": box, "glyph": geometry,
            "anchor_cell_heat_score": float(heat[cell_y, cell_x]),
            "anchor_cell_rank": int(all_ranks[flat]),
            "anchor_cell": [cell_x, cell_y],
            "nearest_peak_distance_px": float(distances[near]),
            "nearest_peak_score": float(ps[near]), "nearest_peak_rank": near + 1,
            "nearest_peak_cell": [int(px[near]), int(py[near])],
            "nearest_dense_distance_px": float(near_all_dist[near_all]),
            "nearest_dense_score": float(all_scores[near_all]),
            "nearest_dense_rank": int(all_ranks[near_all]),
        })
    return rows


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    plan = json.loads((PLAN / "manifest.json").read_text(encoding="utf-8"))
    rows = plan["rows"]
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    if summary.get("accepted_images", 0) < 950:
        raise ValueError("render quality gate failed")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
    records = []
    for index, row in enumerate(sorted(rows, key=lambda r: r["sample_key"]), 1):
        if not (DATA / "samples" / (row["sample_key"] + ".png")).exists():
            continue
        records.extend(_one(model, config, payload, collator, row))
        if index % 100 == 0:
            print(json.dumps({"images": index, "visible_on": len(records)}), flush=True)
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen model state changed")
    by = defaultdict(lambda: defaultdict(list))
    for rec in records:
        status = "matched" if rec["raw_matched"] else "gap"
        group = by[rec["profile_id"]][f"{rec['element']}/{status}"]
        group.append(rec)
    def summarize(items):
        if not items: return {"count": 0}
        def q(key):
            vals = torch.tensor([x[key] for x in items], dtype=torch.float32)
            return {"mean": float(vals.mean()), "median": float(vals.median()),
                    "p90": float(vals.quantile(.9))}
        return {"count": len(items), "anchor_heat_score": q("anchor_cell_heat_score"),
                "anchor_cell_rank": q("anchor_cell_rank"), "nearest_peak_distance_px": q("nearest_peak_distance_px"),
                "nearest_peak_rank": q("nearest_peak_rank"), "nearest_peak_score": q("nearest_peak_score"),
                "nearest_dense_distance_px": q("nearest_dense_distance_px"), "nearest_dense_rank": q("nearest_dense_rank"),
                "glyph_bbox_width_px": q("glyph")} if False else {
                    "count": len(items), "anchor_heat_score": q("anchor_cell_heat_score"),
                    "anchor_cell_rank": q("anchor_cell_rank"), "nearest_peak_distance_px": q("nearest_peak_distance_px"),
                    "nearest_peak_rank": q("nearest_peak_rank"), "nearest_peak_score": q("nearest_peak_score"),
                    "nearest_dense_distance_px": q("nearest_dense_distance_px"), "nearest_dense_rank": q("nearest_dense_rank"),
                }
    result = {"schema_version": "chemvector-oa13-oa33-cap-stroke-pixel-geometry-diagnostic-0.1", "complete": True,
              "read_only": True, "training_performed": False, "optimizer_steps": 0, "gpu_used": True,
              "device": "cuda:0", "sealed_test_read": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": file_digest(INITIAL),
              "state_before": before, "state_after": after, "panel_manifest_sha256": digest(PLAN / "manifest.json"),
              "panel_summary_sha256": digest(DATA / "summary.json"), "visible_on_records": len(records),
              "by_profile_element_status": {p: {k: summarize(v) for k, v in sorted(groups.items())} for p, groups in sorted(by.items())},
              "records": records,
              "diagnostic_contract": {"target_geometry_posthoc_only": True, "ocr_used": False,
                                      "nearest_dense_definition": "all 192x192 predicted anchor cells; no target input",
                                      "nearest_peak_definition": "heatmap local maxima sorted by frozen score"},
              "gate": {"frozen_state_unchanged": before == after, "training_admissible": False,
                       "next": "compare matched versus gap score/rank/distance; no training or decoder change"}}
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "visible_on_records": len(records), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
