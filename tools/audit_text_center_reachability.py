"""Compare dense heatmap evidence at text centers versus atom anchors."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TEXT_CENTER_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-text-center-reachability-20261006"),
))
IMAGE_SIZE = 768.0


def _rank_map(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device=order.device)
    return ranks + 1


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys), key=lambda r: r["sample_key"])
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    records: list[dict] = []
    by_profile: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            _, atom_output, _, raw, _, _, _ = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            mapping, _ = match_atom_targets(raw, tensors, config)
            matched = {int(v) for v in mapping[0].tolist() if int(v) >= 0}
            target_count = int(tensors["atom_mask"].sum())
            missing_text = [i for i in range(target_count) if i not in matched and bool(tensors["atom_text_mask"][0, i])]
            if not missing_text:
                continue
            scores = atom_output["heatmap_logits"].sigmoid()[0, 0].flatten()
            ranks = _rank_map(scores)
            h, w = atom_output["heatmap_logits"].shape[-2:]
            anchors = atom_output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
            centers = atom_output["text_centers"][0].permute(1, 2, 0).reshape(-1, 2)
            target_anchors = tensors["atom_anchors"][0, :target_count]
            target_centers = tensors["atom_text_center"][0, :target_count]
            profile = str(sample["meta"]["profile_id"])
            for target_id in missing_text:
                anchor = target_anchors[target_id]
                center = target_centers[target_id]
                anchor_cell_x = min(w - 1, max(0, int(float(anchor[0]) * w)))
                anchor_cell_y = min(h - 1, max(0, int(float(anchor[1]) * h)))
                center_cell_x = min(w - 1, max(0, int(float(center[0]) * w)))
                center_cell_y = min(h - 1, max(0, int(float(center[1]) * h)))
                anchor_cell = anchor_cell_y * w + anchor_cell_x
                center_cell = center_cell_y * w + center_cell_x
                nearest_anchor = torch.linalg.vector_norm(anchors - anchor[None], dim=1).argmin()
                nearest_center = torch.linalg.vector_norm(centers - center[None], dim=1).argmin()
                record = {
                    "sample_key": row["sample_key"],
                    "profile": profile,
                    "target_id": target_id,
                    "element": str(sample["layers"]["atoms"][target_id]["element"]),
                    "anchor_cell_rank": int(ranks[anchor_cell]),
                    "text_center_cell_rank": int(ranks[center_cell]),
                    "text_center_cell_anchor_distance_px": float(torch.linalg.vector_norm(anchors[center_cell] - anchor) * IMAGE_SIZE),
                    "nearest_pred_anchor_rank": int(ranks[nearest_anchor]),
                    "nearest_pred_anchor_distance_px": float(torch.linalg.vector_norm(anchors[nearest_anchor] - anchor) * IMAGE_SIZE),
                    "nearest_pred_text_center_rank": int(ranks[nearest_center]),
                    "nearest_pred_text_center_distance_px": float(torch.linalg.vector_norm(centers[nearest_center] - center) * IMAGE_SIZE),
                    "target_anchor_px": (anchor * IMAGE_SIZE).tolist(),
                    "target_text_center_px": (center * IMAGE_SIZE).tolist(),
                }
                records.append(record)
                bucket = by_profile[profile]
                bucket["targets"] += 1
                bucket["anchor_rank_le_128"] += int(record["anchor_cell_rank"] <= 128)
                bucket["center_rank_le_128"] += int(record["text_center_cell_rank"] <= 128)
                bucket["nearest_center_rank_le_128"] += int(record["nearest_pred_text_center_rank"] <= 128)
            if index % 200 == 0:
                print(json.dumps({"images": index, "targets": len(records)}), flush=True)
    result = {
        "schema_version": "chemvector-text-center-reachability-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "targets": len(records),
        "summary": {
            "anchor_cell_rank_le_128": sum(r["anchor_cell_rank"] <= 128 for r in records),
            "text_center_cell_rank_le_128": sum(r["text_center_cell_rank"] <= 128 for r in records),
            "nearest_pred_text_center_rank_le_128": sum(r["nearest_pred_text_center_rank"] <= 128 for r in records),
            "anchor_cell_rank_median": float(torch.tensor([r["anchor_cell_rank"] for r in records], dtype=torch.float32).median()) if records else None,
            "text_center_cell_rank_median": float(torch.tensor([r["text_center_cell_rank"] for r in records], dtype=torch.float32).median()) if records else None,
            "nearest_pred_text_center_rank_median": float(torch.tensor([r["nearest_pred_text_center_rank"] for r in records], dtype=torch.float32).median()) if records else None,
            "text_center_cell_anchor_distance_le_6": sum(r["text_center_cell_anchor_distance_px"] <= 6.0 for r in records),
            "text_center_cell_anchor_distance_median_px": float(torch.tensor([r["text_center_cell_anchor_distance_px"] for r in records], dtype=torch.float32).median()) if records else None,
        },
        "by_profile": {key: dict(value) for key, value in sorted(by_profile.items())},
        "records": records,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
