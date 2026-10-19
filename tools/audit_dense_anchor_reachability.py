"""Read-only audit of all dense anchor cells for current-initial raw gaps.

The production decoder only exposes local heatmap peaks.  This audit asks a
separate question: for each target missed by the raw 128 proposal set, does
*any* 192x192 dense cell already regress an anchor within the 6px contract?
It never changes decoding, trains a branch, writes a checkpoint, or reads
sealed identities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

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
    "CHEMVECTOR_DENSE_ANCHOR_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-dense-anchor-reachability-20261006"),
))
IMAGE_SIZE = 768.0
MATCH_RADIUS = 6.0
RANK_K = (128, 256, 512, 1024, 2048)


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed dense replay")
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("Initial checkpoint contract unavailable")

    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys), key=lambda r: r["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")

    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    records: list[dict] = []
    by_style: dict[str, Counter] = defaultdict(Counter)
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, atom_output, _, raw, _, _, _ = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            mapping, _ = match_atom_targets(raw, tensors, config)
            matched = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
            target_count = int(tensors["atom_mask"].sum())
            missing = sorted(set(range(target_count)) - matched)
            if not missing:
                continue

            # Every dense cell, not only local maxima.  ``anchors`` are in
            # normalized 768px image coordinates, one [x,y] per heatmap cell.
            dense_anchors = atom_output["anchors"][0].permute(1, 2, 0).reshape(-1, 2) * IMAGE_SIZE
            dense_scores = atom_output["heatmap_logits"].sigmoid()[0, 0].reshape(-1)
            height, width = atom_output["heatmap_logits"].shape[-2:]
            local_scores = atom_output["heatmap_logits"].sigmoid()[0, 0]
            local_peaks = local_scores == F.max_pool2d(local_scores[None, None], 3, 1, 1)[0, 0]
            target_anchors = tensors["atom_anchors"][0, :target_count] * IMAGE_SIZE
            style = str(sample["meta"]["profile_id"])
            atoms = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}

            # Evaluate all missing targets together to keep the replay cheap.
            distances = torch.cdist(target_anchors[missing].float(), dense_anchors.float())
            nearest_dist, nearest_idx = distances.min(dim=1)
            score_order = torch.argsort(dense_scores, descending=True)
            rank_of_cell = torch.empty_like(score_order)
            rank_of_cell[score_order] = torch.arange(len(score_order), device=score_order.device)
            for local, target_id in enumerate(missing):
                d = float(nearest_dist[local])
                idx = int(nearest_idx[local])
                score_rank = int(rank_of_cell[idx]) + 1
                target_xy = target_anchors[target_id]
                target_cell_x = min(width - 1, max(0, int(float(target_xy[0] / IMAGE_SIZE) * width)))
                target_cell_y = min(height - 1, max(0, int(float(target_xy[1] / IMAGE_SIZE) * height)))
                target_cell_idx = target_cell_y * width + target_cell_x
                target_cell_distance = float(torch.linalg.vector_norm(
                    dense_anchors[target_cell_idx] - target_xy
                ))
                target_cell_rank = int(rank_of_cell[target_cell_idx]) + 1
                within = d <= MATCH_RADIUS
                record = {
                    "sample_key": row["sample_key"],
                    "style": style,
                    "target_id": int(target_id),
                    "element": str(atoms[target_id]["element"]),
                    "role": "text" if bool(atoms[target_id]["visible_text"]) else "implicit",
                    "nearest_all_cell_distance_px": d,
                    "nearest_all_cell_score": float(dense_scores[idx]),
                    "nearest_all_cell_heatmap_rank": score_rank,
                    "all_cell_within_6px": bool(within),
                    "all_cell_in_rank_k": {str(k): bool(score_rank <= k) for k in RANK_K},
                    "gt_cell_distance_px": target_cell_distance,
                    "gt_cell_score": float(dense_scores[target_cell_idx]),
                    "gt_cell_heatmap_rank": target_cell_rank,
                    "gt_cell_is_local_peak": bool(local_peaks[target_cell_y, target_cell_x]),
                }
                records.append(record)
                by_style[style]["targets"] += 1
                by_style[style]["within_6"] += int(within)
                by_style[style]["rank_le_128"] += int(score_rank <= 128)
                by_style[style]["rank_le_512"] += int(score_rank <= 512)
            if index % 200 == 0:
                print(json.dumps({"images": index, "gap_targets": len(records)}), flush=True)

    def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in by_style.values())

    result = {
        "schema_version": "chemvector-dense-anchor-reachability-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "images": len(rows),
        "raw_gap_targets": len(records),
        "summary": {
            "targets": len(records),
            "all_cell_within_6px": sum(bool(r["all_cell_within_6px"]) for r in records),
            "all_cell_rank_le_128": sum(bool(r["all_cell_in_rank_k"]["128"]) for r in records),
            "all_cell_rank_le_512": sum(bool(r["all_cell_in_rank_k"]["512"]) for r in records),
            "all_cell_rank_le_2048": sum(bool(r["all_cell_in_rank_k"]["2048"]) for r in records),
            "all_cell_reachable_but_not_top128": sum(
                bool(r["all_cell_within_6px"]) and not bool(r["all_cell_in_rank_k"]["128"])
                for r in records
            ),
            "gt_cell_within_6px": sum(r["gt_cell_distance_px"] <= MATCH_RADIUS for r in records),
            "gt_cell_rank_le_128": sum(r["gt_cell_heatmap_rank"] <= 128 for r in records),
            "gt_cell_rank_le_512": sum(r["gt_cell_heatmap_rank"] <= 512 for r in records),
            "gt_cell_local_peak": sum(bool(r["gt_cell_is_local_peak"]) for r in records),
        },
        "by_style": {key: dict(value) for key, value in sorted(by_style.items())},
        "records": records,
        "interpretation": {
            "within_6px_any_cell": "dense coordinate evidence exists even if heatmap decoding suppresses it",
            "not_within_6px_any_cell": "upstream dense feature/anchor regression remains insufficient",
            "rank_k": "heatmap score rank of the best-coordinate cell; diagnostic only",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **result["summary"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
