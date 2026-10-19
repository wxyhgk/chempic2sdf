"""Read-only diagnosis of dense peak reachability for raw-gap targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
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
OUTPUT = ROOT / "outputs/audits/chemvector-dense-peak-gap-20261006"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed dense replay")
    if file_digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Initial checkpoint changed")
    fixed = json.loads(COHORT.read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    if len(rows) != 1200:
        raise ValueError(f"Expected fixed 1,200 rows, found {len(rows)}")
    loader = RecordLoader(manifest)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    records = []
    style_totals = defaultdict(Counter)
    element_totals = defaultdict(Counter)
    with torch.no_grad():
        for attempt, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, atom_output, _, raw, _, _, _ = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            mapping, _ = match_atom_targets(raw, tensors, config)
            target_count = int(tensors["atom_mask"].sum())
            matched = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
            missing = sorted(set(range(target_count)) - matched)
            if not missing:
                continue
            heat = atom_output["heatmap_logits"].sigmoid()[0, 0]
            peaks = heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
            peak_y, peak_x = torch.where(peaks)
            peak_scores = heat[peak_y, peak_x]
            order = torch.argsort(peak_scores, descending=True)
            peak_x = peak_x[order]
            peak_y = peak_y[order]
            peak_scores = peak_scores[order]
            anchors = atom_output["anchors"][0].permute(1, 2, 0)[peak_y, peak_x]
            target_anchors = tensors["atom_anchors"][0]
            style = str(sample["meta"]["profile_id"])
            atoms = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            for target_id in missing:
                target = target_anchors[target_id]
                distances = torch.linalg.vector_norm(anchors - target[None], dim=-1) * config.image_size
                nearest_index = int(torch.argmin(distances))
                nearest_distance = float(distances[nearest_index])
                element = str(atoms[target_id]["element"])
                role = "text" if bool(atoms[target_id]["visible_text"]) else "implicit"
                row_record = {
                    "sample_key": row["sample_key"],
                    "style": style,
                    "target_id": target_id,
                    "element": element,
                    "role": role,
                    "target_xy_px": (target * config.image_size).tolist(),
                    "nearest_peak_distance_px": nearest_distance,
                    "nearest_peak_rank": nearest_index + 1,
                    "nearest_peak_score": float(peak_scores[nearest_index]),
                    "nearest_peak_cell": [int(peak_x[nearest_index]), int(peak_y[nearest_index])],
                    "dense_peak_within_6px": nearest_distance <= 6.0,
                    "dense_peak_in_top128": nearest_index < 128,
                    "dense_peak_in_top192": nearest_index < 192,
                    "dense_peak_in_top256": nearest_index < 256,
                    "dense_peak_count": int(len(peak_scores)),
                }
                records.append(row_record)
                style_totals[style].update(
                    targets=1,
                    within_6=int(nearest_distance <= 6.0),
                    top128=int(nearest_index < 128),
                    top192=int(nearest_index < 192),
                    top256=int(nearest_index < 256),
                )
                element_totals[element].update(
                    targets=1,
                    within_6=int(nearest_distance <= 6.0),
                    top128=int(nearest_index < 128),
                )
            if attempt % 200 == 0:
                print(json.dumps({"images": attempt, "gap_targets": len(records)}), flush=True)
    result = {
        "schema_version": "chemvector-dense-peak-gap-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "images": len(rows),
        "raw_gap_targets": len(records),
        "by_style": {key: dict(value) for key, value in sorted(style_totals.items())},
        "by_element": {key: dict(value) for key, value in sorted(element_totals.items())},
        "records": records,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "raw_gap_targets": len(records), "by_style": result["by_style"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
