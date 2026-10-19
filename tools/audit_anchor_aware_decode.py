"""Forward-only shadow of geometry-aware dense-cell decoding.

The production decoder keeps heatmap local maxima.  This audit instead scans a
large score-ranked cell pool and greedily suppresses candidates by *predicted
anchor distance*.  It is target-independent and never changes the production
checkpoint or selector.  The purpose is to test whether the current dense
field already contains useful endpoints that grid-local-max suppression hides.
"""

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
    "CHEMVECTOR_ANCHOR_AWARE_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-anchor-aware-decode-20261006"),
))
POOL = 4096
CAPACITY = 128
RADII_PX = (4.0, 6.0, 8.0, 12.0)


def _target_ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(v) for v in mapping[0].tolist() if int(v) >= 0}


def _shadow(atom_output: dict, radius_px: float, config) -> dict:
    scores = atom_output["heatmap_logits"].sigmoid()[0, 0].flatten()
    order = torch.argsort(scores, descending=True)[: min(POOL, len(scores))]
    height, width = atom_output["heatmap_logits"].shape[-2:]
    anchors = atom_output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    chosen: list[int] = []
    radius = radius_px / config.image_size
    for raw_index in order.tolist():
        point = anchors[raw_index]
        if chosen:
            old = anchors[torch.as_tensor(chosen, device=anchors.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= radius):
                continue
        chosen.append(int(raw_index))
        if len(chosen) >= CAPACITY:
            break
    index = torch.as_tensor(chosen, device=anchors.device, dtype=torch.long)
    y = torch.div(index, width, rounding_mode="floor")
    x = index.remainder(width)
    features = atom_output["object_features"][0].permute(1, 2, 0)[y, x]
    roles = atom_output["role_logits"][0].permute(1, 2, 0)[y, x].softmax(-1)
    centers = atom_output["text_centers"][0].permute(1, 2, 0)[y, x]
    extents = atom_output["text_extents"][0].permute(1, 2, 0)[y, x]
    geometry = torch.cat((anchors[index], roles, centers, extents), dim=-1)[None]
    return {
        "atom_features": features[None],
        "atom_geometry": geometry,
        "atom_scores": scores[index][None],
        "atom_indices": index[None],
        "atom_mask": torch.ones((1, len(index)), dtype=torch.bool, device=index.device),
    }


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
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    totals = {
        str(radius): {
            "targets": 0,
            "raw_baseline_matched": 0,
            "selected_baseline_matched": 0,
            "shadow_matched": 0,
            "shadow_selected_matched": 0,
            "shadow_new_vs_raw": 0,
            "shadow_new_vs_selected": 0,
            "shadow_selected_new_vs_selected": 0,
            "shadow_lost_raw_ids": 0,
            "shadow_lost_selected_ids": 0,
            "shadow_selected_lost_selected_ids": 0,
            "images_with_shadow_new": 0,
            "shadow_candidates": 0,
        }
        for radius in RADII_PX
    }
    style_rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, atom_output, bond_output, raw, bonds, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            raw_ids = _target_ids(raw, tensors, config)
            selected_ids = _target_ids(selected, tensors, config)
            target_count = int(tensors["atom_mask"].sum())
            profile = str(sample["meta"]["profile_id"])
            for radius in RADII_PX:
                shadow = _shadow(atom_output, radius, config)
                shadow_ids = _target_ids(shadow, tensors, config)
                shadow_set = model.set_head(
                    shadow["atom_features"], shadow["atom_geometry"], shadow["atom_scores"], shadow["atom_mask"],
                    bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"],
                )
                shadow_selected = {
                    **shadow,
                    "atom_mask": shadow["atom_mask"] & (
                        shadow_set["atom_presence_logits"] >= payload["provenance"]["atom_threshold"]
                    ),
                }
                shadow_selected_ids = _target_ids(shadow_selected, tensors, config)
                key = str(radius)
                total = totals[key]
                total["targets"] += target_count
                total["raw_baseline_matched"] += len(raw_ids)
                total["selected_baseline_matched"] += len(selected_ids)
                total["shadow_matched"] += len(shadow_ids)
                total["shadow_selected_matched"] += len(shadow_selected_ids)
                total["shadow_new_vs_raw"] += len(shadow_ids - raw_ids)
                total["shadow_new_vs_selected"] += len(shadow_ids - selected_ids)
                total["shadow_selected_new_vs_selected"] += len(shadow_selected_ids - selected_ids)
                total["shadow_lost_raw_ids"] += len(raw_ids - shadow_ids)
                total["shadow_lost_selected_ids"] += len(selected_ids - shadow_ids)
                total["shadow_selected_lost_selected_ids"] += len(selected_ids - shadow_selected_ids)
                total["images_with_shadow_new"] += int(bool(shadow_ids - selected_ids))
                total["shadow_candidates"] += int(shadow["atom_mask"].sum())
                style_rows[f"{profile}/r{key}"]["targets"] += target_count
                style_rows[f"{profile}/r{key}"]["new_vs_selected"] += len(shadow_ids - selected_ids)
                style_rows[f"{profile}/r{key}"]["lost_selected"] += len(selected_ids - shadow_ids)
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)

    result = {
        "schema_version": "chemvector-anchor-aware-decode-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "contract": {
            "pool": POOL,
            "capacity": CAPACITY,
            "selection": "descending heatmap score, greedy suppression by predicted-anchor distance",
            "baseline_unchanged": True,
            "target_independent": True,
        },
        "totals": totals,
        "by_profile": {key: dict(value) for key, value in sorted(style_rows.items())},
        "decision": {
            "admit_training": False,
            "admit_deployment": False,
            "reason": "shadow only; compare candidate recall against immutable selected IDs before any model change",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "totals": totals}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
