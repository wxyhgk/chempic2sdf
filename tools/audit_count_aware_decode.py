"""Forward-only audit of predicted-count, anchor-aware atom decoding."""

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
    "CHEMVECTOR_COUNT_AWARE_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-count-aware-decode-20261006"),
))
POOL = 4096
RADII_PX = (4.0, 6.0, 8.0, 12.0)


def _ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(v) for v in mapping[0].tolist() if int(v) >= 0}


def _shadow(atom_output: dict, bonds: dict, radius_px: float, config, predicted_count: int) -> tuple[dict, torch.Tensor]:
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
        if len(chosen) >= predicted_count:
            break
    index = torch.as_tensor(chosen, device=anchors.device, dtype=torch.long)
    y = torch.div(index, width, rounding_mode="floor")
    x = index.remainder(width)
    features = atom_output["object_features"][0].permute(1, 2, 0)[y, x]
    roles = atom_output["role_logits"][0].permute(1, 2, 0)[y, x].softmax(-1)
    centers = atom_output["text_centers"][0].permute(1, 2, 0)[y, x]
    extents = atom_output["text_extents"][0].permute(1, 2, 0)[y, x]
    shadow = {
        "atom_features": features[None],
        "atom_geometry": torch.cat((anchors[index], roles, centers, extents), dim=-1)[None],
        "atom_scores": scores[index][None],
        "atom_indices": index[None],
        "atom_mask": torch.ones((1, len(index)), dtype=torch.bool, device=index.device),
    }
    return shadow, index


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
    totals = {str(r): defaultdict(int) for r in RADII_PX}
    profiles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, atom_output, _, raw, bonds, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            raw_ids = _ids(raw, tensors, config)
            selected_ids = _ids(selected, tensors, config)
            target_count = int(tensors["atom_mask"].sum())
            predicted_count = int(atom_output["count_logits"][0].argmax().clamp(1, config.maximum_atoms))
            profile = str(sample["meta"]["profile_id"])
            for radius in RADII_PX:
                shadow, _ = _shadow(atom_output, bonds, radius, config, predicted_count)
                shadow_ids = _ids(shadow, tensors, config)
                shadow_set = model.set_head(
                    shadow["atom_features"], shadow["atom_geometry"], shadow["atom_scores"], shadow["atom_mask"],
                    bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"],
                )
                shadow_selected = {**shadow, "atom_mask": shadow["atom_mask"] & (
                    shadow_set["atom_presence_logits"] >= payload["provenance"]["atom_threshold"]
                )}
                shadow_selected_ids = _ids(shadow_selected, tensors, config)
                block = totals[str(radius)]
                block["images"] += 1
                block["targets"] += target_count
                block["predicted_count_sum"] += predicted_count
                block["predicted_count_exact"] += int(predicted_count == target_count)
                block["predicted_count_abs_error"] += abs(predicted_count - target_count)
                block["raw_matched"] += len(raw_ids)
                block["selected_matched"] += len(selected_ids)
                block["shadow_matched"] += len(shadow_ids)
                block["shadow_selected_matched"] += len(shadow_selected_ids)
                block["shadow_new_vs_selected"] += len(shadow_ids - selected_ids)
                block["shadow_selected_new_vs_selected"] += len(shadow_selected_ids - selected_ids)
                block["shadow_selected_lost_selected_ids"] += len(selected_ids - shadow_selected_ids)
                block["shadow_candidates"] += int(shadow["atom_mask"].sum())
                profiles[f"{profile}/r{radius}"]["images"] += 1
                profiles[f"{profile}/r{radius}"]["new_selected"] += len(shadow_selected_ids - selected_ids)
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)
    serial = {}
    for radius, values in totals.items():
        item = dict(values)
        item["predicted_count_mae"] = item["predicted_count_abs_error"] / max(item["images"], 1)
        item["predicted_count_accuracy"] = item["predicted_count_exact"] / max(item["images"], 1)
        serial[radius] = item
    result = {
        "schema_version": "chemvector-count-aware-decode-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "contract": {"pool": POOL, "count_source": "atom.count_logits argmax", "target_independent": True},
        "totals": serial,
        "by_profile": {key: dict(value) for key, value in sorted(profiles.items())},
        "gate": {"bounded_training_admissible": False, "reason": "forward-only decoder audit"},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "totals": serial}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
