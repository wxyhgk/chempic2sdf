"""Read-only readiness audit for an anchor-aware dense sidecar.

Candidates are selected without targets from score-ranked dense cells, using
predicted-anchor distance for suppression and a 12px guard from immutable
selected-baseline proposals.  Matching is then locked: baseline IDs are
assigned first and sidecar candidates may claim only residual targets.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

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
OUTPUT = ROOT / "outputs/audits/chemvector-anchor-aware-sidecar-20261006"
POOL = 4096
SIDE_K = (16, 32, 64)
SIDE_RADIUS_PX = 6.0
BASELINE_GUARD_PX = 12.0
MATCH_RADIUS_PX = 6.0


def _ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(v) for v in mapping[0].tolist() if int(v) >= 0}


def _sidecar(atom_output: dict, baseline_points: torch.Tensor, config, k: int) -> dict:
    scores = atom_output["heatmap_logits"].sigmoid()[0, 0].flatten()
    order = torch.argsort(scores, descending=True)[: min(POOL, len(scores))]
    height, width = atom_output["heatmap_logits"].shape[-2:]
    anchors = atom_output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    chosen: list[int] = []
    side_radius = SIDE_RADIUS_PX / config.image_size
    baseline_guard = BASELINE_GUARD_PX / config.image_size
    for raw_index in order.tolist():
        point = anchors[raw_index]
        if len(baseline_points) and bool(torch.linalg.vector_norm(baseline_points - point, dim=1).min() <= baseline_guard):
            continue
        if chosen:
            old = anchors[torch.as_tensor(chosen, device=anchors.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= side_radius):
                continue
        chosen.append(int(raw_index))
        if len(chosen) >= k:
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


def _locked_side_ids(side: dict, tensors: dict, baseline_ids: set[int], baseline_count: int, config) -> set[int]:
    targets = int(tensors["atom_mask"].sum())
    remaining = [i for i in range(targets) if i not in baseline_ids]
    if not remaining or not int(side["atom_mask"].sum()):
        return set()
    points = side["atom_geometry"][0, :, :2] * config.image_size
    expected = tensors["atom_anchors"][0, remaining, :2] * config.image_size
    cost = torch.cdist(points.float(), expected.float())
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {
        int(remaining[col]) for row, col in zip(rows, cols)
        if float(cost[row, col]) <= MATCH_RADIUS_PX
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
        str(k): {"train": defaultdict(int), "holdout": defaultdict(int), "all": defaultdict(int)}
        for k in SIDE_K
    }
    by_profile: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            _, atom_output, _, _, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            baseline_ids = _ids(selected, tensors, config)
            baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2]
            split = "train" if index <= 120 else "holdout"
            profile = str(sample["meta"]["profile_id"])
            for k in SIDE_K:
                side = _sidecar(atom_output, baseline_points, config, k)
                positives = _locked_side_ids(side, tensors, baseline_ids, int(side["atom_mask"].sum()), config)
                block = totals[str(k)][split]
                all_block = totals[str(k)]["all"]
                count = int(side["atom_mask"].sum())
                for dest in (block, all_block):
                    dest["images"] += 1
                    dest["candidates"] += count
                    dest["positives"] += len(positives)
                    dest["false_positives"] += count - len(positives)
                by_profile[f"{profile}/k{k}"][f"{split}_candidates"] += count
                by_profile[f"{profile}/k{k}"][f"{split}_positives"] += len(positives)
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)

    serial = {}
    for k, splits in totals.items():
        serial[k] = {}
        for split, values in splits.items():
            item = dict(values)
            item["precision"] = item["positives"] / max(item["candidates"], 1)
            serial[k][split] = item
    result = {
        "schema_version": "chemvector-anchor-aware-sidecar-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "contract": {
            "baseline": "selected immutable first",
            "baseline_guard_px": BASELINE_GUARD_PX,
            "sidecar_suppression_px": SIDE_RADIUS_PX,
            "sidecar_match_radius_px": MATCH_RADIUS_PX,
            "pool": POOL,
            "target_independent": True,
        },
        "splits": serial,
        "by_profile": {key: dict(value) for key, value in sorted(by_profile.items())},
        "gate": {
            "minimum_train_positives": 50,
            "precision_floor": 0.85,
            "ready_for_scorer": any(
                serial[str(k)]["train"]["positives"] >= 50 and serial[str(k)]["train"]["precision"] >= 0.85
                for k in SIDE_K
            ),
            "bounded_training_admissible": False,
            "reason": "forward-only sidecar audit; scorer training still requires AUPRC and retention checks",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "splits": serial}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
