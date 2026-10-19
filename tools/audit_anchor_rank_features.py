"""Forward-only comparison of dense anchor ranking contracts.

This audit never replaces the selected baseline.  It appends target-independent
sidecar candidates to the immutable ``selected`` proposals, then evaluates
residual ownership (baseline IDs are assigned first).  It compares the current
heatmap ranking with a few deterministic feature-aware rankings to answer a
narrow question before spending GPU on training: does an existing dense field
contain a cleaner ranking signal than heatmap probability alone?

Only the public fixed 1,200-image cohort is read.  No checkpoint is written.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
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
    "CHEMVECTOR_ANCHOR_RANK_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-anchor-rank-features-20261006"),
))
IMAGE_SIZE = 768.0
POOL = 4096
SIDE_K = (16, 32, 64)
GUARD_PX = 12.0
SUPPRESS_PX = 6.0
MATCH_PX = 6.0


def _ids(proposals: dict, tensors: dict, config) -> set[int]:
    mapping, _ = match_atom_targets(proposals, tensors, config)
    return {int(x) for x in mapping[0].tolist() if int(x) >= 0}


def _score_variants(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return target-independent dense-cell score vectors.

    The first variant is exactly the existing heatmap score.  The other
    variants only use heads already emitted by ``SpatialAtomObjectDetector``;
    they are diagnostics, not proposed deployment defaults.
    """
    logits = output["heatmap_logits"][0, 0]
    probability = logits.sigmoid()
    role = output["role_logits"][0].permute(1, 2, 0).softmax(-1)
    role_conf = role.max(-1).values
    # A local prominence score is intentionally conservative: it cannot create
    # a score above the heatmap itself and exposes whether nearby competitors
    # explain the low rank of useful cells.
    # ``max_pool2d`` includes the center cell and would make the prominence
    # identically zero.  Remove the center explicitly before taking the local
    # competitor maximum.
    patches = F.unfold(logits[None, None], kernel_size=3, padding=1)[0]
    patches[4] = float("-inf")
    neighbor = patches.amax(0).reshape_as(logits)
    prominence = logits - neighbor
    # Offset centrality is a weak, deterministic geometry prior.  A cell whose
    # predicted sub-cell offset is close to its center is less likely to be an
    # unstable border artifact.  It is included only as a diagnostic variant.
    offset = output["anchor_offset_logits"][0].permute(1, 2, 0).sigmoid()
    centrality = 1.0 - (offset - 0.5).abs().amax(-1).mul(2.0).clamp(0.0, 1.0)
    # Rank transformations are monotonic only within each variant.  Coefficients
    # are fixed a priori; no labels are consulted while candidates are chosen.
    return {
        "heatmap": probability.reshape(-1),
        "heatmap_role": (probability * role_conf).reshape(-1),
        "heatmap_prominence": (logits + 0.75 * prominence).reshape(-1),
        "heatmap_role_prominence": (logits + 0.50 * torch.log(role_conf.clamp_min(1e-5))
                                     + 0.75 * prominence).reshape(-1),
        "heatmap_role_offset": (probability * role_conf * (0.75 + 0.25 * centrality)).reshape(-1),
    }


def _sidecar(output: dict[str, torch.Tensor], baseline_points: torch.Tensor,
             scores: torch.Tensor, config, k: int) -> torch.Tensor:
    """Select dense points target-independently, preserving score/point pairing."""
    _, _, height, width = output["heatmap_logits"].shape
    anchors = output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    order = torch.argsort(scores, descending=True)[: min(POOL, len(scores))]
    guard = GUARD_PX / config.image_size
    suppress = SUPPRESS_PX / config.image_size
    chosen: list[int] = []
    for raw_index in order.tolist():
        point = anchors[raw_index]
        if len(baseline_points) and bool(torch.linalg.vector_norm(
                baseline_points - point, dim=1).min() <= guard):
            continue
        if chosen:
            old = anchors[torch.as_tensor(chosen, device=anchors.device)]
            if bool(torch.linalg.vector_norm(old - point, dim=1).min() <= suppress):
                continue
        chosen.append(int(raw_index))
        if len(chosen) >= k:
            break
    return torch.as_tensor(chosen, dtype=torch.long, device=anchors.device)


def _locked_positive(side_indices: torch.Tensor, output: dict[str, torch.Tensor],
                     tensors: dict, baseline_ids: set[int], config) -> set[int]:
    target_count = int(tensors["atom_mask"].sum())
    remaining = [i for i in range(target_count) if i not in baseline_ids]
    if not remaining or not len(side_indices):
        return set()
    anchors = output["anchors"][0].permute(1, 2, 0).reshape(-1, 2)
    points = anchors[side_indices] * config.image_size
    expected = tensors["atom_anchors"][0, remaining] * config.image_size
    cost = torch.cdist(points.float(), expected.float())
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return {int(remaining[col]) for row, col in zip(rows, cols)
            if float(cost[row, col]) <= MATCH_PX}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or current initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text())
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys),
                  key=lambda row: row["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    totals = {name: {str(k): {"images": 0, "candidates": 0, "positives": 0,
                              "false_positives": 0, "residual_targets": 0}
                     for k in SIDE_K} for name in (
                         "heatmap", "heatmap_role", "heatmap_prominence",
                         "heatmap_role_prominence", "heatmap_role_offset")}
    by_profile = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items()
                       if isinstance(value, torch.Tensor)}
            _, output, _, _, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config,
                payload["provenance"]["atom_threshold"],
            )
            baseline_ids = _ids(selected, tensors, config)
            baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2]
            split = "train" if index <= 120 else "holdout"
            profile = str(sample["meta"]["profile_id"])
            variants = _score_variants(output)
            target_count = int(tensors["atom_mask"].sum())
            for name, score in variants.items():
                for k in SIDE_K:
                    indices = _sidecar(output, baseline_points, score, config, k)
                    positives = _locked_positive(indices, output, tensors, baseline_ids, config)
                    block = totals[name][str(k)]
                    block["images"] += 1
                    block["candidates"] += len(indices)
                    block["positives"] += len(positives)
                    block["false_positives"] += len(indices) - len(positives)
                    block["residual_targets"] += target_count - len(baseline_ids)
                    by_profile[f"{profile}/{name}/k{k}/{split}"]["candidates"] += len(indices)
                    by_profile[f"{profile}/{name}/k{k}/{split}"]["positives"] += len(positives)
            if index % 200 == 0:
                print(json.dumps({"images": index}), flush=True)

    serial = {}
    for name, variants in totals.items():
        serial[name] = {}
        for k, values in variants.items():
            item = dict(values)
            item["precision"] = item["positives"] / max(item["candidates"], 1)
            item["residual_recall"] = item["positives"] / max(item["residual_targets"], 1)
            serial[name][k] = item
    result = {
        "schema_version": "chemvector-anchor-rank-features-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "contract": {
            "baseline": "selected immutable first",
            "baseline_guard_px": GUARD_PX,
            "sidecar_suppression_px": SUPPRESS_PX,
            "match_radius_px": MATCH_PX,
            "pool": POOL,
            "target_independent": True,
            "split": "first 120 train, remaining 1080 holdout for diagnostics",
        },
        "variants": serial,
        "by_profile": {key: dict(value) for key, value in sorted(by_profile.items())},
        "gate": {
            "minimum_train_positives": 50,
            "minimum_train_auprc": 0.50,
            "precision_floor": 0.85,
            "old_selected_retention": "by contract (baseline never replaced)",
            "bounded_training_admissible": False,
            "reason": "feature ranking is diagnostic; no labels used for candidate selection and no scorer trained",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "variants": serial}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
