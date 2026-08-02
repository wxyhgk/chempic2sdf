#!/usr/bin/env python3
"""Audit top-K boundary coverage of exact one-component swaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.g3_component_set_ranker import (
    G3ComponentSetRanker,
    build_target_free_components,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1l_minimum_component_replacement import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    _sha256,
    minimum_replacements,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1m-one-move-boundary-reachability"
)
BOUNDARIES = (1, 2, 4, 8, 16)


def selection_exact(compatible, selected_positions) -> bool:
    current = compatible[selected_positions]
    targets = compatible.shape[1]
    rows, columns = linear_sum_assignment((~current.T).to(torch.int64).numpy())
    matched = sum(
        bool(current[column, row])
        for row, column in zip(rows.tolist(), columns.tolist())
    )
    return matched == targets


def boundary_rescue(
    compatible: torch.Tensor,
    score_order: torch.Tensor,
    target_count: int,
    boundary: int,
) -> bool:
    selected = score_order[:target_count]
    unselected = score_order[target_count:]
    drop_values = selected[-min(boundary, len(selected)):]
    add_values = unselected[:min(boundary, len(unselected))]
    for drop in drop_values.tolist():
        remaining = selected[selected != drop]
        for add in add_values.tolist():
            changed = torch.cat((remaining, torch.tensor([add], dtype=torch.long)))
            if selection_exact(compatible, changed):
                return True
    return False


def _stable_score_order(scores, ids):
    identity = ids.argsort(stable=True)
    return identity[scores[identity].argsort(descending=True, stable=True)]


def _empty():
    return {
        "images": 0,
        "one_move_images": 0,
        "covered": {str(value): 0 for value in BOUNDARIES},
        "covered_all": 0,
        "first_boundary": {str(value): 0 for value in (*BOUNDARIES, "all")},
    }


def _finalize(value):
    denominator = value["one_move_images"]
    return {
        **value,
        "coverage_rates": {
            key: count / denominator for key, count in value["covered"].items()
        },
        "coverage_all_rate": value["covered_all"] / denominator,
    }


@torch.no_grad()
def evaluate(atom_head, ranker, vision, manifest, collator, *, device):
    total = _empty()
    sample_digest = hashlib.sha256()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {"atom_anchors": batch["atom_anchors"].to(device),
                   "atom_mask": batch["atom_mask"].to(device)}
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            components = build_target_free_components(candidates)
            ranked = ranker(components)
        fixed = build_hybrid_top128_targets(candidates, targets)
        for index in range(len(batch["keys"])):
            total["images"] += 1
            valid = components["component_mask"][index]
            anchors = components["candidate_anchors"][index, valid].float()
            fixed_mask = fixed["positive_mask"][index]
            target_indices = fixed["matched_target_indices"][index, fixed_mask]
            expected = targets["atom_anchors"][index, target_indices].float()
            compatible = (torch.cdist(anchors, expected) * 768.0 <= 6.0).cpu()
            scores = ranked["ranking_logits"][index, valid].float().cpu()
            ids = ranked["candidate_ids"][index, valid].cpu()
            order = _stable_score_order(scores, ids)
            target_count = len(expected)
            selected_mask = torch.zeros(len(order), dtype=torch.bool)
            selected_mask[order[:target_count]] = True
            if minimum_replacements(compatible, selected_mask) != 1:
                continue
            total["one_move_images"] += 1
            found_first = False
            for boundary in BOUNDARIES:
                rescued = boundary_rescue(compatible, order, target_count, boundary)
                total["covered"][str(boundary)] += int(rescued)
                if rescued and not found_first:
                    total["first_boundary"][str(boundary)] += 1
                    found_first = True
            # The minimum-replacement precondition is an exact all-pair certificate.
            total["covered_all"] += 1
            if not found_first:
                total["first_boundary"]["all"] += 1
    return {**_finalize(total), "sample_keys_sha256": sample_digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-M requires CUDA BF16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-M output directory is not empty")
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("P1-M checkpoint SHA differs")
    device = torch.device("cuda")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ranker = G3ComponentSetRanker().to(device)
    ranker.load_state_dict(payload["component_ranker"], strict=True)
    ranker.eval()
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_result = evaluate(atom_head, ranker, vision, train, collator, device=device)
    route_result = evaluate(atom_head, ranker, vision, route, collator, device=device)
    if (
        train_result["coverage_rates"]["8"] >= 0.90
        and route_result["coverage_rates"]["8"] >= 0.80
    ):
        decision = "top8x8_action_candidates"
    elif (
        train_result["coverage_rates"]["16"] >= 0.95
        and route_result["coverage_rates"]["16"] >= 0.90
    ):
        decision = "top16x16_action_candidates"
    else:
        decision = "full_pair_or_structured_action"
    report = {
        "schema_version": "g3-p1m-one-move-boundary-audit-0.1",
        "scope": "p1k-final-one-move-failures-topk-drop-add",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vision_checkpoint_sha256": vision_sha,
        "train_manifest_sha256": train_audit["sha256"],
        "route_manifest_sha256": route_audit["sha256"],
        "boundaries": list(BOUNDARIES),
        "train5k": train_result,
        "route100": route_result,
        "updates": 0,
        "forbidden_data_read": False,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": decision, "train": train_result["coverage_rates"],
                      "route": route_result["coverage_rates"]}))


if __name__ == "__main__":
    main()
