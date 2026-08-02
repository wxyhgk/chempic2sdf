#!/usr/bin/env python3
"""Audit minimum selected-component replacements needed after P1-K."""

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
    select_ranked_components,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1k_component_set_ranker as p1k
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1l-minimum-component-replacement"
)
CHECKPOINT = p1k.DEFAULT_OUTPUT / "step-006250.pt"
CHECKPOINT_SHA256 = "69243466079b326e94c6c1b356c7dc6c2c5c88aaeeda648b1cac343e0263d4ed"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_replacements(
    compatible: torch.Tensor, selected_mask: torch.Tensor
) -> int | None:
    """Return minimum unselected components in a full target assignment."""

    if compatible.ndim != 2 or selected_mask.shape != (compatible.shape[0],):
        raise ValueError("P1-L compatibility and selection shapes differ")
    components, targets = compatible.shape
    penalty = float(targets + 1)
    costs = (~selected_mask).double()[:, None].expand(components, targets).clone()
    costs[~compatible] = 2.0 * penalty
    rows, columns = linear_sum_assignment(costs.T.numpy())
    assignments = [
        (row, column)
        for row, column in zip(rows.tolist(), columns.tolist())
        if bool(compatible[column, row])
    ]
    if len(assignments) != targets:
        return None
    return sum(not bool(selected_mask[column]) for row, column in assignments)


def _empty():
    return {
        "images": 0,
        "targets": 0,
        "selected_matched_targets": 0,
        "selected_exact_images": 0,
        "oracle_unreachable_images": 0,
        "reachable_failed_images": 0,
        "minimum_replacement_buckets": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5+": 0},
        "failed_targets": 0,
        "failed_targets_single_compatible": 0,
        "failed_targets_multiple_compatible": 0,
    }


def _bucket(value):
    return str(value) if value <= 4 else "5+"


def sample_statistics(compatible, selected_mask):
    selected_compatible = compatible[selected_mask]
    rows, columns = linear_sum_assignment((~selected_compatible.T).to(torch.int64).numpy())
    selected_matched = sum(
        bool(selected_compatible[column, row])
        for row, column in zip(rows.tolist(), columns.tolist())
    )
    target_count = compatible.shape[1]
    replacements = minimum_replacements(compatible, selected_mask)
    missed_target_mask = ~selected_compatible.any(dim=0)
    multiplicity = compatible[:, missed_target_mask].sum(dim=0)
    return {
        "targets": target_count,
        "selected_matched": selected_matched,
        "selected_exact": selected_matched == target_count,
        "minimum_replacements": replacements,
        "failed_targets": int(missed_target_mask.sum()),
        "failed_targets_single_compatible": int((multiplicity == 1).sum()),
        "failed_targets_multiple_compatible": int((multiplicity >= 2).sum()),
    }


def _finalize(value):
    reachable_failed = value["reachable_failed_images"]
    one = value["minimum_replacement_buckets"]["1"]
    two = value["minimum_replacement_buckets"]["2"]
    return {
        **value,
        "selected_atom_recall": value["selected_matched_targets"] / value["targets"],
        "selected_exact_rate": value["selected_exact_images"] / value["images"],
        "one_replacement_rate_among_reachable_failures": one / reachable_failed if reachable_failed else 0.0,
        "one_or_two_rate_among_reachable_failures": (one + two) / reachable_failed if reachable_failed else 0.0,
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
        counts = fixed["representable_counts"].long()
        selected = select_ranked_components(ranked, counts)
        for index in range(len(batch["keys"])):
            valid = components["component_mask"][index]
            component_anchors = components["candidate_anchors"][index, valid].float()
            fixed_mask = fixed["positive_mask"][index]
            target_indices = fixed["matched_target_indices"][index, fixed_mask]
            expected = targets["atom_anchors"][index, target_indices].float()
            compatible = (torch.cdist(component_anchors, expected) * 768.0 <= 6.0).cpu()
            selected_ids = selected["atom_component_ids"][index, selected["atom_mask"][index]].cpu()
            selected_mask = torch.zeros(len(component_anchors), dtype=torch.bool)
            selected_mask[selected_ids] = True
            current = sample_statistics(compatible, selected_mask)
            total["images"] += 1
            total["targets"] += current["targets"]
            total["selected_matched_targets"] += current["selected_matched"]
            total["selected_exact_images"] += int(current["selected_exact"])
            total["failed_targets"] += current["failed_targets"]
            total["failed_targets_single_compatible"] += current["failed_targets_single_compatible"]
            total["failed_targets_multiple_compatible"] += current["failed_targets_multiple_compatible"]
            if current["minimum_replacements"] is None:
                total["oracle_unreachable_images"] += 1
            else:
                bucket = _bucket(current["minimum_replacements"])
                total["minimum_replacement_buckets"][bucket] += 1
                if not current["selected_exact"]:
                    total["reachable_failed_images"] += 1
    return {**_finalize(total), "sample_keys_sha256": sample_digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-L requires CUDA BF16 replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-L output directory is not empty")
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("P1-L checkpoint SHA differs")
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
    one = train_result["one_replacement_rate_among_reachable_failures"]
    one_two = train_result["one_or_two_rate_among_reachable_failures"]
    if one >= 0.70:
        decision = "one_move_action_verifier"
    elif one_two >= 0.90:
        decision = "bounded_two_move_set_corrector"
    else:
        decision = "global_component_set_objective"
    report = {
        "schema_version": "g3-p1l-minimum-component-replacement-audit-0.1",
        "scope": "p1k-final-known-count-minimum-unselected-component-cost",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vision_checkpoint_sha256": vision_sha,
        "train_manifest_sha256": train_audit["sha256"],
        "route_manifest_sha256": route_audit["sha256"],
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
    print(json.dumps({"decision": decision, "train_buckets": train_result["minimum_replacement_buckets"]}))


if __name__ == "__main__":
    main()
