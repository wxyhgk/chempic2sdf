#!/usr/bin/env python3
"""Audit target-free representative anchors for P1-I 3px components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.g3_top128_set_ranker import extract_frozen_top128
from chemvector.evaluation.audits.atoms.audit_g3_p1i_target_free_clustering import (
    cluster_labels,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1j-component-representative"
)
CLUSTER_RADIUS_PIXELS = 3.0
MATCH_RADIUS_PIXELS = 6.0
SCHEMES = ("base_max", "medoid", "base_softmax", "centroid")


def component_representatives(
    anchors: torch.Tensor,
    base_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    labels: torch.Tensor,
    scheme: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one representative anchor and max base score per component."""

    if scheme not in SCHEMES:
        raise ValueError(f"P1-J unknown representative scheme: {scheme}")
    component_count = int(labels.max()) + 1
    representatives = []
    scores = []
    for component in range(component_count):
        members = torch.where(labels == component)[0]
        member_anchors = anchors[members].float()
        member_scores = base_logits[members].float()
        member_ids = candidate_ids[members]
        identity_order = member_ids.argsort(stable=True)
        ordered_scores = member_scores[identity_order]
        score_position = identity_order[
            ordered_scores.argsort(descending=True, stable=True)[0]
        ]
        scores.append(member_scores.max())
        if scheme == "base_max":
            representative = member_anchors[score_position]
        elif scheme == "medoid":
            distance_sums = torch.cdist(member_anchors, member_anchors).sum(dim=1)
            minimum = distance_sums.min()
            tied = torch.where(distance_sums == minimum)[0]
            tied_ids = member_ids[tied]
            representative = member_anchors[tied[tied_ids.argmin()]]
        elif scheme == "base_softmax":
            weights = member_scores.softmax(dim=0)
            representative = (member_anchors * weights[:, None]).sum(dim=0)
        else:
            representative = member_anchors.mean(dim=0)
        representatives.append(representative)
    return torch.stack(representatives), torch.stack(scores)


def _match_count(predicted: torch.Tensor, expected: torch.Tensor) -> int:
    compatible = torch.cdist(predicted.float(), expected.float()) * 768.0 <= 6.0
    rows, columns = linear_sum_assignment((~compatible).to(torch.int64).numpy())
    return sum(
        bool(compatible[row, column])
        for row, column in zip(rows.tolist(), columns.tolist())
    )


def _empty():
    return {
        "images": 0,
        "targets": 0,
        "reachability_matched": 0,
        "reachability_exact_images": 0,
        "known_matched": 0,
        "known_exact_images": 0,
        "components": 0,
    }


def _finalize(value):
    return {
        **value,
        "reachability_atom_recall": value["reachability_matched"]
        / value["targets"],
        "reachability_exact_rate": value["reachability_exact_images"]
        / value["images"],
        "known_count_atom_recall": value["known_matched"] / value["targets"],
        "known_count_exact_rate": value["known_exact_images"] / value["images"],
        "mean_components_per_image": value["components"] / value["images"],
    }


@torch.no_grad()
def evaluate_manifest(atom_head, vision, manifest, collator, *, device):
    totals = {scheme: _empty() for scheme in SCHEMES}
    sample_digest = hashlib.sha256()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device),
            "atom_mask": batch["atom_mask"].to(device),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
        fixed = p1d.build_hybrid_top128_targets(candidates, targets)
        candidate_distances = (
            torch.cdist(
                candidates["candidate_anchors"].float(),
                candidates["candidate_anchors"].float(),
            )
            * 768.0
        ).cpu()
        for index in range(len(batch["keys"])):
            labels = cluster_labels(
                candidate_distances[index], CLUSTER_RADIUS_PIXELS
            )
            fixed_mask = fixed["positive_mask"][index]
            target_indices = fixed["matched_target_indices"][index, fixed_mask]
            expected = targets["atom_anchors"][index, target_indices].float().cpu()
            target_count = len(expected)
            for scheme in SCHEMES:
                representatives, scores = component_representatives(
                    candidates["candidate_anchors"][index].float().cpu(),
                    candidates["base_logits"][index].float().cpu(),
                    candidates["candidate_ids"][index].cpu(),
                    labels,
                    scheme,
                )
                reachable = _match_count(representatives, expected)
                identity_order = torch.arange(len(scores))
                score_order = scores.argsort(descending=True, stable=True)
                selected = representatives[identity_order[score_order[:target_count]]]
                known = _match_count(selected, expected)
                value = totals[scheme]
                value["images"] += 1
                value["targets"] += target_count
                value["reachability_matched"] += reachable
                value["reachability_exact_images"] += int(
                    reachable == target_count
                )
                value["known_matched"] += known
                value["known_exact_images"] += int(known == target_count)
                value["components"] += len(representatives)
    return {
        "images": len(manifest),
        "schemes": {scheme: _finalize(totals[scheme]) for scheme in SCHEMES},
        "sample_keys_sha256": sample_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-J requires CUDA BF16 frozen replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-J output directory is not empty")
    device = torch.device("cuda")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_result = evaluate_manifest(
        atom_head, vision, train, collator, device=device
    )
    route_result = evaluate_manifest(
        atom_head, vision, route, collator, device=device
    )
    priority = {scheme: index for index, scheme in enumerate(SCHEMES)}
    selected = max(
        SCHEMES,
        key=lambda scheme: (
            train_result["schemes"][scheme]["reachability_atom_recall"],
            train_result["schemes"][scheme]["reachability_exact_rate"],
            -priority[scheme],
        ),
    )
    train_selected = train_result["schemes"][selected]
    route_selected = route_result["schemes"][selected]
    passed = (
        train_selected["reachability_atom_recall"] >= 0.999
        and train_selected["reachability_exact_rate"] >= 0.99
        and route_selected["reachability_atom_recall"] >= 0.999
        and route_selected["reachability_exact_rate"] >= 0.99
    )
    decision = (
        "component_representative_go"
        if passed
        else "component_representative_no_go"
    )
    report = {
        "schema_version": "g3-p1j-component-representative-audit-0.1",
        "scope": "p1i-3px-components-target-free-representative",
        "cluster_radius_pixels": CLUSTER_RADIUS_PIXELS,
        "match_radius_pixels": MATCH_RADIUS_PIXELS,
        "selection_priority": list(SCHEMES),
        "selected_scheme": selected,
        "train_manifest_sha256": train_audit["sha256"],
        "route_manifest_sha256": route_audit["sha256"],
        "vision_checkpoint_sha256": vision_sha,
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
    print(json.dumps({"decision": decision, "selected_scheme": selected}))


if __name__ == "__main__":
    main()
