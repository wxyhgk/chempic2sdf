#!/usr/bin/env python3
"""Measure candidate coverage, ranking, and deployed count for dual vertex heads."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

import torch

from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate_node_conditioned_bond import outside_text_objects
from chemvector.geometry import stable_pairwise_distances
from chemvector.grounded_graph import (
    load_grounded_graph_checkpoint,
    radius_aware_linear_assignment,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.pretrain_node_conditioned_bond import KnownEndpointBondCollator
from chemvector.text_object import TextObjectCenterHead, decode_text_objects
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def candidate_matches(
    candidates: list[dict],
    targets: torch.Tensor,
    *,
    radius_pixels: float,
) -> int:
    if not candidates or not len(targets):
        return 0
    points = torch.tensor(
        [(float(item["x"]), float(item["y"])) for item in candidates],
        dtype=torch.float32,
    )
    distances = stable_pairwise_distances(points, targets) * 768.0
    rows, columns = radius_aware_linear_assignment(distances, radius_pixels)
    return sum(
        float(distances[row, column]) <= radius_pixels
        for row, column in zip(rows.tolist(), columns.tolist())
    )


def select_candidates(
    candidates: list[dict], *, threshold: float, maximum: int
) -> list[dict]:
    return [
        item
        for item in sorted(candidates, key=lambda value: float(value["score"]), reverse=True)
        if float(item["score"]) >= threshold
    ][:maximum]


def top_count(candidates: list[dict], count: int) -> list[dict]:
    return sorted(
        candidates, key=lambda value: float(value["score"]), reverse=True
    )[:count]


def merge_candidate_pools(
    text_candidates: list[dict],
    implicit_candidates: list[dict],
    *,
    maximum: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Apply the deployed text-first total candidate capacity."""

    if maximum < 1:
        raise ValueError("Candidate capacity must be positive")
    text = sorted(
        text_candidates, key=lambda value: float(value["score"]), reverse=True
    )[:maximum]
    remaining = maximum - len(text)
    implicit = sorted(
        implicit_candidates, key=lambda value: float(value["score"]), reverse=True
    )[:remaining]
    return text, implicit, text + implicit


def new_totals() -> dict[str, int]:
    return {
        "samples": 0,
        "expected": 0,
        "candidate_count": 0,
        "candidate_matches": 0,
        "candidate_oracle_exact": 0,
        "candidate_missing": 0,
        "images_with_candidate_missing": 0,
        "maximum_candidate_count": 0,
        "candidate_pair_count": 0,
        "capacity_saturated_images": 0,
        "known_count_matches": 0,
        "known_count_exact": 0,
        "deployed_count": 0,
        "deployed_matches": 0,
        "deployed_count_exact": 0,
        "deployed_exact": 0,
    }


def update_totals(
    totals: dict[str, int],
    candidates: list[dict],
    deployed: list[dict],
    targets: torch.Tensor,
    *,
    radius_pixels: float,
    candidate_limit: int | None = None,
) -> dict:
    expected = len(targets)
    candidate_match_count = candidate_matches(
        candidates, targets, radius_pixels=radius_pixels
    )
    known = top_count(candidates, expected)
    known_matches = candidate_matches(known, targets, radius_pixels=radius_pixels)
    deployed_matches = candidate_matches(
        deployed, targets, radius_pixels=radius_pixels
    )
    totals["samples"] += 1
    totals["expected"] += expected
    totals["candidate_count"] += len(candidates)
    totals["candidate_matches"] += candidate_match_count
    totals["candidate_oracle_exact"] += int(candidate_match_count == expected)
    totals["candidate_missing"] += expected - candidate_match_count
    totals["images_with_candidate_missing"] += int(candidate_match_count < expected)
    totals["maximum_candidate_count"] = max(
        totals["maximum_candidate_count"], len(candidates)
    )
    totals["candidate_pair_count"] += len(candidates) * (len(candidates) - 1) // 2
    if candidate_limit is not None:
        totals["capacity_saturated_images"] += int(len(candidates) >= candidate_limit)
    totals["known_count_matches"] += known_matches
    totals["known_count_exact"] += int(
        len(known) == expected and known_matches == expected
    )
    totals["deployed_count"] += len(deployed)
    totals["deployed_matches"] += deployed_matches
    totals["deployed_count_exact"] += int(len(deployed) == expected)
    totals["deployed_exact"] += int(
        len(deployed) == expected and deployed_matches == expected
    )
    return {
        "expected": expected,
        "candidates": len(candidates),
        "candidate_matches": candidate_match_count,
        "candidate_oracle_exact": candidate_match_count == expected,
        "candidate_missing": expected - candidate_match_count,
        "known_count_matches": known_matches,
        "known_count_exact": len(known) == expected and known_matches == expected,
        "deployed": len(deployed),
        "deployed_matches": deployed_matches,
        "deployed_exact": len(deployed) == expected and deployed_matches == expected,
    }


def summarize_totals(totals: dict[str, int]) -> dict:
    samples = max(totals["samples"], 1)
    expected = max(totals["expected"], 1)
    deployed_count = max(totals["deployed_count"], 1)
    deployed_precision = totals["deployed_matches"] / deployed_count
    deployed_recall = totals["deployed_matches"] / expected
    return {
        "samples": totals["samples"],
        "expected_vertices": totals["expected"],
        "average_candidates": totals["candidate_count"] / samples,
        "maximum_candidates": totals["maximum_candidate_count"],
        "average_candidate_pairs": totals["candidate_pair_count"] / samples,
        "capacity_saturated_images": totals["capacity_saturated_images"],
        "candidate_recall": totals["candidate_matches"] / expected,
        "candidate_oracle_exact": totals["candidate_oracle_exact"] / samples,
        "candidate_missing_vertices": totals["candidate_missing"],
        "images_with_candidate_missing": totals["images_with_candidate_missing"],
        "known_count_recall": totals["known_count_matches"] / expected,
        "known_count_exact": totals["known_count_exact"] / samples,
        "deployed_count_exact": totals["deployed_count_exact"] / samples,
        "deployed_precision": deployed_precision,
        "deployed_recall": deployed_recall,
        "deployed_f1": 2
        * deployed_precision
        * deployed_recall
        / max(deployed_precision + deployed_recall, 1e-12),
        "deployed_exact": totals["deployed_exact"] / samples,
    }


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("thresholds must lie between zero and one")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("candidate limits must be positive")
    return values


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-object-checkpoint", required=True)
    parser.add_argument("--implicit-vertex-checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-candidate-threshold", type=float, default=0.05)
    parser.add_argument("--text-deployed-threshold", type=float, default=0.3)
    parser.add_argument(
        "--implicit-thresholds",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.05,0.1,0.2,0.3,0.52"),
    )
    parser.add_argument(
        "--candidate-limits",
        type=parse_int_list,
        default=parse_int_list("128,256"),
    )
    parser.add_argument("--radius-pixels", type=float, default=6.0)
    parser.add_argument("--bbox-expansion", type=float, default=8.0)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.radius_pixels <= 0:
        raise ValueError("samples, batch size, and radius must be positive")

    device = torch.device("cuda")
    text_payload = torch.load(
        args.text_object_checkpoint, map_location="cpu", weights_only=False
    )
    implicit_payload = torch.load(
        args.implicit_vertex_checkpoint, map_location="cpu", weights_only=False
    )
    graph_checkpoint = implicit_payload["grounded_graph_checkpoint"]
    if text_payload["grounded_graph_checkpoint"] != graph_checkpoint:
        raise ValueError("Text and implicit heads use different backbones")
    _, graph_payload = load_grounded_graph_checkpoint(graph_checkpoint, device)
    base, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    base.eval()
    text_head = TextObjectCenterHead(text_payload["input_channels"]).to(device)
    text_head.load_state_dict(text_payload["text_object_head"])
    text_head.eval()
    implicit_head = deepcopy(base.vertex_head).to(device)
    implicit_head.load_state_dict(implicit_payload["implicit_vertex_head"])
    implicit_head.eval()

    dataset = ChemVectorIterableDataset(
        expand_shards(args.validation_shards), sequence_version="0.4"
    )
    records = list(islice(iter(dataset), args.samples))
    if len(records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    collator = KnownEndpointBondCollator()
    keys = [
        f"t{threshold:g}-k{maximum}"
        for threshold in args.implicit_thresholds
        for maximum in args.candidate_limits
    ]
    totals = {
        key: {
            "text": new_totals(),
            "implicit": new_totals(),
            "merged": new_totals(),
        }
        for key in keys
    }
    sample_rows = []
    minimum_threshold = min(
        args.text_candidate_threshold, min(args.implicit_thresholds)
    )
    maximum_limit = max(args.candidate_limits)
    for start in range(0, len(records), args.batch_size):
        samples = records[start : start + args.batch_size]
        pixels = torch.stack(
            [collator._pixels(sample["image"]) for sample in samples]
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = base.vision_encoder.forward_stroke_resolution(pixels)
            text_output = text_head(features)
            implicit_output = implicit_head(features)
        text_batches = decode_text_objects(
            text_output, threshold=minimum_threshold, maximum_objects=maximum_limit
        )
        implicit_batches = decode_vertices(
            implicit_output,
            threshold=minimum_threshold,
            maximum_vertices=maximum_limit,
            nms_kernel_size=1,
        )
        for sample, text_candidates, implicit_candidates in zip(
            samples, text_batches, implicit_batches
        ):
            atoms = sample["layers"]["atoms"]
            visible_targets = torch.tensor(
                [
                    (
                        atom["center"][0] / sample["layers"]["width"],
                        atom["center"][1] / sample["layers"]["height"],
                    )
                    for atom in atoms
                    if atom["visible_text"]
                ],
                dtype=torch.float32,
            ).reshape(-1, 2)
            implicit_targets = torch.tensor(
                [
                    (
                        atom["center"][0] / sample["layers"]["width"],
                        atom["center"][1] / sample["layers"]["height"],
                    )
                    for atom in atoms
                    if not atom["visible_text"]
                ],
                dtype=torch.float32,
            ).reshape(-1, 2)
            row = {"key": sample["key"], "configs": {}}
            for threshold in args.implicit_thresholds:
                for maximum in args.candidate_limits:
                    key = f"t{threshold:g}-k{maximum}"
                    text_pool = select_candidates(
                        text_candidates,
                        threshold=args.text_candidate_threshold,
                        maximum=maximum,
                    )
                    deployed_text = select_candidates(
                        text_candidates,
                        threshold=args.text_deployed_threshold,
                        maximum=maximum,
                    )
                    filtered_implicit = [
                        item
                        for item in implicit_candidates
                        if outside_text_objects(
                            item, deployed_text, args.bbox_expansion
                        )
                    ]
                    implicit_pool = select_candidates(
                        filtered_implicit,
                        threshold=threshold,
                        maximum=maximum,
                    )
                    current_text, current_implicit, merged_candidates = (
                        merge_candidate_pools(
                            text_pool,
                            implicit_pool,
                            maximum=maximum,
                        )
                    )
                    current_deployed_text, _, _ = merge_candidate_pools(
                        deployed_text,
                        [],
                        maximum=maximum,
                    )
                    deployed_filtered_implicit = [
                        item
                        for item in implicit_candidates
                        if outside_text_objects(
                            item, current_deployed_text, args.bbox_expansion
                        )
                    ]
                    current_implicit_deployed = select_candidates(
                        deployed_filtered_implicit,
                        threshold=threshold,
                        maximum=maximum - len(current_deployed_text),
                    )
                    text_result = update_totals(
                        totals[key]["text"],
                        current_text,
                        current_deployed_text,
                        visible_targets,
                        radius_pixels=args.radius_pixels,
                    )
                    implicit_result = update_totals(
                        totals[key]["implicit"],
                        current_implicit,
                        current_implicit_deployed,
                        implicit_targets,
                        radius_pixels=args.radius_pixels,
                    )
                    merged_deployed = current_deployed_text + current_implicit_deployed
                    merged_targets = torch.cat((visible_targets, implicit_targets), dim=0)
                    merged_result = update_totals(
                        totals[key]["merged"],
                        merged_candidates,
                        merged_deployed,
                        merged_targets,
                        radius_pixels=args.radius_pixels,
                        candidate_limit=maximum,
                    )
                    # Override merged metrics with role-aware matches to prevent cross-role credit.
                    role_candidate_matches = candidate_matches(
                        current_text,
                        visible_targets,
                        radius_pixels=args.radius_pixels,
                    ) + candidate_matches(
                        current_implicit,
                        implicit_targets,
                        radius_pixels=args.radius_pixels,
                    )
                    role_candidate_exact = role_candidate_matches == len(merged_targets)
                    role_known_matches = candidate_matches(
                        top_count(current_text, len(visible_targets)),
                        visible_targets,
                        radius_pixels=args.radius_pixels,
                    ) + candidate_matches(
                        top_count(current_implicit, len(implicit_targets)),
                        implicit_targets,
                        radius_pixels=args.radius_pixels,
                    )
                    role_known_exact = role_known_matches == len(merged_targets)
                    role_deployed_matches = candidate_matches(
                        current_deployed_text,
                        visible_targets,
                        radius_pixels=args.radius_pixels,
                    ) + candidate_matches(
                        current_implicit,
                        implicit_targets,
                        radius_pixels=args.radius_pixels,
                    )
                    role_deployed_exact = (
                        len(deployed_text) == len(visible_targets)
                        and len(current_implicit) == len(implicit_targets)
                        and role_deployed_matches == len(merged_targets)
                    )
                    totals[key]["merged"]["candidate_matches"] += (
                        role_candidate_matches - merged_result["candidate_matches"]
                    )
                    totals[key]["merged"]["candidate_oracle_exact"] += (
                        int(role_candidate_exact)
                        - int(merged_result["candidate_oracle_exact"])
                    )
                    role_candidate_missing = (
                        len(merged_targets) - role_candidate_matches
                    )
                    totals[key]["merged"]["candidate_missing"] += (
                        role_candidate_missing - merged_result["candidate_missing"]
                    )
                    totals[key]["merged"]["images_with_candidate_missing"] += (
                        int(role_candidate_missing > 0)
                        - int(merged_result["candidate_missing"] > 0)
                    )
                    totals[key]["merged"]["known_count_matches"] += (
                        role_known_matches - merged_result["known_count_matches"]
                    )
                    totals[key]["merged"]["known_count_exact"] += (
                        int(role_known_exact) - int(merged_result["known_count_exact"])
                    )
                    totals[key]["merged"]["deployed_matches"] += (
                        role_deployed_matches - merged_result["deployed_matches"]
                    )
                    totals[key]["merged"]["deployed_exact"] += (
                        int(role_deployed_exact) - int(merged_result["deployed_exact"])
                    )
                    row["configs"][key] = {
                        "text": text_result,
                        "implicit": implicit_result,
                        "merged": {
                            **merged_result,
                            "candidate_matches": role_candidate_matches,
                            "candidate_missing": role_candidate_missing,
                            "role_candidate_oracle_exact": role_candidate_exact,
                            "role_known_count_exact": role_known_exact,
                            "role_deployed_exact": role_deployed_exact,
                        },
                    }
            sample_rows.append(row)
        if len(sample_rows) % 100 < args.batch_size:
            print(f"evaluated={len(sample_rows)}", flush=True)

    summary = {
        "samples": len(sample_rows),
        "radius_pixels": args.radius_pixels,
        "text_candidate_threshold": args.text_candidate_threshold,
        "text_deployed_threshold": args.text_deployed_threshold,
        "configs": {
            key: {
                role: summarize_totals(role_totals)
                for role, role_totals in current.items()
            }
            for key, current in totals.items()
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
