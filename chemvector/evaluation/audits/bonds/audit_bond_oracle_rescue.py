#!/usr/bin/env python3
"""Measure the deployable skeleton ceiling of a Bond candidate pool."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import torch

from chemvector.audit_bond_endpoint_junction import (
    analyze_endpoint_junctions,
    load_samples,
    summarize,
)
from chemvector.audit_bond_first_decomposition import _atom_centers, _expected_bonds


def candidate_objects(example: dict, threshold: float | None) -> list[dict]:
    objects = []
    for endpoints, score, display, chemical, direction in zip(
        example["endpoints"],
        example["scores"],
        example["display_ids"],
        example["chemical_ids"],
        example["direction_ids"],
    ):
        if threshold is not None and float(score) < threshold:
            continue
        objects.append(
            {
                "endpoints": endpoints.float(),
                "score": float(score),
                "display": int(display),
                "chemical": int(chemical),
                "direction": int(direction),
            }
        )
    return objects


def oracle_select_objects(
    sample: dict,
    example: dict,
    *,
    threshold: float | None,
) -> tuple[list[dict], bool]:
    """Select at most one candidate for each GT Bond pair for ceiling analysis."""

    objects = candidate_objects(example, threshold)
    expected = _expected_bonds(sample)
    expected_by_pair = {item["pair"]: item for item in expected}
    centers = _atom_centers(sample)
    choices: dict[tuple[int, int], tuple[float, float, dict]] = {}
    for item in objects:
        endpoints = item["endpoints"].float().cpu()
        nearest = torch.cdist(endpoints, centers).argmin(-1).tolist()
        pair = tuple(sorted((int(nearest[0]), int(nearest[1]))))
        target = expected_by_pair.get(pair)
        if target is None or pair[0] == pair[1]:
            continue
        forward = torch.linalg.vector_norm(
            (endpoints - target["endpoints"]) * 768.0, dim=-1
        ).sum()
        reverse = torch.linalg.vector_norm(
            (endpoints - target["endpoints"].flip(0)) * 768.0, dim=-1
        ).sum()
        error = float(torch.minimum(forward, reverse))
        rank = (error, -float(item["score"]), item)
        previous = choices.get(pair)
        if previous is None or rank[:2] < previous[:2]:
            choices[pair] = rank
    selected = [choices[pair][2] for pair in sorted(choices)]
    return selected, set(choices) == set(expected_by_pair)


def example_from_objects(key: str, objects: list[dict]) -> dict:
    if objects:
        return {
            "key": key,
            "endpoints": torch.stack([item["endpoints"] for item in objects]),
            "scores": torch.tensor([item["score"] for item in objects]),
            "display_ids": torch.tensor([item["display"] for item in objects]),
            "chemical_ids": torch.tensor([item["chemical"] for item in objects]),
            "direction_ids": torch.tensor([item["direction"] for item in objects]),
        }
    return {
        "key": key,
        "endpoints": torch.empty((0, 2, 2)),
        "scores": torch.empty((0,)),
        "display_ids": torch.empty((0,), dtype=torch.long),
        "chemical_ids": torch.empty((0,), dtype=torch.long),
        "direction_ids": torch.empty((0,), dtype=torch.long),
    }


def evaluate_ceiling(
    samples: list[dict],
    examples: list[dict],
    *,
    threshold: float | None,
    cluster_radius_pixels: float,
    match_radius_pixels: float,
) -> tuple[list[dict], dict]:
    rows = []
    for sample, example in zip(samples, examples):
        if sample["key"] != example["key"]:
            raise ValueError("Bond cache and validation sample order differ")
        selected, covered = oracle_select_objects(
            sample, example, threshold=threshold
        )
        metrics = analyze_endpoint_junctions(
            sample,
            example_from_objects(sample["key"], selected),
            threshold=0.0,
            cluster_radius_pixels=cluster_radius_pixels,
            match_radius_pixels=match_radius_pixels,
            compile_sdf=False,
        )
        rows.append(
            {
                "key": sample["key"],
                "candidate_oracle": covered,
                "selected_objects": len(selected),
                "metrics": metrics,
            }
        )
    metrics = summarize(rows)
    metrics["candidate_oracle_rate"] = sum(
        row["candidate_oracle"] for row in rows
    ) / max(len(rows), 1)
    metrics["topology_exact_on_candidate_oracle"] = sum(
        row["candidate_oracle"] and row["metrics"]["topology_exact"]
        for row in rows
    ) / max(sum(row["candidate_oracle"] for row in rows), 1)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"],
    )
    parser.add_argument(
        "--validation-cache",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-9k/validation-candidates.pt",
    )
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--cluster-radius-pixels", type=float, default=12.0)
    parser.add_argument("--match-radius-pixels", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/audits/chemvector-v0818-bond-oracle-rescue-498",
    )
    args = parser.parse_args()

    samples = load_samples(args.validation_shards, args.samples)
    examples = torch.load(
        args.validation_cache, map_location="cpu", weights_only=False
    )[: args.samples]
    if len(samples) != args.samples or len(examples) != args.samples:
        raise ValueError("Oracle rescue audit loaded fewer samples than requested")
    output = {}
    all_rows = {}
    for name, threshold in (("threshold_020", 0.20), ("top_128", None)):
        rows, metrics = evaluate_ceiling(
            samples,
            examples,
            threshold=threshold,
            cluster_radius_pixels=args.cluster_radius_pixels,
            match_radius_pixels=args.match_radius_pixels,
        )
        output[name] = metrics
        all_rows[name] = rows
        print(name + "=" + json.dumps(metrics, ensure_ascii=False), flush=True)
    result = {
        "schema_version": "bond-oracle-rescue-audit-0.1",
        "metrics": output,
        "protocol": {
            **vars(args),
            "oracle_used_only_for_candidate_selection": True,
            "oracle_features_available_to_deployed_model": False,
            "semantic_endpoints_per_bond_object": 2,
            "uses_mr_tadf_for_tuning": False,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in all_rows.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
