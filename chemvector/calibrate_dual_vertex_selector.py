#!/usr/bin/env python3
"""Calibrate independent REAL thresholds for text and implicit vertices."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.dual_vertex_selector import (
    IMPLICIT_ROLE,
    ROLE_COUNT,
    TEXT_ROLE,
    role_aware_match_count,
    role_suppressed_topk_indices,
)
from chemvector.pretrain_dual_vertex_selector import (
    load_dual_pipeline,
    prepare_cache,
    selector_forward,
    stack_records,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)
from chemvector.train import expand_shards


ROLE_NAMES = {IMPLICIT_ROLE: "implicit", TEXT_ROLE: "text"}


def parse_thresholds(values: list[float], name: str) -> list[float]:
    thresholds = sorted(set(float(value) for value in values))
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError(f"{name} thresholds must lie in (0, 1)")
    return thresholds


def atom_count_bucket(atom_count: int) -> str:
    if atom_count <= 30:
        return "le_30"
    if atom_count <= 70:
        return "31_70"
    return "gt_70"


def summarize_observations(
    observations: dict[int, list[dict]], metadata: list[dict]
) -> dict:
    if set(observations) != set(ROLE_NAMES):
        raise ValueError("Observations must contain every vertex role")
    sample_count = len(metadata)
    if any(len(values) != sample_count for values in observations.values()):
        raise ValueError("Role observation lengths differ from metadata")
    totals = {
        "matches": 0,
        "predicted": 0,
        "expected": 0,
        "exact": 0,
        "total_count_exact": 0,
        "role_count_exact": 0,
    }
    role_totals = {
        role: {"count_exact": 0, "under": 0, "over": 0, "absolute_error": 0}
        for role in ROLE_NAMES
    }
    by_layout: dict[str, dict[str, int]] = {}
    by_atom_count: dict[str, dict[str, int]] = {}
    for index, item in enumerate(metadata):
        predicted_by_role = {
            role: observations[role][index]["predicted"] for role in ROLE_NAMES
        }
        expected_by_role = {
            role: observations[role][index]["expected"] for role in ROLE_NAMES
        }
        matches = sum(observations[role][index]["matches"] for role in ROLE_NAMES)
        predicted = sum(predicted_by_role.values())
        expected = sum(expected_by_role.values())
        exact = predicted == expected == matches
        role_count_exact = all(
            predicted_by_role[role] == expected_by_role[role] for role in ROLE_NAMES
        )
        totals["matches"] += matches
        totals["predicted"] += predicted
        totals["expected"] += expected
        totals["exact"] += int(exact)
        totals["total_count_exact"] += int(predicted == expected)
        totals["role_count_exact"] += int(role_count_exact)
        for role in ROLE_NAMES:
            difference = predicted_by_role[role] - expected_by_role[role]
            role_totals[role]["count_exact"] += int(difference == 0)
            role_totals[role]["under"] += int(difference < 0)
            role_totals[role]["over"] += int(difference > 0)
            role_totals[role]["absolute_error"] += abs(difference)
        for destination, bucket in (
            (by_layout, item["variant"]),
            (by_atom_count, atom_count_bucket(item["atom_count"])),
        ):
            values = destination.setdefault(bucket, {"samples": 0, "exact": 0})
            values["samples"] += 1
            values["exact"] += int(exact)
    precision = totals["matches"] / max(totals["predicted"], 1)
    recall = totals["matches"] / max(totals["expected"], 1)
    return {
        "samples": sample_count,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "exact_rate": totals["exact"] / max(sample_count, 1),
        "total_count_exact_rate": totals["total_count_exact"] / max(sample_count, 1),
        "role_count_exact_rate": totals["role_count_exact"] / max(sample_count, 1),
        "per_role": {
            ROLE_NAMES[role]: {
                "count_exact_rate": values["count_exact"] / max(sample_count, 1),
                "under_images": values["under"],
                "over_images": values["over"],
                "count_mae": values["absolute_error"] / max(sample_count, 1),
            }
            for role, values in role_totals.items()
        },
        "by_layout": {
            name: {
                "samples": values["samples"],
                "exact_rate": values["exact"] / max(values["samples"], 1),
            }
            for name, values in sorted(by_layout.items())
        },
        "by_atom_count": {
            name: {
                "samples": values["samples"],
                "exact_rate": values["exact"] / max(values["samples"], 1),
            }
            for name, values in sorted(by_atom_count.items())
        },
    }


def calibration_key(result: dict) -> tuple[float, float, float, float]:
    return (
        result["exact_rate"],
        result["role_count_exact_rate"],
        result["f1"],
        -result["per_role"]["implicit"]["under_images"],
    )


def role_observations(
    records: list[dict],
    logits: list[torch.Tensor],
    *,
    role: int,
    probability_threshold: float,
) -> list[dict]:
    output = []
    threshold_logit = torch.logit(torch.tensor(probability_threshold))
    for record, scores in zip(records, logits):
        candidate_mask = record["candidate_mask"]
        candidate_roles = record["candidate_roles"]
        eligible = candidate_mask & (candidate_roles == role) & (scores >= threshold_logit)
        counts = torch.zeros(ROLE_COUNT, dtype=torch.long)
        counts[role] = int(eligible.sum())
        selected = role_suppressed_topk_indices(
            scores,
            record["points"],
            eligible,
            candidate_roles,
            counts,
            suppression_radius_pixels=6.0,
            image_size=768,
        )
        matches = role_aware_match_count(
            record["points"][selected],
            candidate_roles[selected],
            record["anchor_points"],
            record["anchor_roles"],
            matching_radius_pixels=6.0,
            image_size=768,
        )
        output.append(
            {
                "predicted": len(selected),
                "expected": int(record["target_role_counts"][role]),
                "matches": matches,
            }
        )
    return output


@torch.inference_mode()
def checkpoint_logits(
    checkpoint: Path,
    expected_config: StructuredSelectorConfig,
    records: list[dict],
    pair_indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = StructuredSelectorConfig(**payload["structured_selector_config"])
    if config != expected_config:
        raise ValueError(f"Selector config differs in {checkpoint}")
    selector = GroundedStructuredSelector(config).to(device)
    selector.load_state_dict(payload["dual_vertex_selector"])
    selector.eval()
    output = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_records(records, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            scores = selector_forward(selector, batch, pair_indices)
        output.extend(scores.float().cpu())
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-checkpoints", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument(
        "--implicit-thresholds",
        nargs="+",
        type=float,
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    )
    parser.add_argument(
        "--text-thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    )
    parser.add_argument("--bbox-expansion-pixels", type=float, default=8.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for selector calibration")
    implicit_thresholds = parse_thresholds(args.implicit_thresholds, "Implicit")
    text_thresholds = parse_thresholds(args.text_thresholds, "Text")
    checkpoints = [Path(path) for pattern in args.selector_checkpoints for path in sorted(Path().glob(pattern))]
    if not checkpoints:
        raise ValueError("No selector checkpoints matched")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = StructuredSelectorConfig(**first_payload["structured_selector_config"])
    base, detector, grounded_selector, graph, text_head, implicit_head = load_dual_pipeline(
        first_payload["text_object_checkpoint"],
        first_payload["implicit_vertex_checkpoint"],
        device,
        maximum_candidates=config.maximum_candidates,
    )

    def dataset():
        return ChemVectorIterableDataset(
            expand_shards(args.validation_shards), sequence_version="0.4"
        )

    loader = DataLoader(
        dataset(),
        batch_size=args.cache_batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=ProposalGraphCollator(),
    )
    records, pair_indices = prepare_cache(
        base,
        detector,
        grounded_selector,
        graph,
        text_head,
        implicit_head,
        loader,
        device,
        samples=args.samples,
        text_candidate_threshold=0.05,
        text_filter_threshold=0.30,
        implicit_candidate_threshold=0.05,
        bbox_expansion_pixels=args.bbox_expansion_pixels,
        matching_radius_pixels=6.0,
        cache_name="calibration",
    )
    raw_records = list(islice(iter(dataset()), args.samples))
    if len(records) != args.samples or len(raw_records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    metadata = [
        {
            "key": sample["key"],
            "variant": sample["render"]["variant"],
            "atom_count": len(sample["layers"]["atoms"]),
        }
        for sample in raw_records
    ]
    if [record["key"] for record in records] != [item["key"] for item in metadata]:
        raise ValueError("Calibration cache and metadata ordering differ")
    pair_indices = pair_indices.to(device)
    checkpoint_results = {}
    overall_best = None
    for checkpoint in checkpoints:
        print(f"checkpoint={checkpoint}", flush=True)
        logits = checkpoint_logits(
            checkpoint,
            config,
            records,
            pair_indices,
            device,
            args.selector_batch_size,
        )
        implicit_observations = {
            threshold: role_observations(
                records, logits, role=IMPLICIT_ROLE, probability_threshold=threshold
            )
            for threshold in implicit_thresholds
        }
        text_observations = {
            threshold: role_observations(
                records, logits, role=TEXT_ROLE, probability_threshold=threshold
            )
            for threshold in text_thresholds
        }
        grid = {}
        best = None
        for implicit_threshold in implicit_thresholds:
            for text_threshold in text_thresholds:
                result = summarize_observations(
                    {
                        IMPLICIT_ROLE: implicit_observations[implicit_threshold],
                        TEXT_ROLE: text_observations[text_threshold],
                    },
                    metadata,
                )
                result["thresholds"] = {
                    "implicit": implicit_threshold,
                    "text": text_threshold,
                }
                key = f"implicit={implicit_threshold:.2f},text={text_threshold:.2f}"
                grid[key] = result
                if best is None or calibration_key(result) > calibration_key(best):
                    best = result
        checkpoint_result = {
            "checkpoint": str(checkpoint),
            "samples_seen": int(
                torch.load(checkpoint, map_location="cpu", weights_only=False).get(
                    "samples_seen", 0
                )
            ),
            "baseline_shared_0.70": grid.get("implicit=0.70,text=0.70"),
            "best": best,
            "grid": grid,
        }
        checkpoint_results[checkpoint.name] = checkpoint_result
        candidate = {**best, "checkpoint": str(checkpoint)}
        if overall_best is None or calibration_key(candidate) > calibration_key(overall_best):
            overall_best = candidate
        print(
            "best="
            + json.dumps(
                {
                    "thresholds": best["thresholds"],
                    "exact_rate": best["exact_rate"],
                    "role_count_exact_rate": best["role_count_exact_rate"],
                    "f1": best["f1"],
                }
            ),
            flush=True,
        )

    baseline = checkpoint_results.get("checkpoint-0045000.pt", {}).get(
        "baseline_shared_0.70"
    )
    output = {
        "samples": args.samples,
        "threshold_grid": {
            "implicit": implicit_thresholds,
            "text": text_thresholds,
        },
        "reference_checkpoint_shared_0.70": baseline,
        "best": overall_best,
        "go_gates": {
            "exact_rate_at_least_0.59": overall_best["exact_rate"] >= 0.59,
            "f1_at_least_0.975": overall_best["f1"] >= 0.975,
            "role_count_exact_at_least_0.62": overall_best[
                "role_count_exact_rate"
            ]
            >= 0.62,
        },
        "checkpoints": checkpoint_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("overall=" + json.dumps(output["best"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
