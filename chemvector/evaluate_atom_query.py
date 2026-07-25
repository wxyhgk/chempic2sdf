#!/usr/bin/env python3
"""Evaluate an Atom Query checkpoint over threshold and tolerance grids."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    apply_atom_query_dense_head,
    apply_atom_query_vision_updates,
    decode_atom_queries,
    decode_atom_queries_by_cardinality,
    decode_atom_queries_by_count_head,
    decode_atom_queries_by_count_regression,
    decode_atom_queries_by_known_counts,
    decode_atom_queries_by_assignment_oracle,
    decode_atom_queries_by_selection_known_counts,
    decode_atom_queries_by_duplicate_penalty_known_counts,
    decode_atom_queries_by_set_count,
    load_atom_query_state,
    snap_queries_to_dense_candidates,
)
from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.evaluate_vertex import parse_positive_floats, parse_thresholds
from chemvector.pretrain_atom_query import query_features
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator
from chemvector.vertex import decode_vertices


@torch.no_grad()
def evaluate(
    base_model,
    detector,
    loader,
    device: torch.device,
    *,
    feature_level: str,
    thresholds: tuple[float, ...],
    tolerances: tuple[float, ...],
    duplicate_penalties: tuple[float, ...],
    dense_model=None,
    dense_uses_query_features: bool = False,
    dense_threshold: float = 0.2,
    snap_distance_pixels: float = 64.0,
) -> dict:
    base_model.eval()
    detector.eval()
    modes = ("raw", "dense_snap") if dense_model is not None else ("raw",)
    totals = {
        (mode, threshold, tolerance): new_detection_totals()
        for mode in modes
        for threshold in thresholds
        for tolerance in tolerances
    }
    cardinality_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    count_head_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    count_regression_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    set_count_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    oracle_count_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    oracle_assignment_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    selection_oracle_totals = {
        tolerance: new_detection_totals() for tolerance in tolerances
    }
    duplicate_penalty_totals = {
        (penalty, tolerance): new_detection_totals()
        for penalty in duplicate_penalties
        for tolerance in tolerances
    }
    selection_component_names = (
        "ranking_prior_only",
        "local_context_only",
        "full_selection",
    )
    selection_component_oracle_totals = {
        (name, tolerance): new_detection_totals()
        for name in selection_component_names
        for tolerance in tolerances
    }
    selection_component_set_count_totals = {
        (name, tolerance): new_detection_totals()
        for name in selection_component_names
        for tolerance in tolerances
    }
    dense_topk_totals = (
        {
            (name, tolerance): new_detection_totals()
            for name in ("set_count", "selection_oracle", "assignment_oracle")
            for tolerance in tolerances
        }
        if dense_model is not None
        else {}
    )
    samples = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = query_features(
                base_model,
                pixels,
                feature_level=feature_level,
            )
            output = detector(features)
            dense_output = None
            if dense_model is not None:
                dense_features = (
                    features
                    if dense_uses_query_features and feature_level == "stroke"
                    else dense_model.vision_encoder.forward_stroke_resolution(pixels)
                )
                dense_output = dense_model.vertex_head(dense_features)
        dense_decoded = (
            decode_vertices(
                dense_output,
                threshold=dense_threshold,
                maximum_vertices=128,
            )
            if dense_output is not None
            else None
        )
        for threshold in thresholds:
            decoded = decode_atom_queries(output, threshold=threshold)
            decoded_modes = {"raw": decoded}
            if dense_decoded is not None:
                decoded_modes["dense_snap"] = [
                    snap_queries_to_dense_candidates(
                        query_items,
                        candidate_items,
                        image_size=batch["pixel_values"].shape[-1],
                        maximum_distance_pixels=snap_distance_pixels,
                    )
                    for query_items, candidate_items in zip(decoded, dense_decoded)
                ]
            for mode, mode_decoded in decoded_modes.items():
                for tolerance in tolerances:
                    update_detection_metrics(
                        totals[(mode, threshold, tolerance)],
                        mode_decoded,
                        batch["anchor_points"],
                        batch["anchor_mask"],
                        batch["aromatic_targets"],
                        batch["pixel_values"].shape[-1],
                        tolerance,
                    )
        cardinality_decoded = decode_atom_queries_by_cardinality(output)
        count_head_decoded = decode_atom_queries_by_count_head(output)
        count_regression_decoded = decode_atom_queries_by_count_regression(output)
        set_count_decoded = decode_atom_queries_by_set_count(output)
        oracle_count_decoded = decode_atom_queries_by_known_counts(
            output,
            batch["anchor_mask"].sum(dim=1),
        )
        oracle_assignment_decoded = decode_atom_queries_by_assignment_oracle(
            output,
            batch["anchor_points"],
            batch["anchor_mask"],
        )
        selection_oracle_decoded = decode_atom_queries_by_selection_known_counts(
            output,
            batch["anchor_mask"].sum(dim=1),
        )
        duplicate_penalty_decoded = {
            penalty: decode_atom_queries_by_duplicate_penalty_known_counts(
                output,
                batch["anchor_mask"].sum(dim=1),
                duplicate_penalty=penalty,
            )
            for penalty in duplicate_penalties
        }
        component_logits = {
            "ranking_prior_only": output["ranking_presence_logits"],
            "local_context_only": (
                output["local_presence_logits"]
                + output["selection_context_logits"]
            ),
            "full_selection": output["selection_logits"],
        }
        component_oracle_decoded = {}
        component_set_count_decoded = {}
        for name, logits in component_logits.items():
            component_output = dict(output)
            component_output["selection_logits"] = logits
            component_oracle_decoded[name] = (
                decode_atom_queries_by_selection_known_counts(
                    component_output,
                    batch["anchor_mask"].sum(dim=1),
                )
            )
            component_set_count_decoded[name] = decode_atom_queries_by_set_count(
                component_output
            )
        dense_topk_decoded = {}
        if dense_decoded is not None:
            def snap_batch(items):
                return [
                    snap_queries_to_dense_candidates(
                        query_items,
                        candidate_items,
                        image_size=batch["pixel_values"].shape[-1],
                        maximum_distance_pixels=snap_distance_pixels,
                    )
                    for query_items, candidate_items in zip(items, dense_decoded)
                ]

            dense_topk_decoded = {
                "set_count": snap_batch(set_count_decoded),
                "selection_oracle": snap_batch(selection_oracle_decoded),
                "assignment_oracle": snap_batch(oracle_assignment_decoded),
            }
        for tolerance in tolerances:
            update_detection_metrics(
                cardinality_totals[tolerance],
                cardinality_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                count_head_totals[tolerance],
                count_head_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                count_regression_totals[tolerance],
                count_regression_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                set_count_totals[tolerance],
                set_count_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                oracle_count_totals[tolerance],
                oracle_count_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                oracle_assignment_totals[tolerance],
                oracle_assignment_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            update_detection_metrics(
                selection_oracle_totals[tolerance],
                selection_oracle_decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
            for penalty, penalty_decoded in duplicate_penalty_decoded.items():
                update_detection_metrics(
                    duplicate_penalty_totals[(penalty, tolerance)],
                    penalty_decoded,
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    batch["pixel_values"].shape[-1],
                    tolerance,
                )
            for name in selection_component_names:
                update_detection_metrics(
                    selection_component_oracle_totals[(name, tolerance)],
                    component_oracle_decoded[name],
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    batch["pixel_values"].shape[-1],
                    tolerance,
                )
                update_detection_metrics(
                    selection_component_set_count_totals[(name, tolerance)],
                    component_set_count_decoded[name],
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    batch["pixel_values"].shape[-1],
                    tolerance,
                )
            for name, dense_decoded_items in dense_topk_decoded.items():
                update_detection_metrics(
                    dense_topk_totals[(name, tolerance)],
                    dense_decoded_items,
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    batch["pixel_values"].shape[-1],
                    tolerance,
                )
        samples += int(pixels.size(0))
    return {
        "samples": samples,
        "modes": {
            mode: {
                f"{tolerance:.4f}": {
                    f"{threshold:.4f}": summarize_detection_metrics(
                        totals[(mode, threshold, tolerance)]
                    )
                    for threshold in thresholds
                }
                for tolerance in tolerances
            }
            for mode in modes
        }
        | {
            "cardinality_topk": {
                f"{tolerance:.4f}": {
                    "expected_count": summarize_detection_metrics(
                        cardinality_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "count_head_topk": {
                f"{tolerance:.4f}": {
                    "predicted_count": summarize_detection_metrics(
                        count_head_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "count_regression_topk": {
                f"{tolerance:.4f}": {
                    "rounded_value": summarize_detection_metrics(
                        count_regression_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "set_count_topk": {
                f"{tolerance:.4f}": {
                    "distribution_mode": summarize_detection_metrics(
                        set_count_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "oracle_count_topk": {
                f"{tolerance:.4f}": {
                    "ground_truth_count": summarize_detection_metrics(
                        oracle_count_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "oracle_assignment_topk": {
                f"{tolerance:.4f}": {
                    "hungarian_truth_selection": summarize_detection_metrics(
                        oracle_assignment_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "selection_oracle_topk": {
                f"{tolerance:.4f}": {
                    "ground_truth_count": summarize_detection_metrics(
                        selection_oracle_totals[tolerance]
                    )
                }
                for tolerance in tolerances
            },
            "selection_duplicate_penalty_topk": {
                f"{tolerance:.4f}": {
                    f"{penalty:.4f}": summarize_detection_metrics(
                        duplicate_penalty_totals[(penalty, tolerance)]
                    )
                    for penalty in duplicate_penalties
                }
                for tolerance in tolerances
            },
            "selection_component_ablation": {
                f"{tolerance:.4f}": {
                    name: {
                        "oracle_count": summarize_detection_metrics(
                            selection_component_oracle_totals[(name, tolerance)]
                        ),
                        "set_count": summarize_detection_metrics(
                            selection_component_set_count_totals[(name, tolerance)]
                        ),
                    }
                    for name in selection_component_names
                }
                for tolerance in tolerances
            },
            "dense_snap_topk": {
                f"{tolerance:.4f}": {
                    name: summarize_detection_metrics(
                        dense_topk_totals[(name, tolerance)]
                    )
                    for name in ("set_count", "selection_oracle", "assignment_oracle")
                }
                for tolerance in tolerances
            }
            if dense_model is not None
            else {},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--reject-keys-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--dense-checkpoint")
    parser.add_argument("--dense-head-only", action="store_true")
    parser.add_argument("--query-dense-head", action="store_true")
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--snap-distance-pixels", type=float, default=64.0)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.3,0.5,0.7"),
    )
    parser.add_argument(
        "--tolerances",
        type=parse_positive_floats,
        default=parse_positive_floats("4,6,8,10,16"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--duplicate-penalties",
        type=parse_positive_floats,
        default=parse_positive_floats("0.5,1,2,4"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    base_model = load_vertex_model(Path(args.base_checkpoint), device).eval()
    payload = torch.load(args.query_checkpoint, map_location="cpu", weights_only=False)
    config = AtomQueryConfig(**payload["atom_query_config"])
    detector = AtomQueryDetector(config).to(device)
    load_atom_query_state(detector, payload["atom_query"])
    apply_atom_query_vision_updates(base_model, payload)
    dense_model = None
    dense_uses_query_features = False
    if args.query_dense_head and args.dense_checkpoint:
        raise ValueError("Query-packed and external dense heads are mutually exclusive")
    if args.query_dense_head:
        apply_atom_query_dense_head(base_model, payload)
        dense_model = base_model
        dense_uses_query_features = True
    if args.dense_checkpoint:
        dense_model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
        if args.dense_head_only:
            dense_model.vision_encoder = base_model.vision_encoder
            dense_uses_query_features = True
    elif args.dense_head_only:
        raise ValueError("--dense-head-only requires --dense-checkpoint")
    expected_channels = base_model.config.vision_dims[
        0 if args.feature_level == "stroke" else 1
    ]
    if config.input_channels != expected_channels:
        raise ValueError("Atom Query checkpoint input channels do not match feature level")
    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.shards),
            sequence_version="0.4",
            reject_keys=load_reject_keys(args.reject_keys_file),
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=VertexCollator(),
    )
    result = evaluate(
        base_model,
        detector,
        loader,
        device,
        feature_level=args.feature_level,
        thresholds=args.thresholds,
        tolerances=args.tolerances,
        duplicate_penalties=args.duplicate_penalties,
        dense_model=dense_model,
        dense_uses_query_features=dense_uses_query_features,
        dense_threshold=args.dense_threshold,
        snap_distance_pixels=args.snap_distance_pixels,
    )
    result.update(
        {
            "base_checkpoint": args.base_checkpoint,
            "query_checkpoint": args.query_checkpoint,
            "query_config": asdict(config),
            "feature_level": args.feature_level,
            "dense_checkpoint": args.dense_checkpoint,
            "dense_head_only": args.dense_head_only,
            "query_dense_head": args.query_dense_head,
            "dense_threshold": args.dense_threshold,
            "snap_distance_pixels": args.snap_distance_pixels,
            "duplicate_penalties": args.duplicate_penalties,
            "shards": expand_shards(args.shards),
            "reject_keys_file": args.reject_keys_file,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
