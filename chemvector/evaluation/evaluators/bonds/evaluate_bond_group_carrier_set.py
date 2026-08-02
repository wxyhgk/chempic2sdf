#!/usr/bin/env python3
"""Evaluate compact bond-group carriers through the existing v0.2 metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from chemvector.audit_primitive_set_geometry import distribution
from chemvector.bond_group_carrier_geometry import decode_bond_group_carriers
from chemvector.bond_group_carrier_set import (
    TEXT_BOX,
    BondGroupCarrierSetConfig,
    BondGroupCarrierSetPredictor,
    bond_group_carrier_set_loss,
    decode_bond_group_carrier_set,
)
from chemvector.bond_group_carrier_set_data import BondGroupCarrierSetCollator
from chemvector.bond_group_set import GROUP_ID_TO_KIND
from chemvector.bond_group_set_data import target_to_visual_svg_v0_2_json
from chemvector.evaluate_bond_group_set import (
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_primitive_set import (
    build_decode_groups,
    normalize_decode_modes,
    normalize_thresholds,
    validate_shards,
)
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_group_carrier_set import (
    CHECKPOINT_SCHEMA_VERSION,
    LOSS_FIELDS,
    SEQUENCE_SCHEMA_VERSION,
    SOURCE_BOND_GROUP_SHA256,
    autocast_context,
    reject_internal_test_metadata,
    targets_to_device,
    validate_resume_checkpoint,
)
from chemvector.train_primitive_set import file_sha256
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "bond-group-carrier-set-evaluation-0.1"
DECODE_MODES = ("count_head", "threshold")


@dataclass(frozen=True, slots=True)
class ExpandedCarrierRows:
    rows: tuple[dict[str, Any], ...]
    invalid_objects: tuple[dict[str, Any], ...]


class BondGroupCarrierSetEvaluationCollator(BondGroupCarrierSetCollator):
    """Retain canonical v0.2 strings and original bond layers for metrics."""

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(samples)
        batch["target_v0_2_sequences"] = [
            target_to_visual_svg_v0_2_json(self._target_payload(sample))
            for sample in samples
        ]
        return batch


def load_bond_group_carrier_set_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[BondGroupCarrierSetPredictor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(checkpoint)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupCarrierSetPredictor.model_type,
        "source_bond_group_checkpoint_sha256": SOURCE_BOND_GROUP_SHA256,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Carrier Set checkpoint metadata differs: {mismatches}")
    vision_payload = checkpoint.get("vision_config")
    config_payload = checkpoint.get("bond_group_carrier_set_config")
    state = checkpoint.get("model")
    if not isinstance(vision_payload, dict) or not isinstance(config_payload, dict):
        raise ValueError("Carrier Set checkpoint is missing model configurations")
    if not isinstance(state, dict):
        raise ValueError("Carrier Set checkpoint is missing model state")
    model = BondGroupCarrierSetPredictor(
        ChemVectorConfig(**vision_payload),
        BondGroupCarrierSetConfig(**config_payload),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def _carrier_tensor(
    value: Any,
    *,
    width: int | None,
    field: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    expected = () if width is None else (width,)
    if tensor.shape != expected:
        raise ValueError(f"{field} must have shape {expected}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{field} contains non-finite values")
    return tensor


def _carrier_invalid_payload(row: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "query_index": int(row.get("query_index", -1)),
        "score": float(row.get("score", 0.0)),
        "kind": str(row.get("kind", "unknown")),
        "member_count": int(row.get("member_count", 0)),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def carrier_rows_to_bond_group_rows(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    max_members: int,
) -> ExpandedCarrierRows:
    """Differentiably expand carrier rows into normalized physical members.

    The returned tensors stay in normalized floating-point coordinates.  The
    existing BondGroupSet v0.2 serializer performs the authoritative q2047 and
    stroke-width quantization, so every downstream metric keeps one protocol.
    """

    if max_members < 1:
        raise ValueError("max_members must be positive")
    expanded: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        object_type = int(row.get("type", -1))
        if object_type == TEXT_BOX or row.get("kind") == "text_box":
            expanded.append(row)
            continue
        try:
            if object_type not in GROUP_ID_TO_KIND or object_type == TEXT_BOX:
                raise ValueError(f"Unsupported carrier type id: {object_type}")
            expected_kind = GROUP_ID_TO_KIND[object_type]
            if row.get("kind") != expected_kind:
                raise ValueError(
                    f"Carrier kind/type mismatch: {row.get('kind')!r} != "
                    f"{expected_kind!r}"
                )
            count = int(row["member_count"])
            points = _carrier_tensor(
                row["carrier_points"],
                width=8,
                field="carrier_points",
            )
            positive = _carrier_tensor(
                row["positive_params"],
                width=2,
                field="positive_params",
            ).to(device=points.device, dtype=points.dtype)
            signed = _carrier_tensor(
                row["signed_params"],
                width=2,
                field="signed_params",
            ).to(device=points.device, dtype=points.dtype)
            stroke = _carrier_tensor(
                row["group_stroke"],
                width=None,
                field="group_stroke",
            ).to(device=points.device, dtype=points.dtype)
            decoded = decode_bond_group_carriers(
                points.unsqueeze(0),
                positive.unsqueeze(0),
                signed.unsqueeze(0),
                torch.tensor([object_type], device=points.device, dtype=torch.long),
                torch.tensor([count], device=points.device, dtype=torch.long),
                stroke.unsqueeze(0),
                max_members=max_members,
            )
            geometry = decoded.member_geometry[0, :count]
            member_stroke = decoded.member_stroke[0, :count]
            if not bool(torch.isfinite(geometry).all()):
                raise ValueError("Expanded member geometry contains non-finite values")
            if not bool(torch.isfinite(member_stroke).all()):
                raise ValueError("Expanded member stroke contains non-finite values")
            row["member_geometry"] = geometry
            row["member_stroke"] = member_stroke
            expanded.append(row)
        except (KeyError, TypeError, ValueError) as error:
            invalid.append(_carrier_invalid_payload(row, error))
            # Keep the selected object visible to the standard evaluator.  An
            # empty physical payload makes it invalid instead of erasing it.
            row["member_geometry"] = torch.empty((0, 8), dtype=torch.float32)
            row["member_stroke"] = torch.empty((0,), dtype=torch.float32)
            expanded.append(row)
    return ExpandedCarrierRows(tuple(expanded), tuple(invalid))


def _white_comparison(
    real: dict[str, torch.Tensor],
    white: dict[str, torch.Tensor],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for field, scale in (
        ("carrier_points", 2047.0),
        ("positive_params", 2047.0),
        ("signed_params", 2047.0),
        ("group_stroke", 63.0),
        ("text_geometry", 2047.0),
    ):
        delta = (real[field].float() - white[field].float()).abs() * scale
        comparisons[f"{field}_abs"] = distribution(delta.reshape(-1).cpu().tolist())
    comparisons["type_argmax_same_rate"] = float(
        (
            real["type_logits"].argmax(dim=-1)
            == white["type_logits"].argmax(dim=-1)
        )
        .float()
        .mean()
    )
    comparisons["member_count_argmax_same_rate"] = float(
        (
            real["member_count_logits"].argmax(dim=-1)
            == white["member_count_logits"].argmax(dim=-1)
        )
        .float()
        .mean()
    )
    comparisons["total_count_argmax_same_rate"] = float(
        (
            real["count_logits"].argmax(dim=-1)
            == white["count_logits"].argmax(dim=-1)
        )
        .float()
        .mean()
    )
    return comparisons


def _mean_loss_totals(
    totals: dict[str, float],
    samples: int,
) -> dict[str, float]:
    return {name: value / samples for name, value in totals.items()}


def _summarize_carrier_invalid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    objects = [
        invalid
        for row in rows
        for invalid in row.get("carrier_decode_invalid_objects", [])
    ]
    return {
        "objects": len(objects),
        "samples": sum(bool(row.get("carrier_decode_invalid_objects")) for row in rows),
        "error_types": dict(Counter(row["error_type"] for row in objects)),
        "examples": objects[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument(
        "--decode-mode",
        action="append",
        choices=DECODE_MODES,
        help="Repeat to select modes; default evaluates count_head first",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--line-tolerance", type=float, default=8.0)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--compare-white", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decode_modes = (
        normalize_decode_modes(args.decode_mode)
        if args.decode_mode is not None
        else DECODE_MODES
    )
    thresholds = normalize_thresholds(args.threshold)
    if args.threshold is not None and "threshold" not in decode_modes:
        raise ValueError("--threshold is only meaningful with threshold decoding")
    decode_groups = build_decode_groups(decode_modes, thresholds)
    # Count-head is the decision metric.  Keep it first even if a helper sorts
    # or normalizes the user's repeated CLI flags.
    decode_groups = sorted(
        decode_groups,
        key=lambda item: (item[0] != "count_head", item[1] or 0.0),
    )
    shards = validate_shards(args.shard)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch/image size must be positive and workers nonnegative")
    if args.line_tolerance < 0:
        raise ValueError("--line-tolerance must be nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_bond_group_carrier_set_model(args.checkpoint, device)
    loader = DataLoader(
        VisualSVGWebDataset(list(shards), limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=BondGroupCarrierSetEvaluationCollator(
            image_size=args.image_size,
            query_count=model.config.queries,
            max_members=model.config.max_members,
        ),
    )
    rows_by_group: dict[tuple[str, float | None], list[dict[str, Any]]] = {
        group: [] for group in decode_groups
    }
    loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    white_loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    white_rows: list[dict[str, Any]] = []
    raw_member_count_matched = 0
    raw_member_count_correct = 0
    raw_member_count_by_type: dict[str, list[int]] = {
        name: [0, 0] for name in GROUP_ID_TO_KIND.values()
    }
    samples_seen = 0
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                reject_internal_test_metadata(meta, key)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            with autocast_context(device):
                output = model(pixels)
                losses = bond_group_carrier_set_loss(
                    output,
                    targets,
                    image_size=args.image_size,
                )
                if args.compare_white:
                    white_output = model(torch.ones_like(pixels))
                    white_losses = bond_group_carrier_set_loss(
                        white_output,
                        targets,
                        image_size=args.image_size,
                    )
            for name in LOSS_FIELDS:
                loss_totals[name] += float(losses[name]) * pixels.size(0)
                if args.compare_white:
                    white_loss_totals[name] += float(white_losses[name]) * pixels.size(0)
            for batch_index, (query_indices, target_indices) in enumerate(
                losses["assignments"]
            ):
                if not len(query_indices):
                    continue
                raw_counts = output["member_count_logits"][
                    batch_index, query_indices
                ].argmax(dim=-1)
                expected_counts = targets["member_counts"][
                    batch_index, target_indices
                ]
                expected_types = targets["types"][batch_index, target_indices]
                raw_member_count_matched += len(query_indices)
                raw_member_count_correct += int((raw_counts == expected_counts).sum())
                for type_id in torch.unique(expected_types).tolist():
                    mask = expected_types == type_id
                    kind = GROUP_ID_TO_KIND[int(type_id)]
                    raw_member_count_by_type[kind][0] += int(mask.sum())
                    raw_member_count_by_type[kind][1] += int(
                        (raw_counts[mask] == expected_counts[mask]).sum()
                    )
            if args.compare_white:
                white_rows.append(_white_comparison(output, white_output))

            for decode_mode, threshold in decode_groups:
                decoded = decode_bond_group_carrier_set(
                    output,
                    threshold=threshold if threshold is not None else 0.5,
                    mode=decode_mode,
                )
                for index, carrier_rows in enumerate(decoded):
                    expanded = carrier_rows_to_bond_group_rows(
                        carrier_rows,
                        max_members=model.config.max_members,
                    )
                    row = evaluate_decoded_sample(
                        sample_key=batch["keys"][index],
                        decode_mode=decode_mode,
                        threshold=threshold,
                        decoded_rows=list(expanded.rows),
                        expected_v0_2_sequence=batch["target_v0_2_sequences"][index],
                        bond_layer=batch["bond_layers"][index],
                        line_tolerance=args.line_tolerance,
                        text_box_iou=args.text_box_iou,
                        max_members=model.config.max_members,
                    )
                    row["carrier_decode_invalid_objects"] = list(
                        expanded.invalid_objects
                    )
                    rows_by_group[(decode_mode, threshold)].append(row)
            samples_seen += pixels.size(0)
            if samples_seen % 10 == 0:
                print(f"evaluated={samples_seen}", flush=True)
    if not samples_seen:
        raise RuntimeError("Carrier Set evaluator read no samples")

    carrier_losses = _mean_loss_totals(loss_totals, samples_seen)
    white_summary = None
    if args.compare_white:
        mean_white = _mean_loss_totals(white_loss_totals, samples_seen)
        white_summary = {
            "losses": mean_white,
            "white_image_loss_ratio": mean_white["loss"]
            / max(carrier_losses["loss"], 1e-8),
            "batch_comparisons": white_rows,
        }

    results = []
    for decode_mode, threshold in decode_groups:
        rows = rows_by_group[(decode_mode, threshold)]
        results.append(
            {
                "decode_mode": decode_mode,
                "threshold": threshold,
                "decision_metric": decode_mode == "count_head",
                **summarize_prediction_rows(rows),
                "carrier_decode_invalid": _summarize_carrier_invalid(rows),
            }
        )
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "model_type": model.model_type,
        "primary_decode_mode": "count_head",
        "config": {
            "queries": model.config.queries,
            "max_members": model.config.max_members,
            "local_query_kernel": model.config.local_query_kernel,
            "endpoint_refine_kernel": model.config.endpoint_refine_kernel,
            "line_tolerance_q": args.line_tolerance,
            "text_box_iou": args.text_box_iou,
        },
        "shards": list(shards),
        "samples": samples_seen,
        "carrier_parameter_losses": carrier_losses,
        "raw_member_count": {
            "matched_objects": raw_member_count_matched,
            "correct": raw_member_count_correct,
            "accuracy": raw_member_count_correct
            / max(raw_member_count_matched, 1),
            "by_type": {
                kind: {
                    "matched_objects": values[0],
                    "correct": values[1],
                    "accuracy": values[1] / max(values[0], 1),
                }
                for kind, values in raw_member_count_by_type.items()
            },
            "note": (
                "Diagnostic raw argmax before fixed group cardinalities and "
                "hashed-wedge count constraints are applied"
            ),
        },
        "decode_results": results,
        "white_image_comparison": white_summary,
        "uses_internal_test": False,
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for group in decode_groups:
            for row in rows_by_group[group]:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
