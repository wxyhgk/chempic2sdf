#!/usr/bin/env python3
"""Formally evaluate v0.9.5 membership with frozen flat geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_proposal_membership import (
    BondGroupProposalMembershipConfig,
    BondGroupProposalMembershipPredictor,
    decode_bond_group_proposal_membership,
)
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    bond_only_expected_v0_2_sequence,
    membership_rows_to_evaluator_rows,
    prepare_proposal_membership_pipeline,
    summarize_restricted_partitions,
)
from chemvector.evaluate_bond_group_set import (
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_primitive_set import validate_shards
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_proposal_membership import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    autocast_context,
    file_sha256,
    freeze_primitive_model,
    load_frozen_primitive_checkpoint,
    oracle_keep_type_output,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    validate_frozen_primitive_checkpoint,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "bond-group-proposal-membership-evaluation-0.1"
DECODE_MODES = ("learned_count", "known_count_oracle_keep_type")


@dataclass(frozen=True, slots=True)
class EvaluationModels:
    membership_model: BondGroupProposalMembershipPredictor
    membership_checkpoint: dict[str, Any]
    membership_checkpoint_sha256: str
    primitive_model: PrimitiveSetPredictor
    primitive_checkpoint: dict[str, Any]
    primitive_checkpoint_path: Path
    primitive_checkpoint_sha256: str


def _strict_membership_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupProposalMembershipPredictor.model_type,
        "frozen_primitive_state_saved": False,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Membership checkpoint metadata differs: {mismatches}")
    for name in ("model", "proposal_membership_config", "primitive_provenance"):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Membership checkpoint has no {name}")
    source = checkpoint.get("source_primitive_checkpoint")
    digest = checkpoint.get("source_primitive_checkpoint_sha256")
    if not isinstance(source, str) or not source:
        raise ValueError("Membership checkpoint has no source primitive path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Membership checkpoint has no valid source primitive SHA256")
    provenance = checkpoint["primitive_provenance"]
    if provenance.get("checkpoint_sha256") != digest:
        raise ValueError("Membership source SHA differs from primitive provenance")
    if provenance.get("model_type") != PrimitiveSetPredictor.model_type:
        raise ValueError("Membership primitive provenance has the wrong model type")
    training_args = checkpoint.get("training_args")
    if not isinstance(training_args, dict):
        raise ValueError("Membership checkpoint has invalid training_args")
    for field in ("train_shards", "validation_shards"):
        values = training_args.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"Membership checkpoint has invalid {field}")
        reject_internal_test_shards(values)


def load_evaluation_models(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    primitive_checkpoint_override: str | Path | None = None,
) -> EvaluationModels:
    """Strictly load membership and its exact frozen primitive source."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Membership checkpoint does not exist: {path}")
    membership_sha = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _strict_membership_checkpoint(checkpoint)
    config = BondGroupProposalMembershipConfig(
        **checkpoint["proposal_membership_config"]
    )
    model = BondGroupProposalMembershipPredictor(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()

    source = (
        Path(primitive_checkpoint_override).expanduser()
        if primitive_checkpoint_override is not None
        else Path(checkpoint["source_primitive_checkpoint"]).expanduser()
    ).resolve()
    source_sha = checkpoint["source_primitive_checkpoint_sha256"]
    primitive, primitive_checkpoint, _ = load_frozen_primitive_checkpoint(
        source,
        expected_sha256=source_sha,
        device=device,
    )
    validate_frozen_primitive_checkpoint(primitive_checkpoint)
    freeze_primitive_model(primitive)
    if config.proposal_feature_dim != primitive.config.d_model:
        raise ValueError(
            "Membership proposal feature width differs from frozen primitive d_model"
        )
    provenance = checkpoint["primitive_provenance"]
    if provenance.get("checkpoint_global_step") != primitive_checkpoint.get(
        "global_step"
    ):
        raise ValueError("Frozen primitive global step differs from provenance")
    return EvaluationModels(
        membership_model=model,
        membership_checkpoint=checkpoint,
        membership_checkpoint_sha256=membership_sha,
        primitive_model=primitive,
        primitive_checkpoint=primitive_checkpoint,
        primitive_checkpoint_path=source,
        primitive_checkpoint_sha256=source_sha,
    )


def _device_tensor_dict(values: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in values.items()
    }


def _restricted_record(summary: dict[str, Any], index: int) -> dict[str, Any]:
    return dict(summary["records"][index])


def evaluate_membership_output_batch(
    output: dict[str, torch.Tensor],
    prepared: dict[str, Any],
    proposal_batch: dict[str, Any],
    source_batch: dict[str, Any],
    *,
    line_tolerance: float,
    text_box_iou: float,
    max_members: int,
    source_primitive_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Decode both formal modes and apply the frozen-geometry evaluator."""

    model_inputs = prepared["model_inputs"]
    targets = prepared["targets"]
    proposal_mask = model_inputs["proposal_mask"].bool()
    known_counts = prepared["known_counts"].to(proposal_mask.device)
    decoded_by_mode = {
        "learned_count": decode_bond_group_proposal_membership(
            output,
            proposal_mask,
            mode="learned_count",
        ),
        "known_count_oracle_keep_type": decode_bond_group_proposal_membership(
            oracle_keep_type_output(output, targets, proposal_mask),
            proposal_mask,
            mode="known_count",
            known_counts=known_counts,
        ),
    }
    result = {mode: [] for mode in DECODE_MODES}
    for mode in DECODE_MODES:
        decoded = decoded_by_mode[mode]
        restricted = summarize_restricted_partitions(decoded, targets)
        for index, membership_rows in enumerate(decoded):
            fitted = membership_rows_to_evaluator_rows(
                membership_rows,
                proposal_batch,
                batch_index=index,
                max_members=max_members,
            )
            row = evaluate_decoded_sample(
                sample_key=source_batch["keys"][index],
                decode_mode=mode,
                threshold=None,
                decoded_rows=list(fitted.rows),
                expected_v0_2_sequence=bond_only_expected_v0_2_sequence(
                    source_batch["primitives"][index]
                ),
                bond_layer=source_batch["bond_layers"][index],
                line_tolerance=line_tolerance,
                text_box_iou=text_box_iou,
                max_members=max_members,
            )
            row["membership_rows"] = membership_rows
            row["carrier_fit_failures"] = list(fitted.fit_failures)
            row["restricted_partition"] = _restricted_record(restricted, index)
            row["known_group_count"] = int(known_counts[index])
            row["oracle_proposal_keep_type"] = (
                mode == "known_count_oracle_keep_type"
            )
            row["source_primitive_checkpoint_sha256"] = source_primitive_sha256
            row["uses_internal_test"] = False
            result[mode].append(row)
    return result


def _merge_metric_counts(
    target: dict[str, int],
    metrics: dict[str, int | float],
) -> None:
    for field in ("true_positive", "predicted", "expected"):
        target[field] += int(metrics[field])


def summarize_restricted_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    group_counts = Counter(true_positive=0, predicted=0, expected=0)
    typed_counts = Counter(true_positive=0, predicted=0, expected=0)
    pair_counts = Counter(true_positive=0, predicted=0, expected=0)
    exact = typed_exact = invalid = 0
    for row in rows:
        record = row["restricted_partition"]
        exact += int(record["partition_exact"])
        typed_exact += int(record["typed_partition_exact"])
        invalid += int(record["invalid_rows"])
        for name, accumulator in (
            ("groups", group_counts),
            ("typed_groups", typed_counts),
            ("pairs", pair_counts),
        ):
            _merge_metric_counts(accumulator, record[name])

    def finalize(values: Counter) -> dict[str, Any]:
        predicted = values["predicted"]
        expected = values["expected"]
        true_positive = values["true_positive"]
        precision = true_positive / max(predicted, 1)
        recall = true_positive / max(expected, 1)
        return {
            **dict(values),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        }
    return {
        "samples": samples,
        "partition_exact": exact / max(samples, 1),
        "typed_partition_exact": typed_exact / max(samples, 1),
        "invalid_rows": invalid,
        "groups": finalize(group_counts),
        "typed_groups": finalize(typed_counts),
        "pairs": finalize(pair_counts),
    }


def summarize_fit_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [
        (row["sample_key"], failure)
        for row in rows
        for failure in row["carrier_fit_failures"]
    ]
    by_kind = Counter(str(value.get("kind", "unknown")) for _, value in failures)
    return {
        "objects": len(failures),
        "samples": len({sample_key for sample_key, _ in failures}),
        "by_kind": dict(sorted(by_kind.items())),
        "examples": [
            {"sample_key": sample_key, **failure}
            for sample_key, failure in failures[:50]
        ],
    }


def _json_default(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_evaluation_outputs(
    output_dir: str | Path,
    metrics: dict[str, Any],
    rows_by_mode: dict[str, list[dict[str, Any]]],
) -> None:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with (path / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for mode in DECODE_MODES:
            for row in rows_by_mode[mode]:
                handle.write(
                    json.dumps(row, ensure_ascii=True, default=_json_default) + "\n"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--primitive-checkpoint")
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--line-tolerance", type=float, default=8.0)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch/image size must be positive and workers nonnegative")
    if args.line_tolerance < 0.0:
        raise ValueError("--line-tolerance must be nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    shards = validate_shards(args.shard)
    reject_internal_test_shards(list(shards))
    device = torch.device(args.device)
    models = load_evaluation_models(
        args.checkpoint,
        device,
        primitive_checkpoint_override=args.primitive_checkpoint,
    )
    membership_model = models.membership_model
    primitive_model = models.primitive_model
    collator = BondGroupProposalMembershipCollator(
        image_size=args.image_size,
        max_proposals=primitive_model.config.queries,
    )
    loader = DataLoader(
        VisualSVGWebDataset(list(shards), limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    rows_by_mode = {mode: [] for mode in DECODE_MODES}
    samples_seen = 0
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                reject_internal_test_metadata(meta, str(key))
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device):
                flat_output = primitive_model(pixels)
            proposal_batch = collator.attach_flat_output(batch, flat_output)
            prepared = prepare_proposal_membership_pipeline(
                proposal_batch,
                flat_output,
                primitive_model.reference_points.detach(),
                group_slots=membership_model.config.group_queries,
            )
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            targets = _device_tensor_dict(prepared["targets"], device)
            prepared = {
                **prepared,
                "model_inputs": model_inputs,
                "targets": targets,
                "known_counts": prepared["known_counts"].to(device),
            }
            with autocast_context(device):
                output = membership_model(**model_inputs)
            evaluated = evaluate_membership_output_batch(
                output,
                prepared,
                proposal_batch,
                batch,
                line_tolerance=args.line_tolerance,
                text_box_iou=args.text_box_iou,
                max_members=membership_model.config.max_members,
                source_primitive_sha256=models.primitive_checkpoint_sha256,
            )
            for mode in DECODE_MODES:
                rows_by_mode[mode].extend(evaluated[mode])
            samples_seen += len(batch["keys"])
            if samples_seen % 10 == 0:
                print(f"evaluated={samples_seen}", flush=True)
    if not samples_seen:
        raise RuntimeError("Proposal Membership evaluator read no samples")

    results = []
    for mode in DECODE_MODES:
        rows = rows_by_mode[mode]
        results.append(
            {
                "decode_mode": mode,
                "threshold": None,
                "oracle_proposal_keep_type": (
                    mode == "known_count_oracle_keep_type"
                ),
                **summarize_prediction_rows(rows),
                "restricted_partition": summarize_restricted_rows(rows),
                "carrier_fit_failures": summarize_fit_failures(rows),
            }
        )
    checkpoint = models.membership_checkpoint
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": models.membership_checkpoint_sha256,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_samples_seen": int(checkpoint.get("samples_seen", -1)),
        "model_type": membership_model.model_type,
        "config": checkpoint["proposal_membership_config"],
        "source_primitive_checkpoint": str(models.primitive_checkpoint_path),
        "source_primitive_checkpoint_sha256": models.primitive_checkpoint_sha256,
        "source_primitive_global_step": int(
            models.primitive_checkpoint.get("global_step", -1)
        ),
        "source_primitive_model_type": primitive_model.model_type,
        "shards": list(shards),
        "samples": samples_seen,
        "line_tolerance_q": args.line_tolerance,
        "text_box_iou": args.text_box_iou,
        "decode_results": results,
        "uses_internal_test": False,
    }
    write_evaluation_outputs(args.output_dir, metrics, rows_by_mode)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
