#!/usr/bin/env python3
"""Formally evaluate v0.9.6 pointer partitions with frozen geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_proposal_membership import PHYSICAL_GROUP_NAMES
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    bond_only_expected_v0_2_sequence,
    membership_rows_to_evaluator_rows,
    summarize_restricted_partitions,
)
from chemvector.bond_group_proposal_pointer import (
    BondGroupProposalPointerConfig,
    BondGroupProposalPointerPredictor,
)
from chemvector.bond_group_proposal_pointer_pipeline import (
    PointerCountResolution,
    decode_proposal_pointer_partition,
    pointer_root_selection_diagnostics,
    prepare_proposal_pointer_pipeline,
)
from chemvector.evaluate_bond_group_set import (
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_primitive_set import validate_shards
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_proposal_pointer import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    assert_frozen_flat_output,
    autocast_context,
    file_sha256,
    load_frozen_primitive_checkpoint,
    oracle_pool_contract,
    reject_internal_test_metadata,
    reject_internal_test_shards,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "bond-group-proposal-pointer-evaluation-0.1"
DECODE_MODES = (
    "known_count_oracle_usable_pool",
    "learned_count_oracle_usable_pool",
)


@dataclass(frozen=True, slots=True)
class EvaluationModels:
    pointer_model: BondGroupProposalPointerPredictor
    pointer_checkpoint: dict[str, Any]
    pointer_checkpoint_sha256: str
    primitive_model: PrimitiveSetPredictor
    primitive_checkpoint: dict[str, Any]
    primitive_checkpoint_path: Path
    primitive_checkpoint_sha256: str


def _strict_pointer_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupProposalPointerPredictor.model_type,
        "frozen_primitive_state_saved": False,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Pointer checkpoint metadata differs: {mismatches}")
    for name in (
        "model",
        "proposal_pointer_config",
        "primitive_provenance",
        "oracle_proposal_pool",
    ):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Pointer checkpoint has no valid {name}")
    source = checkpoint.get("source_primitive_checkpoint")
    digest = checkpoint.get("source_primitive_checkpoint_sha256")
    if not isinstance(source, str) or not source:
        raise ValueError("Pointer checkpoint has no source primitive path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Pointer checkpoint has no valid source primitive SHA256")
    provenance = checkpoint["primitive_provenance"]
    if provenance.get("checkpoint_sha256") != digest:
        raise ValueError("Pointer source SHA differs from primitive provenance")
    if provenance.get("model_type") != PrimitiveSetPredictor.model_type:
        raise ValueError("Pointer primitive provenance has the wrong model type")
    pool = checkpoint["oracle_proposal_pool"]
    if pool.get("source") != "v0.9.5_gt_associated_flat_proposals":
        raise ValueError("Pointer checkpoint uses an unknown proposal pool")
    if pool.get("unmatched_flat_false_positives_in_pool") is not False:
        raise ValueError("Pointer pool must exclude unmatched flat false positives")
    training_args = checkpoint.get("training_args")
    if not isinstance(training_args, dict):
        raise ValueError("Pointer checkpoint has invalid training_args")
    for field in ("train_shards", "validation_shards"):
        values = training_args.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"Pointer checkpoint has invalid {field}")
        reject_internal_test_shards(values)


def load_evaluation_models(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    primitive_checkpoint_override: str | Path | None = None,
) -> EvaluationModels:
    """Strictly load a pointer checkpoint and its exact frozen flat source."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pointer checkpoint does not exist: {path}")
    checkpoint_sha = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _strict_pointer_checkpoint(checkpoint)
    config = BondGroupProposalPointerConfig(
        **checkpoint["proposal_pointer_config"]
    )
    model = BondGroupProposalPointerPredictor(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()

    source = (
        Path(primitive_checkpoint_override).expanduser()
        if primitive_checkpoint_override is not None
        else Path(checkpoint["source_primitive_checkpoint"]).expanduser()
    ).resolve()
    source_sha = checkpoint["source_primitive_checkpoint_sha256"]
    primitive, primitive_checkpoint, provenance = (
        load_frozen_primitive_checkpoint(
            source,
            expected_sha256=source_sha,
            device=device,
        )
    )
    if config.proposal_feature_dim != primitive.config.d_model:
        raise ValueError(
            "Pointer proposal feature width differs from frozen primitive d_model"
        )
    expected_provenance = checkpoint["primitive_provenance"]
    if expected_provenance.get("checkpoint_global_step") != (
        primitive_checkpoint.get("global_step")
    ):
        raise ValueError("Frozen primitive global step differs from provenance")
    if provenance.get("checkpoint_sha256") != source_sha:
        raise ValueError("Loaded primitive provenance differs from pointer source")
    return EvaluationModels(
        pointer_model=model,
        pointer_checkpoint=checkpoint,
        pointer_checkpoint_sha256=checkpoint_sha,
        primitive_model=primitive,
        primitive_checkpoint=primitive_checkpoint,
        primitive_checkpoint_path=source,
        primitive_checkpoint_sha256=source_sha,
    )


def build_evaluation_collator(
    checkpoint: dict[str, Any],
    primitive_model: PrimitiveSetPredictor,
    *,
    image_size: int,
) -> BondGroupProposalMembershipCollator:
    """Recreate and verify the checkpoint's exact conditional proposal pool."""

    training_args = checkpoint["training_args"]
    trained_image_size = training_args.get("image_size")
    if trained_image_size is not None and int(trained_image_size) != image_size:
        raise ValueError(
            f"Evaluation image size {image_size} differs from training "
            f"image size {trained_image_size}"
        )
    pool = checkpoint["oracle_proposal_pool"]
    collator = BondGroupProposalMembershipCollator(
        image_size=image_size,
        decode_mode=str(pool["decode_mode"]),
        proposal_threshold=float(pool["proposal_threshold"]),
        high_confidence_tolerance_q=float(
            pool["high_confidence_tolerance_q"]
        ),
        association_tolerance_q=float(pool["association_tolerance_q"]),
        shell_weight=float(pool["shell_weight"]),
        max_proposals=int(pool["max_proposals"]),
    )
    if collator.max_proposals != primitive_model.config.queries:
        raise ValueError("Pointer pool capacity differs from primitive queries")
    actual = oracle_pool_contract(collator)
    if actual != pool:
        raise ValueError("Reconstructed pointer proposal pool differs from checkpoint")
    return collator


def _device_tensor_dict(
    values: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in values.items()
    }


def _f1_counts(
    true_positive: int,
    predicted: int,
    expected: int,
) -> dict[str, int | float]:
    precision = (
        1.0 if predicted == expected == 0 else true_positive / max(predicted, 1)
    )
    recall = (
        1.0 if predicted == expected == 0 else true_positive / max(expected, 1)
    )
    score = (
        1.0
        if predicted == expected == 0
        else 2.0 * precision * recall / max(precision + recall, 1e-12)
    )
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def _restricted_type_record(
    decoded_rows: list[dict[str, Any]],
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, Any]:
    active = torch.where(targets["group_mask"][batch_index].bool())[0]
    target_rows = []
    for slot in active.tolist():
        members = frozenset(
            torch.where(
                targets["group_membership"][batch_index, slot].bool()
            )[0].tolist()
        )
        target_rows.append(
            (
                members,
                int(targets["group_types"][batch_index, slot]),
                int(targets["member_counts"][batch_index, slot]),
            )
        )
    result = {}
    for group_type, kind in enumerate(PHYSICAL_GROUP_NAMES):
        if group_type == 0:
            continue
        expected = [row for row in target_rows if row[1] == group_type]
        predicted = [
            row for row in decoded_rows if int(row.get("type", 0)) == group_type
        ]
        valid = [
            (
                frozenset(int(value) for value in row["proposal_indices"]),
                group_type,
                int(row.get("member_count", 0)),
            )
            for row in predicted
            if bool(row.get("valid_membership", False))
        ]
        expected_sets = Counter(row[0] for row in expected)
        predicted_sets = Counter(row[0] for row in valid)
        expected_typed = Counter(expected)
        predicted_typed = Counter(valid)
        group_tp = sum((expected_sets & predicted_sets).values())
        typed_tp = sum((expected_typed & predicted_typed).values())
        invalid = len(predicted) - len(valid)
        result[kind] = {
            "partition_exact": invalid == 0 and predicted_sets == expected_sets,
            "typed_partition_exact": (
                invalid == 0 and predicted_typed == expected_typed
            ),
            "invalid_rows": invalid,
            "groups": _f1_counts(group_tp, len(predicted), len(expected)),
            "typed_groups": _f1_counts(
                typed_tp,
                len(predicted),
                len(expected),
            ),
        }
    return result


def _slice_tensor_dict(
    values: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    return {
        name: value[index : index + 1]
        if isinstance(value, torch.Tensor)
        else value
        for name, value in values.items()
    }


def _restricted_record(summary: dict[str, Any], index: int) -> dict[str, Any]:
    return dict(summary["records"][index])


def evaluate_pointer_output_batch(
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
    """Decode both conditional pointer modes and run the frozen evaluator."""

    model_inputs = prepared["model_inputs"]
    targets = prepared["targets"]
    proposal_mask = model_inputs["proposal_mask"].bool()
    usable_mask = prepared["oracle_usable_mask"].bool()
    known_counts = prepared["known_counts"].long()
    known_decoded, known_resolution = decode_proposal_pointer_partition(
        output,
        proposal_mask,
        root_candidate_mask=usable_mask,
        member_candidate_mask=usable_mask,
        mode="known_count",
        known_counts=known_counts,
    )
    learned_decoded, learned_resolution = decode_proposal_pointer_partition(
        output,
        proposal_mask,
        root_candidate_mask=usable_mask,
        member_candidate_mask=usable_mask,
        mode="learned_count",
    )
    decoded_by_mode = {
        DECODE_MODES[0]: (known_decoded, known_resolution),
        DECODE_MODES[1]: (learned_decoded, learned_resolution),
    }
    result = {mode: [] for mode in DECODE_MODES}
    for mode, (decoded, resolution) in decoded_by_mode.items():
        restricted = summarize_restricted_partitions(decoded, targets)
        for index, pointer_rows in enumerate(decoded):
            fitted = membership_rows_to_evaluator_rows(
                pointer_rows,
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
            root_diagnostics = pointer_root_selection_diagnostics(
                _slice_tensor_dict(output, index),
                [pointer_rows],
                _slice_tensor_dict(targets, index),
                usable_mask[index : index + 1],
            )
            row["pointer_rows"] = pointer_rows
            row["carrier_fit_failures"] = list(fitted.fit_failures)
            row["restricted_partition"] = _restricted_record(
                restricted,
                index,
            )
            row["restricted_by_type"] = _restricted_type_record(
                pointer_rows,
                targets,
                index,
            )
            row["root_diagnostics"] = root_diagnostics
            row["known_usable_group_count"] = int(known_counts[index])
            row["raw_predicted_group_count"] = int(
                resolution.raw_counts[index]
            )
            row["feasible_predicted_group_count"] = int(
                resolution.feasible_counts[index]
            )
            row["raw_group_count_infeasible"] = bool(
                resolution.infeasible[index]
            )
            row["matched_proposal_count"] = int(proposal_mask[index].sum())
            row["oracle_usable_proposal_count"] = int(usable_mask[index].sum())
            row["ignored_matched_proposal_count"] = int(
                (proposal_mask[index] & ~usable_mask[index]).sum()
            )
            row["source_primitive_checkpoint_sha256"] = (
                source_primitive_sha256
            )
            row["conditional_on_gt_associated_proposals"] = True
            row["oracle_usable_root_member_pool"] = True
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
    return {
        "samples": samples,
        "partition_exact": exact / max(samples, 1),
        "typed_partition_exact": typed_exact / max(samples, 1),
        "invalid_rows": invalid,
        "groups": _f1_counts(**group_counts),
        "typed_groups": _f1_counts(**typed_counts),
        "pairs": _f1_counts(**pair_counts),
    }


def summarize_restricted_types(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for kind in PHYSICAL_GROUP_NAMES[1:]:
        group_counts = Counter(true_positive=0, predicted=0, expected=0)
        typed_counts = Counter(true_positive=0, predicted=0, expected=0)
        exact = typed_exact = invalid = 0
        for row in rows:
            record = row["restricted_by_type"][kind]
            exact += int(record["partition_exact"])
            typed_exact += int(record["typed_partition_exact"])
            invalid += int(record["invalid_rows"])
            _merge_metric_counts(group_counts, record["groups"])
            _merge_metric_counts(typed_counts, record["typed_groups"])
        expected = typed_counts["expected"]
        result[kind] = {
            "samples": len(rows),
            "partition_exact": exact / max(len(rows), 1),
            "typed_partition_exact": typed_exact / max(len(rows), 1),
            "invalid_rows": invalid,
            "groups": _f1_counts(**group_counts),
            "typed_groups": _f1_counts(**typed_counts),
            "typed_group_exact_recall": (
                typed_counts["true_positive"] / max(expected, 1)
            ),
        }
    return result


def _merge_integer_totals(
    target: dict[str, int],
    values: dict[str, int],
) -> None:
    for name, value in values.items():
        target[name] = target.get(name, 0) + int(value)


def summarize_root_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for row in rows:
        _merge_integer_totals(totals, row["root_diagnostics"])
    samples = max(totals.get("samples", 0), 1)
    expected = totals.get("expected_groups", 0)
    roots = totals.get("selected_roots", 0)
    assigned = totals.get("assigned_members", 0)
    return {
        **totals,
        "distinct_target_group_coverage": (
            totals.get("distinct_target_groups", 0) / max(expected, 1)
        ),
        "root_collision_rate": totals.get("root_collisions", 0)
        / max(roots, 1),
        "distinct_group_coverage_exact_rate": totals.get(
            "distinct_group_coverage_exact_samples",
            0,
        )
        / samples,
        "affinity_self_preference_rate": totals.get(
            "affinity_self_preferred_roots",
            0,
        )
        / max(roots, 1),
        "affinity_majority_agreement_rate": totals.get(
            "affinity_majority_agrees_roots",
            0,
        )
        / max(roots, 1),
        "root_group_member_purity": totals.get(
            "assigned_members_same_as_root_group",
            0,
        )
        / max(assigned, 1),
    }


def summarize_count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    raw_errors = [
        abs(row["raw_predicted_group_count"] - row["known_usable_group_count"])
        for row in rows
    ]
    feasible_errors = [
        abs(
            row["feasible_predicted_group_count"]
            - row["known_usable_group_count"]
        )
        for row in rows
    ]
    return {
        "samples": samples,
        "raw_count_exact": sum(error == 0 for error in raw_errors)
        / max(samples, 1),
        "raw_count_mae": sum(raw_errors) / max(samples, 1),
        "raw_infeasible_samples": sum(
            row["raw_group_count_infeasible"] for row in rows
        ),
        "raw_infeasible_rate": sum(
            row["raw_group_count_infeasible"] for row in rows
        )
        / max(samples, 1),
        "feasible_count_exact": sum(error == 0 for error in feasible_errors)
        / max(samples, 1),
        "feasible_count_mae": sum(feasible_errors) / max(samples, 1),
    }


def summarize_candidate_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "matched_proposals": sum(
            row["matched_proposal_count"] for row in rows
        ),
        "oracle_usable_proposals": sum(
            row["oracle_usable_proposal_count"] for row in rows
        ),
        "ignored_matched_proposals_excluded_from_decode": sum(
            row["ignored_matched_proposal_count"] for row in rows
        ),
        "root_mask": "oracle_usable_proposals",
        "member_mask": "oracle_usable_proposals",
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


def _resolved_training_shards(values: list[str]) -> set[str]:
    resolved = set()
    for value in values:
        matches = glob.glob(value)
        candidates = matches or [value]
        resolved.update(str(Path(path).expanduser().resolve()) for path in candidates)
    return resolved


def _checkpoint_overlap(
    checkpoint: dict[str, Any],
    shards: list[str],
) -> dict[str, list[str]]:
    evaluated = set(shards)
    training_args = checkpoint["training_args"]
    return {
        field: sorted(
            evaluated
            & _resolved_training_shards(training_args.get(field, []))
        )
        for field in ("train_shards", "validation_shards")
    }


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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    shards = validate_shards(args.shard)
    reject_internal_test_shards(list(shards))
    device = _device(args.device)
    models = load_evaluation_models(
        args.checkpoint,
        device,
        primitive_checkpoint_override=args.primitive_checkpoint,
    )
    pointer_model = models.pointer_model
    primitive_model = models.primitive_model
    collator = build_evaluation_collator(
        models.pointer_checkpoint,
        primitive_model,
        image_size=args.image_size,
    )
    loader = DataLoader(
        VisualSVGWebDataset(list(shards), limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    rows_by_mode = {mode: [] for mode in DECODE_MODES}
    sample_key_hash = hashlib.sha256()
    samples_seen = 0
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                reject_internal_test_metadata(meta, str(key))
                sample_key_hash.update(f"{key}\n".encode("utf-8"))
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device):
                flat_output = primitive_model(pixels)
            assert_frozen_flat_output(primitive_model, flat_output)
            proposal_batch = collator.attach_flat_output(batch, flat_output)
            prepared = prepare_proposal_pointer_pipeline(
                proposal_batch,
                flat_output,
                primitive_model.reference_points.detach(),
                max_group_count=pointer_model.config.max_group_count,
            )
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            targets = _device_tensor_dict(prepared["targets"], device)
            prepared = {
                **prepared,
                "model_inputs": model_inputs,
                "targets": targets,
                "known_counts": prepared["known_counts"].to(device),
                "oracle_usable_mask": prepared["oracle_usable_mask"].to(device),
            }
            with autocast_context(device):
                output = pointer_model(**model_inputs)
            evaluated = evaluate_pointer_output_batch(
                output,
                prepared,
                proposal_batch,
                batch,
                line_tolerance=args.line_tolerance,
                text_box_iou=args.text_box_iou,
                max_members=pointer_model.config.max_members,
                source_primitive_sha256=models.primitive_checkpoint_sha256,
            )
            for mode in DECODE_MODES:
                rows_by_mode[mode].extend(evaluated[mode])
            samples_seen += len(batch["keys"])
            if samples_seen % 10 == 0:
                print(f"evaluated={samples_seen}", flush=True)
    if not samples_seen:
        raise RuntimeError("Proposal Pointer evaluator read no samples")

    results = []
    for mode in DECODE_MODES:
        rows = rows_by_mode[mode]
        results.append(
            {
                "decode_mode": mode,
                "threshold": None,
                "oracle_usable_root_member_pool": True,
                **summarize_prediction_rows(rows),
                "restricted_partition": summarize_restricted_rows(rows),
                "restricted_by_type": summarize_restricted_types(rows),
                "root_diagnostics": summarize_root_rows(rows),
                "count_resolution": summarize_count_rows(rows),
                "candidate_pool": summarize_candidate_pool(rows),
                "carrier_fit_failures": summarize_fit_failures(rows),
            }
        )
    checkpoint = models.pointer_checkpoint
    primitive_internal = models.primitive_checkpoint.get("uses_internal_test")
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_scope": (
            "capacity_conditional_gt_associated_oracle_usable_proposal_pool"
        ),
        "conditional_on_gt_associated_proposals": True,
        "oracle_usable_root_member_pool": True,
        "end_to_end_deployment_metric": False,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": models.pointer_checkpoint_sha256,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_samples_seen": int(checkpoint.get("samples_seen", -1)),
        "model_type": pointer_model.model_type,
        "config": checkpoint["proposal_pointer_config"],
        "source_primitive_checkpoint": str(models.primitive_checkpoint_path),
        "source_primitive_checkpoint_sha256": (
            models.primitive_checkpoint_sha256
        ),
        "source_primitive_global_step": int(
            models.primitive_checkpoint.get("global_step", -1)
        ),
        "source_primitive_model_type": primitive_model.model_type,
        "oracle_proposal_pool": checkpoint["oracle_proposal_pool"],
        "shards": list(shards),
        "shard_sha256": {path: file_sha256(Path(path)) for path in shards},
        "sample_key_sha256": sample_key_hash.hexdigest(),
        "checkpoint_shard_overlap": _checkpoint_overlap(checkpoint, list(shards)),
        "samples": samples_seen,
        "line_tolerance_q": args.line_tolerance,
        "text_box_iou": args.text_box_iou,
        "decode_results": results,
        "internal_test_audit": {
            "evaluation_reads_internal_test": False,
            "pointer_checkpoint_uses_internal_test": False,
            "primitive_checkpoint_uses_internal_test": (
                primitive_internal if primitive_internal is not None else "unknown"
            ),
        },
        "uses_internal_test": False,
    }
    write_evaluation_outputs(args.output_dir, metrics, rows_by_mode)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
