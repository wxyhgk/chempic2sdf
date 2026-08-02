#!/usr/bin/env python3
"""Formally evaluate v0.9.7 typed affinities with exact-cover packing.

This evaluator is deliberately conditional.  The model and decoder receive
only the GT-associated oracle-usable frozen proposal pool, and the exact-cover
decoder additionally receives the known number of usable physical groups.
It is therefore a formal bond-only evaluation of grouping and frozen carrier
fitting, not an end-to-end deployment metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_proposal_membership import PHYSICAL_GROUP_NAMES
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    DEFAULT_MAX_MEMBERS,
    bond_only_expected_v0_2_sequence,
    membership_rows_to_evaluator_rows,
    prepare_proposal_membership_pipeline,
    summarize_restricted_partitions,
)
from chemvector.bond_group_typed_affinity import (
    BondGroupTypedAffinityConfig,
    BondGroupTypedAffinityPredictor,
)
from chemvector.bond_group_typed_affinity_packing import (
    TypedAffinityPackingConfig,
    decode_typed_affinity_packing,
)
from chemvector.evaluate_bond_group_set import (
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_primitive_set import validate_shards
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_typed_affinity import (
    CHECKPOINT_SCHEMA_VERSION,
    CONDITIONAL_SCOPE_NAME,
    SEQUENCE_SCHEMA_VERSION,
    assert_frozen_flat_output,
    autocast_context,
    conditional_scope_contract,
    file_sha256,
    load_frozen_primitive_checkpoint,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    restrict_to_oracle_usable_pool,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "bond-group-typed-affinity-formal-evaluation-0.1"
DECODE_MODE = "known_count_oracle_usable_pool_typed_affinity_exact_cover"
FORMAL_LEARNED_AFFINITY_THRESHOLDS = (0.90, 0.95)


@dataclass(frozen=True, slots=True)
class EvaluationModels:
    affinity_model: BondGroupTypedAffinityPredictor
    affinity_checkpoint: dict[str, Any]
    affinity_checkpoint_sha256: str
    primitive_model: PrimitiveSetPredictor
    primitive_checkpoint: dict[str, Any]
    primitive_checkpoint_path: Path
    primitive_checkpoint_sha256: str


def _strict_typed_affinity_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupTypedAffinityPredictor.model_type,
        "frozen_primitive_state_saved": False,
        "candidate_generator_state_saved": False,
        "exact_cover_solver_state_saved": False,
        "formal_sdf_evaluation_in_scope": False,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Typed Affinity checkpoint metadata differs: {mismatches}")
    for name in (
        "model",
        "typed_affinity_config",
        "primitive_provenance",
        "conditional_scope",
        "training_args",
    ):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Typed Affinity checkpoint has no valid {name}")

    source = checkpoint.get("source_primitive_checkpoint")
    digest = checkpoint.get("source_primitive_checkpoint_sha256")
    if not isinstance(source, str) or not source:
        raise ValueError("Typed Affinity checkpoint has no source primitive path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Typed Affinity checkpoint has no valid primitive SHA256")
    provenance = checkpoint["primitive_provenance"]
    if provenance.get("checkpoint_sha256") != digest:
        raise ValueError("Typed Affinity source SHA differs from provenance")
    if provenance.get("model_type") != PrimitiveSetPredictor.model_type:
        raise ValueError("Typed Affinity primitive provenance has the wrong model type")

    scope = checkpoint["conditional_scope"]
    expected_scope_fields = {
        "name": CONDITIONAL_SCOPE_NAME,
        "conditional": True,
        "proposal_mask": "targets.proposal_keep & frozen_proposal_mask",
        "candidate_generator_in_scope": False,
        "exact_cover_solver_in_scope": False,
        "formal_sdf_in_scope": False,
        "end_to_end_metric": False,
    }
    scope_mismatches = {
        name: (scope.get(name), value)
        for name, value in expected_scope_fields.items()
        if scope.get(name) != value
    }
    if scope_mismatches:
        raise ValueError(
            "Typed Affinity conditional scope differs: "
            f"{scope_mismatches}"
        )
    if not isinstance(scope.get("frozen_proposal_pool"), dict):
        raise ValueError("Typed Affinity conditional scope has no proposal pool")

    training_args = checkpoint["training_args"]
    for field in ("train_shards", "validation_shards"):
        values = training_args.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"Typed Affinity checkpoint has invalid {field}")
        reject_internal_test_shards(values)


def load_evaluation_models(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    primitive_checkpoint_override: str | Path | None = None,
) -> EvaluationModels:
    """Strictly load v0.9.7 and the exact frozen primitive checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Typed Affinity checkpoint does not exist: {path}")
    checkpoint_sha = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _strict_typed_affinity_checkpoint(checkpoint)

    config = BondGroupTypedAffinityConfig(**checkpoint["typed_affinity_config"])
    model = BondGroupTypedAffinityPredictor(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()

    source = (
        Path(primitive_checkpoint_override).expanduser()
        if primitive_checkpoint_override is not None
        else Path(checkpoint["source_primitive_checkpoint"]).expanduser()
    ).resolve()
    source_sha = checkpoint["source_primitive_checkpoint_sha256"]
    primitive, primitive_checkpoint, provenance = load_frozen_primitive_checkpoint(
        source,
        expected_sha256=source_sha,
        device=device,
    )
    if config.proposal_feature_dim != primitive.config.d_model:
        raise ValueError(
            "Typed Affinity proposal feature width differs from primitive d_model"
        )
    expected_provenance = checkpoint["primitive_provenance"]
    if expected_provenance.get("checkpoint_global_step") != (
        primitive_checkpoint.get("global_step")
    ):
        raise ValueError("Frozen primitive global step differs from provenance")
    if provenance.get("checkpoint_sha256") != source_sha:
        raise ValueError("Loaded primitive provenance differs from checkpoint source")
    return EvaluationModels(
        affinity_model=model,
        affinity_checkpoint=checkpoint,
        affinity_checkpoint_sha256=checkpoint_sha,
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
    """Reconstruct the exact conditional proposal-pool protocol."""

    trained_image_size = checkpoint["training_args"].get("image_size")
    if trained_image_size is not None and int(trained_image_size) != image_size:
        raise ValueError(
            f"Evaluation image size {image_size} differs from training image size "
            f"{trained_image_size}"
        )
    pool = checkpoint["conditional_scope"]["frozen_proposal_pool"]
    collator = BondGroupProposalMembershipCollator(
        image_size=image_size,
        decode_mode=str(pool["decode_mode"]),
        proposal_threshold=float(pool["proposal_threshold"]),
        high_confidence_tolerance_q=float(pool["high_confidence_tolerance_q"]),
        association_tolerance_q=float(pool["association_tolerance_q"]),
        shell_weight=float(pool["shell_weight"]),
        max_proposals=int(pool["max_proposals"]),
    )
    if collator.max_proposals != primitive_model.config.queries:
        raise ValueError("Typed Affinity pool capacity differs from primitive queries")
    actual_scope = conditional_scope_contract(collator)
    if actual_scope != checkpoint["conditional_scope"]:
        raise ValueError(
            "Reconstructed Typed Affinity conditional scope differs from checkpoint"
        )
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


def _prepare_evaluation_batch(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    primitive_model: PrimitiveSetPredictor,
    collator: BondGroupProposalMembershipCollator,
    *,
    max_group_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal_batch = collator.attach_flat_output(batch, flat_output)
    prepared = prepare_proposal_membership_pipeline(
        proposal_batch,
        flat_output,
        primitive_model.reference_points.detach(),
        group_slots=max_group_count,
    )
    conditional_inputs, full_mask, usable_mask = restrict_to_oracle_usable_pool(
        prepared["model_inputs"],
        prepared["targets"],
    )
    return proposal_batch, {
        **prepared,
        "model_inputs": conditional_inputs,
        "full_proposal_mask": full_mask,
        "oracle_usable_mask": usable_mask,
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
            torch.where(targets["group_membership"][batch_index, slot].bool())[0]
            .detach()
            .cpu()
            .tolist()
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
            "typed_groups": _f1_counts(typed_tp, len(predicted), len(expected)),
        }
    return result


def evaluate_typed_affinity_output_batch(
    output: dict[str, torch.Tensor],
    prepared: dict[str, Any],
    proposal_batch: dict[str, Any],
    source_batch: dict[str, Any],
    *,
    packing_config: TypedAffinityPackingConfig,
    line_tolerance: float,
    text_box_iou: float,
    max_members: int,
    source_primitive_sha256: str,
) -> list[dict[str, Any]]:
    """Pack one model batch and run the frozen bond-only formal evaluator."""

    targets = prepared["targets"]
    usable_mask = prepared["oracle_usable_mask"].bool()
    known_counts = prepared["known_counts"].long()
    full_mask = prepared["full_proposal_mask"].bool()
    decoded_batch: list[list[dict[str, Any]]] = []
    packing_results = []
    for index in range(usable_mask.size(0)):
        packing = decode_typed_affinity_packing(
            {
                "node_group_type_logits": output["node_group_type_logits"][index],
                "pair_logits": output["pair_logits"][index],
            },
            proposal_batch["proposal_geometry"][index],
            proposal_batch["proposal_type"][index],
            usable_mask[index],
            known_group_count=int(known_counts[index]),
            config=packing_config,
        )
        rows = [dict(row) for row in packing.typed_rows]
        decoded_batch.append(rows)
        packing_results.append(packing)

    restricted = summarize_restricted_partitions(decoded_batch, targets)
    evaluated_rows = []
    for index, (typed_rows, packing) in enumerate(
        zip(decoded_batch, packing_results, strict=True)
    ):
        fitted = membership_rows_to_evaluator_rows(
            typed_rows,
            proposal_batch,
            batch_index=index,
            max_members=max_members,
        )
        row = evaluate_decoded_sample(
            sample_key=source_batch["keys"][index],
            decode_mode=DECODE_MODE,
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
        row["typed_rows"] = typed_rows
        row["carrier_fit_failures"] = list(fitted.fit_failures)
        row["restricted_partition"] = dict(restricted["records"][index])
        row["restricted_by_type"] = _restricted_type_record(
            typed_rows,
            targets,
            index,
        )
        row["candidate_counts"] = dict(packing.candidate_counts)
        row["solver"] = {
            "feasible": bool(packing.feasible),
            "objective": packing.objective,
            "latency_ms": float(packing.solver_latency_ms),
            "status": str(packing.solver_status),
            "status_code": int(packing.solver_status_code),
            "message": str(packing.solver_message),
            "timeout": "limit_reached" in str(packing.solver_status),
            "infeasible": not bool(packing.feasible),
        }
        row["known_usable_group_count"] = int(known_counts[index])
        row["matched_proposal_count"] = int(full_mask[index].sum())
        row["oracle_usable_proposal_count"] = int(usable_mask[index].sum())
        row["ignored_matched_proposal_count"] = int(
            (full_mask[index] & ~usable_mask[index]).sum()
        )
        row["source_primitive_checkpoint_sha256"] = source_primitive_sha256
        row["conditional_on_gt_associated_proposals"] = True
        row["oracle_usable_proposal_pool"] = True
        row["known_usable_group_count_given_to_decoder"] = True
        row["end_to_end_deployment_metric"] = False
        row["uses_internal_test"] = False
        evaluated_rows.append(row)
    return evaluated_rows


def _merge_metric_counts(
    target: dict[str, int],
    metrics: dict[str, int | float],
) -> None:
    for field in ("true_positive", "predicted", "expected"):
        target[field] = target.get(field, 0) + int(metrics[field])


def summarize_restricted_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    group_counts: dict[str, int] = {}
    typed_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
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
        group_counts: dict[str, int] = {}
        typed_counts: dict[str, int] = {}
        exact = typed_exact = invalid = 0
        for row in rows:
            record = row["restricted_by_type"][kind]
            exact += int(record["partition_exact"])
            typed_exact += int(record["typed_partition_exact"])
            invalid += int(record["invalid_rows"])
            _merge_metric_counts(group_counts, record["groups"])
            _merge_metric_counts(typed_counts, record["typed_groups"])
        expected = typed_counts.get("expected", 0)
        result[kind] = {
            "samples": len(rows),
            "partition_exact": exact / max(len(rows), 1),
            "typed_partition_exact": typed_exact / max(len(rows), 1),
            "invalid_rows": invalid,
            "groups": _f1_counts(**group_counts),
            "typed_groups": _f1_counts(**typed_counts),
            "group_exact_recall": group_counts.get("true_positive", 0)
            / max(group_counts.get("expected", 0), 1),
            "typed_group_exact_recall": typed_counts.get("true_positive", 0)
            / max(expected, 1),
            "typed_group_exact_definition": (
                "proposal_membership + group_type + original_member_count"
            ),
        }
    return result


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: Iterable[int | float]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0,
                "p50": 0.0, "p95": 0.0, "max": 0.0}
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "mean": total / len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def summarize_candidate_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {name for row in rows for name in row.get("candidate_counts", {})}
    )
    return {
        "per_field": {
            name: _distribution(
                int(row.get("candidate_counts", {}).get(name, 0)) for row in rows
            )
            for name in names
        },
        "matched_proposals": _distribution(
            row["matched_proposal_count"] for row in rows
        ),
        "oracle_usable_proposals": _distribution(
            row["oracle_usable_proposal_count"] for row in rows
        ),
        "ignored_matched_proposals_excluded_from_decode": _distribution(
            row["ignored_matched_proposal_count"] for row in rows
        ),
        "proposal_mask": "targets.proposal_keep & frozen_proposal_mask",
    }


def summarize_solver(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["solver"]["status"]) for row in rows)
    feasible = sum(bool(row["solver"]["feasible"]) for row in rows)
    timeout = sum(bool(row["solver"]["timeout"]) for row in rows)
    infeasible = sum(bool(row["solver"]["infeasible"]) for row in rows)
    return {
        "samples": len(rows),
        "feasible_samples": feasible,
        "feasible_rate": feasible / max(len(rows), 1),
        "infeasible_samples": infeasible,
        "infeasible_rate": infeasible / max(len(rows), 1),
        "timeout_samples": timeout,
        "timeout_rate": timeout / max(len(rows), 1),
        "optimal_samples": statuses.get("optimal", 0),
        "status_counts": dict(sorted(statuses.items())),
        "latency_ms": _distribution(row["solver"]["latency_ms"] for row in rows),
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


def build_decode_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    formal = summarize_prediction_rows(rows)
    restricted = summarize_restricted_rows(rows)
    by_type = summarize_restricted_types(rows)
    return {
        "decode_mode": DECODE_MODE,
        "threshold": None,
        "conditional_on_gt_associated_proposals": True,
        "oracle_usable_proposal_pool": True,
        "known_usable_group_count_given_to_decoder": True,
        "end_to_end_deployment_metric": False,
        **formal,
        "restricted_partition": restricted,
        "restricted_by_type": by_type,
        "candidate_counts": summarize_candidate_counts(rows),
        "solver": summarize_solver(rows),
        "carrier_fit_failures": summarize_fit_failures(rows),
        "headline_metrics": {
            "restricted_partition_exact": restricted["partition_exact"],
            "restricted_typed_partition_exact": restricted[
                "typed_partition_exact"
            ],
            "restricted_group_f1": restricted["groups"]["f1"],
            "restricted_typed_group_f1": restricted["typed_groups"]["f1"],
            "restricted_pair_f1": restricted["pairs"]["f1"],
            "per_type_group_exact_recall": {
                kind: values["group_exact_recall"]
                for kind, values in by_type.items()
            },
            "per_type_group_and_original_member_count_exact_recall": {
                kind: values["typed_group_exact_recall"]
                for kind, values in by_type.items()
            },
            "line_recall_r8": formal["member_line_geometry"]["recall"]["8q"],
            "line_recall_r32": formal["member_line_geometry"]["recall"]["32q"],
            "bond_iou_mean": formal["bond_iou"]["mean"],
            "bond_iou_min": formal["bond_iou"]["min"],
            "invalid_objects": formal["invalid"]["objects"],
            "carrier_fit_failure_objects": summarize_fit_failures(rows)["objects"],
        },
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
    rows: list[dict[str, Any]],
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
        for row in rows:
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
            evaluated & _resolved_training_shards(training_args.get(field, []))
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
    parser.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
    parser.add_argument(
        "--learned-affinity-thresholds",
        type=float,
        nargs="+",
        default=list(FORMAL_LEARNED_AFFINITY_THRESHOLDS),
    )
    parser.add_argument("--solver-time-limit-seconds", type=float, default=5.0)
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
    if min(args.batch_size, args.image_size, args.max_members) < 1 or args.workers < 0:
        raise ValueError("Batch/image/member sizes must be positive and workers nonnegative")
    if args.line_tolerance < 0.0:
        raise ValueError("--line-tolerance must be nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    packing_config = TypedAffinityPackingConfig(
        learned_affinity_thresholds=tuple(args.learned_affinity_thresholds),
        exact_cover_time_limit_seconds=args.solver_time_limit_seconds,
    )

    shards = validate_shards(args.shard)
    reject_internal_test_shards(list(shards))
    device = _device(args.device)
    models = load_evaluation_models(
        args.checkpoint,
        device,
        primitive_checkpoint_override=args.primitive_checkpoint,
    )
    affinity_model = models.affinity_model
    primitive_model = models.primitive_model
    checkpoint = models.affinity_checkpoint
    collator = build_evaluation_collator(
        checkpoint,
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

    rows: list[dict[str, Any]] = []
    sample_key_hash = hashlib.sha256()
    max_group_count = int(checkpoint["training_args"]["max_group_count"])
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"], strict=True):
                reject_internal_test_metadata(meta, str(key))
                sample_key_hash.update(f"{key}\n".encode("utf-8"))
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device):
                flat_output = primitive_model(pixels)
            assert_frozen_flat_output(primitive_model, flat_output)
            proposal_batch, prepared = _prepare_evaluation_batch(
                batch,
                flat_output,
                primitive_model,
                collator,
                max_group_count=max_group_count,
            )
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            with autocast_context(device):
                output = affinity_model(**model_inputs)
            rows.extend(
                evaluate_typed_affinity_output_batch(
                    output,
                    prepared,
                    proposal_batch,
                    batch,
                    packing_config=packing_config,
                    line_tolerance=args.line_tolerance,
                    text_box_iou=args.text_box_iou,
                    max_members=args.max_members,
                    source_primitive_sha256=models.primitive_checkpoint_sha256,
                )
            )
            if len(rows) % 10 == 0:
                print(f"evaluated={len(rows)}", flush=True)
    if not rows:
        raise RuntimeError("Typed Affinity formal evaluator read no samples")

    decode_result = build_decode_result(rows)
    primitive_internal = models.primitive_checkpoint.get("uses_internal_test")
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_scope": (
            "capacity_conditional_gt_associated_oracle_usable_proposal_pool_"
            "known_group_count_exact_cover_formal_bond_only"
        ),
        "conditional_on_gt_associated_proposals": True,
        "oracle_usable_proposal_pool": True,
        "known_usable_group_count_given_to_decoder": True,
        "end_to_end_deployment_metric": False,
        "formal_bond_only_metric": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": models.affinity_checkpoint_sha256,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_samples_seen": int(checkpoint.get("samples_seen", -1)),
        "model_type": affinity_model.model_type,
        "config": checkpoint["typed_affinity_config"],
        "packing_config": asdict(packing_config),
        "training_conditional_scope": checkpoint["conditional_scope"],
        "evaluation_oracle_information": [
            "which frozen proposals have complete usable GT association",
            "known number of usable physical groups for exact-cover decoding",
        ],
        "source_primitive_checkpoint": str(models.primitive_checkpoint_path),
        "source_primitive_checkpoint_sha256": models.primitive_checkpoint_sha256,
        "source_primitive_global_step": int(
            models.primitive_checkpoint.get("global_step", -1)
        ),
        "source_primitive_model_type": primitive_model.model_type,
        "shards": list(shards),
        "shard_sha256": {path: file_sha256(Path(path)) for path in shards},
        "sample_key_sha256": sample_key_hash.hexdigest(),
        "checkpoint_shard_overlap": _checkpoint_overlap(checkpoint, list(shards)),
        "samples": len(rows),
        "line_tolerance_q": args.line_tolerance,
        "text_box_iou": args.text_box_iou,
        "max_members": args.max_members,
        "decode_results": [decode_result],
        "internal_test_audit": {
            "evaluation_reads_internal_test": False,
            "typed_affinity_checkpoint_uses_internal_test": False,
            "primitive_checkpoint_uses_internal_test": (
                primitive_internal if primitive_internal is not None else "unknown"
            ),
        },
        "uses_internal_test": False,
    }
    write_evaluation_outputs(args.output_dir, metrics, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
