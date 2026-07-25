#!/usr/bin/env python3
"""Explicit one-pass validation for a frozen Evidence Carrier pair scorer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    validate_molecule_disjoint_splits,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
    reject_internal_test_shards,
)
from chemvector.evidence_carrier_pair_cache import (
    CACHE_SCHEMA_VERSION,
    load_evidence_carrier_pair_cache,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    score_evidence_carrier_pair_record,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    file_sha256,
    load_pair_scorer_checkpoint,
)


EVALUATION_SCHEMA_VERSION = "evidence-carrier-pair-scorer-validation-0.1"
DEFAULT_VALIDATION_LIMIT = 500
EXPECTED_RENDER_VARIANTS = frozenset(("coordgen", "original", "rdkit", "stress"))


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() != ".json":
        path = path / "validation.json"
    return path.resolve(strict=False)


def _expand_validation_shards(patterns: Sequence[str]) -> list[str]:
    reject_internal_test_shards(patterns)
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        current = matches or [str(Path(pattern).expanduser())]
        paths.extend(current)
    resolved = []
    for value in paths:
        path = Path(value).resolve(strict=False)
        if _contains_internal_test(path):
            raise ValueError("Pair scorer evaluation cannot read internal_test shards")
        if not path.is_file():
            raise FileNotFoundError(f"Validation shard does not exist: {path}")
        resolved.append(str(path))
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Validation shard expansion is empty or contains duplicates")
    return resolved


def _shard_descriptors(paths: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(Path(path).resolve()),
            "size_bytes": Path(path).stat().st_size,
        }
        for path in paths
    ]


def _train_cache_identities(cache: Mapping[str, Any]) -> list[ConditionalPairSampleIdentity]:
    identities = []
    sample_keys: set[str] = set()
    for side in ("fit_records", "calibration_records"):
        records = cache.get(side)
        if not isinstance(records, list):
            raise ValueError(f"Train cache lacks {side}")
        for record in records:
            payload = record.get("identity") if isinstance(record, Mapping) else None
            if not isinstance(payload, Mapping):
                raise ValueError("Train cache record lacks identity metadata")
            identity = ConditionalPairSampleIdentity(**dict(payload))
            if identity.source_split != "train":
                raise ValueError("Train cache contains a non-train identity")
            if identity.sample_key in sample_keys:
                raise ValueError("Train cache contains duplicate sample identities")
            sample_keys.add(identity.sample_key)
            identities.append(identity)
    if not identities:
        raise ValueError("Train cache contains no identities")
    return identities


def _checkpoint_contract(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> tuple[Path, str, float]:
    if checkpoint_path.name != "best.pt":
        raise ValueError("Pair scorer public evaluation requires a frozen best.pt")
    if checkpoint.get("uses_internal_test") is not False:
        raise ValueError("Pair scorer checkpoint internal_test contract is invalid")
    if checkpoint.get("validation_read") is not False:
        raise ValueError("Pair scorer checkpoint has already read validation")
    if checkpoint.get("pairing_profile") != PAIRING_PROFILE_TRAINING_WIDE:
        raise ValueError("Pair scorer checkpoint was not trained with training_wide")
    expected_pairing = asdict(
        conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
    )
    if checkpoint.get("pairing_config") != expected_pairing:
        raise ValueError("Pair scorer checkpoint pairing config is not frozen training_wide")
    threshold = checkpoint.get("gain_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Pair scorer checkpoint gain_threshold is invalid")
    cache_path = Path(str(checkpoint.get("cache_path", ""))).expanduser().resolve()
    if _contains_internal_test(cache_path) or not cache_path.is_file():
        raise ValueError("Pair scorer checkpoint cache path is missing or forbidden")
    cache_sha = file_sha256(cache_path)
    if checkpoint.get("cache_sha256") != cache_sha:
        raise ValueError("Pair scorer checkpoint cache SHA256 no longer matches")
    return cache_path, cache_sha, float(threshold)


def pair_scorer_validation_gate(
    metrics: Mapping[str, Any],
    *,
    sample_count: int,
    expected_sample_count: int,
    render_variants: set[str],
    identity_audit: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Formal public-validation gate; no value participates in calibration."""

    latency = metrics["matching_latency_ms"]
    checks = {
        "sample_count_exact": sample_count == expected_sample_count,
        "all_render_variants_present": render_variants == EXPECTED_RENDER_VARIANTS,
        "train_validation_molecule_overlap_equal_0": (
            identity_audit["molecule_overlap"] == 0
        ),
        "train_validation_source_overlap_equal_0": (
            identity_audit["source_structure_overlap"] == 0
        ),
        "checkpoint_train_cal_gate_is_GO": (
            checkpoint.get("train_cal_gate", {}).get("status") == "GO"
        ),
        "selected_cross_target_equal_0": metrics["selected_cross_target_pairs"] == 0,
        "parallel_2_groups_present": metrics["parallel_2_group_count"] > 0,
        "parallel_2_preservation_equal_1": (
            metrics["parallel_2_distinct_preservation"] == 1.0
        ),
        "selected_precision_at_least_0_999": (
            (metrics["selected_pair_precision"] or 0.0) >= 0.999
        ),
        "multiplicity_2_recall_at_least_0_995": (
            (metrics["multiplicity_2_recall"] or 0.0) >= 0.995
        ),
        "pairable_samples_present": metrics["pairable_sample_count"] > 0,
        "pairable_sample_exact_at_least_0_95": (
            (metrics["pairable_sample_exact"] or 0.0) >= 0.95
        ),
        "support_loss_at_most_0_1pp": (
            (metrics["ordinary_support_loss_pp"] or 0.0) <= 0.1
        ),
        "all_matching_optimal": (
            metrics["matching_optimal_sample_count"] == sample_count
        ),
        "permutation_determinism_equal_1": (
            metrics["permutation_deterministic"] is True
        ),
        "matching_p95_below_20ms": (
            (latency["p95"] if latency["p95"] is not None else math.inf) < 20.0
        ),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": {
            "expected_samples": expected_sample_count,
            "expected_render_variants": sorted(EXPECTED_RENDER_VARIANTS),
            "selected_cross_target_pairs": 0,
            "parallel_2_distinct_preservation": 1.0,
            "selected_pair_precision": 0.999,
            "multiplicity_2_recall": 0.995,
            "pairable_sample_exact": 0.95,
            "maximum_support_loss_pp": 0.1,
            "maximum_matching_p95_ms": 20.0,
        },
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    validation_paths = _expand_validation_shards(args.validation_shards)
    output = _output_path(args.output)
    if output.exists():
        raise FileExistsError(f"Pair scorer validation output already exists: {output}")

    device = torch.device(args.device)
    model, checkpoint = load_pair_scorer_checkpoint(checkpoint_path, device=device)
    model.eval()
    cache_path, cache_sha, gain_threshold = _checkpoint_contract(
        checkpoint_path,
        checkpoint,
    )
    cache = load_evidence_carrier_pair_cache(
        cache_path,
        validate_source_hashes=True,
    )
    if cache.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Checkpoint train cache schema changed")
    if checkpoint.get("cache_schema_version") != cache.get("cache_schema_version"):
        raise ValueError("Checkpoint and train cache schema versions differ")
    train_identities = _train_cache_identities(cache)

    pairing_config = conditional_pair_pairing_config(
        PAIRING_PROFILE_TRAINING_WIDE
    )
    target_config = EvidenceCarrierExtentTargetConfig()
    dataset = ConditionalOrdinaryPairWebDataset(
        validation_paths,
        split="validation",
        render_variant=None,
        validate_hashes=True,
        limit=args.limit,
    )
    identities: list[ConditionalPairSampleIdentity] = []
    evaluations = []
    seen_keys: set[str] = set()
    target_build_count = 0
    for sample in dataset:
        identity = sample.get("identity")
        if not isinstance(identity, ConditionalPairSampleIdentity):
            raise ValueError("Validation dataset returned an unvalidated identity")
        if identity.sample_key in seen_keys:
            raise ValueError("Validation dataset returned a duplicate sample key")
        seen_keys.add(identity.sample_key)
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            target_config=target_config,
            pairing_config=pairing_config,
        )
        target_build_count += 1
        pair_features = target["pair_features"].unsqueeze(0).to(device)
        pair_mask = torch.ones(
            pair_features.shape[:2], dtype=torch.bool, device=device
        )
        pair_logits = model(pair_features, pair_mask)["pair_gains"][0].detach().cpu()
        evaluations.append(
            score_evidence_carrier_pair_record(
                target,
                pair_logits,
                gain_threshold,
                verify_permutation=True,
            )
        )
        identities.append(identity)
    if len(identities) != args.limit or target_build_count != len(identities):
        raise ValueError(
            f"Expected exactly {args.limit} one-pass validation samples, got "
            f"{len(identities)}"
        )

    identity_audit = validate_molecule_disjoint_splits(
        train_identities,
        identities,
        left_name="train_cache",
        right_name="validation",
    )
    metrics = aggregate_evidence_carrier_pair_evaluations(evaluations)
    render_variants = {identity.render_variant for identity in identities}
    gate = pair_scorer_validation_gate(
        metrics,
        sample_count=len(identities),
        expected_sample_count=args.limit,
        render_variants=render_variants,
        identity_audit=identity_audit,
        checkpoint=checkpoint,
    )
    checkpoint_sha = file_sha256(checkpoint_path)
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": gate["status"],
        "protocol": {
            "evaluation_only": True,
            "validation_passes": 1,
            "target_builds_per_sample": 1,
            "target_build_count": target_build_count,
            "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
            "render_mode": "all_variants",
            "gain_threshold_source": "frozen_checkpoint",
            "threshold_recalibrated": False,
            "checkpoint_modified": False,
            "internal_test_read": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_type": checkpoint.get("model_type"),
            "parameter_count": checkpoint.get("parameter_count"),
            "best_epoch": checkpoint.get("best_epoch"),
            "gain_threshold": gain_threshold,
            "cache_path": str(cache_path),
            "cache_sha256": cache_sha,
            "declared_cache_sha256": checkpoint.get("cache_sha256"),
            "train_cal_gate": checkpoint.get("train_cal_gate"),
        },
        "validation_shards": _shard_descriptors(validation_paths),
        "identity": {
            "validation_sample_count": len(identities),
            "validation_molecule_count": len(
                {identity.molecule_id for identity in identities}
            ),
            "validation_source_structure_count": len(
                {identity.source_structure_id for identity in identities}
            ),
            "render_variants": sorted(render_variants),
            "samples": [identity.as_dict() for identity in identities],
            "train_cache_sample_count": len(train_identities),
            "train_validation_zero_overlap": identity_audit,
        },
        "metrics": metrics,
        "go_gate": gate,
        "sample_evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_VALIDATION_LIMIT)
    parser.add_argument("--device", default="cpu")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if isinstance(args.limit, bool) or not isinstance(args.limit, int) or args.limit < 1:
        raise ValueError("Pair scorer validation limit must be a positive integer")
    forbidden = [args.checkpoint, args.output, *args.validation_shards]
    if any(_contains_internal_test(value) for value in forbidden):
        raise ValueError("Pair scorer evaluation paths cannot reference internal_test")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=False)
    if _contains_internal_test(checkpoint) or not checkpoint.is_file():
        raise ValueError("Pair scorer checkpoint is missing or references internal_test")
    output = _output_path(args.output)
    if _contains_internal_test(output):
        raise ValueError("Pair scorer validation output cannot reference internal_test")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate(args)
    print(json.dumps(_plain_json(payload), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_VALIDATION_LIMIT",
    "EVALUATION_SCHEMA_VERSION",
    "build_parser",
    "evaluate",
    "pair_scorer_validation_gate",
]
