#!/usr/bin/env python3
"""One-pass paired B-low versus B+coverage smoke/fresh-dev evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_fresh_dev_data import (
    FreshEvidenceCarrierPackage,
    load_fresh_evidence_carrier_package,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    score_evidence_carrier_pair_record,
)
from chemvector.evidence_carrier_pair_coverage import (
    EvidenceCarrierPairCoverageConfig,
)
from chemvector.evaluate_evidence_carrier_p0_pair_scorer import (
    load_frozen_p0_pair_pipeline,
    score_frozen_p0_pair_target,
)
from chemvector.evaluate_evidence_carrier_pair_coverage import (
    DEVELOPMENT_THRESHOLD,
    aggregate_pair_coverage_evaluations,
    evaluate_pair_coverage_sample,
)
from chemvector.train_evidence_carrier_extent import file_sha256
from chemvector.paths import PROJECT_ROOT


SCHEMA_VERSION = "evidence-carrier-pair-coverage-fresh-paired-0.1"
DEFAULT_COMMITMENT = (
    "Data/sdf/visual_svg_layers_c_v0_1/"
    "smoke_train_32/commitment.smoke.json"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0911-synergistic-pair-coverage-smoke32/paired.json"
)
CHECKPOINT_PATH = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/best.pt"
)
CHECKPOINT_SHA256 = (
    "67ba638de2841089612b428a8cf43b45206cd21f4da9aa7e182fb269160c7234"
)
CALIBRATION_REPORT_PATH = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/calibration.json"
)
CALIBRATION_REPORT_SHA256 = (
    "7a987a423a12b1c8a480be6b90bff7c73b925af31e22a312c8b62e92c134265a"
)
SAFE_CHECKPOINT_THRESHOLD = 2.7005028724670415
COMMITMENT_REGISTRY: dict[str, dict[str, str | None]] = {
    "smoke": {
        "path": (
            "Data/sdf/visual_svg_layers_c_v0_1/"
            "smoke_train_32/commitment.smoke.json"
        ),
        "sha256": (
            "e8c0bf1fbd792b60836ab879ee7a8b2dbee230f0a6a426bf710fee1e0ad6825a"
        ),
    },
    "dev": {
        "path": (
            "Data/sdf/visual_svg_layers_c_v0_1/"
            "fresh_dev/commitment.dev.json"
        ),
        "sha256": (
            "67a2bffd984a95bb925b16126faa04b791a7ab2679abf0e59c92941806920d07"
        ),
    },
}
EXPECTED_IMPLEMENTATION_SHA256 = {
    "evidence_carrier_fresh_dev_data.py": (
        "3891c73fd9ddd97957004bb7faf702faa9b908a1d0ed14d128add5d26ad6511d"
    ),
    "evidence_carrier_conditional_pair_data.py": (
        "8276a7e61f939ed44bc823888656164aa5b4a06bcd91919f4fd4c5ee0f09cae1"
    ),
    "evidence_carrier_pair_evaluation.py": (
        "5f551ae4b3076d53476d4c26f4054864c25da84ba3e4519d9e78fea57c4931de"
    ),
    "evidence_carrier_pair_matching.py": (
        "cb6dfca166b60b182e4aac9d6f94d12114492099a7aa5b2f0a68c1606f9c1487"
    ),
    "evidence_carrier_pair_coverage.py": (
        "bdc8a0f946b639efd0be9840eed6c8821950ada57fff294b35b3b5d7ed87ea2c"
    ),
    "evaluate_evidence_carrier_pair_coverage.py": (
        "6ca864ecc4d2f7c21468bb095150d06256b80c4c0014f8260357704a67976a97"
    ),
    "evaluate_evidence_carrier_p0_pair_scorer.py": (
        "686b21f28b98689dce1b07ad8c045b9d1b79fd9021c95955f5b1637f8c23968b"
    ),
    "evidence_carrier_p0_pair_scorer.py": (
        "c6e963f9251d7d840c32d2be84c016691f59c20ccb65e83009454628a0365e13"
    ),
    "evidence_carrier_pair_scorer.py": (
        "339c9dd6b0701a85e31e70245762dadf3e87c951be9be8d63a2d3ff1578dc3f3"
    ),
}


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _plain_json(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _registered_commitment(commitment_path: str | Path) -> dict[str, str]:
    requested = Path(commitment_path).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    requested = requested.resolve(strict=False)
    for role, record in COMMITMENT_REGISTRY.items():
        registered_path = Path(str(record["path"]))
        if not registered_path.is_absolute():
            registered_path = PROJECT_ROOT / registered_path
        if requested != registered_path.resolve(strict=False):
            continue
        digest = record.get("sha256")
        if not isinstance(digest, str):
            raise ValueError(
                f"Formal {role} commitment has not been pre-registered"
            )
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Registered {role} commitment SHA256 is invalid")
        return {
            "role": role,
            "path": str(registered_path.resolve(strict=False)),
            "sha256": digest,
        }
    raise ValueError("Commitment path is not present in the formal registry")


def _route_configs(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    shared = {
        "development_threshold": DEVELOPMENT_THRESHOLD,
        "verify_permutation": True,
        "target_config": asdict(pipeline["target_config"]),
        "pairing_config": asdict(pipeline["pairing_config"]),
        "pair_model_config": asdict(pipeline["pair_model"].config),
    }
    route_b = {
        **shared,
        "route": "B",
        "selector": "frozen maximum-weight pair matching",
        "coverage": None,
    }
    route_c = {
        **shared,
        "route": "C",
        "selector": "same frozen B matching plus synchronous coverage veto",
        "coverage": asdict(EvidenceCarrierPairCoverageConfig()),
    }
    return {
        "route_b": route_b,
        "route_c": route_c,
        "route_b_config_sha256": _canonical_sha256(route_b),
        "route_c_config_sha256": _canonical_sha256(route_c),
    }


def _tensor_mapping_sha256(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(value):
        item = value[key]
        if not isinstance(item, torch.Tensor):
            continue
        tensor = item.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_low_threshold_commitment(path: Path) -> dict[str, Any]:
    if file_sha256(path) != CALIBRATION_REPORT_SHA256:
        raise ValueError("B calibration report SHA256 differs from v0.9.11")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = payload.get("threshold_trace")
    if not isinstance(trace, list):
        raise ValueError("B calibration report lacks threshold_trace")
    rows = [
        row
        for row in trace
        if math.isclose(
            float(row.get("threshold", math.nan)),
            DEVELOPMENT_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(rows) != 1:
        raise ValueError("Frozen v0.9.11 low threshold is absent or duplicated")
    row = rows[0]
    expected = {
        "selected_cross_target_pairs": 1,
        "multiplicity_2_matched_count": 36399,
        "pairable_sample_exact_count": 707,
    }
    if any(int(row.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("Frozen v0.9.11 threshold trace metrics changed")
    return row


def _implementation_provenance() -> dict[str, dict[str, str]]:
    result = {}
    for filename, expected_sha256 in EXPECTED_IMPLEMENTATION_SHA256.items():
        spec = __import__(
            f"chemvector.{Path(filename).stem}", fromlist=["__file__"]
        )
        path = Path(spec.__file__).resolve()
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Frozen v0.9.11 implementation changed: {filename}")
        result[filename] = {
            "path": str(path),
            "sha256": actual_sha256,
        }
    current = Path(__file__).resolve()
    result[current.name] = {
        "path": str(current),
        "sha256": file_sha256(current),
    }
    return result


def load_committed_v0911_pipeline(*, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(CHECKPOINT_PATH).resolve()
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("B checkpoint SHA256 differs from v0.9.11")
    calibration_path = Path(CALIBRATION_REPORT_PATH).resolve()
    threshold_trace = _validate_low_threshold_commitment(calibration_path)
    implementation = _implementation_provenance()
    pipeline = load_frozen_p0_pair_pipeline(
        checkpoint_path,
        device=device,
        require_train_cal_go=False,
    )
    checkpoint = pipeline["pair_checkpoint"]
    if checkpoint.get("train_cal_gate", {}).get("status") != "NO-GO":
        raise ValueError("Committed B checkpoint train/cal status changed")
    if not math.isclose(
        float(pipeline["pair_threshold"]),
        SAFE_CHECKPOINT_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Committed B safe checkpoint threshold changed")
    pairing_config = conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
    if checkpoint.get("pairing_config") != asdict(pairing_config):
        raise ValueError("Committed B candidate profile changed")
    for name in ("pair_model", "p0_model"):
        model = pipeline[name]
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(f"Frozen Phase A {name} is trainable")
    result = {
        **pipeline,
        "pair_threshold": DEVELOPMENT_THRESHOLD,
        "safe_checkpoint_threshold": SAFE_CHECKPOINT_THRESHOLD,
        "threshold_trace": threshold_trace,
        "pairing_config": pairing_config,
        "implementation": implementation,
        "calibration_report_path": calibration_path,
        "calibration_report_sha256": CALIBRATION_REPORT_SHA256,
    }
    result.update(_route_configs(result))
    return result


def _same_initial_matching(
    baseline: Mapping[str, Any], coverage: Mapping[str, Any]
) -> bool:
    initial = coverage.get("initial_matching")
    return isinstance(initial, Mapping) and (
        initial.get("selected_candidate_indices")
        == baseline.get("selected_candidate_indices")
        and initial.get("selected_anchor_pairs")
        == baseline.get("selected_anchor_pairs")
    )


@torch.inference_mode()
def evaluate_paired_sample(
    pipeline: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    started = perf_counter()
    target = build_conditional_ordinary_pair_target(
        sample["image"],
        sample["primitives"],
        sample["layers"],
        target_config=pipeline["target_config"],
        pairing_config=pipeline["pairing_config"],
    )
    target_latency_ms = 1000.0 * (perf_counter() - started)
    scored_started = perf_counter()
    scored = score_frozen_p0_pair_target(
        pipeline["pair_model"],
        pipeline["p0_model"],
        sample["image"],
        target,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scorer_latency_ms = 1000.0 * (perf_counter() - scored_started)
    record = dict(target)
    record["decoded_segments_pixels"] = scored["decoded_segments_pixels"]
    baseline = score_evidence_carrier_pair_record(
        record,
        scored["pair_scores"],
        DEVELOPMENT_THRESHOLD,
        verify_permutation=True,
    )
    coverage = evaluate_pair_coverage_sample(
        record,
        scored["pair_scores"],
        threshold=DEVELOPMENT_THRESHOLD,
        verify_permutation=True,
    )
    if not _same_initial_matching(baseline, coverage):
        raise RuntimeError("B and C did not share the same initial matching")
    identity = sample["identity"]
    target_sha256 = _tensor_mapping_sha256(target)
    score_sha256 = _tensor_sha256(scored["pair_scores"])
    shared_route_fingerprint = {
        "input_sha256": sample["input_fingerprint"],
        "target_sha256": target_sha256,
        "score_sha256": score_sha256,
    }
    return {
        "identity": identity.as_dict(),
        "input_fingerprint": sample["input_fingerprint"],
        "target_sha256": target_sha256,
        "score_sha256": score_sha256,
        "route_b_fingerprints": dict(shared_route_fingerprint),
        "route_c_fingerprints": dict(shared_route_fingerprint),
        "target_builds": 1,
        "p0_forward_passes": int(scored["p0_forward_passes"]),
        "target_latency_ms": target_latency_ms,
        "p0_pair_scorer_latency_ms": scorer_latency_ms,
        "pair_stage_latency_ms": float(coverage.get("matching_latency_ms", 0.0)),
        "same_initial_matching": True,
        "baseline": baseline,
        "coverage": coverage,
        "delta": {
            "cross_target_pairs": int(coverage["selected_cross_target_pairs"])
            - int(baseline["selected_cross_target_pairs"]),
            "multiplicity_2_matched": int(coverage["multiplicity_2_matched_count"])
            - int(baseline["multiplicity_2_matched_count"]),
            "parallel_2_violations": int(
                coverage["parallel_2_violated_group_count"]
            )
            - int(baseline["parallel_2_violated_group_count"]),
            "ordinary_support_lost": int(
                coverage["ordinary_support_lost_target_count"]
            )
            - int(baseline["ordinary_support_lost_target_count"]),
            "pairable_exact_true_to_false": bool(
                baseline["pairable_sample_eligible"]
                and baseline["pairable_sample_exact"] is True
                and coverage["pairable_sample_exact"] is False
            ),
        },
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else None,
        "p50": float(np.percentile(array, 50)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _paired_route_mismatches(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = ("input_sha256", "target_sha256", "score_sha256")
    counts = {name: 0 for name in names}
    examples: list[dict[str, Any]] = []
    any_mismatch_count = 0
    for item in records:
        route_b = item.get("route_b_fingerprints")
        route_c = item.get("route_c_fingerprints")
        mismatched = []
        if not isinstance(route_b, Mapping) or not isinstance(route_c, Mapping):
            mismatched = list(names)
        else:
            for name in names:
                left = route_b.get(name)
                right = route_c.get(name)
                if not isinstance(left, str) or left != right:
                    mismatched.append(name)
        for name in mismatched:
            counts[name] += 1
        if mismatched:
            any_mismatch_count += 1
            if len(examples) < 8:
                identity = item.get("identity", {})
                examples.append(
                    {
                        "sample_key": (
                            identity.get("sample_key")
                            if isinstance(identity, Mapping)
                            else None
                        ),
                        "fields": mismatched,
                    }
                )
    return {
        "paired_any_mismatch_count": any_mismatch_count,
        "paired_input_mismatch_count": counts["input_sha256"],
        "paired_target_mismatch_count": counts["target_sha256"],
        "paired_score_mismatch_count": counts["score_sha256"],
        "examples": examples,
    }


def _paired_safety_regressions(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_delta_fields = {
        "cross_target_pairs",
        "multiplicity_2_matched",
        "parallel_2_violations",
        "ordinary_support_lost",
        "pairable_exact_true_to_false",
    }
    predicates = {
        "cross_increase": lambda delta: int(delta["cross_target_pairs"]) > 0,
        "m2_decrease": lambda delta: int(delta["multiplicity_2_matched"]) < 0,
        "p2_violation_increase": lambda delta: int(delta["parallel_2_violations"]) > 0,
        "support_loss_increase": lambda delta: int(delta["ordinary_support_lost"]) > 0,
        "pairable_exact_true_to_false": lambda delta: bool(
            delta["pairable_exact_true_to_false"]
        ),
    }
    counts = {name: 0 for name in predicates}
    examples: list[dict[str, Any]] = []
    complete = True
    for item in records:
        delta = item.get("delta")
        if not isinstance(delta, Mapping) or not required_delta_fields.issubset(delta):
            complete = False
            continue
        failed = [name for name, predicate in predicates.items() if predicate(delta)]
        for name in failed:
            counts[name] += 1
        if failed and len(examples) < 8:
            identity = item.get("identity", {})
            examples.append(
                {
                    "sample_key": (
                        identity.get("sample_key")
                        if isinstance(identity, Mapping)
                        else None
                    ),
                    "regressions": failed,
                }
            )
    return {
        "complete": complete,
        "counts": counts,
        "examples": examples,
    }


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def paired_gate(
    package: FreshEvidenceCarrierPackage,
    records: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    coverage_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    values = list(records)
    mismatches = _paired_route_mismatches(values)
    safety_regressions = _paired_safety_regressions(values)
    technical_checks = {
        "sample_count_exact": len(values) == package.sample_count,
        "one_target_build_per_sample": all(
            int(item["target_builds"]) == 1 for item in values
        ),
        "one_p0_forward_per_sample": all(
            int(item["p0_forward_passes"]) == 1 for item in values
        ),
        "same_initial_matching": all(
            bool(item["same_initial_matching"]) for item in values
        ),
        "paired_input_target_score_identical": (
            mismatches["paired_any_mismatch_count"] == 0
        ),
        "paired_delta_complete": safety_regressions["complete"],
        "unique_input_fingerprints": len(
            {str(item["input_fingerprint"]) for item in values}
        )
        == len(values),
        "all_matching_optimal": (
            int(baseline_metrics["matching_optimal_sample_count"]) == len(values)
            and int(coverage_metrics["matching_optimal_sample_count"]) == len(values)
        ),
        "permutation_deterministic": bool(
            baseline_metrics["permutation_deterministic"]
            and coverage_metrics["permutation_deterministic"]
        ),
        "independence_report_verified": (
            package.role == "smoke"
            or bool(getattr(package, "independence_verified", False))
        ),
    }
    regression_counts = safety_regressions["counts"]
    safety_checks = {
        "cross_does_not_increase_in_any_sample": regression_counts[
            "cross_increase"
        ]
        == 0,
        "m2_matched_does_not_decrease_in_any_sample": regression_counts[
            "m2_decrease"
        ]
        == 0,
        "p2_violations_do_not_increase_in_any_sample": regression_counts[
            "p2_violation_increase"
        ]
        == 0,
        "support_loss_does_not_increase_in_any_sample": regression_counts[
            "support_loss_increase"
        ]
        == 0,
        "pairable_exact_never_true_to_false": regression_counts[
            "pairable_exact_true_to_false"
        ]
        == 0,
        "no_vetoed_m2_true_pair": int(
            coverage_metrics["coverage_vetoed_multiplicity_2_pairs"]
        )
        == 0,
    }
    if package.role == "smoke":
        passed = all(technical_checks.values()) and all(safety_checks.values())
        return {
            "status": "SMOKE-PASS" if passed else "SMOKE-FAIL",
            "interpretation": "execution-contract-only",
            "technical_checks": technical_checks,
            "safety_checks": safety_checks,
            "quality_checks": None,
            "mechanism_status": "NOT-EVALUATED",
            "paired_route_mismatches": mismatches,
            "paired_safety_regressions": safety_regressions,
        }

    precision = coverage_metrics["selected_pair_precision"]
    m2_recall = coverage_metrics["multiplicity_2_recall"]
    pairable_exact = coverage_metrics["pairable_sample_exact"]
    p2 = coverage_metrics["parallel_2_distinct_preservation"]
    quality_checks = {
        "precision_at_least_0_999": precision is not None and precision >= 0.999,
        "m2_recall_at_least_0_995": m2_recall is not None and m2_recall >= 0.995,
        "pairable_exact_at_least_0_95": (
            pairable_exact is not None and pairable_exact >= 0.95
        ),
        "p2_preservation_equal_1": p2 == 1.0,
        "support_loss_at_most_0_1pp": float(
            coverage_metrics["ordinary_support_loss_pp"]
        )
        <= 0.1,
        "pair_stage_p95_below_20ms": float(
            coverage_metrics["matching_latency_ms"]["p95"]
        )
        < 20.0,
    }
    baseline_cross = int(baseline_metrics["selected_cross_target_pairs"])
    coverage_cross = int(coverage_metrics["selected_cross_target_pairs"])
    if coverage_cross < baseline_cross:
        mechanism_status = "PASS"
    elif baseline_cross == 0 and coverage_cross == 0:
        mechanism_status = "SAFE-BUT-INCONCLUSIVE"
    else:
        mechanism_status = "FAIL"
    eligible_mechanism_samples = sum(
        int(
            item.get("baseline", {}).get("selected_cross_target_pairs", 0)
            if isinstance(item.get("baseline"), Mapping)
            else 0
        )
        > 0
        for item in values
    )
    improved_mechanism_samples = sum(
        int(item["delta"].get("cross_target_pairs", 0)) < 0
        for item in values
        if isinstance(item.get("delta"), Mapping)
    )
    mechanism_events = {
        "eligible_sample_count": eligible_mechanism_samples,
        "improved_sample_count": improved_mechanism_samples,
        "baseline_cross_pair_count": baseline_cross,
        "coverage_cross_pair_count": coverage_cross,
        "removed_cross_pair_count": baseline_cross - coverage_cross,
        "improved_sample_fraction_95pct_wilson": _wilson_interval(
            improved_mechanism_samples,
            eligible_mechanism_samples,
        ),
    }
    common_pass = (
        all(technical_checks.values())
        and all(safety_checks.values())
        and all(quality_checks.values())
    )
    if common_pass and mechanism_status == "PASS":
        status = "FRESH-DEV-GO"
    elif common_pass and mechanism_status == "SAFE-BUT-INCONCLUSIVE":
        status = "SAFE-BUT-INCONCLUSIVE"
    else:
        status = "FRESH-DEV-NO-GO"
    return {
        "status": status,
        "interpretation": "oracle-routed ordinary-anchor pair stage",
        "technical_checks": technical_checks,
        "safety_checks": safety_checks,
        "quality_checks": quality_checks,
        "mechanism_status": mechanism_status,
        "mechanism_events": mechanism_events,
        "paired_route_mismatches": mismatches,
        "paired_safety_regressions": safety_regressions,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    registration = _registered_commitment(args.commitment)
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Fresh paired output exists: {output}")
    package = load_fresh_evidence_carrier_package(
        registration["path"],
        expected_role=registration["role"],
        expected_commitment_sha256=registration["sha256"],
    )
    device = _device(args.device)
    pipeline = load_committed_v0911_pipeline(device=device)
    records = [
        evaluate_paired_sample(pipeline, sample, device=device)
        for sample in package.iter_samples()
    ]
    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(
        [item["baseline"] for item in records]
    )
    coverage_metrics = aggregate_pair_coverage_evaluations(
        [item["coverage"] for item in records]
    )
    gate = paired_gate(package, records, baseline_metrics, coverage_metrics)
    paired_mismatches = gate["paired_route_mismatches"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "training_steps": 0,
        "parameter_updates": 0,
        "classification": (
            "train-exposed pipeline smoke"
            if package.role == "smoke"
            else "fresh molecule/scaffold/style-disjoint paired dev audit"
        ),
        "independent_generalization": package.independence_verified,
        "role": package.role,
        "protocol": {
            "phase": "A",
            "training_steps": 0,
            "parameter_updates": 0,
            "single_dataset_pass": True,
            "target_builds_per_sample": 1,
            "p0_forwards_per_sample": 1,
            "shared_pair_scores": True,
            "baseline": "B low-threshold matching",
            "treatment": "same B matching plus synchronous coverage veto",
            "threshold": DEVELOPMENT_THRESHOLD,
            "threshold_recalibrated": False,
            "coverage_recalibrated": False,
            "sample_limit_exposed": False,
            "internal_test_read": False,
            "sealed_audit_read": False,
            "registered_commitment": True,
        },
        "input": {
            "commitment_path": str(package.commitment_path),
            "commitment_sha256": package.commitment_sha256,
            "package_input_fingerprint": package.input_fingerprint,
            "fresh_audit_eligible": package.fresh_audit_eligible,
            "samples": package.sample_count,
            "molecules": package.molecule_count,
            "manifest_path": str(package.manifest_path),
            "manifest_sha256": package.manifest_sha256,
            "freeze_path": (
                str(package.freeze_path) if package.freeze_path is not None else None
            ),
            "freeze_sha256": package.freeze_sha256,
            "isolation_path": (
                str(package.isolation_path)
                if package.isolation_path is not None
                else None
            ),
            "isolation_sha256": package.isolation_sha256,
            "independence_verified": package.independence_verified,
            "supporting_artifacts": [
                {
                    "label": artifact.label,
                    "path": str(artifact.path),
                    "sha256": artifact.sha256,
                }
                for artifact in package.supporting_artifacts
            ],
            "shards": [
                {
                    "path": str(shard.path),
                    "sha256": shard.sha256,
                    "samples": shard.sample_count,
                }
                for shard in package.shards
            ],
        },
        "pipeline": {
            "device": str(device),
            "checkpoint_path": str(pipeline["pair_checkpoint_path"]),
            "checkpoint_sha256": pipeline["pair_checkpoint_sha256"],
            "checkpoint_train_cal_status": pipeline["pair_checkpoint"]
            .get("train_cal_gate", {})
            .get("status"),
            "safe_checkpoint_threshold": pipeline["safe_checkpoint_threshold"],
            "development_threshold": DEVELOPMENT_THRESHOLD,
            "calibration_report_path": str(pipeline["calibration_report_path"]),
            "calibration_report_sha256": pipeline[
                "calibration_report_sha256"
            ],
            "p0_checkpoint_path": str(pipeline["p0_checkpoint_path"]),
            "p0_checkpoint_sha256": pipeline["p0_checkpoint_sha256"],
            "route_b_config": pipeline["route_b"],
            "route_c_config": pipeline["route_c"],
            "route_b_config_sha256": pipeline["route_b_config_sha256"],
            "route_c_config_sha256": pipeline["route_c_config_sha256"],
            "implementation": pipeline["implementation"],
        },
        "paired_execution": {
            "sample_count_b": len(records),
            "sample_count_c": len(records),
            "target_build_count": sum(
                int(item["target_builds"]) for item in records
            ),
            "p0_forward_count": sum(
                int(item["p0_forward_passes"]) for item in records
            ),
            **paired_mismatches,
        },
        "baseline_metrics": baseline_metrics,
        "coverage_metrics": coverage_metrics,
        "runtime": {
            "target_build_ms": _latency_summary(
                [float(item["target_latency_ms"]) for item in records]
            ),
            "p0_pair_scorer_ms": _latency_summary(
                [float(item["p0_pair_scorer_latency_ms"]) for item in records]
            ),
            "pair_stage_ms": _latency_summary(
                [float(item["pair_stage_latency_ms"]) for item in records]
            ),
            "pair_stage_scope": (
                "frozen pair matching plus synchronous coverage veto; "
                "not full image inference"
            ),
        },
        "paired_gate": gate,
        "sample_evaluations": records,
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
    parser.add_argument("--commitment", default=DEFAULT_COMMITMENT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "status": report["status"],
                "role": report["role"],
                "baseline_cross": report["baseline_metrics"][
                    "selected_cross_target_pairs"
                ],
                "coverage_cross": report["coverage_metrics"][
                    "selected_cross_target_pairs"
                ],
                "vetoed": report["coverage_metrics"][
                    "coverage_vetoed_pair_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "COMMITMENT_REGISTRY",
    "SCHEMA_VERSION",
    "_registered_commitment",
    "build_parser",
    "evaluate",
    "evaluate_paired_sample",
    "load_committed_v0911_pipeline",
    "main",
    "paired_gate",
]
