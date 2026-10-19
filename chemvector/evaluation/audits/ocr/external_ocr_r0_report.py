"""Candidate-set metrics and frozen promotion gate for external OCR R0."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import platform
import sys
from typing import Any

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest
from chemvector.external_ocr_r0_recognizers import validate_arm_artifact
from chemvector.external_ocr_r0_truth import load_truth_manifest


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "rate": count / total if total else None}


def _candidate_texts(row: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(str(item["text"]) for item in row.get("candidates", []))
    )


def _bucket_metrics(
    items: list[dict[str, Any]], candidate_key: str
) -> dict[str, Any]:
    correct = sum(item["truth"] in item[candidate_key] for item in items)
    result = _rate(correct, len(items))
    result["adequate"] = len(items) >= contract.GROUP_MINIMUM
    return result


def summarize_records(
    crops: list[dict[str, Any]],
    truth_population: list[dict[str, Any]],
    arms: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crop_keys = [row["object_key"] for row in crops]
    arm_maps = {}
    for arm in contract.ARM_NAMES:
        rows = arms[arm]
        if [row["object_key"] for row in rows] != crop_keys:
            raise ValueError(f"Arm keys or ordering differ from crop manifest: {arm}")
        if any(row.get("arm") != arm for row in rows):
            raise ValueError(f"Arm identity differs inside records: {arm}")
        for crop, row in zip(crops, rows):
            if row.get("crop_sha256") != crop["crop"]["sha256"]:
                raise ValueError(f"Arm crop SHA256 differs: {arm}/{crop['object_key']}")
        arm_maps[arm] = {row["object_key"]: row for row in rows}

    truth_by_object = {}
    for row in truth_population:
        object_key = row.get("object_key")
        if object_key is None:
            continue
        if object_key in truth_by_object:
            raise ValueError(
                f"Multiple GT TextObjects map to one predicted crop: {object_key}"
            )
        truth_by_object[object_key] = row
    joined = []
    for crop in crops:
        key = crop["object_key"]
        truth_row = truth_by_object.get(key)
        arm_candidates = {
            arm: _candidate_texts(arm_maps[arm][key]) for arm in contract.ARM_NAMES
        }
        external = list(
            dict.fromkeys(
                arm_candidates["ppocrv5_mobile"]
                + arm_candidates["ppocrv5_server"]
            )
        )
        incumbent = list(
            dict.fromkeys(
                arm_candidates["ctc_v0109_greedy"]
                + arm_candidates["ctc_v0109_top8"]
            )
        )
        joined.append(
            {
                **crop,
                "truth": truth_row["truth"] if truth_row is not None else None,
                "group": truth_row["group"] if truth_row is not None else None,
                "arm_candidates": arm_candidates,
                "incumbent_union": incumbent,
                "external_union": external,
                "combined_union": list(dict.fromkeys(incumbent + external)),
            }
        )

    positives = [row for row in joined if row.get("truth") is not None]
    negatives = [row for row in joined if row.get("truth") is None]
    groups = {
        name: [row for row in positives if row["group"] == name]
        for name in contract.GROUP_NAMES
    }
    arm_metrics = _summarize_arms(joined, positives, negatives, groups)
    union_metrics = _summarize_unions(joined, positives, negatives, groups)

    for row in joined:
        row["incumbent"] = row["incumbent_union"]
        row["combined"] = row["combined_union"]
    paired_overall = _paired(positives)
    paired_groups = {name: _paired(items) for name, items in groups.items()}
    if paired_overall["baseline_only"]:
        raise AssertionError("Candidate union cannot lose a baseline-correct string")
    total = len(positives)
    incremental = paired_overall["union_only"] / total if total else 0.0
    overall_delta_pp = incremental * 100.0
    group_delta_pp = {
        name: (
            values["union_only"] / values["total"] * 100.0
            if values["total"]
            else None
        )
        for name, values in paired_groups.items()
    }
    adequate = all(
        len(groups[name]) >= contract.GROUP_MINIMUM
        for name in contract.REQUIRED_NATURAL_GROUPS
    )
    ability = overall_delta_pp >= contract.OVERALL_DELTA_PP or any(
        len(groups[name]) >= contract.GROUP_MINIMUM
        and group_delta_pp[name] is not None
        and group_delta_pp[name] >= contract.HARD_GROUP_DELTA_PP
        for name in contract.HARD_GROUPS
    )
    decision = (
        "INCONCLUSIVE"
        if not adequate
        else (
            "CANDIDATE-SOURCE-GO"
            if ability and incremental >= contract.INCREMENTAL_COVERAGE
            else "NO-GO"
        )
    )
    population_objects = len(truth_population)
    matched_population = sum(
        row.get("object_key") is not None for row in truth_population
    )
    greedy_missing = sum(
        any(
            text not in row["arm_candidates"]["ctc_v0109_top8"]
            for text in row["arm_candidates"]["ctc_v0109_greedy"]
        )
        for row in joined
    )
    report = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": decision,
        "protocol": _protocol(),
        "population": {
            "gt_text_objects": population_objects,
            "matched_gt_text_objects": matched_population,
            "text_detection_recall": (
                matched_population / population_objects
                if population_objects
                else None
            ),
            "scored_positive_crops": len(positives),
            "negative_crops": len(negatives),
            "groups": {
                name: {
                    "objects": len(items),
                    "adequate": len(items) >= contract.GROUP_MINIMUM,
                }
                for name, items in groups.items()
            },
            "required_adequate_groups": list(contract.REQUIRED_NATURAL_GROUPS),
        },
        "arms": arm_metrics,
        "union": union_metrics,
        "paired": {
            "vs_incumbent_greedy_plus_top8": {
                **paired_overall,
                "groups": paired_groups,
            }
        },
        "ctc_search_diagnostics": {
            "greedy_missing_from_top8": greedy_missing,
            "objects": len(joined),
        },
        "gates": {
            "overall_delta_pp": overall_delta_pp,
            "group_delta_pp": group_delta_pp,
            "incremental_correct_count": paired_overall["union_only"],
            "incremental_coverage": incremental,
            "requirements": {
                "overall_delta_pp": contract.OVERALL_DELTA_PP,
                "modifier_or_open_string_delta_pp": contract.HARD_GROUP_DELTA_PP,
                "incremental_coverage": contract.INCREMENTAL_COVERAGE,
                "group_minimum": contract.GROUP_MINIMUM,
                "required_natural_groups": list(contract.REQUIRED_NATURAL_GROUPS),
            },
            "adequate_groups": adequate,
            "candidate_ability_passed": ability,
            "passed": decision == "CANDIDATE-SOURCE-GO",
            "decision": decision,
        },
    }
    helper_keys = (
        *(f"arm:{arm}" for arm in contract.ARM_NAMES),
        "incumbent",
        "external",
        "combined",
    )
    for row in joined:
        for key in helper_keys:
            row.pop(key, None)
    return report, joined


def _summarize_arms(joined, positives, negatives, groups) -> dict[str, Any]:
    metrics = {}
    for arm in contract.ARM_NAMES:
        key = f"arm:{arm}"
        for row in joined:
            row[key] = row["arm_candidates"][arm]
        counts = [len(row[key]) for row in joined]
        contamination = sum(
            any(
                ("-" in text and "-" not in row["truth"])
                or ("=" in text and "=" not in row["truth"])
                for text in row[key]
            )
            for row in positives
        )
        simple = groups["simple_element"]
        metrics[arm] = {
            "exact": _bucket_metrics(positives, key),
            "groups": {
                name: _bucket_metrics(items, key)
                for name, items in groups.items()
            },
            "negative_nonempty": _rate(
                sum(any(text != "" for text in row[key]) for row in negatives),
                len(negatives),
            ),
            "bond_token_contamination": _rate(contamination, len(positives)),
            "false_modifier_on_simple": _rate(
                sum(
                    any(contract.has_modifier(text) for text in row[key])
                    for row in simple
                ),
                len(simple),
            ),
            "candidate_count": {
                "mean": sum(counts) / len(counts) if counts else 0.0,
                "max": max(counts, default=0),
            },
        }
    return metrics


def _summarize_unions(joined, positives, negatives, groups) -> dict[str, Any]:
    for row in joined:
        row["incumbent"] = row["incumbent_union"]
        row["external"] = row["external_union"]
        row["combined"] = row["combined_union"]
    metrics = {}
    for name, key in (
        ("incumbent_greedy_plus_top8", "incumbent"),
        ("external_only", "external"),
        ("ctc_plus_external", "combined"),
    ):
        counts = [len(row[key]) for row in joined]
        metrics[name] = {
            "exact": _bucket_metrics(positives, key),
            "groups": {
                group: _bucket_metrics(items, key)
                for group, items in groups.items()
            },
            "candidate_count": {
                "mean": sum(counts) / len(counts) if counts else 0.0,
                "max": max(counts, default=0),
            },
            "negative_nonempty": _rate(
                sum(any(text != "" for text in row[key]) for row in negatives),
                len(negatives),
            ),
        }
    return metrics


def _paired(items: list[dict[str, Any]]) -> dict[str, int]:
    values = Counter()
    for row in items:
        base = row["truth"] in row["incumbent"]
        combined = row["truth"] in row["combined"]
        if base and combined:
            values["both_correct"] += 1
        elif base:
            values["baseline_only"] += 1
        elif combined:
            values["union_only"] += 1
        else:
            values["both_wrong"] += 1
    return {
        "both_correct": values["both_correct"],
        "baseline_only": values["baseline_only"],
        "union_only": values["union_only"],
        "both_wrong": values["both_wrong"],
        "total": len(items),
    }


def _protocol() -> dict[str, Any]:
    return {
        "phase": "candidate_coverage_only",
        "bbox_source": "predicted_v0108",
        "ctc_beam": {
            "beam_size": contract.CTC_BEAM_SIZE,
            "top_paths": contract.CTC_TOP_PATHS,
            "maximum_bytes": 24,
            "token_beam": contract.CTC_TOKEN_BEAM,
        },
        "incumbent_candidates": "raw_utf8_dedupe(greedy + prefix_top8)",
        "external_top_paths": 1,
        "group_minimum": contract.GROUP_MINIMUM,
        "required_natural_groups": list(contract.REQUIRED_NATURAL_GROUPS),
        "open_string_evaluation": "separate_open_text_challenge",
        "training_performed": False,
        "routing_performed": False,
        "replacement_performed": False,
        "external_detector_used": False,
    }


def summarize(
    dataset_dir: str | Path,
    truth_manifest_path: str | Path,
    arm_paths: dict[str, str | Path],
    output: str | Path,
) -> dict[str, Any]:
    if set(arm_paths) != set(contract.ARM_NAMES):
        raise ValueError("Summary requires exactly the four frozen arms")
    root, _, crops = load_crop_manifest(dataset_dir)
    _, truth = load_truth_manifest(truth_manifest_path, crops)
    crop_manifest_path = root / "manifest.json"
    arms = {
        name: validate_arm_artifact(name, path, crop_manifest_path)
        for name, path in arm_paths.items()
    }
    report, joined = summarize_records(crops, truth, arms)
    report["provenance"] = {
        "code_revision": contract.git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "finished_utc": contract.utc_now(),
    }
    report["inputs"] = {
        "crop_manifest": {
            "path": str(crop_manifest_path),
            "sha256": contract.sha256_file(crop_manifest_path),
        },
        "truth_manifest": {
            "path": str(truth_manifest_path),
            "sha256": contract.sha256_file(truth_manifest_path),
        },
        "arms": {
            name: {"path": str(path), "sha256": contract.sha256_file(path)}
            for name, path in arm_paths.items()
        },
    }
    output = Path(output)
    joined_path = output.with_name("records.jsonl")
    contract.write_new_jsonl(joined_path, joined)
    report["artifacts"] = {
        "joined_records": {
            "path": str(joined_path),
            "sha256": contract.sha256_file(joined_path),
            "rows": len(joined),
        }
    }
    contract.write_new_json(output, report)
    return report
