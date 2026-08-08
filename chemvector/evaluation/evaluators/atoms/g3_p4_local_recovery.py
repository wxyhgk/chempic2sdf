"""Target-side evaluation and threshold calibration for P4 local recovery."""

from __future__ import annotations

import math

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p4e6_missing_peak_sources import (
    maximum_cardinality_assignments,
)


MATCH_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0
MAXIMUM_ROUTE_ACTIONS = 16


def _matched(predicted: torch.Tensor, expected: torch.Tensor) -> int:
    return sum(
        value is not None
        for value in maximum_cardinality_assignments(
            predicted, expected, radius_pixels=MATCH_RADIUS_PIXELS
        )
    )


def _stable_top_actions(logits: torch.Tensor, threshold: float) -> list[int]:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("P4 recovery logits must be one-dimensional floating point")
    if not bool(torch.isfinite(logits).all()) or not math.isfinite(float(threshold)):
        raise ValueError("P4 recovery logits and threshold must be finite")
    eligible = torch.where(logits.double() >= float(threshold))[0]
    if not len(eligible):
        return []
    order = torch.argsort(logits[eligible], descending=True, stable=True)
    return eligible[order[:MAXIMUM_ROUTE_ACTIONS]].tolist()


def sample_recovery_metrics(record: dict, selected: list[int]) -> dict:
    base = record["base_anchors"].float()
    expected = record["target_anchors"].float()
    offsets = record["offsets"].float()
    source_ids = record["source_ids"].long()
    if base.ndim != 2 or base.shape[-1] != 2 or len(base) != len(offsets):
        raise ValueError("P4 recovery base anchors or offsets have invalid shape")
    if expected.ndim != 2 or expected.shape[-1] != 2:
        raise ValueError("P4 recovery targets have invalid shape")
    if source_ids.shape != (len(base),):
        raise ValueError("P4 recovery source IDs differ from base anchors")
    if len(selected) != len(set(selected)) or any(
        index < 0 or index >= len(base) for index in selected
    ):
        raise ValueError("P4 recovery selected indices are invalid")
    additions = (
        base[selected] + offsets[selected] if selected else base.new_empty((0, 2))
    ).clamp(0.0, 1.0)
    baseline_matched = _matched(base, expected)
    augmented_matched = _matched(torch.cat((base, additions)), expected)
    gain = augmented_matched - baseline_matched
    if gain < 0 or gain > len(selected):
        raise RuntimeError("P4 additive recovery violated monotonic matching")
    oracle_by_source = {
        int(item["base_source_id"]): torch.tensor(item["target_anchor"]).float()
        for item in record.get("oracle_actions", [])
    }
    recovered_oracle = 0
    for row, index in enumerate(selected):
        target = oracle_by_source.get(int(source_ids[index]))
        if target is None:
            continue
        distance = float(torch.linalg.vector_norm((additions[row] - target) * IMAGE_SIZE))
        recovered_oracle += int(distance <= MATCH_RADIUS_PIXELS)
    target_count = len(expected)
    return {
        "actions": len(selected),
        "baseline_matched": baseline_matched,
        "augmented_matched": augmented_matched,
        "matched_gain": gain,
        "useless_actions": len(selected) - gain,
        "baseline_exact": int(baseline_matched == target_count),
        "augmented_exact": int(augmented_matched == target_count),
        "exact_gain": int(augmented_matched == target_count)
        - int(baseline_matched == target_count),
        "oracle_actions": len(oracle_by_source),
        "recovered_oracle": recovered_oracle,
    }


def _empty_totals() -> dict:
    return {
        "images": 0,
        "actions": 0,
        "baseline_matched": 0,
        "augmented_matched": 0,
        "matched_gain": 0,
        "useless_actions": 0,
        "baseline_exact": 0,
        "augmented_exact": 0,
        "exact_gain": 0,
        "oracle_actions": 0,
        "recovered_oracle": 0,
    }


def _merge(totals: dict, current: dict, sign: int = 1) -> None:
    for name in totals:
        if name == "images":
            continue
        totals[name] += sign * int(current[name])


def evaluate_threshold(records: list[dict], threshold: float) -> dict:
    totals = _empty_totals()
    totals["images"] = len(records)
    by_variant: dict[str, dict] = {}
    for record in records:
        selected = _stable_top_actions(record["logits"], threshold)
        current = sample_recovery_metrics(record, selected)
        _merge(totals, current)
        variant = str(record.get("variant", "unknown"))
        bucket = by_variant.setdefault(variant, _empty_totals())
        bucket["images"] += 1
        _merge(bucket, current)
    totals["zero_useless"] = totals["useless_actions"] == 0
    totals["by_variant"] = dict(sorted(by_variant.items()))
    totals["threshold"] = float(threshold)
    return totals


def calibrate_zero_useless_threshold(records: list[dict]) -> tuple[float, dict]:
    """Incrementally scan only actions that can enter each image's top-16 set."""

    if not records:
        raise ValueError("P4 threshold calibration requires records")
    states: list[dict] = []
    events: list[tuple[float, int, int]] = []
    totals = _empty_totals()
    totals["images"] = len(records)
    for record_index, record in enumerate(records):
        logits = record["logits"].float()
        if logits.ndim != 1 or not len(logits) or not bool(torch.isfinite(logits).all()):
            raise ValueError("P4 calibration logits are invalid")
        order = torch.argsort(logits, descending=True, stable=True)[
            :MAXIMUM_ROUTE_ACTIONS
        ]
        events.extend(
            (float(logits[index]), record_index, int(index)) for index in order
        )
        baseline = sample_recovery_metrics(record, [])
        states.append({"selected": [], "metrics": baseline})
        _merge(totals, baseline)
    events.sort(key=lambda item: (-item[0], item[1], item[2]))
    maximum = events[0][0]
    candidates = [
        {
            **totals,
            "threshold": math.nextafter(maximum, math.inf),
            "zero_useless": True,
        }
    ]
    cursor = 0
    while cursor < len(events):
        score = events[cursor][0]
        affected = set()
        while cursor < len(events) and events[cursor][0] == score:
            _, record_index, candidate_index = events[cursor]
            states[record_index]["selected"].append(candidate_index)
            affected.add(record_index)
            cursor += 1
        for record_index in sorted(affected):
            previous = states[record_index]["metrics"]
            _merge(totals, previous, sign=-1)
            current = sample_recovery_metrics(
                records[record_index], states[record_index]["selected"]
            )
            states[record_index]["metrics"] = current
            _merge(totals, current)
        candidates.append(
            {
                **totals,
                "threshold": score,
                "zero_useless": totals["useless_actions"] == 0,
            }
        )
    valid = [item for item in candidates if item["zero_useless"]]
    if not valid:
        raise RuntimeError("P4 threshold calibration lost the no-action state")
    winner = max(
        valid,
        key=lambda item: (
            item["matched_gain"],
            -item["actions"],
            item["threshold"],
        ),
    )
    threshold = float(winner["threshold"])
    repeated = evaluate_threshold(records, threshold)
    for name in (
        "actions",
        "matched_gain",
        "useless_actions",
        "baseline_exact",
        "augmented_exact",
        "exact_gain",
        "oracle_actions",
        "recovered_oracle",
    ):
        if repeated[name] != winner[name]:
            raise RuntimeError(f"P4 incremental calibration differs for {name}")
    audit = {
        "threshold": threshold,
        "scanned_states": len(candidates),
        "valid_zero_useless_states": len(valid),
        "tie_break": [
            "zero_useless",
            "maximum_matched_gain",
            "minimum_action_count",
            "highest_threshold",
        ],
        "metrics": repeated,
    }
    return threshold, audit


__all__ = [
    "calibrate_zero_useless_threshold",
    "evaluate_threshold",
    "sample_recovery_metrics",
]
