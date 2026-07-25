"""Summaries for paired rendering style-sensitivity audits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def font_bucket(font_size: int | None) -> str:
    if font_size is None:
        return "unknown"
    if font_size <= 12:
        return "small"
    if font_size <= 19:
        return "normal"
    if font_size <= 28:
        return "large"
    return "extreme"


def line_width_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 1.8:
        return "thin"
    if value < 2.2:
        return "medium"
    return "thick"


def rotation_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    magnitude = abs(value)
    if magnitude < 4.0:
        return "low"
    if magnitude < 8.0:
        return "medium"
    return "high"


def multiple_bond_offset_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.14:
        return "narrow"
    if value < 0.18:
        return "medium"
    return "wide"


def style_dimensions(row: dict) -> dict[str, str]:
    style = row.get("style", {})
    return {
        "font_family": str(style.get("font_family_id") or "unknown"),
        "font_bucket": font_bucket(style.get("font_size")),
        "line_width_bucket": line_width_bucket(style.get("bond_line_width")),
        "rotation_bucket": rotation_bucket(style.get("rotation_degrees")),
        "multiple_bond_offset_bucket": multiple_bond_offset_bucket(
            style.get("multiple_bond_offset")
        ),
        "layout": str(style.get("layout") or row.get("variant") or "unknown"),
        "kekulize": str(bool(style.get("kekulize"))).lower(),
        "stress_operation": str(style.get("stress_operation") or "none"),
    }


def _totals(rows: Iterable[dict], *, prefix: str) -> dict:
    rows = list(rows)
    presence_name = f"{prefix}presence_exact"
    chemical_name = f"{prefix}chemical_exact"
    samples = len(rows)
    presence = sum(bool(row[presence_name]) for row in rows)
    chemical = sum(bool(row[chemical_name]) for row in rows)
    return {
        "samples": samples,
        "representable": sum(bool(row["representable"]) for row in rows),
        "presence_exact": presence,
        "presence_exact_rate": presence / max(samples, 1),
        "chemical_exact": chemical,
        "chemical_exact_rate": chemical / max(samples, 1),
        "false_positive": sum(int(row[f"{prefix}false_positive"]) for row in rows),
        "false_negative": sum(int(row[f"{prefix}false_negative"]) for row in rows),
        "wrong_type": sum(int(row[f"{prefix}wrong_type"]) for row in rows),
    }


def _paired_summary(
    rows: list[dict], *, prefix: str, expected_renderings: int = 4
) -> dict:
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_molecule[str(row["molecule_id"])].append(row)

    result = {}
    for metric in ("presence_exact", "chemical_exact"):
        field = f"{prefix}{metric}"
        all_exact = none_exact = mixed = 0
        complete_groups = 0
        incomplete_groups = 0
        for molecule_rows in by_molecule.values():
            if len(molecule_rows) != expected_renderings:
                incomplete_groups += 1
                continue
            complete_groups += 1
            values = [bool(row[field]) for row in molecule_rows]
            if all(values):
                all_exact += 1
            elif any(values):
                mixed += 1
            else:
                none_exact += 1
        result[metric] = {
            "molecules": complete_groups,
            "incomplete_molecules_excluded": incomplete_groups,
            "expected_renderings_per_molecule": expected_renderings,
            "all_renderings_exact": all_exact,
            "no_rendering_exact": none_exact,
            "mixed_outcome": mixed,
            "mixed_outcome_rate": mixed / max(complete_groups, 1),
        }
    return result


def summarize_style_sensitivity(rows: list[dict], *, baseline: bool) -> dict:
    prefix = "baseline_" if baseline else ""
    breakdowns = {}
    gaps = {}
    for dimension in (
        "font_family",
        "font_bucket",
        "line_width_bucket",
        "rotation_bucket",
        "multiple_bond_offset_bucket",
        "layout",
        "kekulize",
        "stress_operation",
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[style_dimensions(row)[dimension]].append(row)
        summary = {
            name: _totals(group_rows, prefix=prefix)
            for name, group_rows in sorted(groups.items())
        }
        breakdowns[dimension] = summary
        eligible = [value for value in summary.values() if value["samples"] >= 10]
        gaps[dimension] = {
            "presence_exact_rate_gap": (
                max(value["presence_exact_rate"] for value in eligible)
                - min(value["presence_exact_rate"] for value in eligible)
                if eligible
                else 0.0
            ),
            "chemical_exact_rate_gap": (
                max(value["chemical_exact_rate"] for value in eligible)
                - min(value["chemical_exact_rate"] for value in eligible)
                if eligible
                else 0.0
            ),
        }
    return {
        "overall": _totals(rows, prefix=prefix),
        "paired_renderings": _paired_summary(rows, prefix=prefix),
        "breakdowns": breakdowns,
        "maximum_bucket_gaps": gaps,
    }


__all__ = [
    "font_bucket",
    "line_width_bucket",
    "multiple_bond_offset_bucket",
    "rotation_bucket",
    "style_dimensions",
    "summarize_style_sensitivity",
]
