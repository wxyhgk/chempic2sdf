"""Deterministic, target-independent admission for text-offset residuals.

This is intentionally a CPU-auditable policy, not a trainable scorer.  It
accepts frozen per-candidate geometry exported by a replay and preserves the
baseline prefix.  Ground-truth labels are accepted only by the audit helper;
they never participate in selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class OffsetGeometryPolicy:
    """One fixed gate; values must be registered before a replay."""

    # Frozen replay diagnostics show p(TEXT) is heavily under-confident on
    # dense residual peaks (median ~3e-4); .20 is a single registered probe,
    # not a threshold sweep.  Precision/recovery gates decide whether it is
    # admissible.
    role_text_min: float = 0.20
    peak_prominence_min: float = 0.02
    ink_fraction_min: float = 0.05
    extent_min_px: float = 2.0
    extent_max_px: float = 96.0
    offset_norm_min: float = 0.20
    offset_norm_max: float = 1.75
    duplicate_radius_px: float = 6.0
    maximum_extra_per_image: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.role_text_min <= 1.0:
            raise ValueError("role_text_min must be in [0, 1]")
        if self.peak_prominence_min < 0 or self.ink_fraction_min < 0:
            raise ValueError("prominence and ink thresholds must be nonnegative")
        if not 0 < self.extent_min_px <= self.extent_max_px:
            raise ValueError("invalid extent bounds")
        if not 0 <= self.offset_norm_min <= self.offset_norm_max:
            raise ValueError("invalid offset bounds")
        if self.duplicate_radius_px <= 0 or self.maximum_extra_per_image < 0:
            raise ValueError("invalid duplicate radius or capacity")


def _finite(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate field {key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"candidate field {key!r} must be finite")
    return value


def geometry_score(row: Mapping[str, object], policy: OffsetGeometryPolicy) -> float:
    """Return a frozen geometry score and fail closed on malformed features."""

    role = _finite(row, "role_text_prob")
    prominence = _finite(row, "peak_prominence")
    ink = _finite(row, "ink_fraction")
    extent_x = _finite(row, "text_extent_x_px")
    extent_y = _finite(row, "text_extent_y_px")
    offset_norm = _finite(row, "offset_norm")
    raw_score = _finite(row, "heat_score")
    if not all(0.0 <= value <= 1.0 for value in (role, ink)):
        raise ValueError("role_text_prob and ink_fraction must be in [0, 1]")
    accepted = (
        role >= policy.role_text_min
        and prominence >= policy.peak_prominence_min
        and ink >= policy.ink_fraction_min
        and policy.extent_min_px <= extent_x <= policy.extent_max_px
        and policy.extent_min_px <= extent_y <= policy.extent_max_px
        and policy.offset_norm_min <= offset_norm <= policy.offset_norm_max
    )
    if not accepted:
        return float("-inf")
    # The score is only used to choose among already-admissible residuals; it
    # cannot replace a protected baseline candidate.
    return float(role + 0.25 * min(prominence, 1.0) + 0.25 * min(ink, 1.0)
                 + 0.01 * max(raw_score, 0.0))


def select_offset_residuals(
    rows: Iterable[Mapping[str, object]],
    *,
    policy: OffsetGeometryPolicy = OffsetGeometryPolicy(),
) -> list[dict]:
    """Select residual rows using only frozen fields, preserving protection.

    Required target-independent fields are checked by :func:`geometry_score`.
    ``protected`` rows are never scored or removed.  ``sample_key`` groups the
    capacity, while ``anchor_x_px``/``anchor_y_px`` provide duplicate
    suppression.  Output is stable under row ordering.
    """

    grouped: dict[str, list[dict]] = {}
    for source in rows:
        row = dict(source)
        key = str(row.get("sample_key", ""))
        if not key:
            raise ValueError("candidate sample_key is required")
        row["protected"] = bool(row.get("protected", False))
        if row["protected"]:
            row["admission_score"] = float("inf")
            grouped.setdefault(key, []).append(row)
            continue
        score = geometry_score(row, policy)
        if math.isfinite(score):
            row["admission_score"] = score
            grouped.setdefault(key, []).append(row)

    selected: list[dict] = []
    for key in sorted(grouped):
        protected = [r for r in grouped[key] if r["protected"]]
        residual = [r for r in grouped[key] if not r["protected"]]
        residual.sort(key=lambda r: (-float(r["admission_score"]), str(r.get("candidate_id", ""))))
        chosen: list[dict] = []
        for row in residual:
            x = _finite(row, "anchor_x_px")
            y = _finite(row, "anchor_y_px")
            if any((x - _finite(old, "anchor_x_px")) ** 2
                   + (y - _finite(old, "anchor_y_px")) ** 2
                   <= policy.duplicate_radius_px ** 2 for old in chosen):
                continue
            chosen.append(row)
            if len(chosen) >= policy.maximum_extra_per_image:
                break
        selected.extend(protected)
        selected.extend(chosen)
    return selected


def audit_offset_gate(rows: Iterable[Mapping[str, object]], selected: Iterable[Mapping[str, object]]) -> dict:
    """Compute post-hoc metrics; labels never affect :func:`select_offset_residuals`."""

    all_rows = [dict(row) for row in rows]
    chosen = list(selected)
    chosen_ids = {str(row.get("candidate_id")) for row in chosen}
    labels = {str(row.get("candidate_id")): str(row.get("label", "unknown")) for row in all_rows}
    if len(chosen_ids) != len(chosen):
        raise ValueError("candidate_id must be unique in selected output")
    allowed = {"protected", "strict_positive", "hard_negative", "near_unknown", "selected_unmatched", "unknown"}
    if any(label not in allowed for label in labels.values()):
        raise ValueError("unknown audit label")
    selected_labels = [labels.get(str(row.get("candidate_id")), "unknown") for row in chosen]
    positive = {"protected", "strict_positive"}
    known = positive | {"hard_negative"}
    return {
        "candidates": len(all_rows), "selected": len(chosen),
        "protected_total": sum(label == "protected" for label in labels.values()),
        "protected_retained": sum(label == "protected" for label in selected_labels),
        "protected_retention": sum(label == "protected" for label in selected_labels) / max(sum(label == "protected" for label in labels.values()), 1),
        "strict_positive_total": sum(label == "strict_positive" for label in labels.values()),
        "strict_positive_selected": sum(label == "strict_positive" for label in selected_labels),
        "strict_precision_conservative": sum(label in positive for label in selected_labels) / max(len(chosen), 1),
        "known_precision": sum(label in positive for label in selected_labels) / max(sum(label in known for label in selected_labels), 1),
        "hard_negative_survival": sum(label == "hard_negative" for label in selected_labels) / max(sum(label == "hard_negative" for label in labels.values()), 1),
        "unknown_selected": sum(label not in known for label in selected_labels),
    }


__all__ = ["OffsetGeometryPolicy", "audit_offset_gate", "geometry_score", "select_offset_residuals"]
