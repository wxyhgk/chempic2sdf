"""Inference-safe features and conservative routing for two OCR experts."""

from __future__ import annotations

import math

import torch
from torch import nn

from chemvector.ocr import parse_atom_display_text
from chemvector.pretrain_ctc_ocr import is_modifier_label


COMMON_ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se", "H", "*")
UTILITY_LABELS = ("new_rescue", "new_break", "both_exact", "both_wrong")
OCR_VISUAL_DESCRIPTOR_DIM = 48


def _parsed_features(prefix: str, text: str) -> dict[str, float]:
    parsed = parse_atom_display_text(text)
    values = {
        f"{prefix}_parseable": float(parsed is not None),
        f"{prefix}_formal_charge": float(parsed["formal_charge"] if parsed else 0),
        f"{prefix}_isotope": float(parsed["isotope"] if parsed else 0),
        f"{prefix}_explicit_h": float(parsed["explicit_h_count"] if parsed else 0),
        f"{prefix}_radicals": float(parsed["radical_electrons"] if parsed else 0),
    }
    element = str(parsed["element"]) if parsed else None
    for name in COMMON_ELEMENTS:
        values[f"{prefix}_element_{name}"] = float(element == name)
    return values


def ocr_utility_features(
    old_text: str,
    old_score: float,
    new_text: str,
    new_score: float,
) -> dict[str, float]:
    if not all(math.isfinite(value) for value in (old_score, new_score)):
        raise ValueError("OCR utility scores must be finite")
    old_parsed = parse_atom_display_text(old_text)
    new_parsed = parse_atom_display_text(new_text)
    old_state = tuple(sorted(old_parsed.items())) if old_parsed else None
    new_state = tuple(sorted(new_parsed.items())) if new_parsed else None
    features = {
        "old_score": float(old_score),
        "new_score": float(new_score),
        "score_delta_new_minus_old": float(new_score - old_score),
        "old_bytes": float(len(old_text.encode("utf-8"))),
        "new_bytes": float(len(new_text.encode("utf-8"))),
        "byte_length_delta": float(
            len(new_text.encode("utf-8")) - len(old_text.encode("utf-8"))
        ),
        "texts_equal": float(old_text == new_text),
        "states_equal": float(old_state is not None and old_state == new_state),
        "old_blank": float(not old_text),
        "new_blank": float(not new_text),
        "old_modifier": float(is_modifier_label(old_text)),
        "new_modifier": float(is_modifier_label(new_text)),
    }
    features.update(_parsed_features("old", old_text))
    features.update(_parsed_features("new", new_text))
    return features


def add_ocr_visual_features(
    features: dict[str, float],
    old_descriptor: torch.Tensor,
    new_descriptor: torch.Tensor,
) -> dict[str, float]:
    """Attach paired OCR visual evidence without labels or graph state."""
    old = old_descriptor.detach().float().cpu().flatten()
    new = new_descriptor.detach().float().cpu().flatten()
    if old.numel() != OCR_VISUAL_DESCRIPTOR_DIM or new.numel() != old.numel():
        raise ValueError("OCR visual descriptor shape differs")
    result = dict(features)
    for index, (old_value, new_value) in enumerate(zip(old.tolist(), new.tolist())):
        result[f"old_visual_{index:02d}"] = float(old_value)
        result[f"new_visual_{index:02d}"] = float(new_value)
        result[f"visual_delta_{index:02d}"] = float(new_value - old_value)
    return result


class OCRUtilityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("OCR utility router dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 4), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError("OCR utility router input shape differs")
        return self.network(features).squeeze(-1)


def route_summary(
    probabilities: torch.Tensor,
    labels: list[str],
    domains: list[str],
    threshold: float,
) -> dict:
    selected = probabilities >= threshold

    def summarize(indices: list[int]) -> dict:
        active = [index for index in indices if bool(selected[index])]
        counts = {
            label: sum(labels[index] == label for index in active)
            for label in UTILITY_LABELS
        }
        return {
            "samples": len(indices),
            "selected": len(active),
            **counts,
            "net_text_exact_gain": counts["new_rescue"] - counts["new_break"],
        }

    return {
        "threshold": threshold,
        "overall": summarize(list(range(len(labels)))),
        "domains": {
            domain: summarize(
                [index for index, value in enumerate(domains) if value == domain]
            )
            for domain in sorted(set(domains))
        },
    }


def conservative_frontier(
    probabilities: torch.Tensor,
    labels: list[str],
    domains: list[str],
    break_budgets: tuple[int, ...] = (0, 1, 2, 5, 10),
) -> dict[str, dict | None]:
    thresholds = sorted(
        {1.0, 0.0, *(float(value) for value in probabilities.tolist())},
        reverse=True,
    )
    summaries = [route_summary(probabilities, labels, domains, value) for value in thresholds]
    frontier = {}
    for budget in break_budgets:
        eligible = [
            item for item in summaries if item["overall"]["new_break"] <= budget
        ]
        frontier[str(budget)] = max(
            eligible,
            key=lambda item: (
                item["overall"]["net_text_exact_gain"],
                item["overall"]["new_rescue"],
                item["threshold"],
            ),
            default=None,
        )
    return frontier
