"""Conservative image-style gate for selecting a structure expert."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


STRUCTURE_HEAD_PREFIXES = (
    "vertex_head.",
    "stroke_head.",
    "relation_head.",
)


def extract_structure_overlay(
    base: dict[str, torch.Tensor],
    expert: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return changed structure tensors and reject every non-structure change."""
    if base.keys() != expert.keys():
        missing = sorted(base.keys() - expert.keys())
        added = sorted(expert.keys() - base.keys())
        raise ValueError(f"Expert keys differ: missing={missing}, added={added}")
    overlay = {}
    for key, base_value in base.items():
        expert_value = expert[key]
        if base_value.shape != expert_value.shape:
            raise ValueError(f"Expert tensor shape differs for {key}")
        if torch.equal(base_value, expert_value):
            continue
        if not key.startswith(STRUCTURE_HEAD_PREFIXES):
            raise ValueError(f"Non-structure tensor changed: {key}")
        overlay[key] = expert_value.detach().cpu().clone()
    if not overlay:
        raise ValueError("Expert has no changed structure tensors")
    return overlay


class StructureStyleGate(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if input_channels < 1 or hidden_dim < 1:
            raise ValueError("Gate dimensions must be positive")
        projection_dim = max(hidden_dim // 2, 8)
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, projection_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(projection_dim * 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.size(1) != self.input_channels:
            raise ValueError("Gate input must be a matching BCHW feature map")
        return self.network(features).squeeze(-1)


class PackedStructureHeads(nn.Module):
    """Expert-only copies of the three small structure heads."""

    def __init__(self, base_model) -> None:
        super().__init__()
        self.vertex_head = deepcopy(base_model.vertex_head)
        self.stroke_head = deepcopy(base_model.stroke_head)
        self.relation_head = deepcopy(base_model.relation_head)


class StructureExpertView:
    """Share the base model while exposing packed expert structure heads."""

    def __init__(self, base_model, heads: PackedStructureHeads) -> None:
        self.config = base_model.config
        self.vision_encoder = base_model.vision_encoder
        self.ocr_head = base_model.ocr_head
        self.vertex_head = heads.vertex_head
        self.stroke_head = heads.stroke_head
        self.relation_head = heads.relation_head
        self._packed_heads = heads

    def eval(self) -> StructureExpertView:
        self._packed_heads.eval()
        return self


def load_packed_structure_expert(
    base_model,
    payload: dict,
    device: torch.device,
) -> StructureExpertView:
    if "expert_structure_tensors" not in payload:
        raise ValueError("Checkpoint does not contain a packed structure expert")
    heads = PackedStructureHeads(base_model)
    state = heads.state_dict()
    overlay = payload["expert_structure_tensors"]
    unknown = sorted(set(overlay) - set(state))
    if unknown:
        raise ValueError(f"Packed expert contains unknown tensors: {unknown}")
    for key, value in overlay.items():
        if state[key].shape != value.shape:
            raise ValueError(f"Packed expert tensor shape differs for {key}")
        state[key] = value
    heads.load_state_dict(state, strict=True)
    heads.to(device).eval()
    return StructureExpertView(base_model, heads).eval()


def load_structure_gate(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[StructureStyleGate, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gate = StructureStyleGate(
        payload["input_channels"],
        hidden_dim=payload["hidden_dim"],
    ).to(device)
    gate.load_state_dict(payload["gate"])
    return gate.eval(), payload


@torch.inference_mode()
def expert_route_probability(
    base_model,
    gate: StructureStyleGate,
    pixels: torch.Tensor,
) -> float:
    features = base_model.vision_encoder.forward_stroke_resolution(pixels)
    return float(gate(features).float().sigmoid()[0])


def summarize_gate_scores(
    expert_scores: torch.Tensor,
    base_scores: torch.Tensor,
    thresholds: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict:
    if expert_scores.ndim != 1 or base_scores.ndim != 1:
        raise ValueError("Gate scores must be one-dimensional")
    result = {
        "expert_samples": expert_scores.numel(),
        "base_samples": base_scores.numel(),
        "expert_mean": float(expert_scores.mean()) if expert_scores.numel() else 0.0,
        "base_mean": float(base_scores.mean()) if base_scores.numel() else 0.0,
        "thresholds": {},
    }
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Gate thresholds must be in [0, 1]")
        expert_selected = int((expert_scores >= threshold).sum())
        base_misrouted = int((base_scores >= threshold).sum())
        result["thresholds"][f"{threshold:.4f}"] = {
            "expert_recall": expert_selected / max(expert_scores.numel(), 1),
            "base_false_route_rate": base_misrouted / max(base_scores.numel(), 1),
            "expert_selected": expert_selected,
            "base_misrouted": base_misrouted,
        }
    return result
