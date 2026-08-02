"""Strong target-free three-arm ranker for frozen G3 top-128 Atom sets."""

from __future__ import annotations

import hashlib
import random

import torch
from torch import nn
from torch.nn import functional as F


G3_STRONG_CANDIDATES = 128
G3_STRONG_CURRENT_DIM = 384
G3_STRONG_STROKE_CHANNELS = 96
G3_STRONG_PATCH_SIZE = 9
G3_STRONG_MODEL_DIM = 256
G3_STRONG_SET_RANKER_PARAMETERS = 2_427_713


def _validate_candidates(candidates: dict[str, torch.Tensor]) -> int:
    required = {
        "candidate_ids",
        "candidate_features",
        "candidate_anchors",
        "base_logits",
    }
    missing = required - candidates.keys()
    if missing:
        raise ValueError(f"G3 strong candidates are missing: {sorted(missing)}")
    ids = candidates["candidate_ids"]
    features = candidates["candidate_features"]
    anchors = candidates["candidate_anchors"]
    scores = candidates["base_logits"]
    if ids.ndim != 2 or ids.size(1) != G3_STRONG_CANDIDATES:
        raise ValueError("G3 strong candidate IDs must have shape [batch,128]")
    batch = ids.size(0)
    if ids.dtype != torch.long or features.shape != (
        batch,
        G3_STRONG_CANDIDATES,
        G3_STRONG_CURRENT_DIM,
    ):
        raise ValueError("G3 strong candidate IDs or current features differ")
    if anchors.shape != (batch, G3_STRONG_CANDIDATES, 2) or scores.shape != (
        batch,
        G3_STRONG_CANDIDATES,
    ):
        raise ValueError("G3 strong candidate anchors or base scores differ")
    devices = {ids.device, features.device, anchors.device, scores.device}
    if len(devices) != 1:
        raise ValueError("G3 strong candidate tensors must share one device")
    if not bool(torch.isfinite(features).all()) or not bool(
        torch.isfinite(anchors).all() & torch.isfinite(scores).all()
    ):
        raise ValueError("G3 strong candidate tensors must be finite")
    if bool(((anchors < 0.0) | (anchors > 1.0)).any()):
        raise ValueError("G3 strong candidate anchors must be in [0,1]")
    return batch


@torch.no_grad()
def extract_predicted_anchor_stroke_patches(
    stroke_features: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sample 9x9 frozen stroke patches around predicted candidate anchors."""

    batch = _validate_candidates(candidates)
    if (
        stroke_features.ndim != 4
        or stroke_features.size(0) != batch
        or stroke_features.size(1) != G3_STRONG_STROKE_CHANNELS
    ):
        raise ValueError("G3 stroke features must have shape [batch,96,h,w]")
    if stroke_features.device != candidates["candidate_anchors"].device:
        raise ValueError("G3 stroke features and candidates must share one device")
    if not bool(torch.isfinite(stroke_features).all()):
        raise ValueError("G3 stroke features must be finite")
    height, width = stroke_features.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("G3 stroke feature map dimensions must be positive")
    offset = G3_STRONG_PATCH_SIZE // 2
    y, x = torch.meshgrid(
        torch.arange(-offset, offset + 1, device=stroke_features.device),
        torch.arange(-offset, offset + 1, device=stroke_features.device),
        indexing="ij",
    )
    local = torch.stack(
        (x.to(torch.float32) / width, y.to(torch.float32) / height), dim=-1
    )
    anchors = candidates["candidate_anchors"].detach().float()
    grid = anchors[:, :, None, None, :] + local[None, None]
    grid = grid.mul(2.0).sub(1.0).reshape(
        batch,
        G3_STRONG_CANDIDATES * G3_STRONG_PATCH_SIZE,
        G3_STRONG_PATCH_SIZE,
        2,
    )
    sampled = F.grid_sample(
        stroke_features.detach(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.reshape(
        batch,
        G3_STRONG_STROKE_CHANNELS,
        G3_STRONG_CANDIDATES,
        G3_STRONG_PATCH_SIZE,
        G3_STRONG_PATCH_SIZE,
    ).permute(0, 2, 1, 3, 4).contiguous()


def cap_zero_candidate_patches(candidate_patches: torch.Tensor) -> torch.Tensor:
    """Return the CAP arm's shape-preserving zero patch tensor."""

    _validate_patches(candidate_patches)
    return torch.zeros_like(candidate_patches)


def align_candidate_patches(candidate_patches: torch.Tensor) -> torch.Tensor:
    """Return an unmodified copy for the ALIGN arm."""

    _validate_patches(candidate_patches)
    return candidate_patches.clone()


def deterministic_sham_permutations(
    sample_keys: list[str] | tuple[str, ...],
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build deterministic Sattolo permutations with no candidate fixed points."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("G3 SHAM seed must be an integer")
    if not sample_keys or any(not isinstance(key, str) or not key for key in sample_keys):
        raise ValueError("G3 SHAM sample keys must be non-empty strings")
    rows = []
    for key in sample_keys:
        digest = hashlib.sha256(f"{seed}\n{key}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        permutation = list(range(G3_STRONG_CANDIDATES))
        for index in range(G3_STRONG_CANDIDATES - 1, 0, -1):
            other = rng.randrange(index)
            permutation[index], permutation[other] = (
                permutation[other],
                permutation[index],
            )
        if any(index == value for index, value in enumerate(permutation)):
            raise RuntimeError("G3 SHAM permutation contains a fixed point")
        rows.append(torch.tensor(permutation, dtype=torch.long))
    return torch.stack(rows).to(device=device)


def sham_candidate_patches(
    candidate_patches: torch.Tensor,
    sample_keys: list[str] | tuple[str, ...],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Reassign aligned patches by a deterministic fixed-point-free permutation."""

    batch = _validate_patches(candidate_patches)
    if len(sample_keys) != batch:
        raise ValueError("G3 SHAM keys and patch batch differ")
    permutation = deterministic_sham_permutations(
        sample_keys, seed=seed, device=candidate_patches.device
    )
    index = permutation[..., None, None, None].expand_as(candidate_patches)
    return {
        "candidate_patches": candidate_patches.gather(1, index),
        "permutation": permutation,
    }


def _validate_patches(candidate_patches: torch.Tensor) -> int:
    if candidate_patches.ndim != 5 or candidate_patches.shape[1:] != (
        G3_STRONG_CANDIDATES,
        G3_STRONG_STROKE_CHANNELS,
        G3_STRONG_PATCH_SIZE,
        G3_STRONG_PATCH_SIZE,
    ):
        raise ValueError("G3 candidate patches must have shape [batch,128,96,9,9]")
    if not bool(torch.isfinite(candidate_patches).all()):
        raise ValueError("G3 candidate patches must be finite")
    return candidate_patches.size(0)


class G3StrongTop128SetRanker(nn.Module):
    """Fuse current, geometry, and stroke-patch evidence before set attention."""

    def __init__(self, config) -> None:
        super().__init__()
        if int(config.d_model) != G3_STRONG_CURRENT_DIM:
            raise ValueError("G3 strong ranker requires d_model=384")
        if int(config.maximum_atoms) != G3_STRONG_CANDIDATES:
            raise ValueError("G3 strong ranker requires maximum_atoms=128")
        if int(config.atom_slots_per_cell) != 3:
            raise ValueError("G3 strong ranker requires three Atom slots")
        self.config = config
        self.current_norm = nn.LayerNorm(
            G3_STRONG_CURRENT_DIM, elementwise_affine=False
        )
        self.current_projection = nn.Linear(
            G3_STRONG_CURRENT_DIM, G3_STRONG_MODEL_DIM
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, G3_STRONG_MODEL_DIM),
        )
        self.patch_projection = nn.Sequential(
            nn.Conv2d(G3_STRONG_STROKE_CHANNELS, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, G3_STRONG_MODEL_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.token_norm = nn.LayerNorm(G3_STRONG_MODEL_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=G3_STRONG_MODEL_DIM,
            nhead=8,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=4, enable_nested_tensor=False
        )
        self.residual_head = nn.Linear(G3_STRONG_MODEL_DIM, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.num_parameters() != G3_STRONG_SET_RANKER_PARAMETERS:
            raise RuntimeError("G3 strong set-ranker parameter total changed")

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        candidates: dict[str, torch.Tensor],
        candidate_patches: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = _validate_candidates(candidates)
        if _validate_patches(candidate_patches) != batch:
            raise ValueError("G3 candidates and patches differ")
        if candidate_patches.device != candidates["candidate_ids"].device:
            raise ValueError("G3 candidates and patches must share one device")
        features = candidates["candidate_features"].detach().float()
        anchors = candidates["candidate_anchors"].detach().float()
        base_scores = candidates["base_logits"].detach().float()
        current = self.current_projection(self.current_norm(features))
        geometry = self.geometry_projection(
            torch.cat((anchors, base_scores[..., None]), dim=-1)
        )
        patches = candidate_patches.detach().float().reshape(
            batch * G3_STRONG_CANDIDATES,
            G3_STRONG_STROKE_CHANNELS,
            G3_STRONG_PATCH_SIZE,
            G3_STRONG_PATCH_SIZE,
        )
        patch = self.patch_projection(patches).flatten(1).reshape(
            batch, G3_STRONG_CANDIDATES, G3_STRONG_MODEL_DIM
        )
        hidden = self.token_norm(F.gelu(current + geometry + patch))
        hidden = self.set_encoder(hidden)
        residual = self.residual_head(hidden).squeeze(-1)
        return {
            **candidates,
            "base_logits": base_scores,
            "ranking_residual": residual,
            "ranking_logits": base_scores + residual,
        }
