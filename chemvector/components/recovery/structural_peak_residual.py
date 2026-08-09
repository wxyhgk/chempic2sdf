"""Zero-initialized structural residuals over a frozen peak-ranking prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.recovery.structural_peak_selector import (
    CANDIDATES, CONTEXT_SIZE, IMAGE_SIZE, StructuralPeakSelector,
)


ARMS = ("global", "local17", "multiscale")


def normalized_peak_prior(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or logits.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("P6 residual prior tensors have invalid shapes")
    valid = logits.float().masked_fill(~mask, 0.0); count = mask.sum(dim=1, keepdim=True).float()
    if not bool((count > 0).all()): raise ValueError("P6 residual prior requires candidates")
    mean = valid.sum(dim=1, keepdim=True) / count
    variance = (((valid - mean).square() * mask).sum(dim=1, keepdim=True) / count).sqrt().clamp_min(1e-4)
    return ((valid - mean) / variance).masked_fill(~mask, -torch.inf)


def candidate_centered_patches(values: torch.Tensor, offsets: torch.Tensor, patch_size: int) -> torch.Tensor:
    if values.ndim != 4 or values.shape[1:] != (2, CONTEXT_SIZE, CONTEXT_SIZE): raise ValueError("P6 residual patch source must be [B,2,96,96]")
    if offsets.shape != (len(values), CANDIDATES, 2): raise ValueError("P6 residual patch offsets differ")
    if patch_size not in (17, 41): raise ValueError("P6 residual patch size must be 17 or 41")
    axis = torch.arange(patch_size, device=values.device, dtype=torch.float32) - (patch_size - 1) * 0.5
    y, x = torch.meshgrid(axis, axis, indexing="ij"); local = torch.stack((x, y), dim=-1)
    center_pixels = offsets.float() * IMAGE_SIZE
    grid = (center_pixels[:, :, None, None, :] + local[None, None]) / (CONTEXT_SIZE * 0.5)
    outputs=[]
    for batch_index in range(len(values)):
        outputs.append(F.grid_sample(values[batch_index:batch_index+1].expand(CANDIDATES,-1,-1,-1), grid[batch_index], mode="bilinear", padding_mode="zeros", align_corners=False))
    return torch.stack(outputs)


class _PatchResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.GELU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.output=nn.Linear(32,1); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch,candidates,channels,height,width=patches.shape
        return self.output(self.encoder(patches.reshape(batch*candidates,channels,height,width)).flatten(1)).reshape(batch,candidates)


class StructuralPeakResidual(nn.Module):
    def __init__(self, *, arm: str, cap: float = 2.0) -> None:
        super().__init__()
        if arm not in ARMS: raise ValueError(f"unknown P6 residual arm: {arm}")
        if cap <= 0: raise ValueError("P6 residual cap must be positive")
        self.arm=arm; self.cap=float(cap)
        self.global_branch=StructuralPeakSelector(use_peak_logits=False)
        nn.init.zeros_(self.global_branch.output[-1].weight); nn.init.zeros_(self.global_branch.output[-1].bias)
        self.local17=_PatchResidual() if arm in ("local17","multiscale") else None
        self.local41=_PatchResidual() if arm=="multiscale" else None

    @property
    def parameter_count(self) -> int: return sum(value.numel() for value in self.parameters())

    def forward_details(self,image_crops,score_crops,candidate_offsets,candidate_peak_logits,candidate_mask,source_scores,source_ranks) -> dict[str,torch.Tensor]:
        prior=normalized_peak_prior(candidate_peak_logits,candidate_mask)
        raw=self.global_branch(image_crops,score_crops,candidate_offsets,torch.zeros_like(candidate_peak_logits),candidate_mask,source_scores,source_ranks)
        inputs=torch.cat((image_crops.float(),score_crops.float()),dim=1)
        if self.local17 is not None: raw=raw+self.local17(candidate_centered_patches(inputs,candidate_offsets,17))
        if self.local41 is not None: raw=raw+self.local41(candidate_centered_patches(inputs,candidate_offsets,41))
        residual=self.cap*torch.tanh(raw.float()).masked_fill(~candidate_mask,0.0)
        final=(prior+residual).masked_fill(~candidate_mask,-torch.inf)
        return {"scores":final,"prior":prior,"residual":residual}

    def forward(self,*args,**kwargs) -> torch.Tensor: return self.forward_details(*args,**kwargs)["scores"]


def residual_selection_loss(details:dict[str,torch.Tensor],candidate_offsets:torch.Tensor,candidate_mask:torch.Tensor,target_offsets:torch.Tensor,hard_mask:torch.Tensor) -> dict[str,torch.Tensor]:
    scores,prior,residual=details["scores"],details["prior"],details["residual"]
    if hard_mask.shape!=(len(scores),) or hard_mask.dtype!=torch.bool or not bool(hard_mask.any()) or not bool((~hard_mask).any()): raise ValueError("P6 residual loss requires hard and easy rows")
    distances=torch.linalg.vector_norm((candidate_offsets.float()-target_offsets.float()[:,None])*IMAGE_SIZE,dim=-1)
    compatible=candidate_mask&(distances<=6.0)
    if not bool(compatible[hard_mask].any(dim=1).all()): raise ValueError("P6 residual hard target is unreachable")
    hard_scores=scores[hard_mask]; hard_positive=compatible[hard_mask]
    hard_loss=(torch.logsumexp(hard_scores.float(),dim=1)-torch.logsumexp(hard_scores.float().masked_fill(~hard_positive,-torch.inf),dim=1)).mean()
    easy_scores=scores[~hard_mask]; easy_prior=prior[~hard_mask]; easy_mask=candidate_mask[~hard_mask]
    prior_prob=torch.softmax(easy_prior.float(),dim=1)
    log_difference=(torch.log_softmax(easy_prior.float(),dim=1)-torch.log_softmax(easy_scores.float(),dim=1)).masked_fill(~easy_mask,0.0)
    easy_kl=(prior_prob*log_difference).sum(dim=1).mean()
    winner=easy_prior.argmax(dim=1); winner_score=easy_scores.gather(1,winner[:,None])[:,0]; other=easy_scores.clone(); other.scatter_(1,winner[:,None],-torch.inf); keep_hinge=torch.relu(other.max(dim=1).values-winner_score+0.25).mean()
    residual_l2=residual[candidate_mask].square().mean()
    total=hard_loss+easy_kl+keep_hinge+0.01*residual_l2
    return {"loss":total,"hard_loss":hard_loss,"easy_kl":easy_kl,"keep_hinge":keep_hinge,"residual_l2":residual_l2}


__all__=["ARMS","StructuralPeakResidual","candidate_centered_patches","normalized_peak_prior","residual_selection_loss"]
