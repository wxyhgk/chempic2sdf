"""Input-injected, weight-tied attention over pre-set candidate tokens."""

import copy

import torch
from torch import nn


class CandidateRefiner(nn.Module):
    def __init__(self, input_dim, width=96, heads=4, untied_depth=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width))
        self.injection = nn.Linear(2 * width, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        # Identical initial values, independent Parameters for the untied control.
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(untied_depth)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, evidence, mask, baseline_logits, loops=1):
        if loops < 1 or (len(self.blocks) != 1 and loops != len(self.blocks)):
            raise ValueError("Invalid loop count for this shared/untied model")
        if mask.dtype != torch.bool or mask.shape != evidence.shape[:2]:
            raise ValueError("Invalid candidate mask")
        # Exclude padding before any projection; protect all-empty rows from NaNs.
        safe = mask.clone()
        safe[~safe.any(dim=1), 0] = True
        embedded = self.encoder(evidence.masked_fill(~mask[..., None], 0))
        state = torch.zeros_like(embedded)
        for index in range(loops):
            state = self.injection(torch.cat((state, embedded), dim=-1))
            block = self.blocks[0] if len(self.blocks) == 1 else self.blocks[index]
            state = self.norm(block(state, src_key_padding_mask=~safe))
            state = state.masked_fill(~mask[..., None], 0)
        logits = baseline_logits + self.output(state).squeeze(-1)
        return logits.masked_fill(~mask, -1e4)
