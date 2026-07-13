"""A compact vision encoder and autoregressive ChemMol decoder."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ChemVectorConfig:
    vocab_size: int = 2153
    pad_token_id: int = 0
    d_model: int = 512
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    dropout: float = 0.1
    max_target_length: int = 3072
    vision_dims: tuple[int, ...] = (96, 192, 384, 512)
    vision_depths: tuple[int, ...] = (2, 2, 6, 2)
    visual_aux_channels: int = 16

    def __post_init__(self) -> None:
        if len(self.vision_dims) != len(self.vision_depths):
            raise ValueError("vision_dims and vision_depths must have equal length")
        if self.d_model % self.decoder_heads:
            raise ValueError("d_model must be divisible by decoder_heads")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by 4 for 2D position encoding")


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        hidden = channels * 4
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise_in = nn.Conv2d(channels, hidden, kernel_size=1)
        self.pointwise_out = nn.Conv2d(hidden, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.depthwise(inputs)
        hidden = self.norm(hidden)
        hidden = self.pointwise_in(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.pointwise_out(hidden)
        return residual + hidden


class ChemVisionEncoder(nn.Module):
    """Preserve line geometry while reducing a 768px image to visual tokens."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        dims = config.vision_dims
        depths = config.vision_depths
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        stages = []
        for stage_index, (channels, depth) in enumerate(zip(dims, depths)):
            if stage_index:
                stages.append(
                    nn.Sequential(
                        nn.GroupNorm(1, dims[stage_index - 1]),
                        nn.Conv2d(
                            dims[stage_index - 1],
                            channels,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )
            stages.append(
                nn.Sequential(
                    *(ConvNeXtBlock(channels, config.dropout) for _ in range(depth))
                )
            )
        self.stages = nn.ModuleList(stages)
        self.output_norm = nn.LayerNorm(dims[-1])
        self.projection = nn.Linear(dims[-1], config.d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(pixel_values)
        for stage in self.stages:
            hidden = stage(hidden)
        height, width = hidden.shape[-2:]
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = self.output_norm(hidden)
        hidden = self.projection(hidden)
        return hidden + self._position_encoding(
            height,
            width,
            hidden.size(-1),
            hidden.device,
            hidden.dtype,
        ).unsqueeze(0)

    @staticmethod
    def _position_encoding(
        height: int,
        width: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build deterministic 2D sine/cosine positions in flattened image order."""
        quarter = channels // 4
        frequencies = torch.exp(
            torch.arange(quarter, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(quarter - 1, 1))
        )
        y = torch.linspace(0.0, 2.0 * math.pi, height, device=device)
        x = torch.linspace(0.0, 2.0 * math.pi, width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        x_angles = grid_x.reshape(-1, 1) * frequencies
        y_angles = grid_y.reshape(-1, 1) * frequencies
        encoding = torch.cat(
            (x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()),
            dim=1,
        )
        return encoding.to(dtype=dtype)


class ChemVectorModel(nn.Module):
    """Generate ChemMol grammar tokens with cross-attention to image features."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = ChemVisionEncoder(config)
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_target_length, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.visual_aux_head = nn.Linear(config.d_model, config.visual_aux_channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(pixel_values)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = decoder_input_ids.shape
        if sequence_length > self.config.max_target_length:
            raise ValueError(
                f"Target length {sequence_length} exceeds {self.config.max_target_length}"
            )
        positions = torch.arange(sequence_length, device=decoder_input_ids.device)
        hidden = self.token_embedding(decoder_input_ids)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=decoder_input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        padding_mask = None
        if decoder_attention_mask is not None:
            padding_mask = decoder_attention_mask == 0
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            tgt_is_causal=True,
        )
        return self.output_projection(decoded)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        layer_caches: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode one token while caching prior normalized layer inputs.

        This avoids rerunning every decoder layer over the complete prefix during
        autoregressive inference. PyTorch's MultiheadAttention still projects the
        cached prefix at each step, but computation grows quadratically rather
        than repeatedly decoding the full sequence.
        """
        if token_ids.ndim != 2 or token_ids.size(1) != 1:
            raise ValueError("token_ids must have shape [batch, 1]")
        if not 0 <= position < self.config.max_target_length:
            raise ValueError(f"Position {position} is outside the target context")
        if layer_caches is None:
            layer_caches = [None] * len(self.decoder.layers)
        if len(layer_caches) != len(self.decoder.layers):
            raise ValueError("One cache tensor is required per decoder layer")

        positions = torch.full_like(token_ids, position)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        next_caches: list[torch.Tensor] = []
        for layer, cached_inputs in zip(self.decoder.layers, layer_caches):
            normalized = layer.norm1(hidden)
            key_value = (
                normalized
                if cached_inputs is None
                else torch.cat((cached_inputs, normalized), dim=1)
            )
            self_attention = layer.self_attn(
                normalized,
                key_value,
                key_value,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout1(self_attention)

            normalized = layer.norm2(hidden)
            cross_attention = layer.multihead_attn(
                normalized,
                memory,
                memory,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout2(cross_attention)

            normalized = layer.norm3(hidden)
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normalized)))
            )
            hidden = hidden + layer.dropout3(feed_forward)
            next_caches.append(key_value)

        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return self.output_projection(hidden), next_caches

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        memory: torch.Tensor | None = None,
        teacher_forcing_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.size(1) < 2:
            raise ValueError("input_ids must include at least BOS and EOS")
        if memory is None:
            memory = self.encode_images(pixel_values)
        if teacher_forcing_ids is None:
            teacher_forcing_ids = input_ids
        if teacher_forcing_ids.shape != input_ids.shape:
            raise ValueError("teacher_forcing_ids must have the same shape as input_ids")
        decoder_input_ids = teacher_forcing_ids[:, :-1]
        labels = input_ids[:, 1:]
        decoder_attention_mask = attention_mask[:, :-1] if attention_mask is not None else None
        logits = self.decode(decoder_input_ids, memory, decoder_attention_mask)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.config.pad_token_id,
            reduction="none",
        ).reshape_as(labels)
        valid = labels != self.config.pad_token_id
        loss = token_losses.sum() / valid.sum().clamp_min(1)
        loss_per_sample = token_losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "logits": logits,
            "memory": memory,
            "visual_logits": self.visual_aux_head(memory),
        }

    @staticmethod
    def visual_auxiliary_loss(
        visual_logits: torch.Tensor,
        visual_targets: torch.Tensor,
        positive_weight: float = 20.0,
    ) -> torch.Tensor:
        if visual_logits.shape != visual_targets.shape:
            raise ValueError(
                f"Visual logits {visual_logits.shape} and targets {visual_targets.shape} differ"
            )
        pos_weight = torch.full(
            (visual_logits.size(-1),),
            positive_weight,
            device=visual_logits.device,
            dtype=visual_logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            visual_logits,
            visual_targets.to(dtype=visual_logits.dtype),
            pos_weight=pos_weight,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
