"""Decoder-only visual-prefix model for raster-to-Visual-SVG generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.model import ChemVectorConfig, ChemVisionEncoder


@dataclass(frozen=True, slots=True)
class VisualPrefixMemory:
    prefix: torch.Tensor
    values: torch.Tensor


class VisualPrefixBlock(nn.Module):
    """A pre-norm self-attention block with an incremental decoding path."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.decoder_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.activation_dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.dropout2 = nn.Dropout(config.dropout)

    def feed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm2(hidden)
        transformed = self.linear1(normalized)
        transformed = F.gelu(transformed)
        transformed = self.activation_dropout(transformed)
        return hidden + self.dropout2(self.linear2(transformed))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        visual_length: int,
        drop_text_residual: bool,
        visual_values: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden)
        if drop_text_residual:
            if visual_values is None:
                raise ValueError("visual-only attention requires content-only values")
            visual_normalized = normalized[:, :visual_length]
            visual_attended = self.self_attention(
                visual_normalized,
                visual_normalized,
                visual_normalized,
                need_weights=False,
            )[0]
            text_attended = self.self_attention(
                normalized[:, visual_length:],
                visual_normalized,
                visual_values,
                need_weights=False,
            )[0]
            attended = torch.cat((visual_attended, text_attended), dim=1)
        else:
            attended = self.self_attention(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
            if visual_values is not None:
                grounded_text = self.self_attention(
                    normalized[:, visual_length:],
                    normalized[:, :visual_length],
                    visual_values,
                    need_weights=False,
                )[0]
                attended = torch.cat(
                    (
                        attended[:, :visual_length],
                        attended[:, visual_length:] + grounded_text,
                    ),
                    dim=1,
                )
        attended = self.dropout1(attended)
        if drop_text_residual:
            hidden = torch.cat(
                (
                    hidden[:, :visual_length] + attended[:, :visual_length],
                    attended[:, visual_length:],
                ),
                dim=1,
            )
        else:
            hidden = hidden + attended
        return self.feed_forward(hidden)

    def encode_prefix(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(hidden)
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return self.feed_forward(hidden + self.dropout1(attended)), normalized

    def decode_step(
        self,
        hidden: torch.Tensor,
        cached_normalized_inputs: torch.Tensor,
        *,
        visual_length: int,
        visual_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(hidden)
        key_value = torch.cat((cached_normalized_inputs, normalized), dim=1)
        attended = self.self_attention(
            normalized,
            key_value,
            key_value,
            need_weights=False,
        )[0]
        if visual_values is not None:
            attended = attended + self.self_attention(
                normalized,
                cached_normalized_inputs[:, :visual_length],
                visual_values,
                need_weights=False,
            )[0]
        hidden = self.feed_forward(hidden + self.dropout1(attended))
        return hidden, key_value


class VisualPrefixChemVectorModel(nn.Module):
    """Generate Visual SVG tokens from an image prefix in one attention stream."""

    model_type = "visual-prefix-v0.1"

    def __init__(
        self,
        config: ChemVectorConfig,
        *,
        visual_grid_size: int = 12,
        ink_subgrid: int = 2,
    ) -> None:
        super().__init__()
        if visual_grid_size < 1:
            raise ValueError("visual_grid_size must be positive")
        if ink_subgrid < 1:
            raise ValueError("ink_subgrid must be positive")
        self.config = config
        self.visual_grid_size = visual_grid_size
        self.visual_prefix_tokens = visual_grid_size**2
        self.ink_subgrid = ink_subgrid
        self.vision_encoder = ChemVisionEncoder(config)
        self.raw_ink_projection = nn.Linear(
            2 * ink_subgrid**2,
            config.d_model,
            bias=False,
        )
        self.visual_position_embedding = nn.Embedding(
            self.visual_prefix_tokens,
            config.d_model,
        )
        self.visual_modality_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_target_length, config.d_model)
        self.text_modality_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.layers = nn.ModuleList(
            VisualPrefixBlock(config) for _ in range(config.decoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.primitive_count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 4),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.visual_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.visual_modality_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.text_modality_embedding, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()

    def _cnn_feature_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_encoder.stem(pixel_values)
        for stage in self.vision_encoder.stages:
            hidden = stage(hidden)
        return hidden

    def _raw_ink_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        grayscale = (pixel_values + 1.0).mul(0.5).mean(dim=1, keepdim=True)
        ink = (1.0 - grayscale).clamp(0.0, 1.0)
        fine_size = self.visual_grid_size * self.ink_subgrid
        average_ink = F.adaptive_avg_pool2d(ink, (fine_size, fine_size))
        maximum_ink = F.adaptive_max_pool2d(ink, (fine_size, fine_size))

        def pack(values: torch.Tensor) -> torch.Tensor:
            batch = values.size(0)
            packed = values.reshape(
                batch,
                1,
                self.visual_grid_size,
                self.ink_subgrid,
                self.visual_grid_size,
                self.ink_subgrid,
            )
            packed = packed.permute(0, 2, 4, 3, 5, 1)
            return packed.reshape(
                batch,
                self.visual_prefix_tokens,
                self.ink_subgrid**2,
            )

        descriptors = torch.cat((pack(average_ink), pack(maximum_ink)), dim=-1)
        return self.raw_ink_projection(descriptors)

    def _encode_visual_inputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self._cnn_feature_map(pixel_values)
        pooled = F.adaptive_avg_pool2d(
            feature_map,
            (self.visual_grid_size, self.visual_grid_size),
        )
        content = pooled.flatten(2).transpose(1, 2)
        content = self.vision_encoder.output_norm(content)
        content = self.vision_encoder.projection(content)

        raw_ink = self._raw_ink_tokens(pixel_values)
        content = content + raw_ink

        positions = torch.arange(self.visual_prefix_tokens, device=pixel_values.device)
        prefix = (
            content
            + self.visual_position_embedding(positions).unsqueeze(0)
            + self.visual_modality_embedding.view(1, 1, -1)
        )
        return prefix, raw_ink

    def encode_visual_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._encode_visual_inputs(pixel_values)[0]

    def encode_images(self, pixel_values: torch.Tensor) -> VisualPrefixMemory:
        prefix, values = self._encode_visual_inputs(pixel_values)
        return VisualPrefixMemory(prefix, values)

    @staticmethod
    def _prefix_lm_mask(
        visual_length: int,
        text_length: int,
        device: torch.device,
        attention_layout: str = "causal",
    ) -> torch.Tensor:
        if attention_layout not in {"causal", "visual-only"}:
            raise ValueError(f"Unsupported attention layout: {attention_layout}")
        total = visual_length + text_length
        mask = torch.triu(
            torch.ones(total, total, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[:visual_length, :visual_length] = False
        mask[:visual_length, visual_length:] = True
        if attention_layout == "visual-only":
            mask[visual_length:, visual_length:] = True
        return mask

    def _text_embeddings(self, input_ids: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        length = input_ids.size(1)
        if start_position < 0 or start_position + length > self.config.max_target_length:
            raise ValueError("Text positions are outside the configured target context")
        positions = torch.arange(
            start_position,
            start_position + length,
            device=input_ids.device,
        )
        return (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
            + self.text_modality_embedding.view(1, 1, -1)
        )

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        visual_prefix: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
        attention_layout: str = "causal",
        visual_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_length = decoder_input_ids.size(1)
        if text_length > self.config.max_target_length:
            raise ValueError(
                f"Target length {text_length} exceeds {self.config.max_target_length}"
            )
        visual_length = visual_prefix.size(1)
        hidden = torch.cat(
            (visual_prefix, self._text_embeddings(decoder_input_ids)),
            dim=1,
        )
        key_padding_mask = None
        if decoder_attention_mask is not None:
            visual_mask = torch.zeros(
                decoder_attention_mask.size(0),
                visual_length,
                dtype=torch.bool,
                device=decoder_attention_mask.device,
            )
            key_padding_mask = torch.cat(
                (visual_mask, decoder_attention_mask == 0),
                dim=1,
            )
        attention_mask = self._prefix_lm_mask(
            visual_length,
            text_length,
            hidden.device,
            attention_layout,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                visual_length=visual_length,
                drop_text_residual=attention_layout == "visual-only",
                visual_values=visual_values,
            )
        hidden = self.final_norm(hidden)
        encoded_visual = hidden[:, :visual_length]
        logits = self.output_projection(hidden[:, visual_length:])
        return logits, encoded_visual

    def _initial_layer_caches(
        self,
        visual_prefix: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden = visual_prefix
        caches = []
        for layer in self.layers:
            hidden, normalized = layer.encode_prefix(hidden)
            caches.append(normalized)
        return caches

    def decode_step(
        self,
        token_ids: torch.Tensor,
        position: int,
        visual_prefix: torch.Tensor | VisualPrefixMemory,
        layer_caches: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if token_ids.ndim != 2 or token_ids.size(1) != 1:
            raise ValueError("token_ids must have shape [batch, 1]")
        if not 0 <= position < self.config.max_target_length:
            raise ValueError(f"Position {position} is outside the target context")
        visual_values = None
        if isinstance(visual_prefix, VisualPrefixMemory):
            visual_values = visual_prefix.values
            visual_prefix = visual_prefix.prefix
        if layer_caches is None:
            layer_caches = self._initial_layer_caches(visual_prefix)
        if len(layer_caches) != len(self.layers):
            raise ValueError("One cache tensor is required per visual-prefix layer")
        hidden = self._text_embeddings(token_ids, start_position=position)
        next_caches = []
        for layer, cache in zip(self.layers, layer_caches):
            hidden, next_cache = layer.decode_step(
                hidden,
                cache,
                visual_length=visual_prefix.size(1),
                visual_values=visual_values,
            )
            next_caches.append(next_cache)
        hidden = self.final_norm(hidden)
        return self.output_projection(hidden), next_caches

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        memory: VisualPrefixMemory | None = None,
        teacher_forcing_ids: torch.Tensor | None = None,
        attention_layout: str = "causal",
    ) -> dict[str, torch.Tensor]:
        if input_ids.size(1) < 2:
            raise ValueError("input_ids must include at least BOS and EOS")
        if memory is None:
            visual_prefix, visual_values = self._encode_visual_inputs(pixel_values)
        elif isinstance(memory, VisualPrefixMemory):
            visual_prefix = memory.prefix
            visual_values = memory.values
        else:
            raise TypeError("memory must be VisualPrefixMemory or None")
        if teacher_forcing_ids is None:
            teacher_forcing_ids = input_ids
        if teacher_forcing_ids.shape != input_ids.shape:
            raise ValueError("teacher_forcing_ids must have the same shape as input_ids")
        decoder_input_ids = teacher_forcing_ids[:, :-1]
        labels = input_ids[:, 1:]
        decoder_attention_mask = attention_mask[:, :-1] if attention_mask is not None else None
        logits, encoded_visual = self.decode(
            decoder_input_ids,
            visual_prefix,
            decoder_attention_mask,
            attention_layout,
            visual_values,
        )
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        valid = (
            attention_mask[:, 1:].bool()
            if attention_mask is not None
            else labels != self.config.pad_token_id
        )
        loss = (token_losses * valid).sum() / valid.sum().clamp_min(1)
        loss_per_sample = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        primitive_count_log = self.primitive_count_head(encoded_visual.mean(dim=1))
        return {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "logits": logits,
            "memory": visual_prefix,
            "visual_prefix": visual_prefix,
            "encoded_visual_prefix": encoded_visual,
            "visual_values": visual_values,
            "primitive_count_log": primitive_count_log,
        }

    @staticmethod
    def primitive_count_loss(
        predicted_log_counts: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_log_counts.shape != target_counts.shape:
            raise ValueError("Primitive count predictions and targets must have equal shape")
        return F.smooth_l1_loss(
            predicted_log_counts,
            torch.log1p(target_counts.to(dtype=predicted_log_counts.dtype)),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
