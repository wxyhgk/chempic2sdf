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
    ocr_enabled: bool = False
    ocr_vocab_size: int = 259
    ocr_d_model: int = 192
    ocr_layers: int = 2
    ocr_heads: int = 4
    ocr_ffn_dim: int = 768
    ocr_max_length: int = 24
    ocr_roi_height: int = 4
    ocr_roi_width: int = 8
    ocr_roi_radius: float = 24.0
    stroke_enabled: bool = False
    stroke_classes: int = 5
    stroke_feature_dim: int = 64
    stroke_samples: int = 32
    stroke_band: int = 5
    stroke_band_radius: float = 4.0
    vertex_enabled: bool = False
    vertex_feature_dim: int = 64

    def __post_init__(self) -> None:
        if len(self.vision_dims) != len(self.vision_depths):
            raise ValueError("vision_dims and vision_depths must have equal length")
        if self.d_model % self.decoder_heads:
            raise ValueError("d_model must be divisible by decoder_heads")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by 4 for 2D position encoding")
        if len(self.vision_dims) < 2:
            raise ValueError("OCR requires at least two vision stages")
        if self.ocr_d_model % self.ocr_heads:
            raise ValueError("ocr_d_model must be divisible by ocr_heads")
        if self.ocr_max_length < 3:
            raise ValueError("ocr_max_length must be at least 3")
        if self.ocr_roi_height < 1 or self.ocr_roi_width < 1:
            raise ValueError("OCR ROI dimensions must be positive")
        if self.ocr_roi_radius <= 0:
            raise ValueError("ocr_roi_radius must be positive")
        if self.stroke_classes < 2 or self.stroke_feature_dim < 1:
            raise ValueError("Stroke scorer dimensions are invalid")
        if self.stroke_samples < 2 or self.stroke_band < 1:
            raise ValueError("Stroke sampling dimensions are invalid")
        if self.stroke_band_radius <= 0:
            raise ValueError("stroke_band_radius must be positive")
        if self.vertex_feature_dim < 1:
            raise ValueError("vertex_feature_dim must be positive")


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

    def _forward_features(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high_resolution = self.forward_high_resolution(pixel_values)
        hidden = high_resolution
        for stage in self.stages[3:]:
            hidden = stage(hidden)
        height, width = hidden.shape[-2:]
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = self.output_norm(hidden)
        hidden = self.projection(hidden)
        memory = hidden + self._position_encoding(
            height,
            width,
            hidden.size(-1),
            hidden.device,
            hidden.dtype,
        ).unsqueeze(0)
        return memory, high_resolution

    def forward_high_resolution(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_stroke_resolution(pixel_values)
        return self.forward_high_resolution_from_stroke(hidden)

    def forward_high_resolution_from_stroke(
        self,
        stroke_resolution: torch.Tensor,
    ) -> torch.Tensor:
        hidden = stroke_resolution
        for stage in self.stages[1:3]:
            hidden = stage(hidden)
        return hidden

    def forward_stroke_resolution(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(pixel_values)
        return self.stages[0](hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        memory, _ = self._forward_features(pixel_values)
        return memory

    def forward_with_high_resolution(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_features(pixel_values)

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


class ChemOCRHead(nn.Module):
    """Decode UTF-8 byte sequences from geometry-provided image anchors."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        self.config = config
        self.roi_projection = nn.Linear(config.vision_dims[1], config.ocr_d_model)
        roi_tokens = config.ocr_roi_height * config.ocr_roi_width
        self.roi_position_embedding = nn.Embedding(roi_tokens, config.ocr_d_model)
        self.token_embedding = nn.Embedding(
            config.ocr_vocab_size,
            config.ocr_d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.ocr_max_length, config.ocr_d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.ocr_d_model,
            nhead=config.ocr_heads,
            dim_feedforward=config.ocr_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.ocr_layers,
            norm=nn.LayerNorm(config.ocr_d_model),
        )
        self.output_projection = nn.Linear(
            config.ocr_d_model,
            config.ocr_vocab_size,
            bias=False,
        )
        self.output_projection.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.roi_position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()

    def extract_rois(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_mask: torch.Tensor,
        pixel_size: tuple[int, int],
        *,
        roi_radius: float | None = None,
    ) -> torch.Tensor:
        if anchor_points.ndim != 3 or anchor_points.size(-1) != 2:
            raise ValueError("OCR anchor_points must have shape [batch, anchors, 2]")
        if anchor_mask.shape != anchor_points.shape[:2]:
            raise ValueError("OCR anchor_mask must match anchor_points")
        if feature_map.size(0) != anchor_points.size(0):
            raise ValueError("OCR feature and anchor batch sizes differ")

        pixel_height, pixel_width = pixel_size
        radius = self.config.ocr_roi_radius if roi_radius is None else roi_radius
        if radius <= 0:
            raise ValueError("OCR ROI radius must be positive")
        dtype = feature_map.dtype
        device = feature_map.device
        x_offsets = torch.linspace(
            -radius,
            radius,
            self.config.ocr_roi_width,
            device=device,
            dtype=dtype,
        ) * (2.0 / pixel_width)
        y_offsets = torch.linspace(
            -radius,
            radius,
            self.config.ocr_roi_height,
            device=device,
            dtype=dtype,
        ) * (2.0 / pixel_height)
        grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing="ij")
        offsets = torch.stack((grid_x, grid_y), dim=-1)
        centers = anchor_points.to(device=device, dtype=dtype) * 2.0 - 1.0
        grid = centers[:, :, None, None, :] + offsets[None, None, :, :, :]
        batch_size, anchors = anchor_points.shape[:2]
        grid = grid.reshape(
            batch_size,
            anchors * self.config.ocr_roi_height,
            self.config.ocr_roi_width,
            2,
        )
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size,
            feature_map.size(1),
            anchors,
            self.config.ocr_roi_height,
            self.config.ocr_roi_width,
        ).permute(0, 2, 3, 4, 1)
        selected = sampled[anchor_mask.to(device=device)]
        return selected.reshape(selected.size(0), -1, selected.size(-1))

    def prepare_memory(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_mask: torch.Tensor,
        pixel_size: tuple[int, int],
        *,
        roi_radius: float | None = None,
    ) -> torch.Tensor:
        roi_tokens = self.extract_rois(
            feature_map,
            anchor_points,
            anchor_mask,
            pixel_size,
            roi_radius=roi_radius,
        )
        roi_tokens = self.roi_projection(roi_tokens)
        roi_positions = torch.arange(roi_tokens.size(1), device=roi_tokens.device)
        return roi_tokens + self.roi_position_embedding(roi_positions).unsqueeze(0)

    def decode(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if decoder_input_ids.size(1) >= self.config.ocr_max_length:
            raise ValueError("OCR decoder input exceeds configured maximum length")
        positions = torch.arange(decoder_input_ids.size(1), device=memory.device)
        hidden = self.token_embedding(decoder_input_ids)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                decoder_input_ids.size(1),
                decoder_input_ids.size(1),
                device=memory.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids == self.config.pad_token_id,
            tgt_is_causal=True,
        )
        return self.output_projection(decoded)

    def forward(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_mask: torch.Tensor,
        token_ids: torch.Tensor,
        pixel_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        selected_ids = token_ids.to(device=feature_map.device)[
            anchor_mask.to(device=feature_map.device)
        ]
        if selected_ids.size(0) == 0:
            raise ValueError("OCR batch contains no valid anchors")
        if selected_ids.size(1) > self.config.ocr_max_length:
            raise ValueError("OCR target exceeds configured maximum length")

        memory = self.prepare_memory(
            feature_map,
            anchor_points,
            anchor_mask,
            pixel_size,
        )
        decoder_input_ids = selected_ids[:, :-1]
        labels = selected_ids[:, 1:]
        logits = self.decode(decoder_input_ids, memory)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.config.pad_token_id,
            reduction="none",
        ).reshape_as(labels)
        valid = labels != self.config.pad_token_id
        loss = token_losses.sum() / valid.sum().clamp_min(1)
        return {
            "ocr_loss": loss,
            "ocr_logits": logits,
            "ocr_labels": labels,
        }

    @torch.no_grad()
    def generate(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_mask: torch.Tensor,
        pixel_size: tuple[int, int],
        *,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        roi_radius: float | None = None,
    ) -> torch.Tensor:
        memory = self.prepare_memory(
            feature_map,
            anchor_points,
            anchor_mask,
            pixel_size,
            roi_radius=roi_radius,
        )
        token_ids = torch.full(
            (memory.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(self.config.ocr_max_length - 1):
            next_token = self.decode(token_ids, memory)[:, -1].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
            finished |= next_token == eos_token_id
            if bool(finished.all()):
                break
        return token_ids[:, 1:]


class ChemStrokeScorer(nn.Module):
    """Classify visible line primitives between arbitrary vertex pairs."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.stroke_feature_dim
        self.line_encoder = nn.Sequential(
            nn.Conv2d(config.vision_dims[0], channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(channels, config.stroke_classes),
        )

    def extract_line_bands(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        pixel_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.ndim != 3 or pair_indices.size(-1) != 2:
            raise ValueError("pair_indices must have shape [batch, pairs, 2]")
        if pair_mask.shape != pair_indices.shape[:2]:
            raise ValueError("pair_mask must match pair_indices")
        batch_size, pair_count = pair_indices.shape[:2]
        if anchor_points.size(0) != batch_size or feature_map.size(0) != batch_size:
            raise ValueError("Stroke feature, anchor, and pair batch sizes differ")

        gather_indices = pair_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, 2)
        expanded_anchors = anchor_points[:, None].expand(-1, pair_count, -1, -1)
        endpoints = torch.gather(expanded_anchors, 2, gather_indices)
        start = endpoints[:, :, 0]
        end = endpoints[:, :, 1]
        delta = end - start
        pixel_height, pixel_width = pixel_size
        delta_pixels = torch.stack(
            (delta[..., 0] * pixel_width, delta[..., 1] * pixel_height),
            dim=-1,
        )
        lengths = delta_pixels.norm(dim=-1).clamp_min(1e-6)
        unit_perpendicular = torch.stack(
            (-delta_pixels[..., 1], delta_pixels[..., 0]),
            dim=-1,
        ) / lengths.unsqueeze(-1)
        unit_perpendicular = torch.stack(
            (
                unit_perpendicular[..., 0] / pixel_width,
                unit_perpendicular[..., 1] / pixel_height,
            ),
            dim=-1,
        )

        dtype = feature_map.dtype
        device = feature_map.device
        positions = torch.linspace(
            0.08,
            0.92,
            self.config.stroke_samples,
            device=device,
            dtype=dtype,
        )
        offsets = torch.linspace(
            -self.config.stroke_band_radius,
            self.config.stroke_band_radius,
            self.config.stroke_band,
            device=device,
            dtype=dtype,
        )
        start = start.to(device=device, dtype=dtype)
        delta = delta.to(device=device, dtype=dtype)
        unit_perpendicular = unit_perpendicular.to(device=device, dtype=dtype)
        base = start[:, :, None, :] + delta[:, :, None, :] * positions[None, None, :, None]
        grid = base[:, :, None, :, :] + (
            unit_perpendicular[:, :, None, None, :]
            * offsets[None, None, :, None, None]
        )
        grid = grid * 2.0 - 1.0
        grid = grid.reshape(
            batch_size,
            pair_count * self.config.stroke_band,
            self.config.stroke_samples,
            2,
        )
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size,
            feature_map.size(1),
            pair_count,
            self.config.stroke_band,
            self.config.stroke_samples,
        ).permute(0, 2, 1, 3, 4)
        selected = sampled[pair_mask.to(device=device)]
        geometry = torch.stack(
            (
                delta_pixels[..., 0] / pixel_width,
                delta_pixels[..., 1] / pixel_height,
                lengths / max(pixel_height, pixel_width),
            ),
            dim=-1,
        ).to(device=device, dtype=dtype)
        return selected, geometry[pair_mask.to(device=device)]

    def forward(
        self,
        feature_map: torch.Tensor,
        anchor_points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        pixel_size: tuple[int, int],
    ) -> torch.Tensor:
        line_bands, geometry = self.extract_line_bands(
            feature_map,
            anchor_points,
            pair_indices,
            pair_mask,
            pixel_size,
        )
        line_features = self.line_encoder(line_bands).flatten(1)
        geometry_features = self.geometry_encoder(geometry)
        return self.classifier(torch.cat((line_features, geometry_features), dim=-1))


class ChemVertexDetector(nn.Module):
    """Predict visual vertex centers, sub-cell offsets, and aromatic flags."""

    def __init__(self, config: ChemVectorConfig) -> None:
        super().__init__()
        channels = config.vertex_feature_dim
        self.shared = nn.Sequential(
            nn.Conv2d(config.vision_dims[0], channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.heatmap = nn.Conv2d(channels, 1, kernel_size=1)
        self.offset = nn.Conv2d(channels, 2, kernel_size=1)
        self.aromatic = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.constant_(self.heatmap.bias, -4.6)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(feature_map)
        return {
            "vertex_heatmap_logits": self.heatmap(hidden),
            "vertex_offset_logits": self.offset(hidden),
            "vertex_aromatic_logits": self.aromatic(hidden),
        }


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
        self.ocr_head = ChemOCRHead(config) if config.ocr_enabled else None
        self.stroke_head = ChemStrokeScorer(config) if config.stroke_enabled else None
        self.vertex_head = ChemVertexDetector(config) if config.vertex_enabled else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(pixel_values)

    def encode_images_with_ocr_features(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vision_encoder.forward_with_high_resolution(pixel_values)

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
        ocr_anchor_points: torch.Tensor | None = None,
        ocr_anchor_mask: torch.Tensor | None = None,
        ocr_token_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.size(1) < 2:
            raise ValueError("input_ids must include at least BOS and EOS")
        ocr_features = None
        has_ocr_inputs = any(
            value is not None
            for value in (ocr_anchor_points, ocr_anchor_mask, ocr_token_ids)
        )
        if has_ocr_inputs and not all(
            value is not None
            for value in (ocr_anchor_points, ocr_anchor_mask, ocr_token_ids)
        ):
            raise ValueError("All OCR tensors must be provided together")
        if has_ocr_inputs and self.ocr_head is None:
            raise ValueError("OCR inputs require an OCR-enabled model")
        if memory is None:
            if has_ocr_inputs:
                memory, ocr_features = self.encode_images_with_ocr_features(pixel_values)
            else:
                memory = self.encode_images(pixel_values)
        elif has_ocr_inputs:
            raise ValueError("OCR inputs cannot be combined with precomputed memory")
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
        output = {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "logits": logits,
            "memory": memory,
            "visual_logits": self.visual_aux_head(memory),
        }
        if has_ocr_inputs:
            output.update(
                self.ocr_head(
                    ocr_features,
                    ocr_anchor_points,
                    ocr_anchor_mask,
                    ocr_token_ids,
                    pixel_values.shape[-2:],
                )
            )
        return output

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
