"""Unordered set prediction for visible Visual SVG primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import position_encoding_2d
from chemvector.model import ChemVectorConfig, ChemVisionEncoder


NO_OBJECT = 0
LINE = 1
TRIANGLE = 2
QUAD = 3
TEXT_BOX = 4
PRIMITIVE_TYPES = ("no_object", "line", "triangle", "quad", "text_box")
POINTS_BY_TYPE = {LINE: 2, TRIANGLE: 3, QUAD: 4}

LINE_ENDPOINT_LOSS_WEIGHT = 2.0
LINE_CENTER_LOSS_WEIGHT = 0.25
LINE_LENGTH_LOSS_WEIGHT = 0.25
LINE_ORIENTATION_LOSS_WEIGHT = 0.05
LINE_ENDPOINT_COST_WEIGHT = 2.5
LINE_CENTER_COST_WEIGHT = 0.5
LINE_LENGTH_COST_WEIGHT = 0.25
LINE_ORIENTATION_COST_WEIGHT = 0.1
LINE_HUBER_BETA_PIXELS = 4.0
LINE_ORIENTATION_GATE_PIXELS = 8.0
LINE_HALF_VECTOR_MAX = 0.5


@dataclass(frozen=True, slots=True)
class PrimitiveSetConfig:
    queries: int = 288
    d_model: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 8
    decoder_ffn_dim: int = 1024
    memory_grid_size: int = 48
    dropout: float = 0.0
    local_query_kernel: int = 0
    endpoint_refine_kernel: int = 0
    line_center_vector: bool = False

    def __post_init__(self) -> None:
        if min(
            self.queries,
            self.d_model,
            self.decoder_layers,
            self.decoder_heads,
            self.decoder_ffn_dim,
            self.memory_grid_size,
        ) < 1:
            raise ValueError("Primitive Set dimensions must be positive")
        if self.d_model % self.decoder_heads:
            raise ValueError("Primitive Set heads must divide d_model")
        if self.d_model % 4:
            raise ValueError("Primitive Set d_model must be divisible by four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Primitive Set dropout must be in [0, 1)")
        if self.local_query_kernel < 0 or (
            self.local_query_kernel and self.local_query_kernel % 2 == 0
        ):
            raise ValueError("Primitive Set local query kernel must be zero or odd")
        if self.endpoint_refine_kernel < 0 or (
            self.endpoint_refine_kernel and self.endpoint_refine_kernel % 2 == 0
        ):
            raise ValueError("Primitive Set endpoint refine kernel must be zero or odd")


class PrimitiveSetPredictor(nn.Module):
    """Predict one unordered set of physical drawing primitives per image."""

    model_type = "primitive-set-v0.1"

    def __init__(
        self,
        vision_config: ChemVectorConfig,
        config: PrimitiveSetConfig = PrimitiveSetConfig(),
    ) -> None:
        super().__init__()
        if len(vision_config.vision_dims) < 3:
            raise ValueError("Primitive Set requires a stride-16 vision stage")
        self.vision_config = vision_config
        self.config = config
        self.vision_encoder = ChemVisionEncoder(vision_config)
        high_channels = vision_config.vision_dims[1]
        self.memory_projection = nn.Sequential(
            nn.Conv2d(high_channels, config.d_model, kernel_size=1),
            nn.GroupNorm(1, config.d_model),
            nn.GELU(),
            nn.Conv2d(config.d_model, config.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(1, config.d_model),
        )
        self.raw_ink_projection = nn.Conv2d(2, config.d_model, kernel_size=1, bias=False)
        if config.local_query_kernel:
            self.local_query_projection = nn.Sequential(
                nn.Conv2d(
                    config.d_model,
                    config.d_model,
                    kernel_size=config.local_query_kernel,
                    padding=config.local_query_kernel // 2,
                    groups=config.d_model,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(
                    config.d_model,
                    config.d_model,
                    kernel_size=1,
                    bias=False,
                ),
            )
        if config.endpoint_refine_kernel:
            refine_channels = min(config.d_model, 64)
            self.endpoint_refine_channels = refine_channels
            self.endpoint_refine_projection = nn.Sequential(
                nn.Conv2d(
                    vision_config.vision_dims[0] + 2,
                    refine_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(1, refine_channels),
                nn.GELU(),
            )
            self.endpoint_refine_depthwise = nn.Sequential(
                nn.Conv2d(
                    refine_channels,
                    refine_channels,
                    kernel_size=config.endpoint_refine_kernel,
                    padding=config.endpoint_refine_kernel // 2,
                    groups=refine_channels,
                    bias=False,
                ),
                nn.GroupNorm(1, refine_channels),
                nn.GELU(),
                nn.Conv2d(
                    refine_channels,
                    refine_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(1, refine_channels),
                nn.GELU(),
            )
            self.endpoint_point_embedding = nn.Embedding(4, refine_channels)
            self.endpoint_refine_mlp = nn.Sequential(
                nn.Linear(config.d_model + 2 * refine_channels, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
        self.query_embedding = nn.Embedding(config.queries, config.d_model)
        self.query_position_projection = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.register_buffer(
            "reference_points",
            self._reference_grid(config.queries),
            persistent=True,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.type_head = nn.Linear(config.d_model, len(PRIMITIVE_TYPES))
        self.geometry_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 8),
        )
        if config.line_center_vector:
            self.line_geometry_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 4),
            )
        self.text_geometry_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),
        )
        self.stroke_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.queries + 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.geometry_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.geometry_head[-1].bias)
        if self.config.line_center_vector:
            nn.init.normal_(self.line_geometry_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.line_geometry_head[-1].bias)
        nn.init.normal_(self.text_geometry_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.text_geometry_head[-1].bias)
        if self.config.endpoint_refine_kernel:
            nn.init.normal_(self.endpoint_point_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.endpoint_refine_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.endpoint_refine_mlp[-1].bias)
        with torch.no_grad():
            self.type_head.bias.zero_()
            self.type_head.bias[NO_OBJECT] = math.log(14.0)
            self.stroke_head[-1].bias.fill_(-1.2)
            self.count_head[-1].bias.zero_()

    @staticmethod
    def _reference_grid(queries: int) -> torch.Tensor:
        columns = math.ceil(math.sqrt(queries))
        rows = math.ceil(queries / columns)
        y, x = torch.meshgrid(
            (torch.arange(rows, dtype=torch.float32) + 0.5) / rows,
            (torch.arange(columns, dtype=torch.float32) + 0.5) / columns,
            indexing="ij",
        )
        return torch.stack((x.flatten(), y.flatten()), dim=-1)[:queries]

    @staticmethod
    def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(values / (1.0 - values))

    def _memory_map(
        self,
        pixel_values: torch.Tensor,
        *,
        return_refine_inputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stroke = self.vision_encoder.forward_stroke_resolution(pixel_values)
        high = self.vision_encoder.forward_high_resolution_from_stroke(stroke)
        high = F.adaptive_avg_pool2d(
            high,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        memory_map = self.memory_projection(high)

        grayscale = pixel_values.add(1.0).mul(0.5).mean(dim=1, keepdim=True)
        ink = (1.0 - grayscale).clamp(0.0, 1.0)
        average = F.adaptive_avg_pool2d(
            ink,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        maximum = F.adaptive_max_pool2d(
            ink,
            (self.config.memory_grid_size, self.config.memory_grid_size),
        )
        memory_map = memory_map + self.raw_ink_projection(torch.cat((average, maximum), dim=1))
        if return_refine_inputs:
            return memory_map, stroke, ink
        return memory_map

    def _memory_from_map(self, memory_map: torch.Tensor) -> torch.Tensor:
        memory = memory_map.flatten(2).transpose(1, 2)
        return memory + position_encoding_2d(
            self.config.memory_grid_size,
            self.config.memory_grid_size,
            self.config.d_model,
            memory.device,
            memory.dtype,
        ).unsqueeze(0)

    def _memory(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._memory_from_map(self._memory_map(pixel_values))

    def _local_query_features(
        self,
        memory_map: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.local_query_kernel:
            raise RuntimeError("Local query features are disabled")
        local_map = self.local_query_projection(memory_map)
        grid = references.mul(2.0).sub(1.0)
        grid = grid.view(1, -1, 1, 2).expand(memory_map.size(0), -1, -1, -1)
        sampled = F.grid_sample(
            local_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _endpoint_refine_map(
        self,
        stroke: torch.Tensor,
        ink: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.endpoint_refine_kernel:
            raise RuntimeError("Endpoint refinement is disabled")
        size = stroke.shape[-2:]
        average = F.adaptive_avg_pool2d(ink, size)
        maximum = F.adaptive_max_pool2d(ink, size)
        inputs = torch.cat((stroke, average, maximum), dim=1)
        projected = self.endpoint_refine_projection(inputs)
        return self.endpoint_refine_depthwise(projected)

    def _refine_endpoints(
        self,
        coarse_geometry: torch.Tensor,
        object_features: torch.Tensor,
        refine_map: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.endpoint_refine_kernel:
            raise RuntimeError("Endpoint refinement is disabled")
        batch, queries, _ = coarse_geometry.shape
        coarse_points = coarse_geometry.reshape(batch, queries, 4, 2)
        grid = coarse_points.mul(2.0).sub(1.0).reshape(batch, queries * 4, 1, 2)
        sampled = F.grid_sample(
            refine_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        sampled = sampled.reshape(batch, queries, 4, self.endpoint_refine_channels)
        point_indices = torch.arange(4, device=object_features.device)
        point_features = self.endpoint_point_embedding(point_indices).to(
            object_features.dtype
        )
        point_features = point_features.view(1, 1, 4, -1).expand(
            batch, queries, -1, -1
        )
        object_features = object_features.unsqueeze(2).expand(-1, -1, 4, -1)
        residual = self.endpoint_refine_mlp(
            torch.cat((object_features, sampled, point_features), dim=-1)
        )
        refined = torch.sigmoid(
            self._inverse_sigmoid(coarse_points) + residual
        )
        return refined.flatten(-2)

    def _coarse_line_geometry(
        self,
        object_features: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.line_center_vector:
            raise RuntimeError("Line center-vector geometry is disabled")
        raw = self.line_geometry_head(object_features)
        center = torch.sigmoid(
            self._inverse_sigmoid(references)[None] + raw[..., :2]
        )
        maximum_half_vector = torch.minimum(center, 1.0 - center).clamp_max(
            LINE_HALF_VECTOR_MAX
        )
        half_vector = torch.tanh(raw[..., 2:]) * maximum_half_vector
        endpoints = torch.cat((center - half_vector, center + half_vector), dim=-1)
        return F.pad(endpoints, (0, 4))

    def _refine_line_endpoints(
        self,
        coarse_geometry: torch.Tensor,
        object_features: torch.Tensor,
        refine_map: torch.Tensor,
    ) -> torch.Tensor:
        refined = self._refine_endpoints(
            coarse_geometry,
            object_features,
            refine_map,
        )
        return torch.cat((refined[..., :4], torch.zeros_like(refined[..., 4:])), dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError("Primitive Set images must have shape [batch, 3, h, w]")
        if self.config.endpoint_refine_kernel:
            memory_map, stroke, ink = self._memory_map(
                pixel_values,
                return_refine_inputs=True,
            )
        else:
            memory_map = self._memory_map(pixel_values)
        memory = self._memory_from_map(memory_map)
        references = self.reference_points.to(device=memory.device, dtype=memory.dtype)
        queries = (
            self.query_embedding.weight.to(memory.dtype)
            + self.query_position_projection(references)
        ).unsqueeze(0).expand(pixel_values.size(0), -1, -1)
        if self.config.local_query_kernel:
            queries = queries + self._local_query_features(memory_map, references)
        hidden = self.decoder(queries, memory)
        point_references = references.repeat(1, 4)
        coarse_geometry = torch.sigmoid(
            self._inverse_sigmoid(point_references)[None]
            + self.geometry_head(hidden)
        )
        geometry = coarse_geometry
        if self.config.line_center_vector:
            coarse_line_geometry = self._coarse_line_geometry(hidden, references)
            line_geometry = coarse_line_geometry
        if self.config.endpoint_refine_kernel:
            refine_map = self._endpoint_refine_map(stroke, ink)
            geometry = self._refine_endpoints(
                coarse_geometry,
                hidden,
                refine_map,
            )
            if self.config.line_center_vector:
                line_geometry = self._refine_line_endpoints(
                    coarse_line_geometry,
                    hidden,
                    refine_map,
                )
        text_raw = self.text_geometry_head(hidden)
        text_geometry = torch.zeros_like(geometry)
        text_geometry[..., :2] = torch.sigmoid(
            self._inverse_sigmoid(references)[None] + text_raw[..., :2]
        )
        text_geometry[..., 2:4] = torch.sigmoid(text_raw[..., 2:4] - 2.2)
        output = {
            "type_logits": self.type_head(hidden),
            "count_logits": self.count_head(hidden.mean(dim=1)),
            "geometry": geometry,
            "text_geometry": text_geometry,
            "stroke": self.stroke_head(hidden).squeeze(-1).sigmoid(),
            "object_features": hidden,
            "memory": memory,
        }
        if self.config.endpoint_refine_kernel:
            output["coarse_geometry"] = coarse_geometry
        if self.config.line_center_vector:
            output["line_geometry"] = line_geometry
            if self.config.endpoint_refine_kernel:
                output["coarse_line_geometry"] = coarse_line_geometry
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict:
        return {
            "vision_config": asdict(self.vision_config),
            "primitive_set_config": asdict(self.config),
        }


def load_visual_prefix_vision(
    model: PrimitiveSetPredictor,
    checkpoint: dict,
) -> None:
    """Warm-start only the shared ChemVisionEncoder from a Visual Prefix checkpoint."""

    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Visual Prefix checkpoint does not contain a model state")
    prefix = "vision_encoder."
    vision_state = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not vision_state:
        raise ValueError("Visual Prefix checkpoint has no vision_encoder weights")
    model.vision_encoder.load_state_dict(vision_state, strict=True)


def _polygon_variants(geometry: torch.Tensor, points: int) -> torch.Tensor:
    values = geometry[..., : 2 * points].reshape(*geometry.shape[:-1], points, 2)
    variants = []
    for reverse in (False, True):
        ordered = values.flip(-2) if reverse else values
        for shift in range(points):
            variants.append(ordered.roll(-shift, dims=-2).flatten(-2))
    return torch.stack(variants, dim=-2)


def geometry_variants(geometry: torch.Tensor, primitive_type: int) -> torch.Tensor:
    """Return equivalent active coordinate orderings for one primitive type."""

    if geometry.ndim != 2 or geometry.size(-1) != 8:
        raise ValueError("Primitive geometry must have shape [objects, 8]")
    if primitive_type == LINE:
        direct = geometry[:, :4]
        reverse = geometry[:, :4].reshape(-1, 2, 2).flip(1).flatten(1)
        return torch.stack((direct, reverse), dim=1)
    if primitive_type in (TRIANGLE, QUAD):
        return _polygon_variants(geometry, POINTS_BY_TYPE[primitive_type])
    if primitive_type == TEXT_BOX:
        return geometry[:, None, :4]
    raise ValueError(f"Unsupported primitive type: {primitive_type}")


def active_geometry_size(primitive_type: int) -> int:
    if primitive_type in POINTS_BY_TYPE:
        return 2 * POINTS_BY_TYPE[primitive_type]
    if primitive_type == TEXT_BOX:
        return 4
    raise ValueError(f"Unsupported primitive type: {primitive_type}")


def geometry_shape(geometry: torch.Tensor, primitive_type: int) -> torch.Tensor:
    """Return center-x, center-y, width, height for matching and shape loss."""

    if primitive_type == TEXT_BOX:
        return geometry[..., :4]
    points = POINTS_BY_TYPE.get(primitive_type)
    if points is None:
        raise ValueError(f"Unsupported primitive type: {primitive_type}")
    values = geometry[..., : 2 * points].reshape(*geometry.shape[:-1], points, 2)
    minimum = values.amin(dim=-2)
    maximum = values.amax(dim=-2)
    return torch.cat(((minimum + maximum) * 0.5, maximum - minimum), dim=-1)


def pairwise_line_geometry_cost(
    predicted_geometry: torch.Tensor,
    target_geometry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return endpoint, center, length, and orientation costs for line pairs."""

    if predicted_geometry.ndim != 2 or predicted_geometry.size(-1) != 8:
        raise ValueError("Predicted line geometry must have shape [queries, 8]")
    if target_geometry.ndim != 2 or target_geometry.size(-1) != 8:
        raise ValueError("Target line geometry must have shape [targets, 8]")
    predicted = predicted_geometry[:, :4].reshape(-1, 2, 2)
    expected = target_geometry[:, :4].reshape(-1, 2, 2)
    direct = (predicted[:, None] - expected[None]).norm(dim=-1).amax(dim=-1)
    reversed_distance = (
        predicted[:, None] - expected.flip(1)[None]
    ).norm(dim=-1).amax(dim=-1)
    endpoint = torch.minimum(direct, reversed_distance)

    predicted_center = predicted.mean(dim=1)
    expected_center = expected.mean(dim=1)
    center = (predicted_center[:, None] - expected_center[None]).norm(dim=-1)

    predicted_vector = predicted[:, 1] - predicted[:, 0]
    expected_vector = expected[:, 1] - expected[:, 0]
    predicted_length = predicted_vector.norm(dim=-1)
    expected_length = expected_vector.norm(dim=-1)
    length = (predicted_length[:, None] - expected_length[None]).abs()
    denominator = (
        predicted_length[:, None] * expected_length[None]
    ).clamp_min(torch.finfo(predicted.dtype).eps)
    cosine = (
        predicted_vector[:, None] * expected_vector[None]
    ).sum(dim=-1) / denominator
    orientation = 1.0 - cosine.clamp(-1.0, 1.0).abs()
    return endpoint, center, length, orientation


def line_endpoint_aligned_losses(
    predicted_geometry: torch.Tensor,
    target_geometry: torch.Tensor,
    *,
    image_size: float = 768.0,
    huber_beta_pixels: float = LINE_HUBER_BETA_PIXELS,
    orientation_gate_pixels: float = LINE_ORIENTATION_GATE_PIXELS,
) -> dict[str, torch.Tensor]:
    """Compute evaluation-aligned line losses after endpoint-order selection."""

    if predicted_geometry.ndim != 2 or predicted_geometry.size(-1) != 8:
        raise ValueError("Predicted line geometry must have shape [lines, 8]")
    if target_geometry.shape != predicted_geometry.shape:
        raise ValueError("Predicted and target line geometry must have equal shapes")
    if image_size <= 0.0 or huber_beta_pixels <= 0.0:
        raise ValueError("Line image size and Huber beta must be positive")
    if orientation_gate_pixels <= 0.0:
        raise ValueError("Line orientation gate must be positive")
    zero = predicted_geometry.sum() * 0.0
    if not len(predicted_geometry):
        return {
            "line_endpoint_loss": zero,
            "line_center_loss": zero,
            "line_length_loss": zero,
            "line_orientation_loss": zero,
        }

    predicted = predicted_geometry[:, :4].float().reshape(-1, 2, 2)
    expected = target_geometry[:, :4].float().reshape(-1, 2, 2)
    direct = (predicted - expected).norm(dim=-1).amax(dim=-1)
    reversed_distance = (
        predicted - expected.flip(1)
    ).norm(dim=-1).amax(dim=-1)
    use_reversed = (reversed_distance < direct).detach()
    aligned = torch.where(use_reversed[:, None, None], expected.flip(1), expected)

    endpoint_distance = (predicted - aligned).norm(dim=-1).amax(dim=-1)
    predicted_center = predicted.mean(dim=1)
    expected_center = aligned.mean(dim=1)
    center_distance = (predicted_center - expected_center).norm(dim=-1)
    predicted_vector = predicted[:, 1] - predicted[:, 0]
    expected_vector = aligned[:, 1] - aligned[:, 0]
    predicted_length = predicted_vector.norm(dim=-1)
    expected_length = expected_vector.norm(dim=-1)
    length_distance = (predicted_length - expected_length).abs()

    epsilon = 1.0 / image_size
    denominator = (
        predicted_length.clamp_min(epsilon)
        * expected_length.clamp_min(epsilon)
    )
    cosine = (predicted_vector * expected_vector).sum(dim=-1) / denominator
    orientation_gate = (
        predicted_length.mul(image_size).div(orientation_gate_pixels).clamp(0.0, 1.0)
    ).detach()
    orientation_loss = (
        orientation_gate * (1.0 - cosine.clamp(-1.0, 1.0).abs())
    ).mean()

    def pixel_huber(distance: torch.Tensor) -> torch.Tensor:
        pixel_distance = distance * image_size
        return F.smooth_l1_loss(
            pixel_distance,
            torch.zeros_like(pixel_distance),
            beta=huber_beta_pixels,
        ) / image_size

    return {
        "line_endpoint_loss": pixel_huber(endpoint_distance),
        "line_center_loss": pixel_huber(center_distance),
        "line_length_loss": pixel_huber(length_distance),
        "line_orientation_loss": orientation_loss,
    }


def pairwise_geometry_cost(
    predicted_geometry: torch.Tensor,
    target_geometry: torch.Tensor,
    target_types: torch.Tensor,
    predicted_text_geometry: torch.Tensor | None = None,
    predicted_line_geometry: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute permutation-invariant coordinate and shape costs."""

    if predicted_geometry.ndim != 2 or predicted_geometry.size(-1) != 8:
        raise ValueError("Predicted primitive geometry must have shape [queries, 8]")
    if target_geometry.ndim != 2 or target_geometry.size(-1) != 8:
        raise ValueError("Target primitive geometry must have shape [targets, 8]")
    if target_types.shape != target_geometry.shape[:1]:
        raise ValueError("Primitive target types and geometry differ")
    coordinate = predicted_geometry.new_zeros(
        (predicted_geometry.size(0), target_geometry.size(0))
    )
    shape = torch.zeros_like(coordinate)
    for primitive_type in (LINE, TRIANGLE, QUAD, TEXT_BOX):
        indices = torch.where(target_types == primitive_type)[0]
        if not len(indices):
            continue
        size = active_geometry_size(primitive_type)
        if primitive_type == TEXT_BOX and predicted_text_geometry is not None:
            predicted = predicted_text_geometry
        elif primitive_type == LINE and predicted_line_geometry is not None:
            predicted = predicted_line_geometry
        else:
            predicted = predicted_geometry
        if predicted.shape != predicted_geometry.shape:
            raise ValueError("Specialized and generic primitive geometry differ")
        variants = geometry_variants(target_geometry[indices], primitive_type)
        differences = (
            predicted[:, None, None, :size]
            - variants[None, :, :, :size]
        ).abs().mean(dim=-1)
        coordinate[:, indices] = differences.amin(dim=-1)
        predicted_shape = geometry_shape(predicted, primitive_type)
        target_shape = geometry_shape(target_geometry[indices], primitive_type)
        shape[:, indices] = torch.cdist(predicted_shape.float(), target_shape.float(), p=1) / 4.0
    return coordinate, shape


def primitive_set_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    class_cost_weight: float = 2.0,
    coordinate_cost_weight: float = 5.0,
    shape_cost_weight: float = 2.0,
    stroke_cost_weight: float = 0.5,
    line_geometry_objective: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match every target to one unique query without target-order assumptions."""

    if min(
        class_cost_weight,
        coordinate_cost_weight,
        shape_cost_weight,
        stroke_cost_weight,
    ) < 0.0:
        raise ValueError("Primitive Set matching weights must be nonnegative")
    geometry = output["geometry"].detach().float()
    line_geometry = output.get("line_geometry")
    if line_geometry is not None:
        line_geometry = line_geometry.detach().float()
    probabilities = output["type_logits"].detach().float().softmax(-1)
    strokes = output["stroke"].detach().float()
    target_mask = targets["target_mask"].bool()
    if geometry.size(0) != target_mask.size(0):
        raise ValueError("Primitive Set output and target batches differ")

    assignments = []
    for batch_index in range(geometry.size(0)):
        target_indices = torch.where(target_mask[batch_index])[0]
        if len(target_indices) > geometry.size(1):
            raise ValueError("Primitive targets exceed configured query capacity")
        if not len(target_indices):
            empty = torch.empty(0, device=geometry.device, dtype=torch.long)
            assignments.append((empty, empty))
            continue
        target_types = targets["types"][batch_index, target_indices].long()
        target_geometry = targets["geometry"][batch_index, target_indices].float()
        coordinate_cost, shape_cost = pairwise_geometry_cost(
            geometry[batch_index],
            target_geometry,
            target_types,
            output.get("text_geometry", output["geometry"])[batch_index]
            .detach()
            .float(),
            None if line_geometry is None else line_geometry[batch_index],
        )
        class_cost = -probabilities[batch_index][:, target_types]
        has_stroke = targets["has_stroke"][batch_index, target_indices].float()
        stroke_cost = (
            strokes[batch_index, :, None]
            - targets["stroke"][batch_index, target_indices].float()[None]
        ).abs() * has_stroke[None]
        cost = (
            class_cost_weight * class_cost
            + coordinate_cost_weight * coordinate_cost
            + shape_cost_weight * shape_cost
            + stroke_cost_weight * stroke_cost
        )
        if line_geometry_objective:
            line_indices = torch.where(target_types == LINE)[0]
            if len(line_indices):
                endpoint_cost, center_cost, length_cost, orientation_cost = (
                    pairwise_line_geometry_cost(
                        (
                            geometry[batch_index]
                            if line_geometry is None
                            else line_geometry[batch_index]
                        ),
                        target_geometry[line_indices],
                    )
                )
                cost[:, line_indices] = (
                    class_cost_weight * class_cost[:, line_indices]
                    + LINE_ENDPOINT_COST_WEIGHT * endpoint_cost
                    + LINE_CENTER_COST_WEIGHT * center_cost
                    + LINE_LENGTH_COST_WEIGHT * length_cost
                    + LINE_ORIENTATION_COST_WEIGHT * orientation_cost
                    + stroke_cost_weight * stroke_cost[:, line_indices]
                )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(rows, device=geometry.device, dtype=torch.long),
                target_indices[
                    torch.as_tensor(columns, device=geometry.device, dtype=torch.long)
                ],
            )
        )
    return assignments


def align_matched_geometry(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    primitive_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align line/polygon orderings to the closest equivalent target ordering."""

    aligned = torch.zeros_like(expected)
    active = torch.zeros_like(expected, dtype=torch.bool)
    for primitive_type in (LINE, TRIANGLE, QUAD, TEXT_BOX):
        indices = torch.where(primitive_types == primitive_type)[0]
        if not len(indices):
            continue
        size = active_geometry_size(primitive_type)
        variants = geometry_variants(expected[indices], primitive_type)
        differences = (
            predicted[indices, None, :size] - variants[:, :, :size]
        ).abs().mean(dim=-1)
        best = differences.argmin(dim=-1)
        selected = variants[torch.arange(len(indices), device=predicted.device), best]
        aligned[indices, :size] = selected[:, :size]
        active[indices, :size] = True
    return aligned, active


def primitive_set_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.1,
    coordinate_weight: float = 5.0,
    shape_weight: float = 2.0,
    stroke_weight: float = 0.5,
    cardinality_weight: float = 0.25,
    count_classification_weight: float = 0.25,
    class_cost_weight: float = 2.0,
    coordinate_cost_weight: float = 5.0,
    shape_cost_weight: float = 2.0,
    stroke_cost_weight: float = 0.5,
    image_size: float = 768.0,
    line_geometry_objective: bool = False,
) -> dict[str, torch.Tensor]:
    """DETR-style classification, cardinality, and matched geometry losses."""

    if not 0.0 < no_object_weight <= 1.0:
        raise ValueError("Primitive Set no-object weight must be in (0, 1]")
    if min(
        coordinate_weight,
        shape_weight,
        stroke_weight,
        cardinality_weight,
        count_classification_weight,
    ) < 0.0:
        raise ValueError("Primitive Set loss weights must be nonnegative")
    assignments = primitive_set_assignments(
        output,
        targets,
        class_cost_weight=class_cost_weight,
        coordinate_cost_weight=coordinate_cost_weight,
        shape_cost_weight=shape_cost_weight,
        stroke_cost_weight=stroke_cost_weight,
        line_geometry_objective=line_geometry_objective,
    )
    type_targets = torch.full(
        output["type_logits"].shape[:2],
        NO_OBJECT,
        device=output["type_logits"].device,
        dtype=torch.long,
    )
    predicted_geometry = []
    expected_geometry = []
    matched_types = []
    predicted_strokes = []
    expected_strokes = []
    stroke_masks = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        current_types = targets["types"][batch_index, target_indices].long()
        type_targets[batch_index, query_indices] = current_types
        current_geometry = output["geometry"][batch_index, query_indices]
        if "text_geometry" in output:
            current_geometry = torch.where(
                (current_types == TEXT_BOX)[:, None],
                output["text_geometry"][batch_index, query_indices],
                current_geometry,
            )
        if "line_geometry" in output:
            current_geometry = torch.where(
                (current_types == LINE)[:, None],
                output["line_geometry"][batch_index, query_indices],
                current_geometry,
            )
        predicted_geometry.append(current_geometry)
        expected_geometry.append(targets["geometry"][batch_index, target_indices])
        matched_types.append(current_types)
        predicted_strokes.append(output["stroke"][batch_index, query_indices])
        expected_strokes.append(targets["stroke"][batch_index, target_indices])
        stroke_masks.append(targets["has_stroke"][batch_index, target_indices])

    class_weights = output["type_logits"].new_tensor(
        (no_object_weight, 1.0, 3.0, 4.0, 1.5)
    )
    classification_loss = F.cross_entropy(
        output["type_logits"].transpose(1, 2),
        type_targets,
        weight=class_weights,
    )
    zero = classification_loss * 0.0
    if predicted_geometry:
        predicted_geometry_tensor = torch.cat(predicted_geometry)
        expected_geometry_tensor = torch.cat(expected_geometry).to(
            predicted_geometry_tensor.dtype
        )
        matched_type_tensor = torch.cat(matched_types)
        aligned, active = align_matched_geometry(
            predicted_geometry_tensor,
            expected_geometry_tensor,
            matched_type_tensor,
        )
        if line_geometry_objective:
            line_mask = matched_type_tensor == LINE
            generic_active = active & ~line_mask[:, None]
            if bool(generic_active.any()):
                pixel_errors = (
                    predicted_geometry_tensor.float() - aligned.float()
                ) * image_size
                coordinate_loss = F.smooth_l1_loss(
                    pixel_errors[generic_active],
                    torch.zeros_like(pixel_errors[generic_active]),
                    beta=4.0,
                ) / image_size
            else:
                coordinate_loss = zero
            generic_types = (TRIANGLE, QUAD, TEXT_BOX)
            predicted_shape_parts = [
                geometry_shape(
                    predicted_geometry_tensor[matched_type_tensor == primitive_type],
                    primitive_type,
                )
                for primitive_type in generic_types
                if bool((matched_type_tensor == primitive_type).any())
            ]
            expected_shape_parts = [
                geometry_shape(
                    expected_geometry_tensor[matched_type_tensor == primitive_type],
                    primitive_type,
                )
                for primitive_type in generic_types
                if bool((matched_type_tensor == primitive_type).any())
            ]
            if predicted_shape_parts:
                shape_loss = F.smooth_l1_loss(
                    torch.cat(predicted_shape_parts),
                    torch.cat(expected_shape_parts),
                )
            else:
                shape_loss = zero
            if bool(line_mask.any()):
                line_losses = line_endpoint_aligned_losses(
                    predicted_geometry_tensor[line_mask],
                    expected_geometry_tensor[line_mask],
                    image_size=image_size,
                )
            else:
                line_losses = {
                    "line_endpoint_loss": zero,
                    "line_center_loss": zero,
                    "line_length_loss": zero,
                    "line_orientation_loss": zero,
                }
        else:
            pixel_errors = (
                predicted_geometry_tensor.float() - aligned.float()
            ) * image_size
            coordinate_loss = F.smooth_l1_loss(
                pixel_errors[active],
                torch.zeros_like(pixel_errors[active]),
                beta=4.0,
            ) / image_size
            predicted_shapes = torch.cat(
                [
                    geometry_shape(
                        predicted_geometry_tensor[
                            matched_type_tensor == primitive_type
                        ],
                        primitive_type,
                    )
                    for primitive_type in (LINE, TRIANGLE, QUAD, TEXT_BOX)
                    if bool((matched_type_tensor == primitive_type).any())
                ]
            )
            expected_shapes = torch.cat(
                [
                    geometry_shape(
                        expected_geometry_tensor[
                            matched_type_tensor == primitive_type
                        ],
                        primitive_type,
                    )
                    for primitive_type in (LINE, TRIANGLE, QUAD, TEXT_BOX)
                    if bool((matched_type_tensor == primitive_type).any())
                ]
            )
            shape_loss = F.smooth_l1_loss(predicted_shapes, expected_shapes)
            line_losses = {
                "line_endpoint_loss": zero,
                "line_center_loss": zero,
                "line_length_loss": zero,
                "line_orientation_loss": zero,
            }
        predicted_stroke_tensor = torch.cat(predicted_strokes)
        expected_stroke_tensor = torch.cat(expected_strokes).to(
            predicted_stroke_tensor.dtype
        )
        stroke_mask = torch.cat(stroke_masks).bool()
        stroke_loss = (
            F.smooth_l1_loss(
                predicted_stroke_tensor[stroke_mask],
                expected_stroke_tensor[stroke_mask],
            )
            if bool(stroke_mask.any())
            else zero
        )
    else:
        coordinate_loss = shape_loss = stroke_loss = zero
        line_losses = {
            "line_endpoint_loss": zero,
            "line_center_loss": zero,
            "line_length_loss": zero,
            "line_orientation_loss": zero,
        }

    probabilities = output["type_logits"].float().softmax(-1)
    predicted_counts = (1.0 - probabilities[..., NO_OBJECT]).sum(dim=1)
    expected_counts = targets["target_mask"].sum(dim=1).float()
    cardinality_loss = F.smooth_l1_loss(predicted_counts, expected_counts) / max(
        output["type_logits"].size(1), 1
    )
    count_logits = output.get("count_logits")
    if count_logits is None:
        count_classification_loss = zero
    else:
        expected_shape = (
            output["type_logits"].size(0),
            output["type_logits"].size(1) + 1,
        )
        if count_logits.shape != expected_shape:
            raise ValueError(
                "Primitive Set count logits must have shape "
                f"{expected_shape}, received {tuple(count_logits.shape)}"
            )
        count_classification_loss = F.cross_entropy(
            count_logits.float(),
            expected_counts.long(),
        )
    total = (
        classification_loss
        + coordinate_weight * coordinate_loss
        + shape_weight * shape_loss
        + stroke_weight * stroke_loss
        + cardinality_weight * cardinality_loss
        + count_classification_weight * count_classification_loss
    )
    if line_geometry_objective:
        total = (
            total
            + LINE_ENDPOINT_LOSS_WEIGHT * line_losses["line_endpoint_loss"]
            + LINE_CENTER_LOSS_WEIGHT * line_losses["line_center_loss"]
            + LINE_LENGTH_LOSS_WEIGHT * line_losses["line_length_loss"]
            + LINE_ORIENTATION_LOSS_WEIGHT * line_losses["line_orientation_loss"]
        )
    return {
        "loss": total,
        "classification_loss": classification_loss,
        "coordinate_loss": coordinate_loss,
        "shape_loss": shape_loss,
        "stroke_loss": stroke_loss,
        "cardinality_loss": cardinality_loss,
        "count_loss": count_classification_loss,
        "count_classification_loss": count_classification_loss,
        **line_losses,
        "predicted_count_mean": predicted_counts.mean(),
        "target_count_mean": expected_counts.mean(),
        "assignments": assignments,
    }


@torch.no_grad()
def decode_primitive_set(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.5,
    mode: str = "threshold",
) -> list[list[dict[str, torch.Tensor | float | int]]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Primitive Set threshold must be in [0, 1]")
    if mode not in {"threshold", "count_head"}:
        raise ValueError("Primitive Set decode mode must be 'threshold' or 'count_head'")
    probabilities = output["type_logits"].float().softmax(-1)
    object_scores, object_types = probabilities[..., 1:].max(dim=-1)
    object_types = object_types + 1
    no_object_scores = probabilities[..., NO_OBJECT]
    count_logits = output.get("count_logits")
    if mode == "count_head":
        expected_shape = (probabilities.size(0), probabilities.size(1) + 1)
        if count_logits is None or count_logits.shape != expected_shape:
            received = None if count_logits is None else tuple(count_logits.shape)
            raise ValueError(
                "count_head decoding requires count logits with shape "
                f"{expected_shape}, received {received}"
            )
        predicted_counts = count_logits.float().argmax(dim=-1)
    decoded = []
    for batch_index in range(probabilities.size(0)):
        if mode == "threshold":
            keep = (object_scores[batch_index] >= threshold) & (
                object_scores[batch_index] > no_object_scores[batch_index]
            )
            query_indices = torch.where(keep)[0].tolist()
        else:
            count = int(predicted_counts[batch_index])
            ranking = object_scores[batch_index] - no_object_scores[batch_index]
            query_indices = torch.argsort(
                ranking,
                descending=True,
                stable=True,
            )[:count].tolist()
        rows = []
        for query_index in query_indices:
            primitive_type = int(object_types[batch_index, query_index])
            if primitive_type == TEXT_BOX:
                geometry = output.get("text_geometry", output["geometry"])[
                    batch_index, query_index
                ].detach()
            elif primitive_type == LINE and "line_geometry" in output:
                geometry = output["line_geometry"][batch_index, query_index].detach()
            else:
                geometry = output["geometry"][batch_index, query_index].detach()
            rows.append(
                {
                    "query_index": query_index,
                    "score": float(object_scores[batch_index, query_index]),
                    "type": primitive_type,
                    "geometry": geometry,
                    "stroke": float(output["stroke"][batch_index, query_index]),
                }
            )
        rows.sort(key=lambda row: (-float(row["score"]), int(row["query_index"])))
        decoded.append(rows)
    return decoded
