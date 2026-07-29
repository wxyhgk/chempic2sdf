"""Small byte-level CTC recognizer for cropped chemical labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


CTC_BLANK_ID = 0
CTC_VOCAB_SIZE = 257


def bbox_crop(
    image: Image.Image,
    bbox: list[float] | tuple[float, float, float, float],
    *,
    context: float,
    jitter_x: float = 0.0,
    jitter_y: float = 0.0,
    output_size: tuple[int, int] = (192, 48),
) -> Image.Image:
    """Crop a text box and preserve its aspect ratio on a white canvas."""

    x0, y0, x1, y1 = (float(value) for value in bbox)
    crop = image.crop(
        (
            x0 - context + jitter_x,
            y0 - context + jitter_y,
            x1 + context + jitter_x,
            y1 + context + jitter_y,
        )
    ).convert("RGB")
    maximum_width, maximum_height = output_size
    scale = min(maximum_width / max(crop.width, 1), maximum_height / max(crop.height, 1))
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", output_size, "white")
    canvas.paste(
        resized,
        ((maximum_width - resized.width) // 2, (maximum_height - resized.height) // 2),
    )
    return canvas


def detected_text_crops(
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    text_objects: list[dict[str, float]],
    *,
    text_role: int,
    context: float = 4.0,
) -> tuple[torch.Tensor, list[int]]:
    """Match selected text vertices to detected extents and build OCR crops."""

    if pixels.ndim != 3 or pixels.size(0) != 3:
        raise ValueError("Single-image pixels must have shape [3, height, width]")
    if points.ndim != 2 or points.size(-1) != 2 or roles.shape != points.shape[:1]:
        raise ValueError("Detected OCR points and roles have incompatible shapes")
    text_indices = [index for index, role in enumerate(roles.tolist()) if role == text_role]
    if not text_indices or not text_objects:
        return pixels.new_empty((0, 3, 48, 192)), []

    candidates = []
    for point_index in text_indices:
        for object_index, item in enumerate(text_objects):
            distance = (
                (float(points[point_index, 0]) - float(item["x"])) ** 2
                + (float(points[point_index, 1]) - float(item["y"])) ** 2
            )
            candidates.append((distance, point_index, object_index))
    assignments: dict[int, int] = {}
    assigned_objects = set()
    for _, point_index, object_index in sorted(candidates):
        if point_index in assignments or object_index in assigned_objects:
            continue
        assignments[point_index] = object_index
        assigned_objects.add(object_index)

    array = (
        ((pixels.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    height, width = pixels.shape[-2:]
    crops = []
    matched_indices = []
    for point_index in text_indices:
        object_index = assignments.get(point_index)
        if object_index is None:
            continue
        item = text_objects[object_index]
        half_width = max(float(item["half_width"]) * width, 1.0)
        half_height = max(float(item["half_height"]) * height, 1.0)
        center_x = float(item.get("bbox_x", item["x"])) * width
        center_y = float(item.get("bbox_y", item["y"])) * height
        crop = bbox_crop(
            image,
            (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            ),
            context=context,
        )
        crop_array = np.asarray(crop, dtype=np.float32) / 255.0
        crops.append((torch.from_numpy(crop_array).permute(2, 0, 1) - 0.5) / 0.5)
        matched_indices.append(point_index)
    return torch.stack(crops).to(device=pixels.device), matched_indices


@dataclass(frozen=True, slots=True)
class ChemicalCTCConfig:
    channels: int = 128
    hidden_dim: int = 192
    layers: int = 2
    heads: int = 4
    maximum_bytes: int = 24

    def __post_init__(self) -> None:
        if min(self.channels, self.hidden_dim, self.layers, self.heads) < 1:
            raise ValueError("CTC OCR dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("CTC hidden dimension must be divisible by heads")


def encode_text(text: str, maximum_bytes: int = 24) -> torch.Tensor:
    values = text.encode("utf-8")
    if len(values) > maximum_bytes:
        raise ValueError("CTC target exceeds maximum byte length")
    return torch.tensor([value + 1 for value in values], dtype=torch.long)


def decode_ctc(ids: torch.Tensor) -> str:
    values = []
    previous = CTC_BLANK_ID
    for value in ids.tolist():
        value = int(value)
        if value != CTC_BLANK_ID and value != previous:
            values.append(value - 1)
        previous = value
    return bytes(values).decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class CTCBeamCandidate:
    """One unique decoded text candidate and its CTC sequence log score."""

    text: str
    log_score: float


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def decode_ctc_prefix_beam(
    logits: torch.Tensor,
    *,
    beam_size: int = 10,
    top_paths: int | None = None,
    maximum_bytes: int | None = None,
    token_beam: int | None = None,
) -> list[CTCBeamCandidate]:
    """Decode byte-level CTC logits with prefix beam search on the CPU.

    Scores are unnormalized natural-log sequence probabilities, summed over
    every CTC alignment retained by the beam. Distinct byte sequences that
    decode to the same UTF-8 text are merged before candidates are returned.
    """

    if logits.ndim != 2 or logits.size(-1) != CTC_VOCAB_SIZE:
        raise ValueError("Single CTC logits must have shape [time, vocabulary]")
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if top_paths is None:
        top_paths = beam_size
    if top_paths < 1 or top_paths > beam_size:
        raise ValueError("top_paths must be between one and beam_size")
    if maximum_bytes is not None and maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive when provided")
    if token_beam is not None and not 1 <= token_beam <= CTC_VOCAB_SIZE:
        raise ValueError("token_beam must lie between one and the vocabulary size")

    log_probabilities = logits.detach().float().cpu().log_softmax(-1)
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, -math.inf)
    }
    for tensor_row in log_probabilities:
        if token_beam is None:
            token_ids = range(1, CTC_VOCAB_SIZE)
        else:
            retained = tensor_row[1:].topk(min(token_beam, CTC_VOCAB_SIZE - 1)).indices
            token_ids = (retained + 1).tolist()
        row = tensor_row.tolist()
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (blank_score, nonblank_score) in beams.items():
            total_score = _log_add(blank_score, nonblank_score)

            next_blank, next_nonblank = next_beams.get(
                prefix, (-math.inf, -math.inf)
            )
            next_blank = _log_add(
                next_blank,
                total_score + row[CTC_BLANK_ID],
            )
            next_beams[prefix] = (next_blank, next_nonblank)

            can_extend = maximum_bytes is None or len(prefix) < maximum_bytes
            final_token = prefix[-1] if prefix else None
            for token_id in token_ids:
                token_score = row[token_id]
                if token_id == final_token:
                    same_blank, same_nonblank = next_beams[prefix]
                    same_nonblank = _log_add(
                        same_nonblank,
                        nonblank_score + token_score,
                    )
                    next_beams[prefix] = (same_blank, same_nonblank)
                    if not can_extend:
                        continue
                    extended = prefix + (token_id,)
                    extended_blank, extended_nonblank = next_beams.get(
                        extended, (-math.inf, -math.inf)
                    )
                    extended_nonblank = _log_add(
                        extended_nonblank,
                        blank_score + token_score,
                    )
                    next_beams[extended] = (extended_blank, extended_nonblank)
                    continue

                if not can_extend:
                    continue
                extended = prefix + (token_id,)
                extended_blank, extended_nonblank = next_beams.get(
                    extended, (-math.inf, -math.inf)
                )
                extended_nonblank = _log_add(
                    extended_nonblank,
                    total_score + token_score,
                )
                next_beams[extended] = (extended_blank, extended_nonblank)

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _log_add(*item[1]),
            reverse=True,
        )[:beam_size]
        beams = dict(ranked)

    text_scores: dict[str, float] = {}
    for prefix, (blank_score, nonblank_score) in beams.items():
        text = bytes(token_id - 1 for token_id in prefix).decode(
            "utf-8", errors="replace"
        )
        score = _log_add(blank_score, nonblank_score)
        text_scores[text] = _log_add(text_scores.get(text, -math.inf), score)
    return [
        CTCBeamCandidate(text=text, log_score=score)
        for text, score in sorted(
            text_scores.items(), key=lambda item: item[1], reverse=True
        )[:top_paths]
    ]


def ctc_text_log_score(
    logits: torch.Tensor,
    text: str,
    *,
    normalize_by_bytes: bool = False,
) -> float:
    """Return the exact CTC log probability assigned to one UTF-8 string."""

    if logits.ndim != 2 or logits.size(-1) != CTC_VOCAB_SIZE:
        raise ValueError("Single CTC logits must have shape [time, vocabulary]")
    target = encode_text(text)
    if not len(target):
        score = float(logits.log_softmax(-1)[:, CTC_BLANK_ID].sum())
        return score
    loss = nn.functional.ctc_loss(
        logits.log_softmax(-1).unsqueeze(1),
        target.to(logits.device),
        torch.tensor([logits.size(0)], device=logits.device),
        torch.tensor([len(target)], device=logits.device),
        blank=CTC_BLANK_ID,
        reduction="sum",
        zero_infinity=True,
    )
    score = -float(loss)
    return score / len(target) if normalize_by_bytes else score


def decode_ctc_with_score(logits: torch.Tensor) -> tuple[str, float]:
    """Decode one CTC sequence and return normalized sequence log confidence."""

    if logits.ndim != 2 or logits.size(-1) != CTC_VOCAB_SIZE:
        raise ValueError("Single CTC logits must have shape [time, vocabulary]")
    text = decode_ctc(logits.argmax(-1).detach().cpu())
    return text, ctc_text_log_score(logits, text, normalize_by_bytes=True)


class ChemicalCTCOCR(nn.Module):
    def __init__(self, config: ChemicalCTCConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChemicalCTCConfig()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.config.channels, 3, stride=(2, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.channels, self.config.channels, 3, padding=1),
            nn.GELU(),
        )
        self.input_projection = nn.Linear(self.config.channels, self.config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=self.config.layers,
            norm=nn.LayerNorm(self.config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(self.config.hidden_dim, CTC_VOCAB_SIZE)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.size(1) != 3:
            raise ValueError("CTC OCR pixels must have shape [batch, 3, height, width]")
        features = self.encoder(pixels).mean(dim=2).transpose(1, 2)
        hidden = self.context(self.input_projection(features))
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class PretrainedPixelCTCConfig:
    d_model: int = 192
    layers: int = 1
    heads: int = 4
    maximum_bytes: int = 24
    maximum_time_steps: int = 64
    freeze_pixel_encoder: bool = True

    def __post_init__(self) -> None:
        if min(
            self.d_model,
            self.layers,
            self.heads,
            self.maximum_bytes,
            self.maximum_time_steps,
        ) < 1:
            raise ValueError("Pretrained CTC OCR dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Pretrained CTC hidden dimension must be divisible by heads")


class PretrainedPixelCTCOCR(nn.Module):
    """CTC recognizer reusing the trained v0.8.6 pixel stroke encoder."""

    def __init__(self, config: PretrainedPixelCTCConfig | None = None) -> None:
        super().__init__()
        self.config = config or PretrainedPixelCTCConfig()
        pixel_channels = max(self.config.d_model // 2, 32)
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(3, pixel_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                pixel_channels,
                pixel_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                pixel_channels,
                self.config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )
        self.position_embedding = nn.Embedding(
            self.config.maximum_time_steps,
            self.config.d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=self.config.layers,
            norm=nn.LayerNorm(self.config.d_model),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(self.config.d_model, CTC_VOCAB_SIZE)
        self.pixel_encoder_loaded = False
        self.pixel_encoder_frozen = False
        if self.config.freeze_pixel_encoder:
            self.freeze_pixel_encoder()

    def freeze_pixel_encoder(self) -> None:
        for parameter in self.pixel_encoder.parameters():
            parameter.requires_grad_(False)
        self.pixel_encoder_frozen = True

    def unfreeze_pixel_encoder(self) -> None:
        for parameter in self.pixel_encoder.parameters():
            parameter.requires_grad_(True)
        self.pixel_encoder_frozen = False

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def load_legacy_checkpoint(self, checkpoint: str | Path) -> dict[str, int]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        prefix = "ocr_head.pixel_encoder."
        pixel_state = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if not pixel_state:
            raise ValueError("Checkpoint has no OCR pixel encoder weights")
        self.pixel_encoder.load_state_dict(pixel_state, strict=True)
        self.pixel_encoder_loaded = True

        output_key = "ocr_head.output_projection.weight"
        initialized_bytes = 0
        if output_key in state:
            legacy_output = state[output_key]
            if legacy_output.ndim != 2 or legacy_output.size(1) != self.config.d_model:
                raise ValueError("Legacy OCR output projection shape is incompatible")
            with torch.no_grad():
                byte_count = min(256, legacy_output.size(0) - 3)
                self.output.weight[1 : byte_count + 1].copy_(
                    legacy_output[3 : byte_count + 3]
                )
                initialized_bytes = byte_count
        return {
            "pixel_tensors": len(pixel_state),
            "initialized_bytes": initialized_bytes,
        }

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.size(1) != 3:
            raise ValueError("CTC OCR pixels must have shape [batch, 3, height, width]")
        if not self.pixel_encoder_loaded:
            raise RuntimeError("Pretrained pixel encoder checkpoint has not been loaded")
        if self.pixel_encoder_frozen:
            with torch.no_grad():
                features = self.pixel_encoder(pixels)
        else:
            features = self.pixel_encoder(pixels)
        sequence = features.mean(dim=2).transpose(1, 2)
        if sequence.size(1) > self.config.maximum_time_steps:
            raise ValueError("Pixel feature sequence exceeds maximum_time_steps")
        positions = torch.arange(sequence.size(1), device=sequence.device)
        hidden = self.context(sequence + self.position_embedding(positions).unsqueeze(0))
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class ConditionedPixelCTCConfig:
    condition_channels: int = 0
    d_model: int = 192
    layers: int = 1
    heads: int = 4
    maximum_bytes: int = 24
    maximum_time_steps: int = 64
    height_attention: bool = False

    def __post_init__(self) -> None:
        if self.condition_channels < 0:
            raise ValueError("condition_channels must be nonnegative")
        if min(
            self.d_model,
            self.layers,
            self.heads,
            self.maximum_bytes,
            self.maximum_time_steps,
        ) < 1:
            raise ValueError("Conditioned CTC OCR dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Conditioned CTC hidden dimension must be divisible by heads")


class ConditionedPixelCTCOCR(nn.Module):
    """Oracle probe that adds mask/anchor evidence to a frozen RGB encoder."""

    def __init__(self, config: ConditionedPixelCTCConfig) -> None:
        super().__init__()
        self.config = config
        pixel_channels = max(config.d_model // 2, 32)
        self.condition_adapter = (
            nn.Conv2d(config.condition_channels, 3, 1, bias=False)
            if config.condition_channels
            else None
        )
        if self.condition_adapter is not None:
            nn.init.zeros_(self.condition_adapter.weight)
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(3, pixel_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                pixel_channels,
                pixel_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                pixel_channels,
                config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )
        self.height_score = (
            nn.Conv2d(config.d_model, 1, 1)
            if config.height_attention
            else None
        )
        if self.height_score is not None:
            nn.init.zeros_(self.height_score.weight)
            nn.init.zeros_(self.height_score.bias)
        self.position_embedding = nn.Embedding(
            config.maximum_time_steps, config.d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(config.d_model, CTC_VOCAB_SIZE)
        self.pixel_encoder.requires_grad_(False)

    def load_pretrained_ctc_checkpoint(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["pretrained_ctc_ocr"]
        compatible = {
            key: value
            for key, value in state.items()
            if key in self.state_dict() and self.state_dict()[key].shape == value.shape
        }
        incompatible = self.load_state_dict(compatible, strict=False)
        allowed_missing = set()
        if self.condition_adapter is not None:
            allowed_missing.add("condition_adapter.weight")
        if self.height_score is not None:
            allowed_missing.update(("height_score.weight", "height_score.bias"))
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Conditioned OCR initialization is incompatible: "
                f"missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.pixel_encoder.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected_channels = 3 + self.config.condition_channels
        if inputs.ndim != 4 or inputs.size(1) != expected_channels:
            raise ValueError(
                f"Conditioned OCR inputs must have {expected_channels} channels"
            )
        rgb = inputs[:, :3]
        if self.condition_adapter is not None:
            rgb = rgb + self.condition_adapter(inputs[:, 3:])
        features = self.pixel_encoder(rgb)
        if self.height_score is None:
            sequence = features.mean(dim=2).transpose(1, 2)
        else:
            weights = self.height_score(features).softmax(dim=2)
            sequence = (features * weights).sum(dim=2).transpose(1, 2)
        if sequence.size(1) > self.config.maximum_time_steps:
            raise ValueError("Pixel feature sequence exceeds maximum_time_steps")
        positions = torch.arange(sequence.size(1), device=sequence.device)
        hidden = self.context(
            sequence + self.position_embedding(positions).unsqueeze(0)
        )
        return self.output(hidden)


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    input_lengths = torch.full(
        (logits.size(0),), logits.size(1), device=logits.device, dtype=torch.long
    )
    return nn.functional.ctc_loss(
        logits.log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=CTC_BLANK_ID,
        zero_infinity=True,
    )
