#!/usr/bin/env python3
"""Generate clean chemical-label crops for OCR pretraining."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import string

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from chemvector.ocr import OCRTextCodec


FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
)

STANDARD_LABELS = (
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "Se",
    "Si",
    "NH",
    "NH2",
    "NH+",
    "N+",
    "N-",
    "OH",
    "O-",
    "O+",
    "SH",
    "S+",
    "BH",
    "BH2",
    "B-",
)

CHEMICAL_ALIASES = (
    "R",
    "R1",
    "R2",
    "R3",
    "X",
    "Y",
    "Z",
    "A",
    "M",
    "Ar",
    "Het",
    "Ph",
    "Me",
    "Et",
    "Ac",
    "Boc",
    "tBu",
    "Ts",
    "n",
    "m",
    "x",
    "*",
    "?",
    "[M]",
    "R(n)",
)

RANDOM_CHARACTERS = string.ascii_letters + string.digits + "+-()[]*'"


@dataclass(frozen=True, slots=True)
class SyntheticOCRConfig:
    canvas_size: int = 128
    roi_radius: int = 24
    minimum_font_size: int = 10
    maximum_font_size: int = 20
    blank_probability: float = 0.20
    standard_probability: float = 0.35
    alias_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.canvas_size < self.roi_radius * 2:
            raise ValueError("Canvas must contain the complete OCR ROI")
        if not 0.0 <= self.blank_probability <= 1.0:
            raise ValueError("blank_probability must be in [0, 1]")
        if self.blank_probability + self.standard_probability + self.alias_probability > 1:
            raise ValueError("Synthetic OCR category probabilities exceed 1")


def available_fonts(paths: tuple[str, ...] = FONT_PATHS) -> tuple[str, ...]:
    fonts = tuple(path for path in paths if Path(path).is_file())
    if not fonts:
        raise FileNotFoundError("No supported OCR training fonts were found")
    return fonts


def sample_label(rng: random.Random, config: SyntheticOCRConfig) -> tuple[str, str]:
    value = rng.random()
    if value < config.blank_probability:
        return "", "blank"
    value -= config.blank_probability
    if value < config.standard_probability:
        return rng.choice(STANDARD_LABELS), "standard"
    value -= config.standard_probability
    if value < config.alias_probability:
        return rng.choice(CHEMICAL_ALIASES), "alias"
    length = rng.randint(1, 4)
    return "".join(rng.choice(RANDOM_CHARACTERS) for _ in range(length)), "random"


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    font_size: int,
    maximum_width: int,
    minimum_size: int,
) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    size = font_size
    while True:
        font = ImageFont.truetype(font_path, size=size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= maximum_width or size <= minimum_size:
            return font, box
        size -= 1


def _draw_bond(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    order: int,
    width: int,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(math.hypot(dx, dy), 1.0)
    offset_x = -dy / length * 2.2
    offset_y = dx / length * 2.2
    offsets = {
        1: (0.0,),
        2: (-0.5, 0.5),
        3: (-1.0, 0.0, 1.0),
    }[order]
    for multiplier in offsets:
        shift_x = offset_x * multiplier
        shift_y = offset_y * multiplier
        draw.line(
            (
                start[0] + shift_x,
                start[1] + shift_y,
                end[0] + shift_x,
                end[1] + shift_y,
            ),
            fill=0,
            width=width,
        )


def _draw_bonds(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    center: tuple[float, float],
    text_size: tuple[int, int],
    blank: bool,
) -> None:
    half_width = max(text_size[0] / 2, 2)
    half_height = max(text_size[1] / 2, 2)
    base_angle = rng.uniform(0, 2 * math.pi)
    bond_count = rng.randint(1, 4)
    for index in range(bond_count):
        angle = base_angle + index * 2 * math.pi / bond_count + rng.uniform(-0.22, 0.22)
        unit_x = math.cos(angle)
        unit_y = math.sin(angle)
        if blank:
            start_distance = 0.0
        else:
            horizontal = half_width / max(abs(unit_x), 1e-3)
            vertical = half_height / max(abs(unit_y), 1e-3)
            start_distance = min(horizontal, vertical) + rng.uniform(1.5, 3.5)
        end_distance = start_distance + rng.uniform(18.0, 36.0)
        start = (
            center[0] + unit_x * start_distance,
            center[1] + unit_y * start_distance,
        )
        end = (
            center[0] + unit_x * end_distance,
            center[1] + unit_y * end_distance,
        )
        order = rng.choices((1, 2, 3), weights=(0.75, 0.20, 0.05), k=1)[0]
        _draw_bond(draw, start, end, order, rng.choice((1, 2)))


def render_synthetic_ocr_sample(
    text: str,
    *,
    rng: random.Random,
    config: SyntheticOCRConfig,
    font_paths: tuple[str, ...] | None = None,
) -> Image.Image:
    fonts = font_paths or available_fonts()
    image = Image.new("L", (config.canvas_size, config.canvas_size), color=255)
    draw = ImageDraw.Draw(image)
    center = config.canvas_size / 2
    jittered_center = (
        center + rng.uniform(-2.0, 2.0),
        center + rng.uniform(-2.0, 2.0),
    )
    text_width = text_height = 0
    font = None
    if text:
        font, box = _fit_font(
            draw,
            text,
            rng.choice(fonts),
            rng.randint(config.minimum_font_size, config.maximum_font_size),
            config.roi_radius * 2 - 4,
            config.minimum_font_size,
        )
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
    _draw_bonds(
        draw,
        rng,
        jittered_center,
        (text_width, text_height),
        blank=not text,
    )
    if text and font is not None:
        draw.text(jittered_center, text, fill=0, font=font, anchor="mm")
    return image.convert("RGB")


class SyntheticOCRDataset(Dataset):
    def __init__(
        self,
        samples: int,
        *,
        seed: int = 20260714,
        config: SyntheticOCRConfig | None = None,
    ) -> None:
        if samples < 1:
            raise ValueError("Synthetic OCR dataset must contain at least one sample")
        self.samples = samples
        self.seed = seed
        self.config = config or SyntheticOCRConfig()
        self.font_paths = available_fonts()

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        rng = random.Random(self.seed + index * 1_000_003)
        text, category = sample_label(rng, self.config)
        image = render_synthetic_ocr_sample(
            text,
            rng=rng,
            config=self.config,
            font_paths=self.font_paths,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1)
        return {
            "pixel_values": (pixel_values - 0.5) / 0.5,
            "text": text,
            "category": category,
        }


def collate_synthetic_ocr(
    samples: list[dict],
    codec: OCRTextCodec,
) -> dict[str, torch.Tensor | list[str]]:
    encoded = [codec.encode(sample["text"]) for sample in samples]
    maximum_length = max(len(item) for item in encoded)
    token_ids = torch.full(
        (len(samples), 1, maximum_length),
        codec.PAD_TOKEN_ID,
        dtype=torch.long,
    )
    for index, item in enumerate(encoded):
        token_ids[index, 0, : len(item)] = torch.tensor(item, dtype=torch.long)
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "ocr_anchor_points": torch.full((len(samples), 1, 2), 0.5),
        "ocr_anchor_mask": torch.ones((len(samples), 1), dtype=torch.bool),
        "ocr_token_ids": token_ids,
        "texts": [sample["text"] for sample in samples],
        "categories": [sample["category"] for sample in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/synthetic-ocr-preview")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    dataset = SyntheticOCRDataset(args.samples, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        sample = dataset[index]
        image = ((sample["pixel_values"] * 0.5 + 0.5) * 255).byte()
        image = Image.fromarray(image.permute(1, 2, 0).numpy())
        safe_text = sample["text"].replace("/", "_") or "blank"
        image.save(output_dir / f"{index:04d}-{sample['category']}-{safe_text}.png")


if __name__ == "__main__":
    main()
