"""Image preparation and tensor conversion shared by inference pipelines."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label
import torch


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def tensor_to_image(pixel_values: torch.Tensor) -> Image.Image:
    array = (
        (pixel_values.detach().cpu().float().clamp(-1, 1) * 0.5 + 0.5)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def prepare_input_image(
    image: Image.Image,
    *,
    image_size: int = 768,
    content_size: int = 548,
) -> Image.Image:
    """Remove a page frame, preserve aspect ratio, and center on a white canvas."""

    if not 1 <= content_size <= image_size:
        raise ValueError("content_size must be within the output canvas")
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    source = np.asarray(white.convert("RGB")).copy()
    dark = source.min(axis=2) < 160
    components, _ = label(dark)
    for component_id, slices in enumerate(find_objects(components), 1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        width_fraction = (x_slice.stop - x_slice.start) / source.shape[1]
        height_fraction = (y_slice.stop - y_slice.start) / source.shape[0]
        touches_boundary = (
            x_slice.start == 0
            or y_slice.start == 0
            or x_slice.stop == source.shape[1]
            or y_slice.stop == source.shape[0]
        )
        if touches_boundary and width_fraction >= 0.9 and height_fraction >= 0.9:
            source[components == component_id] = 255
    cleaned = Image.fromarray(source)
    scale = min(content_size / cleaned.width, content_size / cleaned.height)
    resized_size = (
        max(1, round(cleaned.width * scale)),
        max(1, round(cleaned.height * scale)),
    )
    resized = cleaned.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), "white")
    canvas.paste(
        resized,
        (
            (image_size - resized.width) // 2,
            (image_size - resized.height) // 2,
        ),
    )
    return canvas
