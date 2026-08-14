"""Deterministic rasterization and production-input hashing for micro-scenes."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from chemvector.core.image_io import image_tensor, prepare_input_image

from .finite_micro_scene_v0_1 import CANVAS_SIZE, Scene


LINE_WIDTH = 3
PRODUCTION_IMAGE_SIZE = 768
PRODUCTION_CONTENT_SIZE = 548


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_scene(scene: Scene) -> Image.Image:
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
    drawing = ImageDraw.Draw(image)
    for bond in scene.bonds:
        for member in bond.members:
            drawing.line(
                (*member.begin, *member.end),
                fill=(0, 0, 0, 255),
                width=LINE_WIDTH,
            )
    return image


def raw_raster_digest(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    header = json.dumps(
        {"mode": "RGBA", "shape": [rgba.height, rgba.width, 4]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256(header + b"\0" + rgba.tobytes())


def prepare_production_image(image: Image.Image) -> Image.Image:
    return prepare_input_image(
        image,
        image_size=PRODUCTION_IMAGE_SIZE,
        content_size=PRODUCTION_CONTENT_SIZE,
    )


def production_tensor_digest(
    image: Image.Image,
    *,
    image_is_prepared: bool = False,
) -> tuple[str, dict[str, Any]]:
    prepared = image.convert("RGB") if image_is_prepared else prepare_production_image(image)
    if prepared.size != (PRODUCTION_IMAGE_SIZE, PRODUCTION_IMAGE_SIZE):
        raise ValueError(
            "A prepared production image must be exactly "
            f"{PRODUCTION_IMAGE_SIZE}x{PRODUCTION_IMAGE_SIZE}"
        )
    tensor = image_tensor(prepared).detach().cpu().contiguous()
    array = np.asarray(tensor.numpy(), dtype="<f4", order="C")
    header_record = {
        "coordinate_order": "CHW",
        "dtype": "float32-le",
        "normalization": "RGB/255 -> (x-0.5)/0.5",
        "shape": list(array.shape),
    }
    header = json.dumps(header_record, sort_keys=True, separators=(",", ":")).encode("ascii")
    return _sha256(header + b"\0" + array.tobytes(order="C")), header_record


def observation_for_scene(
    scene: Scene,
    *,
    image_is_prepared: bool = False,
) -> dict[str, Any]:
    raw = render_scene(scene)
    if image_is_prepared:
        prepared = prepare_production_image(raw)
        tensor_digest, tensor_header = production_tensor_digest(
            prepared, image_is_prepared=True
        )
    else:
        tensor_digest, tensor_header = production_tensor_digest(raw)
    return {
        "raw_raster_sha256": raw_raster_digest(raw),
        "production_tensor_sha256": tensor_digest,
        "tensor_header": tensor_header,
    }


def preprocessing_source_audit() -> dict[str, str]:
    source_path = Path(inspect.getsourcefile(prepare_input_image) or "").resolve()
    if not source_path.is_file():
        raise RuntimeError("Could not locate chemvector.core.image_io source")
    repository_root = Path(__file__).resolve().parents[3]
    try:
        relative_source = source_path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise RuntimeError("Production image source is outside the repository") from error
    return {
        "source_path": relative_source,
        "source_sha256": _sha256(source_path.read_bytes()),
        "prepare_input_image_sha256": _sha256(
            inspect.getsource(prepare_input_image).encode("utf-8")
        ),
        "image_tensor_sha256": _sha256(inspect.getsource(image_tensor).encode("utf-8")),
    }


__all__ = [
    "LINE_WIDTH",
    "PRODUCTION_CONTENT_SIZE",
    "PRODUCTION_IMAGE_SIZE",
    "observation_for_scene",
    "prepare_production_image",
    "preprocessing_source_audit",
    "production_tensor_digest",
    "raw_raster_digest",
    "render_scene",
]
