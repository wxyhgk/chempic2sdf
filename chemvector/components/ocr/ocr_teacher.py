#!/usr/bin/env python3
"""Generate OCR teacher labels for geometry-provided text anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


SCHEMA_VERSION = "chemocr-teacher-0.1"
DEFAULT_MODEL = "en_PP-OCRv5_mobile_rec"


def load_rgb_white(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def prepare_anchor_crop(
    image: Image.Image,
    x: float,
    y: float,
    *,
    crop_size: int = 28,
    padding: int = 12,
    output_size: tuple[int, int] = (192, 96),
) -> np.ndarray:
    if crop_size < 1 or padding < 0:
        raise ValueError("crop_size must be positive and padding must be nonnegative")
    half = crop_size / 2
    crop = image.crop((x - half, y - half, x + half, y + half))
    crop = ImageOps.expand(crop, border=padding, fill="white")
    crop = crop.resize(output_size, Image.Resampling.LANCZOS)
    return np.asarray(crop)


def load_anchors(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    anchors = payload["anchors"] if isinstance(payload, dict) else payload
    if not isinstance(anchors, list):
        raise ValueError("Anchor JSON must be a list or contain an anchors list")
    normalized = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, dict) or "x" not in anchor or "y" not in anchor:
            raise ValueError(f"Anchor {index} must contain numeric x and y values")
        normalized.append(
            {
                "id": anchor.get("id", index),
                "x": float(anchor["x"]),
                "y": float(anchor["y"]),
            }
        )
    return normalized


def recognize_anchors(
    image_path: str | Path,
    anchors: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
    crop_size: int = 28,
    padding: int = 12,
    batch_size: int = 32,
    device: str = "cpu",
    enable_mkldnn: bool = False,
) -> dict[str, Any]:
    try:
        from paddleocr import TextRecognition
    except ImportError as error:
        raise RuntimeError(
            "PaddleOCR 3.x is required; run this tool with .venv-ocr/bin/python"
        ) from error

    image = load_rgb_white(image_path)
    crops = [
        prepare_anchor_crop(
            image,
            anchor["x"],
            anchor["y"],
            crop_size=crop_size,
            padding=padding,
        )
        for anchor in anchors
    ]
    recognizer = TextRecognition(
        model_name=model_name,
        device=device,
        enable_mkldnn=enable_mkldnn,
    )
    results = list(recognizer.predict(crops, batch_size=batch_size)) if crops else []
    predictions = []
    for anchor, result in zip(anchors, results):
        payload = result.json["res"]
        predictions.append(
            {
                **anchor,
                "raw_text": payload["rec_text"],
                "score": round(float(payload["rec_score"]), 8),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "image": str(image_path),
        "model": model_name,
        "crop_size": crop_size,
        "padding": padding,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--anchors-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--crop-size", type=int, default=28)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable-mkldnn", action="store_true")
    args = parser.parse_args()

    output = recognize_anchors(
        args.image,
        load_anchors(args.anchors_json),
        model_name=args.model,
        crop_size=args.crop_size,
        padding=args.padding,
        batch_size=args.batch_size,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
