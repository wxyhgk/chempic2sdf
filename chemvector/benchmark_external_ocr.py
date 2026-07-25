#!/usr/bin/env python3
"""Prepare and evaluate external OCR recognizers on chemical text crops."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps


def crop_text(image: Image.Image, center: list[float], crop_size: int) -> Image.Image:
    half = crop_size / 2.0
    x, y = center
    crop = image.crop((x - half, y - half, x + half, y + half))
    crop = ImageOps.expand(crop, border=12, fill="white")
    return crop.resize((192, 96), Image.Resampling.LANCZOS)


def prepare(args: argparse.Namespace) -> None:
    from chemvector.data import ChemVectorIterableDataset
    from chemvector.train import expand_shards

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    dataset = ChemVectorIterableDataset(
        expand_shards(args.shards), sequence_version="0.4"
    )
    for sample in islice(iter(dataset), args.samples):
        image = sample["image"].convert("RGB")
        for atom in sample["layers"]["atoms"]:
            if not atom["visible_text"] or not str(atom.get("text", "")):
                continue
            truth = str(atom["text"])
            for crop_size in args.crop_sizes:
                key = f"{sample['key']}-a{int(atom['atom_id']):03d}-c{crop_size}"
                relative = Path(f"crop-{crop_size}") / f"{key}.png"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                crop_text(image, atom["center"], crop_size).save(path)
                records.append(
                    {
                        "key": key,
                        "sample_key": sample["key"],
                        "atom_id": int(atom["atom_id"]),
                        "truth": truth,
                        "crop_size": crop_size,
                        "image": str(relative),
                    }
                )
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "source_samples": args.samples,
        "crops": len(records),
        "crop_sizes": args.crop_sizes,
        "labels": dict(sorted(Counter(row["truth"] for row in records).items())),
    }
    (output / "prepare-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def label_group(text: str) -> str:
    if re.fullmatch(r"Br|Cl|Se|Si|[BCFINOPS]", text):
        return "simple_element"
    if re.search(r"H|[+-]|\d", text):
        return "modifier"
    return "open_string"


def has_modifier(text: str) -> bool:
    return bool(re.search(r"H|[+-]|\d", text))


def run_paddle(args: argparse.Namespace) -> None:
    from paddleocr import TextRecognition

    root = Path(args.dataset_dir)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    recognizer = TextRecognition(
        model_name=args.model,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
    )
    output_rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        images = [np.asarray(Image.open(root / row["image"]).convert("RGB")) for row in batch]
        predictions = list(recognizer.predict(images, batch_size=args.batch_size))
        for row, prediction in zip(batch, predictions):
            result = prediction.json["res"]
            predicted = str(result["rec_text"])
            output_rows.append(
                {
                    **row,
                    "prediction": predicted,
                    "score": float(result["rec_score"]),
                    "exact": predicted == row["truth"],
                    "group": label_group(row["truth"]),
                    "false_modifier": (
                        label_group(row["truth"]) == "simple_element"
                        and has_modifier(predicted)
                    ),
                }
            )
        if start and start % (args.batch_size * 20) == 0:
            print(f"recognized={start}/{len(records)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    buckets = {}
    for crop_size in sorted({row["crop_size"] for row in output_rows}):
        selected = [row for row in output_rows if row["crop_size"] == crop_size]
        groups = {}
        for group in ("simple_element", "modifier", "open_string"):
            values = [row for row in selected if row["group"] == group]
            groups[group] = {
                "samples": len(values),
                "exact_rate": sum(row["exact"] for row in values) / max(len(values), 1),
            }
        buckets[str(crop_size)] = {
            "samples": len(selected),
            "exact_rate": sum(row["exact"] for row in selected) / max(len(selected), 1),
            "false_modifier_rate_on_simple_elements": sum(
                row["false_modifier"] for row in selected
            ) / max(sum(row["group"] == "simple_element" for row in selected), 1),
            "mean_score": sum(row["score"] for row in selected) / max(len(selected), 1),
            "groups": groups,
        }
    summary = {"model": args.model, "device": args.device, "crop_sizes": buckets}
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--shards", nargs="+", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--samples", type=int, default=100)
    prepare_parser.add_argument("--crop-sizes", nargs="+", type=int, default=[18, 22, 26, 30])
    prepare_parser.set_defaults(function=prepare)
    paddle_parser = subparsers.add_parser("paddle")
    paddle_parser.add_argument("--dataset-dir", required=True)
    paddle_parser.add_argument("--output", required=True)
    paddle_parser.add_argument("--model", default="en_PP-OCRv5_mobile_rec")
    paddle_parser.add_argument("--device", default="cpu")
    paddle_parser.add_argument("--batch-size", type=int, default=64)
    paddle_parser.add_argument("--enable-mkldnn", action="store_true")
    paddle_parser.set_defaults(function=run_paddle)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
