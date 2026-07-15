#!/usr/bin/env python3
"""Audit MolGrapher keypoints and build a deterministic clean overfit subset."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import html
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ocr import atom_display_text
from chemvector.train import expand_shards


def median_bond_length(render: dict) -> float:
    lengths = [
        math.hypot(
            float(bond["end_image_x"]) - float(bond["begin_image_x"]),
            float(bond["end_image_y"]) - float(bond["begin_image_y"]),
        )
        for bond in render["bonds"]
    ]
    return float(np.median(lengths)) if lengths else 32.0


def nearest_dark_distance(
    dark: np.ndarray,
    x: float,
    y: float,
    *,
    radius: int,
) -> float:
    center_x = round(x)
    center_y = round(y)
    left = max(0, center_x - radius)
    right = min(dark.shape[1], center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(dark.shape[0], center_y + radius + 1)
    points = np.argwhere(dark[top:bottom, left:right])
    if not len(points):
        return float(radius + 1)
    points = points.astype(np.float32)
    points[:, 0] += top
    points[:, 1] += left
    distances = np.hypot(points[:, 1] - x, points[:, 0] - y)
    return float(distances.min())


def expected_ink_region(render: dict, bond_length: float) -> np.ndarray:
    width = int(render["width"])
    height = int(render["height"])
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    bond_width = max(10, min(28, round(bond_length * 0.15)))
    atom_radius = max(14, min(42, round(bond_length * 0.24)))
    for bond in render["bonds"]:
        draw.line(
            (
                float(bond["begin_image_x"]),
                float(bond["begin_image_y"]),
                float(bond["end_image_x"]),
                float(bond["end_image_y"]),
            ),
            fill=1,
            width=bond_width,
        )
    for atom in render["atoms"]:
        x = float(atom["image_x"])
        y = float(atom["image_y"])
        draw.ellipse(
            (x - atom_radius, y - atom_radius, x + atom_radius, y + atom_radius),
            fill=1,
        )
    return np.asarray(mask, dtype=bool)


def small_annotation_components(
    unexplained: np.ndarray,
    anchors: list[tuple[float, float]],
) -> list[dict]:
    labels, count = ndimage.label(unexplained, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    for component_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        pixels = int((labels[slices] == component_id).sum())
        if not (12 <= height <= 46 and 3 <= width <= 42 and 20 <= pixels <= 900):
            continue
        center_x = (slices[1].start + slices[1].stop - 1) / 2
        center_y = (slices[0].start + slices[0].stop - 1) / 2
        distance = min(
            (math.hypot(center_x - x, center_y - y) for x, y in anchors),
            default=float("inf"),
        )
        if 10 <= distance <= 72:
            components.append(
                {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                    "width": width,
                    "height": height,
                    "pixels": pixels,
                    "nearest_anchor_distance": round(distance, 3),
                }
            )
    return components


def analyze_sample(sample: dict, dark_threshold: int) -> tuple[dict, np.ndarray]:
    image = sample["image"]
    gray = np.asarray(image.convert("L"))
    dark = gray < dark_threshold
    render = sample["render"]
    chemmol = sample["chemmol"]
    atoms_by_id = {int(atom["id"]): atom for atom in chemmol["atoms"]}
    bond_length = median_bond_length(render)
    covered = expected_ink_region(render, bond_length)
    unexplained = dark & ~covered
    dark_pixels = int(dark.sum())
    anchor_records = []
    for position in render["atoms"]:
        atom = atoms_by_id[int(position["atom_id"])]
        text = atom_display_text(atom)
        distance = nearest_dark_distance(
            dark,
            float(position["image_x"]),
            float(position["image_y"]),
            radius=48,
        )
        anchor_records.append(
            {
                "atom_id": int(position["atom_id"]),
                "element": atom["element"],
                "display_text": text,
                "role": "visible_text" if text else "implicit_junction",
                "image_x": float(position["image_x"]),
                "image_y": float(position["image_y"]),
                "nearest_dark_pixels": round(distance, 3),
            }
        )
    maximum_anchor_distance = max(
        (item["nearest_dark_pixels"] for item in anchor_records),
        default=0.0,
    )
    implicit_distances = [
        item["nearest_dark_pixels"]
        for item in anchor_records
        if item["role"] == "implicit_junction"
    ]
    visible_distances = [
        item["nearest_dark_pixels"]
        for item in anchor_records
        if item["role"] == "visible_text"
    ]
    unexplained_fraction = int(unexplained.sum()) / max(dark_pixels, 1)
    annotation_components = small_annotation_components(
        unexplained,
        [
            (float(item["image_x"]), float(item["image_y"]))
            for item in render["atoms"]
        ],
    )
    flags = []
    warnings = []
    if any(item > 48.0 for item in implicit_distances + visible_distances):
        flags.append("anchor_without_nearby_ink")
    if any(item > 10.0 for item in implicit_distances):
        warnings.append("implicit_carbon_likely_rendered_as_text")
    if any(item > max(20.0, bond_length * 0.28) for item in visible_distances):
        warnings.append("large_or_offset_visible_label")
    if unexplained_fraction > 0.08:
        warnings.append("unexplained_ink")
    probable_index_threshold = max(4, math.ceil(len(render["atoms"]) * 0.45))
    if len(annotation_components) >= probable_index_threshold:
        flags.append("probable_atom_indices")
    if any(
        not 0 <= item[axis] < render["width" if axis == "image_x" else "height"]
        for item in anchor_records
        for axis in ("image_x", "image_y")
    ):
        flags.append("anchor_out_of_bounds")
    record = {
        "key": sample["key"],
        "molecule_id": chemmol["molecule_id"],
        "atoms": len(render["atoms"]),
        "bonds": len(render["bonds"]),
        "visible_text_atoms": len(visible_distances),
        "implicit_atoms": len(implicit_distances),
        "multiple_bonds": sum(
            bond["display_type"] in {"DOUBLE", "TRIPLE"}
            for bond in render["bonds"]
        ),
        "median_bond_length": round(bond_length, 3),
        "maximum_anchor_dark_distance": round(maximum_anchor_distance, 3),
        "mean_anchor_dark_distance": round(
            float(np.mean([item["nearest_dark_pixels"] for item in anchor_records])),
            3,
        ),
        "unexplained_ink_fraction": round(unexplained_fraction, 6),
        "small_annotation_components": annotation_components,
        "flags": flags,
        "warnings": warnings,
        "anchors": anchor_records,
    }
    return record, unexplained


def render_overlay(sample: dict, record: dict, unexplained: np.ndarray) -> Image.Image:
    array = np.asarray(sample["image"].convert("RGB")).copy()
    array[unexplained] = (
        0.35 * array[unexplained] + 0.65 * np.asarray([255, 0, 200])
    ).astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for bond in sample["render"]["bonds"]:
        draw.line(
            (
                bond["begin_image_x"],
                bond["begin_image_y"],
                bond["end_image_x"],
                bond["end_image_y"],
            ),
            fill=(0, 150, 255),
            width=2,
        )
    for anchor in record["anchors"]:
        x = anchor["image_x"]
        y = anchor["image_y"]
        color = (220, 30, 30) if anchor["role"] == "visible_text" else (0, 145, 60)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
        label = f'{anchor["atom_id"]}:{anchor["element"]}'
        box = draw.textbbox((x + 7, y - 8), label)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x + 7, y - 8), label, fill=color)
    return image


def quality_key(record: dict) -> tuple:
    return (
        bool(record["flags"]),
        len(record["warnings"]),
        record["unexplained_ink_fraction"],
        record["maximum_anchor_dark_distance"],
        record["key"],
    )


def selection_bucket(record: dict) -> tuple[str, bool, bool]:
    atom_count = record["atoms"]
    size = "small" if atom_count <= 10 else "medium" if atom_count <= 25 else "large"
    return size, bool(record["visible_text_atoms"]), bool(record["multiple_bonds"])


def select_clean_records(records: list[dict], count: int) -> list[dict]:
    eligible = [
        record
        for record in records
        if not record["flags"] and 3 <= record["atoms"] <= 64
    ]
    buckets: dict[tuple[str, bool, bool], deque[dict]] = {}
    grouped: dict[tuple[str, bool, bool], list[dict]] = defaultdict(list)
    for record in eligible:
        grouped[selection_bucket(record)].append(record)
    for bucket, values in grouped.items():
        buckets[bucket] = deque(sorted(values, key=quality_key))
    selected = []
    while buckets and len(selected) < count:
        for bucket in sorted(tuple(buckets)):
            values = buckets[bucket]
            if values:
                selected.append(values.popleft())
                if len(selected) >= count:
                    break
            if not values:
                del buckets[bucket]
    return selected


def copy_subset(shards: list[str], keys: set[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    copied_keys = set()
    with tarfile.open(temporary, "w") as destination:
        for shard in shards:
            with tarfile.open(shard) as source:
                for member in source:
                    key = member.name.split(".", 1)[0]
                    if key not in keys or not member.isfile():
                        continue
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable member: {shard}:{member.name}")
                    payload = stream.read()
                    info = tarfile.TarInfo(member.name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.mode = 0o644
                    destination.addfile(info, io.BytesIO(payload))
                    copied_keys.add(key)
    if copied_keys != keys:
        missing = sorted(keys - copied_keys)
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Subset keys were not found: {missing}")
    temporary.replace(output)


def write_index(output: Path, records: list[dict]) -> None:
    cards = []
    for record in records:
        flags = ", ".join(record["flags"]) or "none"
        warnings = ", ".join(record["warnings"]) or "none"
        cards.append(
            '<article data-flag="{}"><img src="overlays/{}.png" loading="lazy">'
            '<div><strong>{}</strong></div><code>atoms={} text={} extra={:.3f} flags={} warnings={}</code></article>'.format(
                "flagged" if record["flags"] else "clean",
                html.escape(record["key"]),
                html.escape(record["key"]),
                record["atoms"],
                record["visible_text_atoms"],
                record["unexplained_ink_fraction"],
                html.escape(flags),
                html.escape(warnings),
            )
        )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MolGrapher 标签审计</title><style>
body{font-family:system-ui,sans-serif;margin:20px;color:#202124}nav{position:sticky;top:0;background:white;padding:8px 0}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
article{border:1px solid #bbb;padding:8px}img{display:block;width:100%;height:auto}code{overflow-wrap:anywhere}
body[data-mode="clean"] article[data-flag="flagged"],body[data-mode="flagged"] article[data-flag="clean"]{display:none}
</style><h1>MolGrapher 500 张标签审计</h1>
<p>青线是标注键，绿圈是隐式碳/交点，红圈是可见文字原子，洋红色是宽松几何覆盖之外的墨迹。</p>
<nav><button onclick="document.body.dataset.mode='all'">全部</button>
<button onclick="document.body.dataset.mode='flagged'">仅异常</button>
<button onclick="document.body.dataset.mode='clean'">仅候选干净样本</button></nav><main>""" + "".join(cards) + "</main></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--dark-threshold", type=int, default=220)
    parser.add_argument("--subset-output")
    parser.add_argument("--subset-size", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1 or args.subset_size < 1:
        raise ValueError("sample counts must be positive")
    if not 1 <= args.dark_threshold <= 254:
        raise ValueError("dark-threshold must be in [1, 254]")

    shards = expand_shards(args.shards)
    output = Path(args.output_dir)
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    records = []
    dataset = ChemMolWebDataset(
        shards,
        validate_legacy_hash=False,
        sequence_version="0.4",
    )
    for sample in dataset:
        if len(records) >= args.samples:
            break
        record, unexplained = analyze_sample(sample, args.dark_threshold)
        render_overlay(sample, record, unexplained).save(
            overlays / f'{sample["key"]}.png'
        )
        records.append(record)
    if len(records) != args.samples:
        raise ValueError(f"Requested {args.samples} samples, found {len(records)}")

    selected = select_clean_records(records, args.subset_size)
    if args.subset_output and len(selected) != args.subset_size:
        raise ValueError(
            f"Only {len(selected)} clean samples available for subset of {args.subset_size}"
        )
    selected_keys = {record["key"] for record in selected}
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            row = dict(record)
            row["selected_for_overfit"] = record["key"] in selected_keys
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    flag_counts: dict[str, int] = defaultdict(int)
    warning_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for flag in record["flags"]:
            flag_counts[flag] += 1
        for warning in record["warnings"]:
            warning_counts[warning] += 1
    summary = {
        "samples": len(records),
        "flagged_samples": sum(bool(record["flags"]) for record in records),
        "flagged_rate": sum(bool(record["flags"]) for record in records) / len(records),
        "flag_counts": dict(sorted(flag_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "maximum_anchor_dark_distance": max(
            record["maximum_anchor_dark_distance"] for record in records
        ),
        "mean_unexplained_ink_fraction": float(
            np.mean([record["unexplained_ink_fraction"] for record in records])
        ),
        "selected_overfit_samples": len(selected),
        "selected_keys": [record["key"] for record in selected],
        "shards": shards,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_index(output, records)
    if args.subset_output:
        subset_output = Path(args.subset_output)
        copy_subset(shards, selected_keys, subset_output)
        manifest = {
            "schema_version": "molgrapher-overfit-subset-0.1",
            "samples": len(selected),
            "source_audit": str(output / "summary.json"),
            "source_shards": shards,
            "selection": "deterministic round-robin across size/text/multiple-bond buckets",
            "keys": [record["key"] for record in selected],
        }
        subset_output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
