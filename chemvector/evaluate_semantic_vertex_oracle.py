#!/usr/bin/env python3
"""Measure whether separating text objects from bond ink improves vertex detection."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tarfile

from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.geometry import stable_pairwise_distances
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.image_io import image_tensor
from chemvector.oracle_semantic_layers import bond_alpha_to_ink_image
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.vertex import decode_vertices
from chemvector.text_object import TextObjectCenterHead, decode_text_objects


SUFFIXES = (".layers.json", ".bond.png", ".png")


def rows_for_splits(root: Path, splits: set[str], samples: int) -> list[dict[str, str]]:
    with (root / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in splits]
    rows.sort(key=lambda row: (row["split"], row["sample_key"]))
    return rows[:samples]


def load_samples(root: Path, rows: list[dict[str, str]]):
    by_shard: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(row["output_shard"], []).append(row)
    for shard, shard_rows in sorted(by_shard.items()):
        wanted = {row["sample_key"]: row for row in shard_rows}
        payloads: dict[str, dict[str, bytes]] = {}
        with tarfile.open(root / shard) as archive:
            for member in archive:
                for suffix in SUFFIXES:
                    if member.name.endswith(suffix):
                        key = member.name[: -len(suffix)]
                        if key not in wanted:
                            break
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise ValueError(f"Cannot read {member.name}")
                        payloads.setdefault(key, {})[suffix] = handle.read()
                        if len(payloads[key]) == len(SUFFIXES):
                            yield wanted[key], payloads.pop(key)
                        break


def outside_text_boxes(item: dict, boxes: list[list[float]], expansion: float) -> bool:
    x = float(item["x"]) * 768.0
    y = float(item["y"]) * 768.0
    return not any(
        x0 - expansion <= x <= x1 + expansion
        and y0 - expansion <= y <= y1 + expansion
        for x0, y0, x1, y1 in boxes
    )


def summarize(records: list[dict], mode: str) -> dict:
    values = [record[mode] for record in records]
    matches = sum(value["matches"] for value in values)
    predicted = sum(value["predicted"] for value in values)
    expected = sum(value["expected"] for value in values)
    precision = matches / max(predicted, 1)
    recall = matches / max(expected, 1)
    return {
        "samples": len(values),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "vertex_exact_rate": sum(value["exact"] for value in values) / max(len(values), 1),
    }


def compare(points: list[tuple[float, float]], targets: list[tuple[float, float]]) -> dict:
    predicted = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
    expected = torch.tensor(targets, dtype=torch.float32).reshape(-1, 2)
    matches = 0
    if len(predicted) and len(expected):
        distances = stable_pairwise_distances(predicted, expected) * 768.0
        rows, columns = linear_sum_assignment(distances.numpy())
        matches = sum(float(distances[row, column]) <= 6.0 for row, column in zip(rows, columns))
    return {
        "matches": matches,
        "predicted": len(predicted),
        "expected": len(expected),
        "exact": matches == len(expected) and len(predicted) == len(expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="Data/sdf/chemvision_layers_v0_1/pilot_10k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("validation", "internal_test"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bbox-expansion", type=float, default=8.0)
    parser.add_argument("--text-object-checkpoint")
    parser.add_argument("--implicit-vertex-checkpoint")
    parser.add_argument("--implicit-threshold", type=float, default=0.52)
    args = parser.parse_args()

    device = torch.device("cuda")
    _, graph_payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    text_head = None
    if args.text_object_checkpoint:
        text_payload = torch.load(
            args.text_object_checkpoint, map_location="cpu", weights_only=False
        )
        text_head = TextObjectCenterHead(text_payload["input_channels"]).to(device)
        text_head.load_state_dict(text_payload["text_object_head"])
        text_head.eval()
    implicit_head = None
    if args.implicit_vertex_checkpoint:
        implicit_payload = torch.load(
            args.implicit_vertex_checkpoint, map_location="cpu", weights_only=False
        )
        implicit_head = deepcopy(model.vertex_head).to(device)
        implicit_head.load_state_dict(implicit_payload["implicit_vertex_head"])
        implicit_head.eval()
    rows = rows_for_splits(Path(args.dataset_root), set(args.splits), args.samples)
    records = []
    pending = []

    def evaluate_batch(batch) -> None:
        combined_pixels = torch.stack([image_tensor(item["combined"]) for item in batch]).to(device)
        bond_pixels = torch.stack([image_tensor(item["bond"]) for item in batch]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            combined_features = model.vision_encoder.forward_stroke_resolution(combined_pixels)
            combined_output = model.vertex_head(combined_features)
            implicit_output = implicit_head(combined_features) if implicit_head is not None else None
            bond_output = model.vertex_head(model.vision_encoder.forward_stroke_resolution(bond_pixels))
            predicted_text_output = text_head(combined_features) if text_head is not None else None
        combined_candidates = decode_vertices(combined_output, threshold=0.05, maximum_vertices=256, nms_kernel_size=1)
        implicit_candidate_batches = (
            decode_vertices(implicit_output, threshold=0.05, maximum_vertices=256, nms_kernel_size=1)
            if implicit_output is not None
            else [None] * len(batch)
        )
        deployed_implicit_batches = (
            decode_vertices(implicit_output, threshold=args.implicit_threshold, maximum_vertices=256, nms_kernel_size=1)
            if implicit_output is not None
            else [None] * len(batch)
        )
        bond_candidates = decode_vertices(bond_output, threshold=0.05, maximum_vertices=256, nms_kernel_size=1)
        predicted_text_batches = (
            decode_text_objects(predicted_text_output, threshold=0.3)
            if predicted_text_output is not None
            else [None] * len(batch)
        )
        for item, combined, bond, predicted_text, implicit_candidates, deployed_implicit in zip(
            batch, combined_candidates, bond_candidates, predicted_text_batches,
            implicit_candidate_batches, deployed_implicit_batches,
        ):
            atoms = item["layers"]["atoms"]
            targets = [(float(atom["center"][0]) / 768.0, float(atom["center"][1]) / 768.0) for atom in atoms]
            visible = [atom for atom in atoms if atom["visible_text"]]
            implicit = [atom for atom in atoms if not atom["visible_text"]]
            visible_points = [(float(atom["center"][0]) / 768.0, float(atom["center"][1]) / 768.0) for atom in visible]
            implicit_points = [(float(atom["center"][0]) / 768.0, float(atom["center"][1]) / 768.0) for atom in implicit]
            boxes = [atom["bbox"] for atom in visible]
            implicit_count = len(atoms) - len(visible)
            combined_filtered = [candidate for candidate in combined if outside_text_boxes(candidate, boxes, args.bbox_expansion)]
            bond_filtered = [candidate for candidate in bond if outside_text_boxes(candidate, boxes, args.bbox_expansion)]
            modes = {
                "combined": combined[: len(atoms)],
                "combined_text_object": combined_filtered[:implicit_count],
                "bond_text_object": bond_filtered[:implicit_count],
            }
            record = {
                "sample_key": item["row"]["sample_key"],
                "font_bucket": item["row"]["font_bucket"],
                "visible_text_count": len(visible),
            }
            for name, candidates in modes.items():
                points = [(float(candidate["x"]), float(candidate["y"])) for candidate in candidates]
                if name != "combined":
                    points += visible_points
                record[name] = compare(points, targets)
            record["combined_implicit"] = compare(
                [(float(candidate["x"]), float(candidate["y"])) for candidate in combined_filtered[:implicit_count]],
                implicit_points,
            )
            record["bond_implicit"] = compare(
                [(float(candidate["x"]), float(candidate["y"])) for candidate in bond_filtered[:implicit_count]],
                implicit_points,
            )
            if predicted_text is not None:
                predicted_boxes = [
                    [
                        (obj["x"] - obj["half_width"]) * 768.0,
                        (obj["y"] - obj["half_height"]) * 768.0,
                        (obj["x"] + obj["half_width"]) * 768.0,
                        (obj["y"] + obj["half_height"]) * 768.0,
                    ]
                    for obj in predicted_text
                ]
                predicted_filtered = [
                    candidate
                    for candidate in combined
                    if outside_text_boxes(candidate, predicted_boxes, args.bbox_expansion)
                ]
                merged = [
                    (float(candidate["x"]), float(candidate["y"]))
                    for candidate in predicted_filtered[:implicit_count]
                ] + [(float(obj["x"]), float(obj["y"])) for obj in predicted_text]
                record["predicted_text_object"] = compare(merged, targets)
                record["predicted_text_only"] = compare(
                    [(float(obj["x"]), float(obj["y"])) for obj in predicted_text],
                    visible_points,
                )
            if implicit_candidates is not None:
                record["implicit_only"] = compare(
                    [(float(obj["x"]), float(obj["y"])) for obj in deployed_implicit],
                    implicit_points,
                )
            if implicit_candidates is not None and predicted_text is not None:
                predicted_boxes = [
                    [
                        (obj["x"] - obj["half_width"]) * 768.0,
                        (obj["y"] - obj["half_height"]) * 768.0,
                        (obj["x"] + obj["half_width"]) * 768.0,
                        (obj["y"] + obj["half_height"]) * 768.0,
                    ]
                    for obj in predicted_text
                ]
                filtered_implicit = [
                    obj for obj in implicit_candidates
                    if outside_text_boxes(obj, predicted_boxes, args.bbox_expansion)
                ]
                filtered_deployed = [
                    obj for obj in deployed_implicit
                    if outside_text_boxes(obj, predicted_boxes, args.bbox_expansion)
                ]
                text_points = [(float(obj["x"]), float(obj["y"])) for obj in predicted_text]
                record["predicted_both_known_implicit_count"] = compare(
                    [(float(obj["x"]), float(obj["y"])) for obj in filtered_implicit[:implicit_count]] + text_points,
                    targets,
                )
                record["predicted_both"] = compare(
                    [(float(obj["x"]), float(obj["y"])) for obj in filtered_deployed] + text_points,
                    targets,
                )
            records.append(record)

    for row, payloads in load_samples(Path(args.dataset_root), rows):
        pending.append({
            "row": row,
            "combined": Image.open(io.BytesIO(payloads[".png"])).convert("RGB"),
            "bond": bond_alpha_to_ink_image(Image.open(io.BytesIO(payloads[".bond.png"])).convert("L")),
            "layers": json.loads(payloads[".layers.json"]),
        })
        if len(pending) == args.batch_size:
            evaluate_batch(pending)
            pending = []
            if len(records) % 100 == 0:
                print(f"evaluated={len(records)}", flush=True)
    if pending:
        evaluate_batch(pending)

    modes = (
        "combined",
        "combined_text_object",
        "bond_text_object",
        "combined_implicit",
        "bond_implicit",
    )
    if text_head is not None:
        modes += ("predicted_text_object", "predicted_text_only")
    if implicit_head is not None:
        modes += ("implicit_only",)
    if text_head is not None and implicit_head is not None:
        modes += ("predicted_both_known_implicit_count", "predicted_both")
    summary = {
        "samples": len(records),
        "matching_radius_pixels": 6.0,
        "bbox_expansion_pixels": args.bbox_expansion,
        "modes": {mode: summarize(records, mode) for mode in modes},
        "large_text": {
            mode: summarize(
                [record for record in records if record["font_bucket"] in {"large", "extreme"}],
                mode,
            )
            for mode in modes
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "records.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
