#!/usr/bin/env python3
"""Compare Atom proposal recall by nearest-neighbor spacing across datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import islice
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.data import ChemVectorIterableDataset, structure_labels
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
)
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector
from chemvector.train_atom_bond_scene import _image_tensor
from chemvector.train_spatial_atom_object import load_grounded_vision
from chemvector.train_spatial_incidence import DEFAULT_ATOM, DEFAULT_GRAPH


BUCKETS = (
    ("lt_8", 0.0, 8.0),
    ("8_12", 8.0, 12.0),
    ("12_16", 12.0, 16.0),
    ("16_20", 16.0, 20.0),
    ("20_24", 20.0, 24.0),
    ("24_32", 24.0, 32.0),
    ("32_48", 32.0, 48.0),
    ("ge_48", 48.0, float("inf")),
)


def distance_bucket(distance: float) -> str:
    for name, lower, upper in BUCKETS:
        if lower <= distance < upper:
            return name
    raise ValueError(f"Cannot bucket distance: {distance}")


def sample_record(sample: dict, *, domain: str) -> dict:
    labels = structure_labels(sample)
    atoms = sorted(labels["atoms"], key=lambda item: int(item["atom_id"]))
    bonds = {
        tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        for bond in labels["bonds"]
    }
    return {
        "key": sample["key"],
        "domain": domain,
        "group": str(labels.get("variant", sample.get("meta", {}).get("variant", "unknown"))),
        "image": sample["image"].convert("RGB"),
        "anchors": np.asarray([atom["center"] for atom in atoms], dtype=np.float32),
        "bonds": bonds,
    }


def webdataset_records(shards: list[str], samples: int | None, domain: str):
    iterator = iter(ChemVectorIterableDataset(shards, sequence_version="0.4"))
    if samples is not None:
        iterator = islice(iterator, samples)
    for sample in iterator:
        yield sample_record(sample, domain=domain)


def mr_tadf_records(manifest: Path, samples: int | None):
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if samples is not None:
        rows = rows[:samples]
    for row in rows:
        annotation = json.loads((root / row["annotation_file"]).read_text())
        atoms = sorted(annotation["atoms"], key=lambda item: int(item["atom_id"]))
        bonds = {
            tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
            for bond in annotation["bonds"]
        }
        with Image.open(root / row["image_file"]) as opened:
            image = opened.convert("RGB")
        yield {
            "key": row["molecule_id"],
            "domain": "mr_tadf_v1",
            "group": row["difficulty"],
            "image": image,
            "anchors": np.asarray(
                [[atom["image_x"], atom["image_y"]] for atom in atoms],
                dtype=np.float32,
            ),
            "bonds": bonds,
        }


def empty_totals() -> dict:
    return {
        "images": 0,
        "image_exact": 0,
        "atoms": 0,
        "matched_atoms": 0,
        "proposal_count": 0,
        "same_stride4_cell_pairs": 0,
        "buckets": defaultdict(Counter),
    }


def update_totals(totals: dict, record: dict, matched: set[int], proposal_count: int) -> None:
    anchors = record["anchors"]
    count = len(anchors)
    totals["images"] += 1
    totals["image_exact"] += int(len(matched) == count)
    totals["atoms"] += count
    totals["matched_atoms"] += len(matched)
    totals["proposal_count"] += proposal_count
    cells = np.floor(anchors / 4.0).astype(np.int32)
    cell_counts = Counter(map(tuple, cells.tolist()))
    totals["same_stride4_cell_pairs"] += sum(
        value * (value - 1) // 2 for value in cell_counts.values() if value > 1
    )
    distances = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(count), nearest]
    image_buckets: dict[str, list[int]] = defaultdict(list)
    for atom_id, (neighbor, distance) in enumerate(zip(nearest, nearest_distance)):
        bucket = distance_bucket(float(distance))
        image_buckets[bucket].append(atom_id)
        row = totals["buckets"][bucket]
        row["atoms"] += 1
        row["matched_atoms"] += int(atom_id in matched)
        pair = tuple(sorted((atom_id, int(neighbor))))
        row["nearest_bonded"] += int(pair in record["bonds"])
        row["nearest_nonbonded"] += int(pair not in record["bonds"])
    for bucket, atom_ids in image_buckets.items():
        row = totals["buckets"][bucket]
        row["images"] += 1
        row["image_bucket_exact"] += int(all(atom_id in matched for atom_id in atom_ids))


def finalize(totals: dict) -> dict:
    images = max(totals["images"], 1)
    atoms = max(totals["atoms"], 1)
    buckets = {}
    for name, _, _ in BUCKETS:
        row = totals["buckets"].get(name, Counter())
        bucket_atoms = max(row["atoms"], 1)
        bucket_images = max(row["images"], 1)
        buckets[name] = {
            **dict(row),
            "atom_fraction": row["atoms"] / atoms,
            "atom_recall_6px": row["matched_atoms"] / bucket_atoms,
            "image_bucket_exact_rate": row["image_bucket_exact"] / bucket_images,
            "nearest_nonbonded_fraction": row["nearest_nonbonded"] / bucket_atoms,
        }
    return {
        "images": totals["images"],
        "atoms": totals["atoms"],
        "atom_recall_6px": totals["matched_atoms"] / atoms,
        "image_exact_rate": totals["image_exact"] / images,
        "mean_proposals": totals["proposal_count"] / images,
        "same_stride4_cell_pairs": totals["same_stride4_cell_pairs"],
        "buckets": buckets,
    }


@torch.no_grad()
def evaluate_domain(
    records,
    *,
    vision: torch.nn.Module,
    atom_head: torch.nn.Module,
    config: PredictedProposalConfig,
    device: torch.device,
    batch_size: int,
    domain: str,
) -> dict:
    totals = empty_totals()
    groups = defaultdict(empty_totals)
    batch = []

    def evaluate_batch(items: list[dict]) -> None:
        pixels = torch.stack([_image_tensor(item["image"]) for item in items]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            output = atom_head(features)
            proposals = decode_predicted_atom_proposals(output, config)
        for index, item in enumerate(items):
            proposal_indices = torch.where(proposals["atom_mask"][index])[0]
            predicted = (
                proposals["atom_geometry"][index, proposal_indices, :2].float().cpu().numpy()
                * 768.0
            )
            expected = item["anchors"]
            matched = set()
            if len(predicted) and len(expected):
                cost = np.linalg.norm(predicted[:, None, :] - expected[None, :, :], axis=-1)
                rows, columns = linear_sum_assignment(cost)
                matched = {
                    int(column)
                    for row, column in zip(rows, columns)
                    if float(cost[row, column]) <= config.atom_match_radius_px
                }
            update_totals(totals, item, matched, len(proposal_indices))
            update_totals(groups[item["group"]], item, matched, len(proposal_indices))

    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            evaluate_batch(batch)
            batch = []
            if totals["images"] % 1000 == 0:
                print(f"{domain}: {totals['images']} images", flush=True)
    if batch:
        evaluate_batch(batch)
    return {
        **finalize(totals),
        "groups": {name: finalize(group) for name, group in sorted(groups.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity-shard",
        default="Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar",
    )
    parser.add_argument(
        "--chemvision-shards",
        nargs="+",
        default=[
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-000000.tar",
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-000001.tar",
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-000002.tar",
        ],
    )
    parser.add_argument(
        "--mr-tadf-manifest", default="Data/benchmarks/mr_tadf_v1/manifest.csv"
    )
    parser.add_argument("--capacity-samples", type=int, default=100)
    parser.add_argument("--chemvision-samples", type=int, default=9000)
    parser.add_argument("--mr-tadf-samples", type=int, default=1100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument(
        "--output",
        default="outputs/audits/chemvector-v0966-atom-spacing-distribution/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Atom spacing audit requires CUDA")
    device = torch.device("cuda")
    vision, vision_config, _ = load_grounded_vision(
        Path(args.grounded_graph_checkpoint), device
    )
    payload = torch.load(args.atom_checkpoint, map_location="cpu", weights_only=False)
    atom_head = SpatialAtomObjectDetector(
        SpatialAtomObjectConfig(**payload["config"])
    ).to(device)
    atom_head.load_state_dict(payload["head"], strict=True)
    atom_head.eval().requires_grad_(False)
    if atom_head.config.input_channels != vision_config.vision_dims[0]:
        raise ValueError("Atom head and vision dimensions differ")
    config = PredictedProposalConfig(maximum_atoms=128, maximum_bonds=1)
    domains = {
        "actual_capacity100": evaluate_domain(
            webdataset_records(
                [args.capacity_shard], args.capacity_samples, "actual_capacity100"
            ),
            vision=vision,
            atom_head=atom_head,
            config=config,
            device=device,
            batch_size=args.batch_size,
            domain="actual_capacity100",
        ),
        "available_chemvision_train": evaluate_domain(
            webdataset_records(
                args.chemvision_shards,
                args.chemvision_samples,
                "available_chemvision_train",
            ),
            vision=vision,
            atom_head=atom_head,
            config=config,
            device=device,
            batch_size=args.batch_size,
            domain="available_chemvision_train",
        ),
        "mr_tadf_v1": evaluate_domain(
            mr_tadf_records(Path(args.mr_tadf_manifest), args.mr_tadf_samples),
            vision=vision,
            atom_head=atom_head,
            config=config,
            device=device,
            batch_size=args.batch_size,
            domain="mr_tadf_v1",
        ),
    }
    report = {
        "schema_version": "atom-spacing-audit-0.1",
        "atom_checkpoint": args.atom_checkpoint,
        "atom_checkpoint_training_args": payload.get("training_args"),
        "match_radius_pixels": config.atom_match_radius_px,
        "nearest_neighbor_buckets_pixels": [
            [name, lower, None if np.isinf(upper) else upper]
            for name, lower, upper in BUCKETS
        ],
        "domains": domains,
        "mr_tadf_used_for_training_or_tuning": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
