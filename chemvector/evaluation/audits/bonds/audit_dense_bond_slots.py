#!/usr/bin/env python3
"""Audit how many Bond Objects share one dense detector grid cell."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

from chemvector.train import expand_shards


def maximum_cell_occupancy(render: dict, grid_size: int) -> int:
    width = float(render["width"])
    height = float(render["height"])
    occupancy = Counter()
    for bond in render["bonds"]:
        center_x = (float(bond["begin_image_x"]) + float(bond["end_image_x"])) / 2
        center_y = (float(bond["begin_image_y"]) + float(bond["end_image_y"])) / 2
        cell_x = min(grid_size - 1, max(0, int(center_x / width * grid_size)))
        cell_y = min(grid_size - 1, max(0, int(center_y / height * grid_size)))
        occupancy[(cell_y, cell_x)] += 1
    return max(occupancy.values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--current-slots", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.grid_size < 1 or args.current_slots < 1:
        raise ValueError("grid-size and current-slots must be positive")

    histogram = Counter()
    overflow_examples = []
    sample_count = 0
    stop = False
    for shard in expand_shards(args.shards):
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.name.endswith(".render.json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"Cannot read {member.name} from {shard}")
                render = json.load(handle)
                maximum = maximum_cell_occupancy(render, args.grid_size)
                histogram[maximum] += 1
                if maximum > args.current_slots and len(overflow_examples) < 100:
                    overflow_examples.append(
                        {
                            "sample": member.name.removesuffix(".render.json"),
                            "shard": str(shard),
                            "maximum_cell_occupancy": maximum,
                            "bonds": len(render["bonds"]),
                            "variant": render.get("variant"),
                        }
                    )
                sample_count += 1
                if args.samples is not None and sample_count >= args.samples:
                    stop = True
                    break
        if stop:
            break

    overflow_count = sum(
        count for occupancy, count in histogram.items() if occupancy > args.current_slots
    )
    summary = {
        "schema_version": "dense-bond-slot-audit-0.1",
        "samples": sample_count,
        "grid_size": args.grid_size,
        "current_slots": args.current_slots,
        "required_slots": max(histogram, default=0),
        "occupancy_histogram": {
            str(occupancy): count for occupancy, count in sorted(histogram.items())
        },
        "overflow_samples": overflow_count,
        "overflow_examples": overflow_examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
