#!/usr/bin/env python3
"""Export the fixed ChemVision validation split as a standard SDF benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from Data.sdf.chemscene_v0_1 import build_chemscene_v0_1, sdf_from_chemscene_v0_1
from chemvector.audit_bond_endpoint_junction import load_samples


FIELDS = (
    "molecule_id",
    "sample_id",
    "dataset",
    "subset",
    "image_file",
    "sdf_file",
    "image_sha256",
    "sdf_sha256",
    "width",
    "height",
    "atoms",
    "bonds",
    "status",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"],
    )
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument(
        "--output-dir",
        default="Data/benchmarks/internal_chemvision_v1",
    )
    args = parser.parse_args()

    samples = load_samples(args.validation_shards, args.samples)
    if len(samples) != args.samples:
        raise ValueError("Validation benchmark loaded fewer samples than requested")
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    sdf_dir = output_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        molecule_id = f"chemvision_{sample['key']}"
        image_path = image_dir / f"{molecule_id}.png"
        sdf_path = sdf_dir / f"{molecule_id}.sdf"
        sample["image"].save(image_path, format="PNG")
        scene = build_chemscene_v0_1(
            sample["chemmol"], sample["render"], sample.get("layers")
        )
        sdf = sdf_from_chemscene_v0_1(scene)
        sdf_path.write_text(sdf, encoding="utf-8")
        image_payload = image_path.read_bytes()
        sdf_payload = sdf.encode("utf-8")
        rows.append(
            {
                "molecule_id": molecule_id,
                "sample_id": sample["key"],
                "dataset": "chemvision_layers_v0_1",
                "subset": "validation",
                "image_file": str(image_path.relative_to(output_dir)),
                "sdf_file": str(sdf_path.relative_to(output_dir)),
                "image_sha256": sha256_bytes(image_payload),
                "sdf_sha256": sha256_bytes(sdf_payload),
                "width": sample["image"].width,
                "height": sample["image"].height,
                "atoms": len(sample["chemmol"]["atoms"]),
                "bonds": len(sample["chemmol"]["bonds"]),
                "status": "ready",
            }
        )
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported={len(rows)} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
