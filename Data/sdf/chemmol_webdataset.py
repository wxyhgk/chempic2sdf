#!/usr/bin/env python3
"""Minimal WebDataset adapter for ChemMol training samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Iterator

import webdataset as wds
from PIL import Image

try:
    from .chemmol_sequence import (
        SCHEMA_VERSION,
        serialize_chemmolseq,
        validate_chemmolseq_roundtrip,
    )
except ImportError:
    from chemmol_sequence import (
        SCHEMA_VERSION,
        serialize_chemmolseq,
        validate_chemmolseq_roundtrip,
    )


class ChemMolWebDataset:
    def __init__(
        self,
        shards: str | list[str],
        *,
        validate_legacy_hash: bool = True,
        validate_target_roundtrip: bool = False,
        shardshuffle: bool = False,
        sample_shuffle_buffer: int = 0,
    ) -> None:
        self.shards = shards
        self.validate_legacy_hash = validate_legacy_hash
        self.validate_target_roundtrip = validate_target_roundtrip
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer

    def __iter__(self) -> Iterator[dict]:
        dataset = wds.WebDataset(
            self.shards,
            shardshuffle=self.shardshuffle,
            empty_check=False,
        )
        if self.sample_shuffle_buffer:
            dataset = dataset.shuffle(self.sample_shuffle_buffer)
        for sample in dataset:
            chemmol = json.loads(sample["chemmol.json"])
            render = json.loads(sample["render.json"])
            meta = json.loads(sample["meta.json"])
            molecule_ids = {
                chemmol["molecule_id"],
                render["molecule_id"],
                meta["molecule_id"],
            }
            if len(molecule_ids) != 1:
                raise ValueError(f"Molecule ID mismatch in {sample['__key__']}")
            if self.validate_legacy_hash:
                legacy_sequence = sample["chemmolseq.txt"].decode("utf-8")
                sequence_hash = hashlib.sha256(legacy_sequence.encode("utf-8")).hexdigest()
                if sequence_hash != meta["chemmolseq_sha256"]:
                    raise ValueError(f"Legacy ChemMolSeq hash mismatch in {sample['__key__']}")
            sequence = serialize_chemmolseq(chemmol, render)
            if self.validate_target_roundtrip:
                validate_chemmolseq_roundtrip(
                    sequence,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )
            image = Image.open(io.BytesIO(sample["png"])).convert("RGB")
            image.load()
            yield {
                "key": sample["__key__"],
                "image": image,
                "target_sequence": sequence,
                "target_schema_version": SCHEMA_VERSION,
                "chemmol": chemmol,
                "render": render,
                "meta": meta,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--validate-target-roundtrip", action="store_true")
    args = parser.parse_args()
    count = 0
    for sample in ChemMolWebDataset(
        args.shard,
        validate_target_roundtrip=args.validate_target_roundtrip,
    ):
        if sample["image"].size != (768, 768):
            raise ValueError(f"Unexpected image size in {sample['key']}")
        count += 1
        if count >= args.limit:
            break
    print(f"Loaded and validated {count} samples")


if __name__ == "__main__":
    main()
