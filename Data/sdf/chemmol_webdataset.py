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
        SCHEMA_VERSION as V2_SCHEMA_VERSION,
        serialize_chemmolseq,
        validate_chemmolseq_roundtrip,
    )
    from .chemmol_sequence_v3 import (
        SCHEMA_VERSION as V3_SCHEMA_VERSION,
        serialize_chemmolseq_v3,
        validate_chemmolseq_v3_roundtrip,
    )
except ImportError:
    from chemmol_sequence import (
        SCHEMA_VERSION as V2_SCHEMA_VERSION,
        serialize_chemmolseq,
        validate_chemmolseq_roundtrip,
    )
    from chemmol_sequence_v3 import (
        SCHEMA_VERSION as V3_SCHEMA_VERSION,
        serialize_chemmolseq_v3,
        validate_chemmolseq_v3_roundtrip,
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
        sequence_version: str = "0.2",
        reject_keys: set[str] | None = None,
    ) -> None:
        self.shards = shards
        self.validate_legacy_hash = validate_legacy_hash
        self.validate_target_roundtrip = validate_target_roundtrip
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer
        if sequence_version not in {"0.2", "0.3"}:
            raise ValueError(f"Unsupported ChemMolSeq version: {sequence_version}")
        self.sequence_version = sequence_version
        self.reject_keys = reject_keys or set()

    def __iter__(self) -> Iterator[dict]:
        dataset = wds.WebDataset(
            self.shards,
            shardshuffle=self.shardshuffle,
            empty_check=False,
        )
        if self.sample_shuffle_buffer:
            dataset = dataset.shuffle(self.sample_shuffle_buffer)
        for sample in dataset:
            if sample["__key__"] in self.reject_keys:
                continue
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
            if self.sequence_version == "0.3":
                sequence = serialize_chemmolseq_v3(chemmol, render)
                schema_version = V3_SCHEMA_VERSION
            else:
                sequence = serialize_chemmolseq(chemmol, render)
                schema_version = V2_SCHEMA_VERSION
            if self.validate_target_roundtrip:
                validator = (
                    validate_chemmolseq_v3_roundtrip
                    if self.sequence_version == "0.3"
                    else validate_chemmolseq_roundtrip
                )
                validator(sequence, chemmol["canonical_isomeric_smiles"], chemmol["inchikey"])
            image = Image.open(io.BytesIO(sample["png"])).convert("RGB")
            image.load()
            yield {
                "key": sample["__key__"],
                "image": image,
                "target_sequence": sequence,
                "target_schema_version": schema_version,
                "chemmol": chemmol,
                "render": render,
                "meta": meta,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--validate-target-roundtrip", action="store_true")
    parser.add_argument("--sequence-version", choices=("0.2", "0.3"), default="0.2")
    args = parser.parse_args()
    count = 0
    for sample in ChemMolWebDataset(
        args.shard,
        validate_target_roundtrip=args.validate_target_roundtrip,
        sequence_version=args.sequence_version,
    ):
        if sample["image"].size != (768, 768):
            raise ValueError(f"Unexpected image size in {sample['key']}")
        count += 1
        if count >= args.limit:
            break
    print(f"Loaded and validated {count} samples")


if __name__ == "__main__":
    main()
