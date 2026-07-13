#!/usr/bin/env python3
"""Compute ChemMolSeq v0.2 token-length statistics over the full dataset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chemmol_sequence import serialize_chemmolseq
from chemmol_tokenizer import ChemMolTokenizer


CHEMMOL_SUFFIX = "_original.chemmol.json"
RENDER_SUFFIX = "_original.render.json"


def analyze_shard(path_text: str) -> dict:
    path = Path(path_text)
    tokenizer = ChemMolTokenizer()
    pending: dict[str, dict] = {}
    lengths = []
    atom_counts = []
    bond_counts = []
    maximum = None
    fields = {
        "elements": Counter(),
        "chiral_tags": Counter(),
        "bond_types": Counter(),
        "display_types": Counter(),
        "bond_stereo": Counter(),
        "bond_directions": Counter(),
    }

    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith(CHEMMOL_SUFFIX):
                key = member.name[: -len(".chemmol.json")]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read {member.name}")
                pending[key] = json.loads(extracted.read())
            elif member.name.endswith(RENDER_SUFFIX):
                key = member.name[: -len(".render.json")]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read {member.name}")
                render = json.loads(extracted.read())
                chemmol = pending.pop(key)
                sequence = serialize_chemmolseq(chemmol, render)
                length = len(tokenizer.encode(sequence))
                lengths.append(length)
                atom_counts.append(len(chemmol["atoms"]))
                bond_counts.append(len(chemmol["bonds"]))
                if maximum is None or length > maximum["tokens"]:
                    maximum = {
                        "sample_key": key,
                        "molecule_id": chemmol["molecule_id"],
                        "tokens": length,
                        "atoms": len(chemmol["atoms"]),
                        "bonds": len(chemmol["bonds"]),
                    }
                fields["elements"].update(atom["element"] for atom in chemmol["atoms"])
                fields["chiral_tags"].update(atom["chiral_tag"] for atom in chemmol["atoms"])
                fields["bond_types"].update(bond["type"] for bond in chemmol["bonds"])
                fields["display_types"].update(bond["display_type"] for bond in render["bonds"])
                fields["bond_stereo"].update(bond["stereo"] for bond in chemmol["bonds"])
                fields["bond_directions"].update(bond["direction"] for bond in render["bonds"])
    return {
        "lengths": lengths,
        "atom_counts": atom_counts,
        "bond_counts": bond_counts,
        "maximum": maximum,
        "fields": {name: dict(values) for name, values in fields.items()},
    }


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
    return sorted(values)[index]


def describe(values: list[int]) -> dict:
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="webdataset_full_v0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    shards = sorted((root / "shards").glob("*.tar"))
    tokenizer = ChemMolTokenizer()
    all_lengths: list[int] = []
    all_atoms: list[int] = []
    all_bonds: list[int] = []
    maximum = None
    fields: dict[str, Counter] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(analyze_shard, str(path)) for path in shards]
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            all_lengths.extend(result["lengths"])
            all_atoms.extend(result["atom_counts"])
            all_bonds.extend(result["bond_counts"])
            candidate = result["maximum"]
            if maximum is None or candidate["tokens"] > maximum["tokens"]:
                maximum = candidate
            for field, counts in result["fields"].items():
                fields.setdefault(field, Counter()).update(counts)
            completed += 1
            if completed % 25 == 0 or completed == len(shards):
                print(f"Analyzed shards: {completed}/{len(shards)}", flush=True)

    if len(all_lengths) != 180_000:
        raise ValueError(f"Expected 180,000 molecules, got {len(all_lengths)}")
    result = {
        "status": "passed",
        "schema_version": "chemmolseq-0.2",
        "molecules": len(all_lengths),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "token_lengths": describe(all_lengths),
        "atom_counts": describe(all_atoms),
        "bond_counts": describe(all_bonds),
        "maximum_sequence": maximum,
        "field_counts": {
            field: dict(sorted(counts.items())) for field, counts in sorted(fields.items())
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
