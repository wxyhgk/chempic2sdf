#!/usr/bin/env python3
"""Validate ChemMolSeq v0.2 against one image variant for every molecule."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chemmol_sequence import (
    parse_chemmolseq,
    serialize_chemmolseq,
    validate_chemmolseq_roundtrip,
)


CHEMMOL_SUFFIX = "_original.chemmol.json"
RENDER_SUFFIX = "_original.render.json"


def validate_shard(path_text: str) -> dict:
    path = Path(path_text)
    pending: dict[str, dict] = {}
    molecules = 0
    atoms = 0
    bonds = 0
    radicals = 0
    stereo_bonds = 0

    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith(CHEMMOL_SUFFIX):
                key = member.name[: -len(".chemmol.json")]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read {member.name} from {path.name}")
                pending[key] = json.loads(extracted.read())
            elif member.name.endswith(RENDER_SUFFIX):
                key = member.name[: -len(".render.json")]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read {member.name} from {path.name}")
                render = json.loads(extracted.read())
                chemmol = pending.pop(key, None)
                if chemmol is None:
                    raise ValueError(f"Missing ChemMol JSON before {member.name} in {path.name}")
                sequence = serialize_chemmolseq(chemmol, render)
                parsed = parse_chemmolseq(sequence)
                validate_chemmolseq_roundtrip(
                    sequence,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )
                molecules += 1
                atoms += len(parsed.atoms)
                bonds += len(parsed.bonds)
                radicals += sum(atom.radical_electrons for atom in parsed.atoms)
                stereo_bonds += sum(
                    bond.stereo not in {"STEREONONE", "STEREOANY"}
                    for bond in parsed.bonds
                )

    if pending:
        raise ValueError(f"Unmatched original ChemMol JSON in {path.name}: {next(iter(pending))}")
    return {
        "shard": path.name,
        "molecules": molecules,
        "atoms": atoms,
        "bonds": bonds,
        "radical_electrons": radicals,
        "stereo_bonds": stereo_bonds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="webdataset_full_v0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    shards = sorted((root / "shards").glob("*.tar"))
    if args.limit_shards is not None:
        shards = shards[: args.limit_shards]
    if not shards:
        raise ValueError(f"No tar shards found under {root / 'shards'}")

    totals = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_shard, str(path)): path for path in shards}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            for key in ("molecules", "atoms", "bonds", "radical_electrons", "stereo_bonds"):
                totals[key] += result[key]
            completed += 1
            if completed % 25 == 0 or completed == len(shards):
                print(f"Validated ChemMolSeq v0.2 shards: {completed}/{len(shards)}", flush=True)

    result = {
        "status": "passed",
        "schema_version": "chemmolseq-0.2",
        "shards": len(shards),
        **dict(totals),
    }
    if args.limit_shards is None and totals["molecules"] != 180_000:
        raise ValueError(f"Expected 180,000 molecules, got {totals['molecules']}")
    output_text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(output_text, end="")
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
