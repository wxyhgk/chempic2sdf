#!/usr/bin/env python3
"""Fill strict-validation gaps in the full v0 WebDataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import fastjsonschema
from rdkit import Chem, RDLogger, rdBase

import build_webdataset_sample as pipeline
import build_full_webdataset as full
import prepare_samples as filters


ROOT = Path("webdataset_full_v0")
RAW = {
    "chembl": Path("extracted/raw/chembl_37.sdf"),
    "coconut": Path("extracted/raw/coconut_sdf_2d_lite-07-2026.sdf"),
}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.perf_counter()
    plan_rows = load_rows(ROOT / "plan_molecules.csv")
    molecule_rows = load_rows(ROOT / "molecules.csv")
    sample_rows = load_rows(ROOT / "samples.csv")
    valid_ids = {row["molecule_id"] for row in molecule_rows}
    missing_plan = [row for row in plan_rows if row["molecule_id"] not in valid_ids]
    source_deficits = Counter(row["source"] for row in missing_plan)
    missing_slots: dict[str, list[int]] = defaultdict(list)
    for row in missing_plan:
        missing_slots[row["split"]].append(int(row["split_index"]))
    for slots in missing_slots.values():
        slots.sort(reverse=True)

    excluded_smiles = {row["canonical_isomeric_smiles"] for row in plan_rows}
    scaffold_split = {row["scaffold"]: row["split"] for row in molecule_rows}
    chemmol_validator = fastjsonschema.compile(
        json.loads(Path("schemas/chemmol-0.1.schema.json").read_text())
    )
    render_validator = fastjsonschema.compile(
        json.loads(Path("schemas/render-0.1.schema.json").read_text())
    )
    accepted = []
    accepted_counts = Counter()
    candidate_rejections = []

    for source in ("chembl", "coconut"):
        target = source_deficits[source]
        supplier = Chem.SDMolSupplier(
            str(RAW[source]), sanitize=True, removeHs=False, strictParsing=False
        )
        for source_index, mol in enumerate(supplier, 1):
            if accepted_counts[source] >= target:
                break
            if mol is None:
                continue
            item, reason = filters.analyze(mol, source, source_index)
            if item is None or item.smiles in excluded_smiles:
                continue
            if source == "coconut" and filters.coconut_tier(item) is None:
                continue
            record = None
            try:
                record = pipeline.canonicalize_record(source, item.molecule, item.smiles, source_index)
                fixed_split = scaffold_split.get(record.scaffold)
                if fixed_split is not None:
                    if not missing_slots[fixed_split]:
                        continue
                    split = fixed_split
                else:
                    available = [name for name, slots in missing_slots.items() if slots]
                    if not available:
                        break
                    split = max(
                        available,
                        key=lambda name: (
                            len(missing_slots[name]),
                            pipeline.stable_int(pipeline.SEED, record.scaffold, name),
                        ),
                    )
                chemmol = pipeline.build_chemmol(record)
                chemmol_validator(chemmol)
                pipeline.validate_chemical_roundtrip(chemmol)
                rendered = []
                maximum_delta = 0.0
                for variant in pipeline.VARIANTS:
                    png, render = pipeline.build_render(record, variant, source_index)
                    render_validator(render)
                    delta = pipeline.validate_geometry_roundtrip(chemmol, render)
                    maximum_delta = max(maximum_delta, delta)
                    rendered.append(
                        (variant, png, render, pipeline.serialize_chemmolseq(chemmol, render))
                    )
            except Exception as error:
                candidate_rejections.append(
                    {
                        "source": source,
                        "source_index": source_index,
                        "molecule_id": record.molecule_id if record is not None else "",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            record.split = split
            split_index = missing_slots[split].pop()
            scaffold_split[record.scaffold] = split
            excluded_smiles.add(record.smiles)
            accepted.append((record, split_index, chemmol, rendered, maximum_delta))
            accepted_counts[source] += 1
            if len(accepted) % 100 == 0:
                print(
                    f"supplement: accepted {len(accepted)}/{sum(source_deficits.values())}; "
                    f"chembl={accepted_counts['chembl']}/{source_deficits['chembl']}; "
                    f"coconut={accepted_counts['coconut']}/{source_deficits['coconut']}",
                    flush=True,
                )

    remaining = {split: len(slots) for split, slots in missing_slots.items() if slots}
    if accepted_counts != source_deficits or remaining:
        raise RuntimeError(
            f"Could not fill deficits: accepted={dict(accepted_counts)}, "
            f"source_deficits={dict(source_deficits)}, remaining_slots={remaining}"
        )

    supplement_results = []
    for split in pipeline.SPLIT_FRACTIONS:
        records = [entry for entry in accepted if entry[0].split == split]
        if not records:
            continue
        stem = f"{split}-supplement-000000"
        tar_path = ROOT / "shards" / f"{stem}.tar"
        temp_path = tar_path.with_suffix(".tar.tmp")
        structure_dir = ROOT / "structures" / stem
        structure_dir.mkdir(parents=True, exist_ok=True)
        new_molecules = []
        new_samples = []
        max_delta = 0.0
        with tarfile.open(temp_path, "w") as archive:
            for record, split_index, chemmol, rendered, molecule_delta in records:
                structure_name = (
                    f"{pipeline.clean_identifier(record.source)}_"
                    f"{pipeline.clean_identifier(record.source_id)}.sdf"
                )
                structure_ref = f"structures/{stem}/{structure_name}"
                structure_path = ROOT / structure_ref
                writer = Chem.SDWriter(str(structure_path))
                writer.write(record.mol)
                writer.close()
                chemmol["structure_ref"] = structure_ref
                new_molecules.append(
                    {
                        "molecule_id": record.molecule_id,
                        "source": record.source,
                        "source_id": record.source_id,
                        "split": split,
                        "split_index": split_index,
                        "shard": f"shards/{stem}.tar",
                        "scaffold": record.scaffold,
                        "stratum": record.stratum,
                        "canonical_isomeric_smiles": record.smiles,
                        "inchikey": record.inchikey,
                        "structure_ref": structure_ref,
                    }
                )
                for variant, png, render, sequence in rendered:
                    sample_key = f"{split}_{split_index:06d}_{variant}"
                    meta = {
                        "schema_version": "sample-meta-0.1",
                        "sample_key": sample_key,
                        "molecule_id": record.molecule_id,
                        "source": record.source,
                        "source_id": record.source_id,
                        "split": split,
                        "variant": variant,
                        "structure_ref": structure_ref,
                        "rdkit_version": rdBase.rdkitVersion,
                        "seed": pipeline.SEED,
                        "chemmolseq_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    }
                    pipeline.tar_add_bytes(archive, f"{sample_key}.png", png)
                    pipeline.tar_add_bytes(archive, f"{sample_key}.chemmol.json", pipeline.json_bytes(chemmol))
                    pipeline.tar_add_bytes(archive, f"{sample_key}.render.json", pipeline.json_bytes(render))
                    pipeline.tar_add_bytes(archive, f"{sample_key}.meta.json", pipeline.json_bytes(meta))
                    pipeline.tar_add_bytes(archive, f"{sample_key}.chemmolseq.txt", sequence.encode())
                    new_samples.append(
                        {
                            "sample_key": sample_key,
                            "molecule_id": record.molecule_id,
                            "split": split,
                            "variant": variant,
                            "shard": f"shards/{stem}.tar",
                            "color_mode": render["render_parameters"]["color_mode"],
                            "layout": render["render_parameters"]["layout"],
                            "stress_operation": render["render_parameters"]["stress_operation"] or "",
                        }
                    )
                max_delta = max(max_delta, molecule_delta)
        os.replace(temp_path, tar_path)
        molecule_rows.extend(new_molecules)
        sample_rows.extend(new_samples)
        supplement_results.append(
            {
                "split": split,
                "molecules": len(new_molecules),
                "samples": len(new_samples),
                "shard": f"shards/{stem}.tar",
                "bytes": tar_path.stat().st_size,
                "max_geometry_delta_pixels": max_delta,
            }
        )

    molecule_rows.sort(key=lambda row: (row["split"], int(row["split_index"])))
    sample_rows.sort(key=lambda row: row["sample_key"])
    full.write_csv(ROOT / "molecules.csv", molecule_rows)
    full.write_csv(ROOT / "samples.csv", sample_rows)
    full.write_csv(ROOT / "supplement_rejected.csv", candidate_rejections)

    summary = json.loads((ROOT / "summary.json").read_text())
    summary.update(
        {
            "valid_molecules": len(molecule_rows),
            "samples": len(sample_rows),
            "source_counts": dict(Counter(row["source"] for row in molecule_rows)),
            "split_counts": dict(Counter(row["split"] for row in molecule_rows)),
            "variant_counts": dict(Counter(row["variant"] for row in sample_rows)),
            "color_mode_counts": dict(Counter(row["color_mode"] for row in sample_rows)),
            "stress_operation_counts": dict(
                Counter(row["stress_operation"] for row in sample_rows if row["stress_operation"])
            ),
            "supplement_molecules": len(accepted),
            "supplement_candidate_rejections": len(candidate_rejections),
            "supplement_shards": len(supplement_results),
            "supplement_bytes": sum(result["bytes"] for result in supplement_results),
            "supplement_elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    summary["bytes"] = sum(path.stat().st_size for path in (ROOT / "shards").glob("*.tar"))
    summary["shards"] = len(list((ROOT / "shards").glob("*.tar")))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (ROOT / "supplement_summary.json").write_text(
        json.dumps(
            {
                "source_deficits": dict(source_deficits),
                "accepted_counts": dict(accepted_counts),
                "candidate_rejections": len(candidate_rejections),
                "results": supplement_results,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
