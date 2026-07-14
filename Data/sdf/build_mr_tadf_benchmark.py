#!/usr/bin/env python3
"""Build a deterministic MR-TADF image benchmark from the audited whitelist."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


MANIFEST_FIELDS = (
    "molecule_id",
    "image_file",
    "annotation_file",
    "sdf_file",
    "source_sdf",
    "inchikey",
    "canonical_isomeric_smiles",
    "formula",
    "atoms",
    "bonds",
    "rings",
    "molecular_weight",
    "maximum_to_median_bond_ratio",
    "difficulty",
)


def enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def difficulty_bucket(maximum_to_median_bond_ratio: float) -> str:
    if maximum_to_median_bond_ratio >= 2.5:
        return "extreme_long_bond"
    if maximum_to_median_bond_ratio >= 1.75:
        return "long_bond"
    return "standard_bond_length"


def load_molecule(path: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SDF is unreadable or empty")
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("SDF contains multiple components")
    if mol.GetNumConformers() != 1:
        raise ValueError("SDF must contain exactly one conformer")
    conformer = mol.GetConformer()
    maximum_abs_z = max(
        abs(conformer.GetAtomPosition(index).z) for index in range(mol.GetNumAtoms())
    )
    if maximum_abs_z > 1e-4:
        raise ValueError(f"SDF is not strictly 2D: max_abs_z={maximum_abs_z:.6g}")
    return mol


def render_molecule(
    mol: Chem.Mol,
    *,
    molecule_id: str,
    width: int = 768,
    height: int = 768,
) -> tuple[bytes, dict]:
    chemical_bonds = {
        bond.GetIdx(): enum_name(bond.GetBondType()) for bond in mol.GetBonds()
    }
    kekulized = True
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol),
            kekulize=True,
            addChiralHs=False,
            wedgeBonds=True,
        )
    except Exception:
        kekulized = False
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol),
            kekulize=False,
            addChiralHs=False,
            wedgeBonds=True,
        )

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.rotate = 0.0
    options.bondLineWidth = 2.0
    options.fixedFontSize = 16
    options.padding = 0.08
    options.addStereoAnnotation = False
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.useBWAtomPalette()
    drawer.DrawMolecule(prepared)

    atom_rows = []
    for atom in prepared.GetAtoms():
        point = drawer.GetDrawCoords(atom.GetIdx())
        atom_rows.append(
            {
                "atom_id": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": bool(mol.GetAtomWithIdx(atom.GetIdx()).GetIsAromatic()),
                "image_x": round(point.x, 3),
                "image_y": round(point.y, 3),
            }
        )
    bond_rows = []
    for bond in prepared.GetBonds():
        begin = drawer.GetDrawCoords(bond.GetBeginAtomIdx())
        end = drawer.GetDrawCoords(bond.GetEndAtomIdx())
        bond_rows.append(
            {
                "bond_id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "display_type": enum_name(bond.GetBondType()),
                "bond_type": chemical_bonds[bond.GetIdx()],
                "direction": enum_name(bond.GetBondDir()),
                "begin_image_x": round(begin.x, 3),
                "begin_image_y": round(begin.y, 3),
                "end_image_x": round(end.x, 3),
                "end_image_y": round(end.y, 3),
            }
        )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    image = Image.open(io.BytesIO(png))
    image.load()
    if image.size != (width, height):
        raise ValueError(f"Rendered image has unexpected size: {image.size}")
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink_pixels = int((gray < 245).sum())
    if ink_pixels < 100:
        raise ValueError(f"Rendered image is blank: ink_pixels={ink_pixels}")
    annotation = {
        "schema_version": "mr-tadf-render-0.1",
        "molecule_id": molecule_id,
        "width": width,
        "height": height,
        "atoms": atom_rows,
        "bonds": bond_rows,
        "render_parameters": {
            "layout": "source_sdf_2d",
            "kekulize": kekulized,
            "rotation_degrees": 0.0,
            "bond_line_width": 2.0,
            "font_size": 16,
            "padding": 0.08,
            "color_mode": "black_white",
        },
        "ink_pixels": ink_pixels,
    }
    return png, annotation


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sample_outputs(
    *,
    png: bytes,
    annotation: dict,
    source_path: Path,
    image_path: Path,
    annotation_path: Path,
    sdf_path: Path,
) -> None:
    """Publish one sample atomically enough that failures leave no partial sample."""
    outputs = (image_path, annotation_path, sdf_path)
    temporary = tuple(path.with_name(f".{path.name}.tmp") for path in outputs)
    try:
        temporary[0].write_bytes(png)
        temporary[1].write_text(
            json.dumps(annotation, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(source_path, temporary[2])
        for staged, final in zip(temporary, outputs):
            os.replace(staged, final)
    except Exception:
        for path in (*temporary, *outputs):
            path.unlink(missing_ok=True)
        raise


def build_benchmark(
    approved_manifest: Path,
    source_dir: Path,
    output_dir: Path,
    *,
    width: int = 768,
    height: int = 768,
    limit: int | None = None,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Benchmark output directory is not empty: {output_dir}")
    with approved_manifest.open(newline="", encoding="utf-8") as handle:
        approved = list(csv.DictReader(handle))
    if limit is not None:
        approved = approved[:limit]

    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    ground_truth_dir = output_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    runtime_rejections = []
    difficulty_counts = Counter()
    for row in approved:
        molecule_id = row["molecule_id"]
        source_path = source_dir / row["sdf_file"]
        try:
            mol = load_molecule(source_path)
            png, annotation = render_molecule(
                mol,
                molecule_id=molecule_id,
                width=width,
                height=height,
            )
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey != row["inchikey"]:
                raise ValueError(
                    f"InChIKey changed after loading: {inchikey} != {row['inchikey']}"
                )
            ratio = float(row["maximum_to_median_bond_ratio"])
            difficulty = difficulty_bucket(ratio)
            image_path = image_dir / f"{molecule_id}.png"
            annotation_path = annotation_dir / f"{molecule_id}.json"
            sdf_path = ground_truth_dir / f"{molecule_id}.sdf"
            write_sample_outputs(
                png=png,
                annotation=annotation,
                source_path=source_path,
                image_path=image_path,
                annotation_path=annotation_path,
                sdf_path=sdf_path,
            )
            manifest_rows.append(
                {
                    "molecule_id": molecule_id,
                    "image_file": str(image_path.relative_to(output_dir)),
                    "annotation_file": str(annotation_path.relative_to(output_dir)),
                    "sdf_file": str(sdf_path.relative_to(output_dir)),
                    "source_sdf": str(source_path),
                    "inchikey": inchikey,
                    "canonical_isomeric_smiles": Chem.MolToSmiles(
                        mol,
                        canonical=True,
                        isomericSmiles=True,
                    ),
                    "formula": rdMolDescriptors.CalcMolFormula(mol),
                    "atoms": mol.GetNumAtoms(),
                    "bonds": mol.GetNumBonds(),
                    "rings": rdMolDescriptors.CalcNumRings(mol),
                    "molecular_weight": round(Descriptors.MolWt(mol), 6),
                    "maximum_to_median_bond_ratio": ratio,
                    "difficulty": difficulty,
                }
            )
            difficulty_counts[difficulty] += 1
        except Exception as error:
            runtime_rejections.append(
                {
                    "molecule_id": molecule_id,
                    "sdf_file": row["sdf_file"],
                    "error": str(error),
                }
            )

    write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, manifest_rows)
    write_csv(
        output_dir / "runtime_rejected.csv",
        ("molecule_id", "sdf_file", "error"),
        runtime_rejections,
    )
    manifest_digest = hashlib.sha256(
        (output_dir / "manifest.csv").read_bytes()
    ).hexdigest()
    summary = {
        "schema_version": "mr-tadf-benchmark-0.1",
        "approved_candidates": len(approved),
        "rendered_samples": len(manifest_rows),
        "runtime_rejections": len(runtime_rejections),
        "width": width,
        "height": height,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "manifest_sha256": manifest_digest,
        "source_policy": "Only audited approved.csv rows are eligible.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approved-manifest",
        default="Data/MR-TADF_SDF/audit/approved.csv",
    )
    parser.add_argument("--source-dir", default="Data/MR-TADF_SDF")
    parser.add_argument("--output-dir", default="Data/benchmarks/mr_tadf_v1")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.width < 64 or args.height < 64:
        raise ValueError("Benchmark image dimensions must be at least 64 pixels")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    RDLogger.DisableLog("rdApp.*")
    summary = build_benchmark(
        Path(args.approved_manifest),
        Path(args.source_dir),
        Path(args.output_dir),
        width=args.width,
        height=args.height,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
