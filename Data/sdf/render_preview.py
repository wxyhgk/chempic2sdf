#!/usr/bin/env python3
"""Render four depiction variants for ten representative molecules."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


INPUT = Path("prepared/train/chembl_150k.sdf.gz")
OUTPUT = Path("preview_images")
SIZE = 512
CATEGORIES = ("B", "Se", "Si", "P", "S", "halogen", "charged", "polycyclic", "stereo", "base")
VARIANTS = (
    {"name": "original", "layout": "original", "line_width": 1.6, "font": 16, "padding": 0.08, "rotate": 0},
    {"name": "rdkit", "layout": "rdkit", "line_width": 2.0, "font": 17, "padding": 0.06, "rotate": 0},
    {"name": "coordgen", "layout": "coordgen", "line_width": 2.4, "font": 18, "padding": 0.05, "rotate": 0},
    {"name": "rotated", "layout": "rdkit", "line_width": 1.8, "font": 16, "padding": 0.10, "rotate": 12},
)


def category_for(mol: Chem.Mol, missing: set[str]) -> str | None:
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    checks = {
        "B": "B" in elements,
        "Se": "Se" in elements,
        "Si": "Si" in elements,
        "P": "P" in elements,
        "S": "S" in elements,
        "halogen": bool(elements & {"F", "Cl", "Br", "I"}),
        "charged": any(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "polycyclic": mol.GetRingInfo().NumRings() >= 4,
        "stereo": any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
        ),
        "base": elements <= {"C", "N", "O"},
    }
    for category in CATEGORIES:
        if category in missing and checks[category]:
            return category
    return None


def select_molecules() -> list[tuple[str, Chem.Mol]]:
    selected: dict[str, Chem.Mol] = {}
    with gzip.open(INPUT, "rb") as stream:
        supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            category = category_for(mol, set(CATEGORIES) - selected.keys())
            if category is not None:
                selected[category] = Chem.Mol(mol)
            if len(selected) == len(CATEGORIES):
                break
    missing = set(CATEGORIES) - selected.keys()
    if missing:
        raise RuntimeError(f"Could not find preview molecules for: {sorted(missing)}")
    return [(category, selected[category]) for category in CATEGORIES]


def prepare_layout(mol: Chem.Mol, layout: str) -> Chem.Mol:
    result = Chem.Mol(mol)
    if layout == "rdkit":
        rdDepictor.Compute2DCoords(result, canonOrient=True, clearConfs=True)
    elif layout == "coordgen":
        result.RemoveAllConformers()
        rdCoordGen.AddCoords(result)
    elif result.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(result, canonOrient=True, clearConfs=True)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    try:
        return rdMolDraw2D.PrepareMolForDrawing(
            result, kekulize=True, addChiralHs=False, wedgeBonds=True
        )
    except Exception:
        return rdMolDraw2D.PrepareMolForDrawing(
            result, kekulize=False, addChiralHs=False, wedgeBonds=True
        )


def render(mol: Chem.Mol, config: dict, destination: Path) -> None:
    prepared = prepare_layout(mol, config["layout"])
    drawer = rdMolDraw2D.MolDraw2DCairo(SIZE, SIZE)
    options = drawer.drawOptions()
    options.bondLineWidth = config["line_width"]
    options.fixedFontSize = config["font"]
    options.padding = config["padding"]
    options.rotate = config["rotate"]
    options.addStereoAnnotation = False
    options.useBWAtomPalette()
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    destination.write_bytes(drawer.GetDrawingText())


def make_contact_sheet(rows: list[dict], part: int) -> Path:
    subset = rows[(part - 1) * 5 : part * 5]
    thumb = 256
    label_width = 180
    header_height = 42
    row_height = 286
    canvas = Image.new("RGB", (label_width + thumb * 4, header_height + row_height * 5), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    for column, variant in enumerate(VARIANTS):
        draw.text(
            (label_width + column * thumb + 8, 12),
            variant["name"],
            fill="black",
            font=font,
        )
    for row_index, row in enumerate(subset):
        y = header_height + row_index * row_height
        draw.text((8, y + 90), row["category"], fill="black", font=font)
        draw.text((8, y + 116), row["source_id"][:21], fill="#444444", font=small_font)
        for column, image_path in enumerate(row["images"]):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                canvas.paste(image, (label_width + column * thumb, y))
    output = OUTPUT / f"contact_sheet_{part}.png"
    canvas.save(output, format="PNG", optimize=False)
    return output


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    selected = select_molecules()
    rows = []
    manifest_rows = []

    for index, (category, mol) in enumerate(selected, 1):
        source_id = mol.GetProp("dataset_source_id")
        molecule_dir = OUTPUT / f"{index:02d}_{category}"
        molecule_dir.mkdir()
        image_paths = []
        for variant in VARIANTS:
            path = molecule_dir / f"{variant['name']}.png"
            render(mol, variant, path)
            image_paths.append(path)
            manifest_rows.append(
                {
                    "molecule_index": index,
                    "category": category,
                    "source_id": source_id,
                    "canonical_smiles": mol.GetProp("canonical_smiles"),
                    "variant": variant["name"],
                    "image_path": str(path),
                    "bytes": path.stat().st_size,
                }
            )
        rows.append(
            {"category": category, "source_id": source_id, "images": image_paths}
        )

    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    sheets = [make_contact_sheet(rows, 1), make_contact_sheet(rows, 2)]
    elapsed = time.perf_counter() - started
    summary = {
        "molecules": len(selected),
        "images": len(manifest_rows),
        "image_size": [SIZE, SIZE],
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(len(manifest_rows) / elapsed, 2),
        "average_png_bytes": round(
            sum(row["bytes"] for row in manifest_rows) / len(manifest_rows)
        ),
        "contact_sheets": [str(path) for path in sheets],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
