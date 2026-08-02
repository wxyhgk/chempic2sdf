#!/usr/bin/env python3
"""Run detected ChemSVG generation, visible-line repair, and SDF export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import torch

from Data.sdf.chemsvg_sequence_v4 import parse_chemsvg_v4
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from chemvector.evaluate import load_model
from chemvector.generation_v4 import generate_detected_chemsvg_v4
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.repair_v4 import repair_chemsvg_v4_from_image
from chemvector.sdf_io import write_sdf_v4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--image-is-prepared", action="store_true")
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--stroke-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for detected inference")

    device = torch.device("cuda")
    model, tokenizer = load_model(Path(args.checkpoint), device)
    if not isinstance(tokenizer, ChemSVGTokenizerV4):
        raise ValueError("Detected inference requires a ChemSVG v0.4 checkpoint")
    source_image = Image.open(args.image)
    if args.image_is_prepared:
        image = source_image.convert("RGB")
    else:
        image = prepare_input_image(
            source_image,
            image_size=args.image_size,
            content_size=args.content_size,
        )
    if image.size != (args.image_size, args.image_size):
        image = image.resize(
            (args.image_size, args.image_size),
            Image.Resampling.BICUBIC,
        )
    pixels = image_tensor(image).unsqueeze(0).to(device)
    started = time.perf_counter()
    generated = generate_detected_chemsvg_v4(
        model,
        pixels,
        tokenizer,
        vertex_threshold=args.vertex_threshold,
        stroke_threshold=args.stroke_threshold,
        ocr_roi_radius=args.ocr_roi_radius,
    )
    if generated.sequence is None:
        raise RuntimeError(generated.error or "Detected generation produced no sequence")
    repaired = repair_chemsvg_v4_from_image(generated.sequence, image)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction-generated.chemsvg.txt").write_text(
        generated.sequence,
        encoding="utf-8",
    )
    (output_dir / "prediction-repaired.chemsvg.txt").write_text(
        repaired.sequence,
        encoding="utf-8",
    )
    write_sdf_v4(repaired.sequence, output_dir / "prediction-repaired.sdf")
    image.save(output_dir / "input-prepared.png")

    mol = Chem.MolFromMolFile(
        str(output_dir / "prediction-repaired.sdf"),
        removeHs=False,
    )
    if mol is None:
        raise RuntimeError("Cannot read the repaired SDF")
    drawer = rdMolDraw2D.MolDraw2DCairo(args.image_size, args.image_size)
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    (output_dir / "prediction-repaired.png").write_bytes(drawer.GetDrawingText())
    generated_parsed = parse_chemsvg_v4(generated.sequence)
    metrics = {
        "seconds": time.perf_counter() - started,
        "ocr_roi_radius": args.ocr_roi_radius,
        "generated_vertices": len(generated_parsed.vertices),
        "generated_strokes": len(generated_parsed.strokes),
        "generated_chemically_valid": generated.chemically_valid,
        "generated_error": generated.error,
        "repaired_vertices": repaired.vertices,
        "repaired_strokes": repaired.strokes,
        "removed_coordinates": repaired.removed_coordinates,
        "components": len(Chem.GetMolFrags(mol)),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "inchikey": Chem.MolToInchiKey(mol),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
