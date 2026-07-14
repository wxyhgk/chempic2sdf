#!/usr/bin/env python3
"""CPU-oriented parallel ChemSVG inference without autoregressive decoding."""

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

from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.parallel_v6 import (
    deployment_parameter_count,
    generate_parallel_chemsvg_v6,
    strip_autoregressive_modules,
)
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image
from chemvector.sdf_io import write_sdf_v4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--image-is-prepared", action="store_true")
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    started = time.perf_counter()
    model, _ = load_model(Path(args.checkpoint), device)
    strip_autoregressive_modules(model)
    load_seconds = time.perf_counter() - started
    source = Image.open(args.image)
    image = (
        source.convert("RGB")
        if args.image_is_prepared
        else prepare_input_image(
            source,
            image_size=args.image_size,
            content_size=args.content_size,
        )
    )
    if image.size != (args.image_size, args.image_size):
        image = image.resize(
            (args.image_size, args.image_size),
            Image.Resampling.BICUBIC,
        )
    pixels = image_tensor(image).unsqueeze(0).to(device)

    started = time.perf_counter()
    parallel = generate_parallel_chemsvg_v6(
        model,
        pixels,
        vertex_threshold=args.vertex_threshold,
        edge_threshold=args.edge_threshold,
        ocr_roi_radius=args.ocr_roi_radius,
    )
    repair_started = time.perf_counter()
    repaired = repair_chemsvg_v4_from_image(
        parallel.sequence,
        image,
        config=InkGraphRepairConfig(aromatic_vote_source="vertices"),
    )
    repair_seconds = time.perf_counter() - repair_started
    inference_seconds = time.perf_counter() - started

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "input-prepared.png")
    (output_dir / "prediction-parallel.chemsvg.txt").write_text(
        parallel.sequence,
        encoding="utf-8",
    )
    (output_dir / "prediction-repaired.chemsvg.txt").write_text(
        repaired.sequence,
        encoding="utf-8",
    )
    sdf_path = output_dir / "prediction-repaired.sdf"
    write_sdf_v4(repaired.sequence, sdf_path)
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise RuntimeError("Cannot read repaired SDF")
    drawer = rdMolDraw2D.MolDraw2DCairo(args.image_size, args.image_size)
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    (output_dir / "prediction-repaired.png").write_bytes(drawer.GetDrawingText())

    metrics = {
        "device": args.device,
        "deployment_parameters": deployment_parameter_count(model),
        "load_seconds": load_seconds,
        "vertex_ocr_seconds": parallel.vertex_ocr_seconds,
        "pair_seconds": parallel.pair_seconds,
        "repair_seconds": repair_seconds,
        "inference_seconds": inference_seconds,
        "vertices": repaired.vertices,
        "candidate_strokes": len(parallel.strokes),
        "repaired_strokes": repaired.strokes,
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
