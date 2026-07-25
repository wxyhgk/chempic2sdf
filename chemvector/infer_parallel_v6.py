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
from chemvector.proposal_keep import load_proposal_keep_checkpoint
from chemvector.proposal_graph import load_proposal_graph_checkpoint
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
    parser.add_argument("--vertex-nms-kernel-size", type=int, default=3)
    parser.add_argument("--proposal-keep-checkpoint")
    parser.add_argument("--proposal-keep-threshold", type=float, default=0.74)
    parser.add_argument("--proposal-graph-checkpoint")
    parser.add_argument("--proposal-graph-threshold", type=float, default=0.87)
    parser.add_argument("--proposal-graph-edge-prior-threshold", type=float)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument(
        "--repair-candidate-mode",
        choices=("all", "model", "model_connect"),
        default="all",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if args.vertex_nms_kernel_size < 1 or args.vertex_nms_kernel_size % 2 == 0:
        raise ValueError("vertex_nms_kernel_size must be a positive odd integer")
    if not 0.0 < args.proposal_keep_threshold < 1.0:
        raise ValueError("proposal_keep_threshold must be in (0, 1)")
    if not 0.0 < args.proposal_graph_threshold < 1.0:
        raise ValueError("proposal_graph_threshold must be in (0, 1)")
    if (
        args.proposal_graph_edge_prior_threshold is not None
        and not 0.0 < args.proposal_graph_edge_prior_threshold < 1.0
    ):
        raise ValueError("proposal_graph_edge_prior_threshold must be in (0, 1)")
    if args.proposal_graph_edge_prior_threshold is not None and not args.proposal_graph_checkpoint:
        raise ValueError("Proposal graph edge prior requires --proposal-graph-checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    started = time.perf_counter()
    model, _ = load_model(Path(args.checkpoint), device)
    strip_autoregressive_modules(model)
    proposal_keep_head = None
    if args.proposal_keep_checkpoint:
        proposal_keep_head, proposal_payload = load_proposal_keep_checkpoint(
            args.proposal_keep_checkpoint,
            device,
        )
        if Path(proposal_payload["dense_checkpoint"]).resolve() != Path(
            args.checkpoint
        ).resolve():
            raise ValueError("Proposal keep dense checkpoint does not match --checkpoint")
    proposal_graph_head = None
    if args.proposal_graph_checkpoint:
        proposal_graph_head, graph_payload = load_proposal_graph_checkpoint(
            args.proposal_graph_checkpoint,
            device,
        )
        if Path(graph_payload["dense_checkpoint"]).resolve() != Path(
            args.checkpoint
        ).resolve():
            raise ValueError("Proposal graph dense checkpoint does not match --checkpoint")
        expected_keep = graph_payload.get("proposal_keep_checkpoint")
        if expected_keep and not args.proposal_keep_checkpoint:
            raise ValueError("Proposal graph checkpoint requires --proposal-keep-checkpoint")
        if expected_keep and Path(expected_keep).resolve() != Path(
            args.proposal_keep_checkpoint
        ).resolve():
            raise ValueError("Proposal graph and keep checkpoints do not match")
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
        vertex_nms_kernel_size=args.vertex_nms_kernel_size,
        proposal_keep_head=proposal_keep_head,
        proposal_keep_threshold=args.proposal_keep_threshold,
        proposal_graph_head=proposal_graph_head,
        proposal_graph_threshold=args.proposal_graph_threshold,
        proposal_graph_edge_prior_threshold=args.proposal_graph_edge_prior_threshold,
        edge_threshold=args.edge_threshold,
        ocr_roi_radius=args.ocr_roi_radius,
    )
    repair_started = time.perf_counter()
    repaired = repair_chemsvg_v4_from_image(
        parallel.sequence,
        image,
        pair_predictions=parallel.pair_predictions,
        config=InkGraphRepairConfig(
            aromatic_vote_source="vertices",
            candidate_pair_mode=args.repair_candidate_mode,
        ),
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
        "model_bridge_strokes": repaired.model_bridge_strokes,
        "edge_threshold": args.edge_threshold,
        "proposal_graph_edge_prior_threshold": (
            args.proposal_graph_edge_prior_threshold
        ),
        "repair_candidate_mode": args.repair_candidate_mode,
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
