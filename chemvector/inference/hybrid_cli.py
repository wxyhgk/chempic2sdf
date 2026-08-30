"""Command-line contract for single-image ChemScene hybrid inference."""

from __future__ import annotations

import argparse


def build_hybrid_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
    )
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--expert-bond-checkpoint")
    parser.add_argument("--style-gate-checkpoint")
    parser.add_argument("--style-gate-threshold", type=float, default=0.90)
    parser.add_argument("--bond-utility-router-checkpoint")
    parser.add_argument(
        "--bond-utility-break-budget",
        choices=("0", "1", "2", "5", "10"),
        default="0",
    )
    parser.add_argument("--text-interior-bond-veto", action="store_true")
    parser.add_argument("--assembler-checkpoint")
    parser.add_argument("--graph-corrector-checkpoint")
    parser.add_argument("--ocr-checkpoint")
    parser.add_argument("--ctc-ocr-checkpoint")
    parser.add_argument("--expert-ctc-ocr-checkpoint")
    parser.add_argument("--invalid-fallback-ctc-ocr-checkpoint")
    parser.add_argument("--ocr-utility-router-checkpoint")
    parser.add_argument("--ocr-utility-router-threshold", type=float, default=0.95)
    parser.add_argument("--text-bbox-checkpoint")
    parser.add_argument("--ctc-anchor-fallback-margin", type=float)
    parser.add_argument("--bond-set-selector-checkpoint")
    parser.add_argument("--endpoint-grouper-checkpoint")
    parser.add_argument("--line-band-checkpoint")
    parser.add_argument("--semantic-bond-relation-checkpoint")
    parser.add_argument("--semantic-bond-existence-checkpoint")
    parser.add_argument("--semantic-bond-existence-threshold", type=float, default=0.60)
    parser.add_argument("--semantic-bond-existence-chunk-size", type=int, default=256)
    parser.add_argument(
        "--semantic-bond-relation-mode",
        choices=("chemical_all", "aromatic_ring_only"),
        default="chemical_all",
    )
    parser.add_argument("--semantic-aromatic-mean-threshold", type=float, default=0.72)
    parser.add_argument(
        "--semantic-aromatic-minimum-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--junction-projection-score-source",
        choices=("detector", "selector"),
        default="detector",
    )
    parser.add_argument(
        "--junction-bond-type-source",
        choices=("detector", "line_band"),
        default="detector",
    )
    parser.add_argument(
        "--graph-correction-mode",
        choices=("none", "type_only", "full", "invalid_only"),
        default="none",
    )
    parser.add_argument("--graph-keep-threshold", type=float)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--image-is-prepared", action="store_true")
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument(
        "--junction-source",
        choices=(
            "vertex",
            "bond_endpoints",
            "junction_aware_bonds",
            "semantic_relations",
            "junction_semantic_relations",
        ),
        default="vertex",
    )
    parser.add_argument("--endpoint-cluster-radius-pixels", type=float, default=12.0)
    parser.add_argument("--text-attach-radius-pixels", type=float, default=24.0)
    parser.add_argument(
        "--atom-decode-mode",
        choices=("greedy", "structured_invalid_only"),
        default="greedy",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("nearest", "global_unique", "learned", "learned_rescue"),
        default="nearest",
    )
    return parser
