from __future__ import annotations

import pytest

from chemvector.hybrid_cli import build_hybrid_parser


def test_minimum_cli_preserves_frozen_defaults() -> None:
    args = build_hybrid_parser().parse_args(
        ["--image", "input.png", "--output-dir", "result"]
    )
    assert args.device == "cuda"
    assert args.image_size == 768
    assert args.content_size == 548
    assert args.junction_source == "vertex"
    assert args.assignment_mode == "nearest"
    assert args.graph_correction_mode == "none"
    assert args.atom_decode_mode == "greedy"
    assert args.bond_utility_break_budget == "0"


def test_optional_expert_and_junction_arguments_remain_available() -> None:
    args = build_hybrid_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            "result",
            "--expert-bond-checkpoint",
            "expert.pt",
            "--style-gate-checkpoint",
            "gate.pt",
            "--bond-set-selector-checkpoint",
            "selector.pt",
            "--endpoint-grouper-checkpoint",
            "grouper.pt",
            "--line-band-checkpoint",
            "line.pt",
            "--junction-source",
            "junction_aware_bonds",
        ]
    )
    assert args.expert_bond_checkpoint == "expert.pt"
    assert args.style_gate_checkpoint == "gate.pt"
    assert args.junction_source == "junction_aware_bonds"


def test_required_paths_and_closed_choices_are_enforced() -> None:
    with pytest.raises(SystemExit):
        build_hybrid_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_hybrid_parser().parse_args(
            [
                "--image",
                "input.png",
                "--output-dir",
                "result",
                "--junction-source",
                "unknown",
            ]
        )
