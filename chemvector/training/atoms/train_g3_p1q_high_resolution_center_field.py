#!/usr/bin/env python3
"""Capacity-test a 384x384 local Atom center field over frozen P1-A features."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    G3AtomCenterFieldHighResolutionResidual,
)
from chemvector.training.atoms import train_g3_p1p_atom_center_field_local_residual as p1p


MODEL_TYPE = "chemvector-g3-p1q-high-resolution-center-field-overfit32"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1q-high-resolution-center-field-overfit32")


@torch.no_grad()
def high_resolution_step0_parity(field, atom_head, hidden) -> dict[str, float | bool]:
    hidden = hidden.detach().float()
    output = field(hidden)
    expected_center = F.interpolate(
        atom_head.presence(hidden)[:, 0:1],
        scale_factor=2.0,
        mode="bilinear",
        align_corners=False,
    )
    center_error = float((output["center_logits"] - expected_center).abs().max())
    offset_error = float(output["offset_logits"].abs().max())
    return {
        "center_max_abs_error": center_error,
        "offset_zero_max_abs_error": offset_error,
        "exact": center_error == 0.0 and offset_error == 0.0,
    }


def build_parser():
    parser = p1p.build_parser()
    parser.set_defaults(output_dir=OUTPUT)
    return parser


def main() -> None:
    p1p.run_capacity(
        build_parser().parse_args(),
        field_factory=G3AtomCenterFieldHighResolutionResidual,
        expected_output=OUTPUT,
        model_type=MODEL_TYPE,
        parity_fn=high_resolution_step0_parity,
    )


if __name__ == "__main__":
    main()
