#!/usr/bin/env python3
"""Calibrate an 8.6k nonlinear existence MLP on frozen P2-J queries."""

from __future__ import annotations

from pathlib import Path

from chemvector.components.atoms.objects.g3_atom_query_set import G3ExistenceMLP
from chemvector.training.atoms import train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m


MODEL_TYPE = "chemvector-g3-p2p-frozen-pointer-existence-mlp-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
TRAINABLE_PARAMETERS = 8_577
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-P",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2p-frozen-pointer-existence-mlp-0.1",
    "audit_schema_version": "g3-p2p-capacity64-audit-0.1",
    "scope": "frozen P2-J representation, nonlinear existence-head capacity",
    "existence_objective": "fixed-pointer-label-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "frozen_pointer_existence_mlp",
    "existence_head_factory": G3ExistenceMLP,
    "existence_head_type": "layernorm64-linear128-gelu-linear1",
    "trainable_parameters": TRAINABLE_PARAMETERS,
}


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
