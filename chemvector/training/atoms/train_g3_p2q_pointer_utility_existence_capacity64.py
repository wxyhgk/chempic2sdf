#!/usr/bin/env python3
"""Calibrate existence from frozen query state plus explicit pointer utilities."""

from __future__ import annotations

from pathlib import Path

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3PointerUtilityExistenceModel,
)
from chemvector.training.atoms import train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m


MODEL_TYPE = "chemvector-g3-p2q-pointer-utility-existence-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
TRAINABLE_PARAMETERS = 9_877
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-Q",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2q-pointer-utility-existence-0.1",
    "audit_schema_version": "g3-p2q-capacity64-audit-0.1",
    "scope": "frozen P2-J representation plus target-independent pointer utilities",
    "existence_objective": "fixed-pointer-label-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "pointer_utility_existence",
    "model_wrapper_factory": G3PointerUtilityExistenceModel,
    "model_wrapper_type": "frozen-query-plus-10d-pointer-utility",
    "existence_head_type": "layernorm74-linear128-gelu-linear1",
    "trainable_parameters": TRAINABLE_PARAMETERS,
}


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
