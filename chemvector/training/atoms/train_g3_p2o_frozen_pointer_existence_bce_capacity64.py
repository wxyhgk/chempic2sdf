#!/usr/bin/env python3
"""Calibrate the frozen P2-J existence head with fixed-label BCE only."""

from __future__ import annotations

from pathlib import Path

from chemvector.training.atoms import train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m


MODEL_TYPE = "chemvector-g3-p2o-frozen-pointer-existence-bce-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-O",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2o-frozen-pointer-existence-bce-0.1",
    "audit_schema_version": "g3-p2o-capacity64-audit-0.1",
    "scope": "frozen P2-J representation, fixed-label weighted-BCE-only isolation",
    "existence_objective": "fixed-pointer-label-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "frozen_pointer_existence_bce",
}


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
