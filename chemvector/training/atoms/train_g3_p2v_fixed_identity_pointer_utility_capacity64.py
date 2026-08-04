#!/usr/bin/env python3
"""Test pointer-utility existence capacity against the correct frozen identity."""

from __future__ import annotations

from pathlib import Path

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3PointerUtilityExistenceModel,
)
from chemvector.training.atoms import (
    train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m,
)
from chemvector.training.atoms import (
    train_g3_p2r_p2j_identity_existence_capacity64 as p2r,
)
from chemvector.training.atoms import (
    train_g3_p2s_fixed_identity_listwise_existence_capacity64 as p2s,
)


MODEL_TYPE = "chemvector-g3-p2v-fixed-identity-pointer-utility-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
TRAINABLE_PARAMETERS = 9_877
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-V",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2v-fixed-identity-pointer-utility-0.1",
    "audit_schema_version": "g3-p2v-capacity64-audit-0.1",
    "scope": "frozen P2-J queries plus pointer utility, correct fixed P2-R identity",
    "existence_objective": "fixed-identity-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "fixed_identity_pointer_utility",
    "query_matching": "frozen-p2r-p2j-existence-aware-action-cache",
    "existence_removed_from_matching_cost": False,
    "model_wrapper_factory": G3PointerUtilityExistenceModel,
    "model_wrapper_type": "frozen-query-plus-10d-pointer-utility",
    "existence_head_type": "layernorm74-linear128-gelu-linear1",
    "trainable_parameters": TRAINABLE_PARAMETERS,
    "prepare_training_context": p2s.prepare_training_context,
    "collate_fn": p2r.collate,
    "training_step_fn": p2r.training_step,
}


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
