#!/usr/bin/env python3
"""Resume P2-V with the same optimizer to a fixed 4,800-step endpoint."""

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
from chemvector.training.atoms import (
    train_g3_p2v_fixed_identity_pointer_utility_capacity64 as p2v,
)


MODEL_TYPE = "chemvector-g3-p2w-fixed-identity-pointer-utility-step4800"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p2v.OUTPUT / "step-001600.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "fab3efb9d06a613dac1d61f0b524d162a0223ea20143a7e5b5cb20148d1893a7"
)
TARGET_STEPS = 4_800
EXPECTED_EXPOSURE = 19_200
EXPECTED_ADDITIONAL_EPOCHS = 200
TRAINABLE_PARAMETERS = 9_877
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-W",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2w-fixed-identity-pointer-utility-step4800-0.1",
    "audit_schema_version": "g3-p2w-step4800-audit-0.1",
    "scope": "P2-V exact optimizer continuation to precommitted 4,800-step endpoint",
    "existence_objective": "fixed-identity-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "fixed_identity_pointer_utility_step4800",
    "query_matching": "frozen-p2r-p2j-existence-aware-action-cache",
    "existence_removed_from_matching_cost": False,
    "model_wrapper_factory": G3PointerUtilityExistenceModel,
    "model_wrapper_type": "frozen-query-plus-10d-pointer-utility",
    "existence_head_type": "layernorm74-linear128-gelu-linear1",
    "trainable_parameters": TRAINABLE_PARAMETERS,
    "prepare_training_context": p2s.prepare_training_context,
    "collate_fn": p2r.collate,
    "training_step_fn": p2r.training_step,
    "source_checkpoint": SOURCE_CHECKPOINT,
    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    "source_is_wrapped": True,
    "resume_optimizer": True,
    "target_steps": TARGET_STEPS,
    "expected_exposure": EXPECTED_EXPOSURE,
    "expected_epochs": EXPECTED_ADDITIONAL_EPOCHS,
}


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
