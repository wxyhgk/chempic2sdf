#!/usr/bin/env python3
"""Continue P2-J with the same optimizer from 4,800 to 8,000 steps."""

from __future__ import annotations

from pathlib import Path

from chemvector.training.atoms import train_g3_p2i_unique_pointer_set_capacity64 as p2i
from chemvector.training.atoms import train_g3_p2j_pointer_matched_query_set_capacity64 as p2j


MODEL_TYPE = "chemvector-g3-p2k-pointer-matched-query-set-step8000"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
P2J_CHECKPOINT = p2j.OUTPUT / "step-004800.pt"
P2J_CHECKPOINT_SHA256 = (
    "a4e5031ae4994b73c62a9ca97625c720aa06d2f816cf322b109fed20f5f2ad0a"
)
PROTOCOL = {
    **p2j.PROTOCOL,
    "experiment": "P2-K",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2k-pointer-matched-query-set-step8000-0.1",
    "audit_schema_version": "g3-p2k-step8000-audit-0.1",
    "scope": "same-image capacity, P2-J fixed-optimizer budget continuation",
    "changed_variable": "training budget 4,800 to 8,000 steps",
    "decision_prefix": "pointer_matched_query_set_step8000",
    "resume_checkpoint": P2J_CHECKPOINT,
    "resume_checkpoint_sha256": P2J_CHECKPOINT_SHA256,
    "resume_model_type": p2j.MODEL_TYPE,
    "resume_step": 4_800,
    "target_steps": 8_000,
    "same_architecture_data_initialization_budget_as_p2h": False,
}


def main() -> None:
    args = p2i.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2i.run_experiment(args, protocol=PROTOCOL, training_step_fn=p2j.training_step)


if __name__ == "__main__":
    main()
