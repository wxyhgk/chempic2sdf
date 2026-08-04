#!/usr/bin/env python3
"""Capacity-test one unified candidate-or-NULL pointer language for Atom queries."""

from __future__ import annotations

from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3NullPointerAtomQuerySet,
    g3_null_pointer_query_set_loss,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p2i_unique_pointer_set_capacity64 as p2i
from chemvector.training.atoms import train_g3_p2j_pointer_matched_query_set_capacity64 as p2j
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
)


MODEL_TYPE = "chemvector-g3-p2n-null-pointer-query-set-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
EXPECTED_INITIAL_SHA256 = (
    "c2be5b6f71882bbf08fe5f9fece347215523f9cb25d581d62374ee0707e0b8b3"
)
PROTOCOL = {
    **p2j.PROTOCOL,
    "experiment": "P2-N",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2n-null-pointer-query-set-capacity64-0.1",
    "audit_schema_version": "g3-p2n-capacity64-audit-0.1",
    "scope": "same-image capacity, unified candidate-or-NULL pointer language",
    "changed_variable": "replace independent existence head with NULL pointer class",
    "matching": "pointer-mass-query-to-gt-with-global-unique-candidate-or-null",
    "pointer_target": "matched-unique-candidate-unmatched-null",
    "decision_prefix": "null_pointer_query_set",
    "existence_objective": "logsumexp-candidate-pointer-minus-null-pointer",
    "same_architecture_data_initialization_budget_as_p2h": False,
    "model_factory": G3NullPointerAtomQuerySet,
    "expected_initial_sha256": EXPECTED_INITIAL_SHA256,
}


def training_step(model, optimizer, batch, *, max_grad_norm: float) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = g3_null_pointer_query_set_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-N gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> None:
    args = p2i.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2i.run_experiment(args, protocol=PROTOCOL, training_step_fn=training_step)


if __name__ == "__main__":
    main()
