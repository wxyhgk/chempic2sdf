#!/usr/bin/env python3
"""Capacity-test pointer-mass query matching with globally unique peak targets."""

from __future__ import annotations

from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    g3_atom_query_set_loss,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p2i_unique_pointer_set_capacity64 as p2i
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
)


MODEL_TYPE = "chemvector-g3-p2j-pointer-matched-query-set-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
PROTOCOL = {
    "experiment": "P2-J",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2j-pointer-matched-query-set-capacity64-0.1",
    "audit_schema_version": "g3-p2j-capacity64-audit-0.1",
    "scope": "same-image capacity, P2-I controlled query-matching ablation",
    "changed_variable": "pointer-compatible probability mass query-to-GT matching",
    "matching": (
        "hungarian-pointer-mass-query-to-gt-then-global-unique-compatible-pointer"
    ),
    "pointer_target": "one-distinct-compatible-peak-per-pointer-matched-query",
    "decision_prefix": "pointer_matched_query_set",
}


def training_step(model, optimizer, batch, *, max_grad_norm: float) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = g3_atom_query_set_loss(
            output,
            batch["candidate_anchors"],
            batch["atom_anchors"],
            batch["full_atom_mask"],
            unique_pointers=True,
            query_matching="pointer",
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-J gradient norm is not finite")
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
