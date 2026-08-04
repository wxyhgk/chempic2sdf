#!/usr/bin/env python3
"""Train P2-J existence with a frozen identity and smooth listwise ordering."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.training.atoms import (
    train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m,
)
from chemvector.training.atoms import (
    train_g3_p2r_p2j_identity_existence_capacity64 as p2r,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p2s-fixed-identity-listwise-existence-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
P2R_ACTION_CACHE = p2r.OUTPUT / "p2j_identity_action_cache.json"
P2R_ACTION_CACHE_SHA256 = (
    "d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca"
)
PAIRWISE_MARGIN = 1.0
PAIRWISE_WEIGHT = 1.0
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-S",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2s-fixed-identity-listwise-existence-0.1",
    "audit_schema_version": "g3-p2s-capacity64-audit-0.1",
    "scope": "frozen P2-J representation and P2-R identity, smooth ordering ablation",
    "existence_objective": "fixed-identity-weighted-bce-plus-smooth-all-pairs-listwise",
    "boundary_weight": 0.0,
    "decision_prefix": "fixed_identity_listwise_existence",
    "query_matching": "frozen-p2r-p2j-existence-aware-action-cache",
    "existence_removed_from_matching_cost": False,
    "prepare_training_context": None,
    "collate_fn": p2r.collate,
    "training_step_fn": None,
}


def prepare_training_context(model, domains, *, device, output_dir):
    del model, device
    if file_sha256(P2R_ACTION_CACHE) != P2R_ACTION_CACHE_SHA256:
        raise ValueError("P2-S P2-R Action Cache SHA256 differs")
    payload = json.loads(P2R_ACTION_CACHE.read_text(encoding="utf-8"))
    lookup = p2r.validate_action_cache(payload)
    expected_keys = {
        str(record["sample_key"])
        for records in domains.values()
        for record in records
    }
    if set(lookup) != expected_keys:
        raise RuntimeError("P2-S capacity samples differ from frozen Action Cache")
    copied = copy.deepcopy(payload)
    path = output_dir / "p2j_identity_action_cache.json"
    _atomic_json(path, copied)
    if file_sha256(path) != P2R_ACTION_CACHE_SHA256:
        raise RuntimeError("P2-S copied Action Cache bytes differ")
    return lookup, {
        "path": str(path.resolve()),
        "source_path": str(P2R_ACTION_CACHE.resolve()),
        "file_sha256": P2R_ACTION_CACHE_SHA256,
        "entries_sha256": payload["entries_sha256"],
        "sample_count": len(lookup),
        "positive_queries": payload["positive_queries"],
        "matching_existence_cost_weight": 1.0,
        "pairwise_margin": PAIRWISE_MARGIN,
        "pairwise_weight": PAIRWISE_WEIGHT,
    }


def smooth_fixed_identity_listwise_loss(
    existence_logits: torch.Tensor,
    fixed_targets: torch.Tensor,
    *,
    margin: float = PAIRWISE_MARGIN,
    pairwise_weight: float = PAIRWISE_WEIGHT,
) -> dict[str, torch.Tensor]:
    base = p2r.fixed_identity_existence_loss(existence_logits, fixed_targets)
    if margin < 0.0 or pairwise_weight < 0.0:
        raise ValueError("P2-S listwise margin and weight must be nonnegative")
    ordering_losses = []
    for logits, mask in zip(existence_logits, fixed_targets):
        differences = logits[~mask, None] - logits[None, mask] + margin
        normalized_logsumexp = (
            torch.logsumexp(differences.flatten(), dim=0)
            - math.log(differences.numel())
        )
        ordering_losses.append(F.softplus(normalized_logsumexp))
    ordering_loss = torch.stack(ordering_losses).mean()
    return {
        **base,
        "loss": base["object_loss"] + pairwise_weight * ordering_loss,
        "listwise_ordering_loss": ordering_loss,
    }


def training_step(
    model, optimizer, batch, *, max_grad_norm: float, training_context
) -> dict:
    if training_context is None:
        raise RuntimeError("P2-S requires the frozen P2-R Action Cache")
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2r.p2h.model_candidates(batch))
        losses = smooth_fixed_identity_listwise_loss(
            output["existence_logits"], batch["fixed_query_targets"]
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.existence_head.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-S gradient norm is not finite")
    optimizer.step()
    if any(
        value.grad is not None
        for name, value in model.named_parameters()
        if not name.startswith("existence_head.")
    ):
        raise RuntimeError("P2-S produced gradients outside existence head")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


PROTOCOL.update({
    "prepare_training_context": prepare_training_context,
    "training_step_fn": training_step,
})


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
