#!/usr/bin/env python3
"""Calibrate P2-J existence against its frozen existence-aware query identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_query_set import (
    hungarian_pointer_query_targets,
)
from chemvector.training.atoms import (
    train_g3_p2h_atom_query_set_capacity64 as p2h,
)
from chemvector.training.atoms import (
    train_g3_p2m_frozen_pointer_existence_head_capacity64 as p2m,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)


MODEL_TYPE = "chemvector-g3-p2r-p2j-identity-existence-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
QUERY_COUNT = 128
TRAINABLE_PARAMETERS = 65
PROTOCOL = {
    **p2m.P2M_PROTOCOL,
    "experiment": "P2-R",
    "model_type": MODEL_TYPE,
    "output": OUTPUT,
    "checkpoint_schema_version": "g3-p2r-p2j-identity-existence-0.1",
    "audit_schema_version": "g3-p2r-capacity64-audit-0.1",
    "scope": "frozen P2-J representation and frozen P2-J existence-aware identity",
    "existence_objective": "p2j-identity-action-cache-weighted-bce-only",
    "boundary_weight": 0.0,
    "decision_prefix": "p2j_identity_existence",
    "query_matching": "frozen-p2j-existence-aware-action-cache",
    "existence_removed_from_matching_cost": False,
    "prepare_training_context": None,
    "collate_fn": None,
    "training_step_fn": None,
}


def action_cache_digest(entries: list[dict]) -> str:
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_action_cache(payload: dict) -> dict[str, torch.Tensor]:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P2-R Action Cache has no entries")
    if payload.get("entries_sha256") != action_cache_digest(entries):
        raise ValueError("P2-R Action Cache digest differs")
    lookup = {}
    positive_queries = 0
    for entry in entries:
        key = entry.get("sample_key")
        mask = entry.get("query_mask")
        if not isinstance(key, str) or not key or key in lookup:
            raise ValueError("P2-R Action Cache sample keys are invalid")
        if (
            not isinstance(mask, list)
            or len(mask) != QUERY_COUNT
            or any(value not in (0, 1) for value in mask)
        ):
            raise ValueError("P2-R Action Cache query mask is invalid")
        tensor = torch.tensor(mask, dtype=torch.bool)
        if int(tensor.sum()) != entry.get("positive_queries"):
            raise ValueError("P2-R Action Cache positive count differs")
        lookup[key] = tensor
        positive_queries += int(tensor.sum())
    if payload.get("positive_queries") != positive_queries:
        raise ValueError("P2-R Action Cache total positive count differs")
    return lookup


@torch.no_grad()
def generate_action_cache_entries(model, domains, *, device: torch.device) -> list[dict]:
    model.eval()
    entries = []
    for domain, records in domains.items():
        for start in range(0, len(records), 4):
            current = records[start : start + 4]
            batch = p2h.collate(current, device)
            with _autocast_context(device, "bf16"):
                output = model(p2h.model_candidates(batch))
            assignments = hungarian_pointer_query_targets(
                output,
                batch["candidate_anchors"],
                batch["atom_anchors"],
                batch["full_atom_mask"],
                existence_cost_weight=1.0,
            )
            for index, (query_indices, _) in enumerate(assignments):
                mask = torch.zeros(QUERY_COUNT, dtype=torch.uint8)
                mask[query_indices.cpu()] = 1
                expected = int(batch["full_atom_mask"][index].sum())
                if int(mask.sum()) != expected:
                    raise RuntimeError("P2-R P2-J identity count differs from GT")
                entries.append({
                    "domain": domain,
                    "sample_key": str(current[index]["sample_key"]),
                    "positive_queries": expected,
                    "query_mask": mask.tolist(),
                })
    return entries


def prepare_training_context(model, domains, *, device, output_dir):
    first = generate_action_cache_entries(model, domains, device=device)
    repeated = generate_action_cache_entries(model, domains, device=device)
    if repeated != first:
        raise RuntimeError("P2-R P2-J identity cache is not deterministic")
    payload = {
        "schema_version": "g3-p2r-p2j-query-identity-action-cache-0.1",
        "source_checkpoint": str(p2m.P2J_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": p2m.P2J_CHECKPOINT_SHA256,
        "query_count": QUERY_COUNT,
        "sample_count": len(first),
        "positive_queries": sum(item["positive_queries"] for item in first),
        "entries_sha256": action_cache_digest(first),
        "repeat_generation_equal": True,
        "entries": first,
    }
    lookup = validate_action_cache(payload)
    if len(lookup) != 64 or payload["positive_queries"] != 1_774:
        raise RuntimeError("P2-R Action Cache capacity domain changed")
    path = output_dir / "p2j_identity_action_cache.json"
    _atomic_json(path, payload)
    audit = {
        "path": str(path.resolve()),
        "schema_version": payload["schema_version"],
        "sample_count": len(lookup),
        "positive_queries": payload["positive_queries"],
        "entries_sha256": payload["entries_sha256"],
        "repeat_generation_equal": True,
        "matching_existence_cost_weight": 1.0,
    }
    return lookup, audit


def collate(records, device, action_cache):
    batch = p2h.collate(records, device)
    try:
        masks = [action_cache[str(record["sample_key"])] for record in records]
    except KeyError as error:
        raise KeyError(f"P2-R sample is absent from Action Cache: {error}") from error
    batch["fixed_query_targets"] = torch.stack(masks).to(device)
    return batch


def fixed_identity_existence_loss(
    existence_logits: torch.Tensor, fixed_targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    if existence_logits.shape != fixed_targets.shape or fixed_targets.dtype != torch.bool:
        raise ValueError("P2-R fixed existence targets have invalid shape or dtype")
    if bool((fixed_targets.sum(dim=1) < 1).any()) or bool(
        (fixed_targets.sum(dim=1) >= fixed_targets.size(1)).any()
    ):
        raise ValueError("P2-R fixed existence targets need positive and negative queries")
    targets = fixed_targets.to(existence_logits.dtype)
    weights = torch.where(
        fixed_targets, torch.ones_like(existence_logits),
        torch.full_like(existence_logits, 0.1),
    )
    loss = F.binary_cross_entropy_with_logits(
        existence_logits, targets, weight=weights, reduction="sum"
    ) / weights.sum().clamp_min(1.0)
    boundary_exact = sum(
        int(bool(logits[mask].min() > logits[~mask].max()))
        for logits, mask in zip(existence_logits.detach(), fixed_targets)
    )
    return {
        "loss": loss,
        "object_loss": loss,
        "boundary_exact_images": loss.new_tensor(float(boundary_exact)),
        "matched_queries": loss.new_tensor(float(fixed_targets.sum())),
    }


def training_step(
    model, optimizer, batch, *, max_grad_norm: float, training_context
) -> dict:
    if training_context is None:
        raise RuntimeError("P2-R requires its frozen Action Cache")
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(batch["candidate_anchors"].device, "bf16"):
        output = model(p2h.model_candidates(batch))
        losses = fixed_identity_existence_loss(
            output["existence_logits"], batch["fixed_query_targets"]
        )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.existence_head.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P2-R gradient norm is not finite")
    optimizer.step()
    if any(
        value.grad is not None
        for name, value in model.named_parameters()
        if not name.startswith("existence_head.")
    ):
        raise RuntimeError("P2-R produced gradients outside existence head")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


PROTOCOL.update({
    "prepare_training_context": prepare_training_context,
    "collate_fn": collate,
    "training_step_fn": training_step,
})


def main() -> None:
    args = p2m.build_parser(output=OUTPUT, description=__doc__).parse_args()
    p2m.run_experiment(args, protocol=PROTOCOL)


if __name__ == "__main__":
    main()
