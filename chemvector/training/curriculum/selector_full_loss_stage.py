"""Fixed ablation: original full set loss, unchanged selector update safeguards.

The prior SHA-bound experiment modules are immutable. Reuse their scope,
sampling and gates; this isolated control replaces only the scalar objective.
"""

import torch
from chemvector.proposal_object_set import proposal_object_set_loss
from .selector_retention_stage import (
    THRESHOLD, CORE_KEYS, SEED, tensors_to, configure, parameters, frozen_digest,
    call_selector, require_training_cache, training_order, local_gate, final_gate,
)


def objective(output, raw, bonds, labels):
    original = proposal_object_set_loss(output, labels["atom_presence_targets"], labels["bond_presence_targets"],
        raw["atom_mask"], bonds["bond_mask"], labels["target_atom_refs"], labels["bond_supervision_mask"],
        atom_decision_threshold=THRESHOLD, bond_decision_threshold=-.5)["loss"]
    return original, original, original.new_zeros(())


def update(model, cache, optimizer, limit, manifest_sha, device, reserve):
    require_training_cache(cache, manifest_sha)
    ps = parameters(model)
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in ps}:
        raise ValueError("Optimizer scope differs from selector")
    if not isinstance(optimizer, torch.optim.AdamW) or any(
            g["lr"] != 5e-6 or g["eps"] != 1e-8 or g["weight_decay"] != 0
            or g["betas"] != (.9, .999) or g["amsgrad"] or g["maximize"]
            for g in optimizer.param_groups):
        raise ValueError("Optimizer differs from fixed AdamW recipe")
    optimizer.zero_grad(set_to_none=True)
    raw, bonds, labels = [tensors_to(cache[k], device) for k in ("raw", "bonds", "labels")]
    output = call_selector(model.set_head, raw, bonds)
    loss, full, keep = objective(output, raw, bonds, labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite loss; no optimizer update")
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in ps):
        raise ValueError("Nonfinite gradient; no optimizer update")
    if not any(p.grad is not None for p in ps):
        raise ValueError("Missing selector gradients")
    norm = torch.nn.utils.clip_grad_norm_(ps, 1., error_if_nonfinite=True)
    limit.consume()
    reserve(limit.consumed)  # Durable intent precedes optimizer.step.
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in ps):
        raise ValueError("Nonfinite selector weights")
    return dict(loss=float(loss.detach()), original_full_loss=float(full.detach()),
                retention_loss=float(keep.detach()), gradient_norm=float(norm))
