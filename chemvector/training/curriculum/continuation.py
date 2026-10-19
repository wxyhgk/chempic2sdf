"""Continuation loss wiring for EXISTING ChemVector components; no new heads.

Proposal cells and set decisions are predicted, never replaced by target atoms.
This module computes losses; it does not load data, grant admission or run a GPU.
"""

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.dense_bond_object import MultiScaleDenseBondObjectDetector, dense_bond_object_loss
from chemvector.direct_atom_pair_bond import direct_atom_pair_loss, enumerate_unordered_atom_pairs
from chemvector.direct_atom_role import enumerate_selected_atoms
from chemvector.direct_atom_text_bbox import direct_atom_text_bbox_loss
from chemvector.predicted_incidence import (
    associate_predicted_proposals, decode_predicted_atom_proposals, decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.spatial_atom_object import spatial_atom_object_loss
from chemvector.train_direct_atom_pair_bond import build_pair_targets, match_atom_targets, select_training_pairs


class ContinuationComponents(nn.Module):
    """Container only: every child is an existing, checkpoint-compatible module."""

    def __init__(self, *, vision, atom, bond, set_head, pair, role, bbox):
        super().__init__()
        self.update_recipe = "legacy-all-v1"
        if not isinstance(bond, MultiScaleDenseBondObjectDetector):
            raise ValueError("This continuation recipe requires the deployed multiscale bond head")
        if bond.type_base is not None or bond.preserve_base_types:
            raise ValueError("Frozen type-base bond variants need a separate gradient audit")
        for name, module in (("vision", vision), ("atom", atom), ("bond", bond),
                             ("set_head", set_head), ("pair", pair), ("role", role), ("bbox", bbox)):
            setattr(self, name, module)
        self.requires_grad_(False)
        # Direct structure uses stride 4/8/16 only. Do not optimize the unused
        # stride-32 stage or token projection merely because they are in vision.
        self.vision.stem.requires_grad_(True)
        for stage in self.vision.stages[:5]:
            stage.requires_grad_(True)
        for name in ("atom", "bond", "set_head", "pair", "role", "bbox"):
            getattr(self, name).requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for module in (*self.vision.stages[5:], self.vision.output_norm, self.vision.projection):
            module.eval()
        if self.update_recipe == "freeze-role-v1":
            self.role.eval()
        return self

    def configure_update_recipe(self, recipe):
        """Opt-in fixed role freeze; no architecture, OCR or threshold changes."""
        if recipe not in {"legacy-all-v1", "freeze-role-v1"}:
            raise ValueError("Unknown continuation update recipe")
        if any(p.grad is not None for p in self.parameters()):
            raise ValueError("Configure update recipe before gradients or optimizer creation")
        self.update_recipe = recipe
        self.role.requires_grad_(recipe == "legacy-all-v1")
        self.train(self.training)
        return self

    def parameter_groups(self):
        # A single provisional recipe, not a hyperparameter search. Backbone
        # updates are smaller than existing head updates. An actual run still
        # requires reviewed data/mix and uses the budgeted launcher.
        rates = {"vision": 1e-6, "atom": 2e-5, "bond": 1e-5, "set_head": 5e-6,
                 "pair": 2e-5, "role": 2e-5, "bbox": 2e-5}
        groups = [{"name": name, "lr": rate, "params": [p for p in getattr(self, name).parameters()
                                                           if p.requires_grad]}
                  for name, rate in rates.items()]
        groups = [g for g in groups if g["params"]]
        identities = [id(p) for g in groups for p in g["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(identities) != len(set(identities)) or set(identities) != expected:
            raise ValueError("Continuation optimizer parameter coverage mismatch")
        return groups


def structure_forward(model, pixels, config, atom_threshold):
    """Same predicted structure path as deployment, with autograd left enabled."""
    features = model.vision.forward_stroke_resolution(pixels)
    atom_output = model.atom(features)
    stride8 = model.vision.forward_high_resolution_from_stroke(features)
    stride16 = stride8
    for stage in model.vision.stages[3:5]:
        stride16 = stage(stride16)
    bond_output = model.bond(features, stride8, stride16)
    atoms = decode_predicted_atom_proposals(atom_output, config)
    bonds = decode_predicted_bond_proposals(bond_output, config)
    set_output = model.set_head(
        atoms["atom_features"], atoms["atom_geometry"], atoms["atom_scores"], atoms["atom_mask"],
        bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"],
    )
    selected = {**atoms, "atom_mask": atoms["atom_mask"] & (set_output["atom_presence_logits"] >= atom_threshold)}
    return features, atom_output, bond_output, atoms, bonds, set_output, selected


def continuation_losses(model, batch, config, *, atom_threshold=0.0, bond_threshold=-0.5):
    features, atom_output, bond_output, atoms, bonds, set_output, selected = structure_forward(
        model, batch["pixel_values"], config, atom_threshold)
    # Matching produces labels only. The returned features retain their graph;
    # differentiating through Hungarian/discrete proposal choices is not claimed.
    proposals = associate_predicted_proposals(atoms, bonds, batch, config)
    atom_losses = spatial_atom_object_loss(atom_output, batch)
    bond_targets = {"endpoints": batch["bond_ports"], "display": batch["bond_display"],
                    "chemical": batch["bond_chemical"], "direction": batch["bond_direction"],
                    "mask": batch["bond_mask"]}
    bond_losses = dense_bond_object_loss(bond_output, bond_targets, heatmap_set_weight=.1,
        slot_matching="hungarian", cell_count_weight=.05, duplicate_weight=.02)
    set_losses = proposal_object_set_loss(set_output, proposals["atom_presence_targets"],
        proposals["bond_presence_targets"], atoms["atom_mask"], bonds["bond_mask"],
        proposals["target_atom_refs"], proposals["bond_supervision_mask"],
        atom_decision_threshold=atom_threshold, bond_decision_threshold=bond_threshold)
    pairs, rows = enumerate_unordered_atom_pairs(selected["atom_mask"])
    # Raw/set supervision keeps its own assignment above. Re-match only retained
    # candidates for pair AND text labels: a closer rejected raw candidate must
    # not consume their GT match and turn a real bond into a sampled negative.
    # This changes labels only; predicted candidates and matching radii stay fixed.
    with torch.no_grad():
        mapping, _ = match_atom_targets(selected, batch, config)
    pair_targets = build_pair_targets(pairs, rows, mapping, batch, selected)
    chosen = select_training_pairs(pairs, rows, selected, pair_targets, negative_ratio=6, minimum_negatives=48)
    # An empty branch is explicitly counted, not supplied oracle candidates or
    # synthetic zero gradients to make coverage appear complete.
    zero = atom_losses["loss"] * 0
    pair_loss = role_loss = bbox_loss = zero
    if len(chosen):
        pair_output = model.pair(features, selected["atom_features"], selected["atom_geometry"][..., :2],
                                selected["atom_scores"], pairs[chosen], rows[chosen])
        pair_loss = direct_atom_pair_loss(pair_output, pair_targets["presence"][chosen],
                                         pair_targets["chemical"][chosen])["loss"]
    atom_indices, sample_indices = enumerate_selected_atoms(selected["atom_mask"])
    mapped = mapping[sample_indices, atom_indices]
    valid = mapped >= 0
    atom_indices, sample_indices, mapped = atom_indices[valid], sample_indices[valid], mapped[valid]
    text = batch["atom_text_mask"][sample_indices, mapped]
    if len(atom_indices):
        # Role and bbox learn their own tasks; do not let crop classification
        # move the atom anchor to a convenient letter fragment.
        anchors = selected["atom_geometry"][..., :2].detach()
        if model.update_recipe != "freeze-role-v1":
            role_logits = model.role(batch["pixel_values"], anchors, atom_indices, sample_indices)
            labels = batch["atom_roles"][sample_indices, mapped].float()
            balance = ((labels == 0).sum().clamp_min(1) / (labels == 1).sum().clamp_min(1)).clamp(1, 8)
            role_loss = F.binary_cross_entropy_with_logits(role_logits.float(), labels, pos_weight=balance)
        if bool(text.any()):
            output = model.bbox(batch["pixel_values"], anchors, atom_indices[text], sample_indices[text])
            bbox_loss = direct_atom_text_bbox_loss(output, anchors[sample_indices[text], atom_indices[text]],
                batch["atom_text_bbox"][sample_indices[text], mapped[text]], model.bbox.config)["loss"]
    losses = {"atom": atom_losses["loss"], "bond": bond_losses["loss"], "set_head": set_losses["loss"],
              "pair": pair_loss, "role": role_loss, "bbox": bbox_loss}
    losses["loss"] = sum(losses.values())
    if any(not bool(torch.isfinite(value)) for value in losses.values()):
        raise ValueError("Non-finite continuation loss; no optimizer step is allowed")
    coverage = {
        "target_atoms": int(batch["atom_mask"].sum()), "target_bonds": int(batch["bond_mask"].sum()),
        "raw_predicted_atoms": int(atoms["atom_mask"].sum()), "selected_atoms": int(selected["atom_mask"].sum()),
        "matched_raw_atoms": int(proposals["atom_target_matched"].sum()), "matched_selected_atoms": len(mapped),
        "pair_rows": len(chosen), "reachable_positive_pairs": int(pair_targets["presence"].sum()),
        "role_rows": len(mapped), "bbox_rows": int(text.sum()),
        "role_update_rows": 0 if model.update_recipe == "freeze-role-v1" else len(mapped),
        "role_frozen": model.update_recipe == "freeze-role-v1",
        "oracle_candidates_injected": False,
    }
    return losses, coverage


def gradient_report(model):
    result = {}
    for group in model.parameter_groups():
        parameters = group["params"]
        present = [p.grad for p in parameters if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in present)
        if not finite:
            raise ValueError(f"Non-finite gradient in {group['name']}")
        result[group["name"]] = {"parameters": sum(p.numel() for p in parameters),
            "tensors": len(parameters), "with_grad": len(present),
            "nonzero_grad_tensors": sum(bool(torch.count_nonzero(g)) for g in present),
            "grad_l2": sum(float(g.double().square().sum()) for g in present) ** .5,
            "finite": finite}
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("Frozen continuation parameters received gradients")
    return result


def train_step(model, batch, config, optimizer, *, admission, step_limit,
               atom_threshold=0.0, bond_threshold=-0.5):
    """Guarded update primitive, not a data-release authority or full launcher.

The launcher binds the approved report to its exact data manifest. This primitive
additionally refuses diagnostic batches and unapproved reports.
"""
    from .admission import require_training_admission
    from .runtime import StepLimit

    require_training_admission(admission)
    if batch.get("purpose") != "training" or batch.get("training_authorized") is not True:
        raise ValueError("Diagnostic/quarantine batches cannot perform optimizer updates")
    if not isinstance(step_limit, StepLimit) or step_limit.consumed >= step_limit.maximum:
        raise RuntimeError("Optimizer step budget exhausted or missing")
    expected = {id(p) for g in model.parameter_groups() for p in g["params"]}
    actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer includes missing, duplicate or frozen parameters")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, coverage = continuation_losses(model, batch, config,
        atom_threshold=atom_threshold, bond_threshold=bond_threshold)
    losses["loss"].backward()
    gradients = gradient_report(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    # A failed optimizer call still consumes its reservation; no hidden retry.
    step_limit.consume()
    optimizer.step()
    return {"losses": {k: float(v.detach()) for k, v in losses.items()}, "coverage": coverage,
            "gradients": gradients, "gradient_norm_before_clip": float(norm),
            "optimizer_steps": step_limit.consumed}
