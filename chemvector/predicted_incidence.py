"""Target-independent spatial proposals for Atom/Bond incidence training."""

from __future__ import annotations

from dataclasses import dataclass, replace

from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PredictedProposalConfig:
    maximum_atoms: int = 128
    maximum_bonds: int = 128
    atom_threshold: float = 0.01
    bond_threshold: float = 0.01
    atom_match_radius_px: float = 6.0
    bond_match_radius_px: float = 12.0
    image_size: float = 768.0

    def __post_init__(self) -> None:
        if min(self.maximum_atoms, self.maximum_bonds) < 1:
            raise ValueError("Proposal capacities must be positive")
        if not 0.0 <= self.atom_threshold <= 1.0:
            raise ValueError("atom_threshold must be in [0, 1]")
        if not 0.0 <= self.bond_threshold <= 1.0:
            raise ValueError("bond_threshold must be in [0, 1]")
        if min(
            self.atom_match_radius_px,
            self.bond_match_radius_px,
            self.image_size,
        ) <= 0.0:
            raise ValueError("Matching radii and image size must be positive")


def _local_peak_topk(
    logits: torch.Tensor,
    *,
    maximum: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = logits.sigmoid()
    peaks = probabilities == F.max_pool2d(
        probabilities, kernel_size=3, stride=1, padding=1
    )
    scores = probabilities.masked_fill(~peaks, -1.0)
    batch = scores.size(0)
    flat = scores.flatten(1)
    count = min(maximum, flat.size(1))
    values, indices = flat.topk(count, dim=1)
    mask = values >= threshold
    if count < maximum:
        padding = maximum - count
        values = F.pad(values, (0, padding), value=-1.0)
        indices = F.pad(indices, (0, padding), value=0)
        mask = F.pad(mask, (0, padding), value=False)
    if values.shape != (batch, maximum):
        raise RuntimeError("Local peak decoder returned an invalid shape")
    return values, indices, mask


def decode_predicted_atom_proposals(
    output: dict[str, torch.Tensor],
    config: PredictedProposalConfig,
) -> dict[str, torch.Tensor]:
    scores, indices, mask = _local_peak_topk(
        output["heatmap_logits"],
        maximum=config.maximum_atoms,
        threshold=config.atom_threshold,
    )
    _, _, height, width = output["heatmap_logits"].shape
    spatial_size = height * width
    spatial_indices = indices.remainder(spatial_size)
    y = torch.div(spatial_indices, width, rounding_mode="floor")
    x = spatial_indices.remainder(width)
    batch = torch.arange(indices.size(0), device=indices.device)[:, None]
    features = output["object_features"].permute(0, 2, 3, 1)[batch, y, x]
    anchors = output["anchors"].permute(0, 2, 3, 1)[batch, y, x]
    roles = output["role_logits"].permute(0, 2, 3, 1)[batch, y, x].softmax(-1)
    text_centers = output["text_centers"].permute(0, 2, 3, 1)[batch, y, x]
    text_extents = output["text_extents"].permute(0, 2, 3, 1)[batch, y, x]
    geometry = torch.cat((anchors, roles, text_centers, text_extents), dim=-1)
    return {
        "atom_scores": scores,
        "atom_indices": spatial_indices,
        "atom_features": features,
        "atom_geometry": geometry,
        "atom_mask": mask,
    }


def fuse_predicted_atom_proposals(
    primary_output: dict[str, torch.Tensor],
    rescue_output: dict[str, torch.Tensor],
    config: PredictedProposalConfig,
    *,
    primary_capacity: int = 128,
    rescue_capacity: int = 16,
    dedup_radius_px: float = 6.0,
) -> dict[str, torch.Tensor]:
    """Append geometrically novel rescue peaks without shifting primary features.

    The deployed set/incidence heads were trained on the primary AtomObject feature
    space. Rescue coordinates therefore keep geometry from the higher-recall head,
    while their visual feature vector is sampled from the primary head at the same
    spatial cell. Existing primary candidates are never removed.
    """

    if min(primary_capacity, rescue_capacity) < 1:
        raise ValueError("Atom fusion capacities must be positive")
    if primary_capacity + rescue_capacity > config.maximum_atoms:
        raise ValueError("Fused Atom capacities exceed maximum_atoms")
    if dedup_radius_px <= 0.0:
        raise ValueError("Atom fusion dedup radius must be positive")
    if primary_output["heatmap_logits"].shape != rescue_output["heatmap_logits"].shape:
        raise ValueError("Primary and rescue Atom heatmaps differ in shape")

    primary = decode_predicted_atom_proposals(
        primary_output,
        replace(config, maximum_atoms=primary_capacity),
    )
    rescue = decode_predicted_atom_proposals(
        rescue_output,
        replace(config, maximum_atoms=max(primary_capacity, rescue_capacity * 4)),
    )
    batch_size = primary["atom_mask"].size(0)
    feature_dim = primary["atom_features"].size(-1)
    geometry_dim = primary["atom_geometry"].size(-1)
    maximum = config.maximum_atoms
    device = primary["atom_features"].device
    features = primary["atom_features"].new_zeros(
        (batch_size, maximum, feature_dim)
    )
    geometry = primary["atom_geometry"].new_zeros(
        (batch_size, maximum, geometry_dim)
    )
    scores = primary["atom_scores"].new_full((batch_size, maximum), -1.0)
    indices = torch.zeros((batch_size, maximum), device=device, dtype=torch.long)
    mask = torch.zeros((batch_size, maximum), device=device, dtype=torch.bool)
    sources = torch.zeros((batch_size, maximum), device=device, dtype=torch.long)
    _, _, height, width = primary_output["heatmap_logits"].shape
    primary_feature_map = primary_output["object_features"]
    primary_probabilities = primary_output["heatmap_logits"].sigmoid()[:, 0]

    for batch_index in range(batch_size):
        selected = torch.where(primary["atom_mask"][batch_index])[0]
        count = min(len(selected), primary_capacity)
        if count:
            selected = selected[:count]
            features[batch_index, :count] = primary["atom_features"][
                batch_index, selected
            ]
            geometry[batch_index, :count] = primary["atom_geometry"][
                batch_index, selected
            ]
            scores[batch_index, :count] = primary["atom_scores"][
                batch_index, selected
            ]
            indices[batch_index, :count] = primary["atom_indices"][
                batch_index, selected
            ]
            mask[batch_index, :count] = True
        rescued = 0
        for candidate in torch.where(rescue["atom_mask"][batch_index])[0]:
            if rescued >= rescue_capacity or count >= maximum:
                break
            anchor = rescue["atom_geometry"][batch_index, candidate, :2]
            if count:
                distance = torch.linalg.vector_norm(
                    (geometry[batch_index, :count, :2] - anchor) * config.image_size,
                    dim=-1,
                )
                if bool((distance <= dedup_radius_px).any()):
                    continue
            spatial_index = int(rescue["atom_indices"][batch_index, candidate])
            cell_y, cell_x = divmod(spatial_index, width)
            features[batch_index, count] = primary_feature_map[
                batch_index, :, cell_y, cell_x
            ]
            geometry[batch_index, count] = rescue["atom_geometry"][
                batch_index, candidate
            ]
            scores[batch_index, count] = torch.maximum(
                rescue["atom_scores"][batch_index, candidate],
                primary_probabilities[batch_index, cell_y, cell_x],
            )
            indices[batch_index, count] = spatial_index
            mask[batch_index, count] = True
            sources[batch_index, count] = 1
            count += 1
            rescued += 1
    return {
        "atom_scores": scores,
        "atom_indices": indices,
        "atom_features": features,
        "atom_geometry": geometry,
        "atom_mask": mask,
        "atom_sources": sources,
    }


def decode_predicted_bond_proposals(
    output: dict[str, torch.Tensor],
    config: PredictedProposalConfig,
) -> dict[str, torch.Tensor]:
    scores, indices, mask = _local_peak_topk(
        output["heatmap_logits"],
        maximum=config.maximum_bonds,
        threshold=config.bond_threshold,
    )
    batch_size, slots, height, width = output["heatmap_logits"].shape
    spatial_size = height * width
    slots_indices = torch.div(indices, spatial_size, rounding_mode="floor")
    spatial_indices = indices.remainder(spatial_size)
    y = torch.div(spatial_indices, width, rounding_mode="floor")
    x = spatial_indices.remainder(width)
    batch = torch.arange(batch_size, device=indices.device)[:, None]
    features = output["object_features"].permute(0, 2, 3, 1)[batch, y, x]
    endpoints = output["endpoints"].permute(0, 1, 4, 5, 2, 3)[
        batch, slots_indices, y, x
    ]
    display = output["display_logits"].permute(0, 1, 3, 4, 2)[
        batch, slots_indices, y, x
    ].softmax(-1)
    chemical = output["chemical_logits"].permute(0, 1, 3, 4, 2)[
        batch, slots_indices, y, x
    ].softmax(-1)
    return {
        "bond_scores": scores,
        "bond_indices": indices,
        "bond_features": features,
        "bond_endpoints": endpoints,
        "bond_types": torch.cat((display, chemical), dim=-1),
        "bond_mask": mask,
    }


def _target_bond_endpoints(targets: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = torch.arange(
        targets["bond_atom_refs"].size(0),
        device=targets["bond_atom_refs"].device,
    )[:, None, None]
    return targets["atom_anchors"][batch, targets["bond_atom_refs"]]


def associate_predicted_proposals(
    atom_proposals: dict[str, torch.Tensor],
    bond_proposals: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: PredictedProposalConfig,
) -> dict[str, torch.Tensor]:
    """Create supervision without using GT to choose proposal cells or counts."""

    batch_size, maximum_bonds = bond_proposals["bond_mask"].shape
    target_refs = torch.zeros(
        (batch_size, maximum_bonds, 2),
        device=targets["atom_anchors"].device,
        dtype=torch.long,
    )
    supervised_bonds = torch.zeros_like(bond_proposals["bond_mask"])
    atom_presence_targets = torch.zeros_like(atom_proposals["atom_mask"])
    atom_target_indices = torch.full_like(
        atom_proposals["atom_indices"], -1, dtype=torch.long
    )
    bond_presence_targets = torch.zeros_like(bond_proposals["bond_mask"])
    bond_target_indices = torch.full_like(
        bond_proposals["bond_indices"], -1, dtype=torch.long
    )
    bond_target_reversed = torch.zeros_like(bond_proposals["bond_mask"])
    atom_target_matched = torch.zeros_like(targets["atom_mask"])
    bond_target_matched = torch.zeros_like(targets["bond_mask"])
    incidence_target_eligible = torch.zeros_like(targets["bond_mask"])
    target_endpoints = _target_bond_endpoints(targets)

    for batch_index in range(batch_size):
        atom_indices = torch.where(atom_proposals["atom_mask"][batch_index])[0]
        atom_count = int(targets["atom_mask"][batch_index].sum())
        target_to_proposal = torch.full(
            (atom_count,),
            -1,
            device=targets["atom_anchors"].device,
            dtype=torch.long,
        )
        if len(atom_indices) and atom_count:
            predicted_atoms = atom_proposals["atom_geometry"][
                batch_index, atom_indices, :2
            ]
            expected_atoms = targets["atom_anchors"][batch_index, :atom_count]
            cost = torch.cdist(
                predicted_atoms.float(), expected_atoms.float()
            ) * config.image_size
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            for row, column in zip(rows, columns):
                if float(cost[row, column].detach()) <= config.atom_match_radius_px:
                    proposal_index = atom_indices[row]
                    target_to_proposal[column] = proposal_index
                    atom_target_indices[batch_index, proposal_index] = int(column)
                    atom_target_matched[batch_index, column] = True
                    atom_presence_targets[batch_index, proposal_index] = True

        bond_indices = torch.where(bond_proposals["bond_mask"][batch_index])[0]
        bond_count = int(targets["bond_mask"][batch_index].sum())
        if not len(bond_indices) or not bond_count:
            continue
        predicted_bonds = bond_proposals["bond_endpoints"][
            batch_index, bond_indices
        ].float()
        expected_bonds = target_endpoints[batch_index, :bond_count].float()
        forward = torch.linalg.vector_norm(
            (predicted_bonds[:, None] - expected_bonds[None]) * config.image_size,
            dim=-1,
        ).amax(-1)
        reverse = torch.linalg.vector_norm(
            (
                predicted_bonds[:, None]
                - expected_bonds[None].flip(2)
            )
            * config.image_size,
            dim=-1,
        ).amax(-1)
        cost = torch.minimum(forward, reverse)
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, column in zip(rows, columns):
            if float(cost[row, column].detach()) > config.bond_match_radius_px:
                continue
            proposal_index = bond_indices[row]
            bond_target_matched[batch_index, column] = True
            bond_presence_targets[batch_index, proposal_index] = True
            bond_target_indices[batch_index, proposal_index] = int(column)
            bond_target_reversed[batch_index, proposal_index] = (
                reverse[row, column] < forward[row, column]
            )
            refs = targets["bond_atom_refs"][batch_index, column]
            mapped = target_to_proposal[refs]
            if bool((mapped < 0).any()):
                continue
            incidence_target_eligible[batch_index, column] = True
            supervised_bonds[batch_index, proposal_index] = True
            target_refs[batch_index, proposal_index] = (
                mapped.flip(0) if reverse[row, column] < forward[row, column] else mapped
            )

    return {
        **atom_proposals,
        **bond_proposals,
        "bond_supervision_mask": supervised_bonds,
        "atom_presence_targets": atom_presence_targets,
        "atom_target_indices": atom_target_indices,
        "bond_presence_targets": bond_presence_targets,
        "bond_target_indices": bond_target_indices,
        "bond_target_reversed": bond_target_reversed,
        "target_atom_refs": target_refs,
        "atom_target_matched": atom_target_matched,
        "bond_target_matched": bond_target_matched,
        "incidence_target_eligible": incidence_target_eligible,
    }
