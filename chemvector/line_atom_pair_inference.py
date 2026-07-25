"""Append learned line-only AtomPairs as first-class Bond proposals."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from chemvector.line_segment_rescue import AtomPairLineCandidate


def _pad_dim1(tensor: torch.Tensor, count: int, value: float | int | bool) -> torch.Tensor:
    if count == 0:
        return tensor.clone()
    shape = list(tensor.shape)
    shape[1] = count
    padding = tensor.new_full(shape, value)
    return torch.cat((tensor, padding), dim=1)


def append_line_atom_pair_proposals(
    proposals: dict[str, torch.Tensor],
    set_output: dict[str, torch.Tensor],
    line_output: dict[str, torch.Tensor],
    line_candidates: list[tuple[AtomPairLineCandidate, ...]],
    sample_indices: torch.Tensor,
    *,
    line_threshold: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict]]:
    """Append line candidates with fixed endpoint incidence and learned types.

    Line presence logits are shifted by the checkpoint threshold so the combined
    decoder can continue using the AtomPair group threshold of zero.
    """

    batch_size, base_bonds = proposals["bond_mask"].shape
    if len(line_candidates) != batch_size:
        raise ValueError("line_candidates must contain one tuple per batch item")
    if len(sample_indices) != len(line_output["presence_logits"]):
        raise ValueError("Line output and sample indices differ")
    maximum_line = max((len(items) for items in line_candidates), default=0)
    if maximum_line == 0:
        return dict(proposals), dict(set_output), [
            {"candidates": 0, "selected": 0, "items": []}
            for _ in range(batch_size)
        ]

    combined_proposals = dict(proposals)
    combined_output = dict(set_output)
    combined_proposals["bond_mask"] = _pad_dim1(
        proposals["bond_mask"], maximum_line, False
    )
    combined_proposals["bond_features"] = _pad_dim1(
        proposals["bond_features"], maximum_line, 0.0
    )
    combined_proposals["bond_endpoints"] = _pad_dim1(
        proposals["bond_endpoints"], maximum_line, 0.0
    )
    combined_proposals["bond_types"] = _pad_dim1(
        proposals["bond_types"], maximum_line, 0.0
    )
    combined_proposals["bond_scores"] = _pad_dim1(
        proposals["bond_scores"], maximum_line, 0.0
    )
    if "bond_indices" in proposals:
        combined_proposals["bond_indices"] = _pad_dim1(
            proposals["bond_indices"], maximum_line, -1
        )
    combined_output["bond_presence_logits"] = _pad_dim1(
        set_output["bond_presence_logits"], maximum_line, -1e4
    )
    combined_output["incidence_logits"] = _pad_dim1(
        set_output["incidence_logits"], maximum_line, -1e4
    )

    display = F.softmax(line_output["display_logits"].float(), dim=-1)
    chemical = F.softmax(line_output["chemical_logits"].float(), dim=-1)
    shifted_presence = line_output["presence_logits"].float() - line_threshold
    diagnostics = []
    flat_index = 0
    for batch_index, candidates in enumerate(line_candidates):
        expected_indices = torch.where(sample_indices == batch_index)[0]
        if len(expected_indices) != len(candidates):
            raise ValueError("Line candidate order differs from feature sample indices")
        items = []
        for local_index, (candidate, output_index) in enumerate(
            zip(candidates, expected_indices.tolist())
        ):
            proposal_index = base_bonds + local_index
            begin, end = candidate.atom_pair
            combined_proposals["bond_mask"][batch_index, proposal_index] = True
            combined_proposals["bond_endpoints"][batch_index, proposal_index] = (
                proposals["atom_geometry"][batch_index, [begin, end], :2]
            )
            combined_proposals["bond_types"][batch_index, proposal_index] = torch.cat(
                (display[output_index], chemical[output_index])
            ).to(combined_proposals["bond_types"].dtype)
            combined_proposals["bond_scores"][batch_index, proposal_index] = (
                shifted_presence[output_index].sigmoid().to(
                    combined_proposals["bond_scores"].dtype
                )
            )
            combined_output["bond_presence_logits"][batch_index, proposal_index] = (
                shifted_presence[output_index].to(
                    combined_output["bond_presence_logits"].dtype
                )
            )
            combined_output["incidence_logits"][
                batch_index, proposal_index, 0, begin
            ] = 1e4
            combined_output["incidence_logits"][
                batch_index, proposal_index, 1, end
            ] = 1e4
            items.append(
                {
                    "proposal": proposal_index,
                    "atom_pair": [begin, end],
                    "raw_presence_logit": float(
                        line_output["presence_logits"][output_index]
                    ),
                    "shifted_presence_logit": float(shifted_presence[output_index]),
                    "display_type": int(display[output_index].argmax()),
                    "chemical_type": int(chemical[output_index].argmax()),
                    "support_segments": len(candidate.support_segment_indices),
                    "line_length_pixels": candidate.maximum_support_length_pixels,
                }
            )
            flat_index += 1
        diagnostics.append(
            {
                "candidates": len(candidates),
                "selected": sum(
                    item["shifted_presence_logit"] >= 0.0 for item in items
                ),
                "items": items,
            }
        )
    if flat_index != len(line_output["presence_logits"]):
        raise ValueError("Not all line outputs were appended")
    return combined_proposals, combined_output, diagnostics

