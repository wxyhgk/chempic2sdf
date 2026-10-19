"""Manual CPU tensors only: no module construction, checkpoint or real image."""

import pytest
import torch

from chemvector.components.atoms.objects.spatial_atom_object import spatial_atom_object_loss
from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig, _local_peak_topk, decode_predicted_atom_proposals,
)
from chemvector.components.vision.vertex import build_vertex_heatmap, focal_heatmap_loss
from tools.selector_localization_contract import inspect_decoder_cell


def manual_output(probabilities, *, target_xy=(2.25, 3.75)):
    """Construct exactly the output coordinate formula, without a model."""
    height, width = probabilities.shape
    offsets = torch.zeros(1, 2, height, width, requires_grad=True)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    anchors = torch.stack(((x + offsets.sigmoid()[0, 0]) / width,
                           (y + offsets.sigmoid()[0, 1]) / height), dim=0)[None]
    output = dict(heatmap_logits=torch.logit(probabilities)[None, None],
        anchor_offset_logits=offsets, anchors=anchors,
        object_features=torch.arange(height * width).reshape(1, 1, height, width).float(),
        role_logits=torch.zeros(1, 2, height, width), text_centers=anchors,
        text_extents=torch.full((1, 2, height, width), .02), count_logits=torch.zeros(1, 2))
    target = torch.tensor([[[target_xy[0] / width, target_xy[1] / height]]])
    targets = dict(atom_anchors=target, atom_mask=torch.tensor([[True]]),
        atom_roles=torch.ones(1, 1, dtype=torch.long), atom_text_mask=torch.tensor([[False]]),
        atom_text_center=target.clone(), atom_text_bbox=torch.zeros(1, 1, 4))
    return output, targets


@pytest.mark.parametrize("candidate_xy", [(3, 3), (4, 3), (6, 6)])
def test_offset_supervision_has_support_at_gt_cell_not_wrong_candidate_cell(candidate_xy):
    probabilities = torch.full((8, 8), .001)
    cx, cy = candidate_xy
    probabilities[cy, cx] = .9
    output, targets = manual_output(probabilities)
    losses = spatial_atom_object_loss(output, targets)
    gradient = torch.autograd.grad(losses["loss"], output["anchor_offset_logits"])[0]
    assert bool((gradient[0, :, 3, 2] != 0).all())
    assert gradient[0, 0, 3, 2] > 0  # .5 -> GT x offset .25
    assert gradient[0, 1, 3, 2] < 0  # .5 -> GT y offset .75
    assert torch.equal(gradient[0, :, cy, cx], torch.zeros(2))
    assert torch.count_nonzero(gradient) == 2


def test_gaussian_neighbor_is_weighted_negative_not_extra_offset_positive():
    output, targets = manual_output(torch.full((8, 8), .5))
    heatmap = build_vertex_heatmap(targets["atom_anchors"], targets["atom_mask"], 8, 8, 1)
    assert heatmap[0, 0, 3, 2] == 1
    assert 0 < heatmap[0, 0, 3, 3] < 1
    logits = torch.zeros_like(heatmap, requires_grad=True)
    gradient = torch.autograd.grad(focal_heatmap_loss(logits, heatmap), logits)[0]
    assert gradient[0, 0, 3, 2] < 0
    assert gradient[0, 0, 3, 3] > 0
    offset_gradient = torch.autograd.grad(spatial_atom_object_loss(output, targets)["anchor_offset_loss"],
                                         output["anchor_offset_logits"])[0]
    assert torch.equal(offset_gradient[0, :, 3, 3], torch.zeros(2))


def test_geometrically_reachable_wrong_cell_still_has_no_direct_offset_gradient():
    probabilities = torch.full((8, 8), .001)
    probabilities[3, 4] = .9
    output, targets = manual_output(probabilities, target_xy=(2.75, 3.5))
    config = PredictedProposalConfig(maximum_atoms=1, image_size=32.)
    raw = decode_predicted_atom_proposals(output, config)
    assert raw["atom_indices"].tolist() == [[3 * 8 + 4]]
    assert bool(raw["atom_mask"].all())
    distance = (raw["atom_geometry"][0, 0, :2] - targets["atom_anchors"][0, 0]).norm() * 32
    assert float(distance.detach()) == pytest.approx(7.)
    # The cell's left boundary is 5 px from the target, but finite sigmoid
    # offsets cannot cross its own cell. This is feasibility, not a real update.
    hypothetical = torch.tensor([(4 + .1) / 8, (3 + .5) / 8])
    assert float((hypothetical - targets["atom_anchors"][0, 0]).norm() * 32) < 6
    gradient = torch.autograd.grad(spatial_atom_object_loss(output, targets)["anchor_offset_loss"],
                                   output["anchor_offset_logits"])[0]
    assert torch.equal(gradient[0, :, 3, 4], torch.zeros(2))


def test_decoder_geometry_uses_selected_cell_without_snapping_to_target():
    probabilities = torch.full((8, 8), .001)
    probabilities[3, 4] = .9
    output, targets = manual_output(probabilities)
    raw = decode_predicted_atom_proposals(output, PredictedProposalConfig(maximum_atoms=1))
    assert torch.equal(raw["atom_geometry"][0, 0, :2], output["anchors"][0, :, 3, 4])
    assert not torch.equal(raw["atom_geometry"][0, 0, :2], targets["atom_anchors"][0, 0])
    # Downstream coordinate losses CAN differentiate through selected offsets.
    # Zero direct GT-cell loss support does not imply zero parameter gradient
    # from every possible downstream loss or shared feature path.
    gradient = torch.autograd.grad(raw["atom_geometry"][..., :2].sum(),
                                   output["anchor_offset_logits"])[0]
    assert bool((gradient[0, :, 3, 4] > 0).all())
    assert torch.equal(gradient[0, :, 3, 2], torch.zeros(2))


def inspect(probabilities, cell, *, maximum=4, threshold=.01):
    return inspect_decoder_cell(torch.logit(probabilities)[None, None], cell,
                                maximum=maximum, threshold=threshold)


def test_above_threshold_cell_can_be_suppressed_by_neighbor_despite_spare_capacity():
    probabilities = torch.full((5, 5), .001)
    probabilities[2, 2], probabilities[2, 3] = .7, .9
    record = inspect(probabilities, 12, maximum=25)
    assert record["passes_threshold"] and not record["raw_active"]
    assert record["first_excluding_stage"] == "local_peak_suppressed"


def test_retained_local_peak_can_fail_threshold_with_spare_capacity():
    probabilities = torch.full((5, 5), .001)
    probabilities[2, 2] = .009
    record = inspect(probabilities, 12)
    assert record["is_local_peak"] and record["in_topk_before_threshold"]
    assert not record["passes_threshold"] and not record["raw_active"]
    assert record["first_excluding_stage"] == "below_threshold"


def test_isolated_above_threshold_peak_can_be_excluded_only_by_topk():
    probabilities = torch.full((7, 7), .001)
    probabilities[0, 0], probabilities[0, 4], probabilities[4, 0] = .9, .8, .7
    record = inspect(probabilities, 4 * 7, maximum=2)
    assert record["is_local_peak"] and record["passes_threshold"]
    assert record["strictly_higher_peak_count"] == 2
    assert not record["in_topk_before_threshold"]
    assert record["first_excluding_stage"] == "topk_excluded"


def test_equal_plateau_cells_are_peaks_and_topk_ties_need_actual_indices():
    probabilities = torch.full((3, 3), .8)
    records = [inspect(probabilities, i, maximum=2) for i in range(9)]
    assert all(r["is_local_peak"] and r["ties_at_topk_boundary"] for r in records)
    assert all(r["strictly_higher_peak_count"] == 0 for r in records)
    assert all(r["equal_peak_count_including_self_if_peak"] == 9 for r in records)
    assert sum(r["raw_active"] for r in records) == 2
    # No assertion on which two tied grid IDs torch.topk chooses.


def test_threshold_comparison_is_inclusive_at_exact_representable_score():
    probabilities = torch.full((3, 3), .001)
    probabilities[1, 1] = .5
    record = inspect(probabilities, 4, maximum=1, threshold=.5)
    assert record["raw_active"] and record["passes_threshold"]


def test_padding_does_not_create_extra_active_candidates():
    probabilities = torch.tensor([[.1, .2], [.3, .9]])
    logits = torch.logit(probabilities)[None, None]
    values, indices, mask = _local_peak_topk(logits, maximum=7, threshold=.01)
    assert values.shape == indices.shape == mask.shape == (1, 7)
    assert mask.sum() == 1 and not bool(mask[0, 4:].any())
    record = inspect_decoder_cell(logits, 0, maximum=7, threshold=.01)
    assert record["padding_slots"] == 3 and not record["raw_active"]


@pytest.mark.parametrize("bad", [torch.zeros(3, 3), torch.zeros(2, 1, 3, 3),
                                torch.full((1, 1, 3, 3), float("nan")),
                                torch.zeros(1, 1, 3, 3, dtype=torch.long)])
def test_observer_rejects_missing_or_invalid_dense_evidence(bad):
    with pytest.raises(ValueError, match="CPU heatmap"):
        inspect_decoder_cell(bad, 0, maximum=1, threshold=.01)


@pytest.mark.parametrize("cell,maximum,threshold", [(-1, 1, .01), (9, 1, .01),
                                                    (0, 0, .01), (0, 1, 2.)])
def test_observer_rejects_invalid_cell_or_decoder_contract(cell, maximum, threshold):
    with pytest.raises(ValueError):
        inspect_decoder_cell(torch.zeros(1, 1, 3, 3), cell, maximum=maximum, threshold=threshold)
