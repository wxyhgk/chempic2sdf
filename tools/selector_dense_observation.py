"""Image-only dense observation and later, separate GT-assisted explanation.

No optimization, alternative decoder, candidate rescue, or changed thresholds.
Historical CPU/GPU parity is recorded, never assumed from close coordinates.
"""

import math

import torch

from chemvector.predicted_incidence import _local_peak_topk, decode_predicted_atom_proposals
from tools.selector_localization_contract import inspect_decoder_cell
from tools.selector_node_semantics import assignment


@torch.no_grad()
def image_only_snapshot(model, pixels, config):
    """One vision/atom pass only; this API cannot accept target annotations."""
    if (pixels.device.type != 'cpu' or pixels.dtype != torch.float32
            or pixels.shape != (1, 3, 768, 768) or not torch.isfinite(pixels).all()
            or model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Frozen eval model and one finite float32 CPU image required')
    features = model.vision.forward_stroke_resolution(pixels)
    output = model.atom(features)
    raw = decode_predicted_atom_proposals(output, config)
    return {**{k: output[k].detach().cpu().clone()
               for k in ('heatmap_logits', 'anchor_offset_logits', 'anchors')},
            **{'raw_' + k: raw['atom_' + k].detach().cpu().clone()
               for k in ('indices', 'scores', 'mask', 'geometry')}}


def validate_snapshot(snapshot, config):
    expected = {'heatmap_logits', 'anchor_offset_logits', 'anchors',
                'raw_indices', 'raw_scores', 'raw_mask', 'raw_geometry'}
    if set(snapshot) != expected:
        raise ValueError('Exact dense snapshot fields required')
    if any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.requires_grad
           for v in snapshot.values()):
        raise ValueError('Detached CPU tensors required')
    heat = snapshot['heatmap_logits']
    if heat.ndim != 4 or heat.shape[:2] != (1, 1) or min(heat.shape[-2:]) < 1:
        raise ValueError('One dense heatmap required')
    h, w = heat.shape[-2:]
    shapes = dict(anchor_offset_logits=(1, 2, h, w), anchors=(1, 2, h, w),
        raw_indices=(1, config.maximum_atoms), raw_scores=(1, config.maximum_atoms),
        raw_mask=(1, config.maximum_atoms), raw_geometry=(1, config.maximum_atoms, 8))
    if any(tuple(snapshot[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError('Dense snapshot shape differs from original configuration')
    if (snapshot['raw_mask'].dtype != torch.bool or snapshot['raw_indices'].dtype != torch.int64
            or any(v.dtype != torch.float32 or not torch.isfinite(v).all()
                   for k, v in snapshot.items() if k not in ('raw_mask', 'raw_indices'))):
        raise ValueError('Finite float32 values and original index/mask types required')
    scores, indices, mask = _local_peak_topk(heat, maximum=config.maximum_atoms,
                                          threshold=config.atom_threshold)
    if any(not torch.equal(snapshot['raw_' + k], value)
           for k, value in (('scores', scores), ('indices', indices), ('mask', mask))):
        raise ValueError('Saved decoder does not reproduce from dense heatmap')
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    offsets = snapshot['anchor_offset_logits'].sigmoid()
    anchors = torch.stack(((x + offsets[0, 0]) / w, (y + offsets[0, 1]) / h))[None]
    selected = anchors.flatten(2).transpose(1, 2)[0, indices[0], :]
    if (not torch.equal(anchors, snapshot['anchors'])
            or not torch.equal(selected, snapshot['raw_geometry'][0, :, :2])):
        raise ValueError('Saved anchor formula/decoder coordinates differ')


@torch.no_grad()
def compare_history(snapshot, geometry, config):
    """Report exact numeric parity AND the unchanged discrete target contract."""
    validate_snapshot(snapshot, config)
    if geometry['image_size'] != config.image_size or geometry['match_radius_px'] != config.atom_match_radius_px:
        raise ValueError('Historical metric changed')
    saved = geometry['candidates']
    old_slots = [c['slot'] for c in saved]
    if old_slots != sorted(set(old_slots)) or len({c['grid_id'] for c in saved}) != len(saved):
        raise ValueError('Unique sorted historical slots/cells required')
    new_slots = snapshot['raw_mask'][0].nonzero().flatten().tolist()
    new = [dict(slot=s, grid_id=int(snapshot['raw_indices'][0, s]),
                geometry=snapshot['raw_geometry'][0, s].tolist(),
                raw_score=float(snapshot['raw_scores'][0, s])) for s in new_slots]
    old_by_grid, new_by_grid = ({c['grid_id']: c for c in cs} for cs in (saved, new))
    if len(new_by_grid) != len(new):
        raise ValueError('Duplicate active decoded grid')
    common = sorted(set(old_by_grid) & set(new_by_grid))
    max_xy = max((max(abs(old_by_grid[g]['geometry'][i] - new_by_grid[g]['geometry'][i])
                      * config.image_size for i in (0, 1)) for g in common), default=None)
    max_geometry = max((max(abs(a - b) for a, b in zip(old_by_grid[g]['geometry'], new_by_grid[g]['geometry']))
                        for g in common), default=None)
    max_score = max((abs(old_by_grid[g]['raw_score'] - new_by_grid[g]['raw_score']) for g in common), default=None)
    anchors = torch.tensor(geometry['target_anchors'], dtype=torch.float32).reshape(-1, 2)
    def costs(candidates):
        xy = torch.tensor([c['geometry'][:2] for c in candidates], dtype=torch.float32).reshape(-1, 2)
        return torch.cdist(xy, anchors) * config.image_size
    old_cost, new_cost = costs(saved), costs(new)
    def edges(candidates, cost):
        return {(c['grid_id'], int(t)) for i, c in enumerate(candidates)
                for t in (cost[i] <= config.atom_match_radius_px).nonzero().flatten().tolist()}
    old_edges, new_edges = edges(saved, old_cost), edges(new, new_cost)
    old_missing = (old_cost > config.atom_match_radius_px).all(0).nonzero().flatten().tolist()
    new_missing = (new_cost > config.atom_match_radius_px).all(0).nonzero().flatten().tolist()
    fresh = dict(geometry, candidates=new)
    old_map, new_map = assignment(geometry, old_slots), assignment(fresh, new_slots)
    if old_map != {c['slot']: c['raw_target'] for c in saved}:
        raise ValueError('Historical raw assignment did not reconstruct')
    ordered = [(c['slot'], c['grid_id']) for c in saved] == [(c['slot'], c['grid_id']) for c in new]
    exact = ordered and all(c['geometry'] == n['geometry'] and c['raw_score'] == n['raw_score']
                            for c, n in zip(saved, new))
    return dict(historical_active_count=len(saved), cpu_active_count=len(new),
        active_slots_equal=old_slots == new_slots, ordered_slot_grid_equal=ordered,
        missing_historical_grid_ids=sorted(set(old_by_grid) - set(new_by_grid)),
        extra_cpu_grid_ids=sorted(set(new_by_grid) - set(old_by_grid)),
        common_grid_count=len(common), max_common_xy_change_px=max_xy,
        max_common_geometry_change=max_geometry, max_common_score_change=max_score,
        candidate_numbers_bitwise_equal=exact,
        lost_eligible_grid_target_edges=sorted(old_edges - new_edges),
        gained_eligible_grid_target_edges=sorted(new_edges - old_edges),
        historical_missing_target_ids=old_missing, cpu_missing_target_ids=new_missing,
        original_assignment_equal=old_map == new_map,
        historical_discrete_contract_reproduced=ordered and old_edges == new_edges and old_map == new_map)


@torch.no_grad()
def explain_targets(snapshot, geometry, target_ids, config):
    """Post-snapshot GT interpretation; never feeds or replaces predictions."""
    parity = compare_history(snapshot, geometry, config)
    if (len(set(target_ids)) != len(target_ids)
            or any(type(t) is not int or t not in parity['historical_missing_target_ids'] for t in target_ids)):
        raise ValueError('Unique historically missing target IDs required')
    heat = snapshot['heatmap_logits']
    h, w = heat.shape[-2:]
    if h != 192 or w != 192 or config.image_size != 768 or config.atom_match_radius_px != 6:
        raise ValueError('Fixed stride4/768/6px observation only')
    probabilities = heat.sigmoid()[0, 0]
    anchors = snapshot['anchors'][0].permute(1, 2, 0).reshape(-1, 2)
    raw_slots = snapshot['raw_mask'][0].nonzero().flatten()
    records = []
    for target_id in target_ids:
        target = torch.tensor(geometry['target_anchors'][target_id], dtype=torch.float32)
        cells = (target * torch.tensor([w, h])).clamp(0, 192 - 1e-4)
        x, y = map(int, cells)
        gt_grid = y*w + x
        direct = torch.linalg.vector_norm(anchors - target, dim=-1) * config.image_size
        def cell(grid):
            cy, cx = divmod(grid, w)
            result = inspect_decoder_cell(heat, grid, maximum=config.maximum_atoms, threshold=config.atom_threshold)
            neighborhood = [dict(grid_id=yy*w+xx, score=float(probabilities[yy, xx]))
                for yy in range(max(0, cy-1), min(h, cy+2))
                for xx in range(max(0, cx-1), min(w, cx+2))]
            result.update(x=cx, y=cy, neighborhood=neighborhood,
                local_max_grid_ids=[n['grid_id'] for n in neighborhood if n['score'] == result['local_maximum_score']],
                offset_logits=snapshot['anchor_offset_logits'][0, :, cy, cx].tolist(),
                sigmoid_offset=snapshot['anchor_offset_logits'][0, :, cy, cx].sigmoid().tolist(),
                anchor=anchors[grid].tolist(), direct_target_distance_px=float(direct[grid]))
            return result
        old_nearest = min(geometry['candidates'], key=lambda c: math.dist(c['geometry'][:2], target.tolist()))
        if len(raw_slots):
            raw_xy = snapshot['raw_geometry'][0, raw_slots, :2]
            nearest_index = int(torch.linalg.vector_norm(raw_xy - target, dim=-1).argmin())
            nearest_slot = int(raw_slots[nearest_index])
            nearest_grid = int(snapshot['raw_indices'][0, nearest_slot])
        else:
            nearest_slot, nearest_grid = None, None
        near_ids = (direct <= config.atom_match_radius_px).nonzero().flatten().tolist()
        records.append(dict(target_id=target_id, saved_target=target.tolist(), target_cell=[x, y],
            expected_gt_cell_offset=(cells - torch.tensor([x, y])).tolist(), gt_cell=cell(gt_grid),
            historical_nearest_slot=old_nearest['slot'], historical_nearest_grid_id=old_nearest['grid_id'],
            historical_nearest_cell_now=cell(old_nearest['grid_id']),
            cpu_nearest_slot=nearest_slot, cpu_nearest_cell=cell(nearest_grid) if nearest_grid is not None else None,
            dense_cells_within_6px=[cell(g) for g in near_ids],
            cpu_still_missing=target_id in parity['cpu_missing_target_ids'],
            historical_explanation_supported=parity['historical_discrete_contract_reproduced'],
            note='Cell/peak decisions are observed mechanisms, not a measured training cause'))
    return dict(parity=parity, targets=records, training_authorized=False, uses_ground_truth_after_snapshot=True)
