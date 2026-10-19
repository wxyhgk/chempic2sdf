"""Target-path and saved-heatmap checks; no renderer, model, or training."""

from copy import deepcopy
import math

from PIL import Image
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.curriculum.data import CurriculumCollator, _point, validate_records
from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss


def legacy_bridge(graph, layers):
    """Same current annotations in the old schema, NOT historical training data.

    Preserve chemical IDs and all anchors. The old scene sorts atoms spatially;
    callers must undo that permutation explicitly before comparing tensors.
    No old training/validation shard is read or synthesized as evidence.
    """
    if (graph['molecule_id'] != layers['molecule_id'] or layers['width'] != 768
            or layers['height'] != 768):
        raise ValueError('Same molecule and original 768 canvas required')
    atoms = sorted(layers['atoms'], key=lambda a: a['atom_id'])
    if [a['atom_id'] for a in atoms] != list(range(len(atoms))):
        raise ValueError('Contiguous explicit atom IDs required')
    quantized = [_point(a['center']) for a in atoms]
    if len(set(quantized)) != len(quantized):
        raise ValueError('Legacy coordinate identity is ambiguous')
    chemistry = dict(molecule_id=graph['molecule_id'],
        atoms=[dict(deepcopy(a), id=a['atom_id']) for a in sorted(graph['atoms'], key=lambda a:a['atom_id'])],
        bonds=[dict(deepcopy(b), id=b['bond_id'], type=b['chemical_type'])
               for b in sorted(graph['bonds'], key=lambda b:b['bond_id'])])
    render = dict(molecule_id=graph['molecule_id'], width=768, height=768,
        atoms=[dict(atom_id=a['atom_id'], image_x=a['center'][0], image_y=a['center'][1],
                    quantized_x=q[0], quantized_y=q[1]) for a, q in zip(atoms, quantized)],
        bonds=[dict(bond_id=b['bond_id'], display_type=b['drawing']['bond_type'],
                    direction=b['drawing']['direction']) for b in layers['bonds']])
    return chemistry, render


def compare_target_paths(graph, layers, text_records, meta):
    """Use both existing collators, with a white placeholder never inferred."""
    validate_records(graph, layers, text_records)
    chemmol, render = legacy_bridge(graph, layers)
    placeholder = Image.new('RGB', (768, 768), 'white')
    sample = dict(key=layers['sample_key'], image=placeholder, graph=graph, layers=layers,
                  text_records=text_records, meta=meta, purpose='diagnostic')
    config = AtomBondSceneConfig()
    current = CurriculumCollator(config, purpose='diagnostic')([sample])
    legacy = AtomBondSceneCollator(config)([dict(key=sample['key'], image=placeholder,
                                               chemmol=chemmol, render=render, layers=layers)])
    n = len(graph['atoms'])
    expected = {tuple(_point(a['center'])): a['atom_id'] for a in layers['atoms']}
    order = [expected[tuple(a['anchor'])] for a in legacy['scenes'][0]['atom_objects']]
    if sorted(order) != list(range(n)):
        raise ValueError('Legacy spatial order does not preserve all source identities')
    inverse = [order.index(i) for i in range(n)]
    anchor = legacy['atom_anchors'][:, inverse]
    new_anchor = current['atom_anchors'][:, :n]
    old_heat = build_vertex_heatmap(anchor, torch.ones(1, n, dtype=torch.bool), 192, 192, 1)
    new_heat = build_vertex_heatmap(new_anchor, current['atom_mask'][:, :n], 192, 192, 1)
    old_text = legacy['atom_text_center'][0, inverse]
    new_text = current['atom_text_center'][0, :n]
    visible = current['atom_text_mask'][0, :n]
    return dict(anchors_equal=torch.equal(anchor, new_anchor), heatmaps_equal=torch.equal(old_heat, new_heat),
        roles_equal=torch.equal(legacy['atom_roles'][0, inverse], current['atom_roles'][0, :n]),
        text_mask_equal=torch.equal(legacy['atom_text_mask'][0, inverse], visible),
        legacy_spatial_slot_to_source_id=order, atom_count=n, visible_text_count=int(visible.sum()),
        source_ids_with_changed_text_center=[i for i in range(n) if visible[i] and not torch.equal(old_text[i], new_text[i])],
        current_anchors=new_anchor, current_heatmap=new_heat,
        legacy_text_centers=old_text, current_text_centers=new_text,
        bridge_is_current_annotations_not_historical_examples=True, training_authorized=False)


def saved_heatmap_probe(snapshot, geometry, target_ids, layers, target_heatmap):
    """d(existing heatmap loss)/d(detached saved logits), not parameter gradients.

    Use ALL original target atoms as supervision; do not treat other atoms as
    negatives by constructing a target heatmap from just the ten failure nodes.
    """
    logits = snapshot['heatmap_logits']
    if (logits.shape != (1, 1, 192, 192) or logits.dtype != torch.float32
            or logits.device.type != 'cpu' or not torch.isfinite(logits).all()
            or logits.requires_grad or target_heatmap.shape != logits.shape
            or target_heatmap.dtype != torch.float32 or target_heatmap.device.type != 'cpu'
            or not torch.isfinite(target_heatmap).all() or target_heatmap.requires_grad):
        raise ValueError('Detached finite CPU float32 saved logits/targets required')
    anchors = torch.tensor(geometry['target_anchors'], dtype=torch.float32)[None]
    expected = build_vertex_heatmap(anchors, torch.ones(anchors.shape[:2], dtype=torch.bool), 192, 192, 1)
    if not torch.equal(expected, target_heatmap):
        raise ValueError('Heatmap must contain all original target anchors')
    if (len(set(target_ids)) != len(target_ids)
            or any(type(t) is not int or not 0 <= t < anchors.shape[1] for t in target_ids)):
        raise ValueError('Unique in-range target IDs required')
    z = logits.detach().clone().requires_grad_(True)
    loss = focal_heatmap_loss(z, target_heatmap)
    gradient, = torch.autograd.grad(loss, z)
    probabilities = logits.sigmoid()[0, 0]
    atoms = {a['atom_id']: a for a in layers['atoms']}
    positive_ids = {}
    for i, xy in enumerate(anchors[0]):
        x, y = (xy * 192).clamp(0, 192-1e-4).long().tolist()
        positive_ids.setdefault(y*192+x, []).append(i)
    records = []
    for target_id in target_ids:
        xy = anchors[0, target_id]
        x, y = (xy * 192).clamp(0, 192-1e-4).long().tolist()
        region = probabilities[max(0, y-1):y+2, max(0, x-1):x+2]
        maxima = []
        gt_grad = float(gradient[0, 0, y, x])
        for cy in range(max(0, y-1), min(192, y+2)):
            for cx in range(max(0, x-1), min(192, x+2)):
                if probabilities[cy, cx] != region.max() or (cx, cy) == (x, y):
                    continue
                d = float(gradient[0, 0, cy, cx])
                maxima.append(dict(grid_id=cy*192+cx, score=float(probabilities[cy, cx]),
                    target_heatmap=float(target_heatmap[0, 0, cy, cx]),
                    positive_source_atom_ids=positive_ids.get(cy*192+cx, []),
                    dloss_dlogit=d, independent_descent_gt_minus_neighbor_margin_direction=-gt_grad+d))
        atom = atoms[target_id]
        box = atom['bbox']
        center = [(box[0]+box[2])/2, (box[1]+box[3])/2] if box else None
        nearest = min(geometry['candidates'], key=lambda c: math.dist(c['geometry'][:2], xy.tolist()))
        source_point = [v*767 for v in nearest['geometry'][:2]]
        records.append(dict(target_id=target_id, text=atom['text'], gt_grid_id=y*192+x,
            gt_heatmap_target=float(target_heatmap[0, 0, y, x]), gt_score=float(probabilities[y, x]),
            gt_dloss_dlogit=gt_grad, local_max_neighbors=maxima,
            original_raw_nearest_slot=nearest['slot'],
            nearest_to_source_chemical_anchor_px=math.dist(source_point, atom['center']),
            nearest_to_source_ink_bbox_center_px=math.dist(source_point, center) if center else None,
            source_chemical_anchor_to_ink_bbox_center_px=math.dist(atom['center'], center) if center else None))
    return dict(existing_heatmap_loss=float(loss.detach()), all_target_count=anchors.shape[1],
        positive_cell_count=int((target_heatmap == 1).sum()), targets=records,
        parameter_gradients_computed=False, optimizer_steps=0,
        scope='Per-image saved-logit derivatives only; not shared-model update or recovery evidence')
