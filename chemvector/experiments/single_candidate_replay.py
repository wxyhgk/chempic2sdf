"""Offline one-candidate intervention, never a production admission policy.

Selection consumes only a previously committed prediction-side review queue.
The intervention changes one mask bit, not coordinates, scores or features.
"""

import math

import torch


def first_review_candidate(queue):
    if (queue.get('action') != 'review_only' or queue.get('graph_edits') != 0
            or queue.get('status') != 'requests_ready' or len(queue.get('requests', [])) != 1):
        raise ValueError('Exactly one prediction-only review request required')
    request = queue['requests'][0]
    candidates = request['candidates']
    slot = request['first_review_slot']
    if (type(slot) is not int or slot < 0 or not candidates or candidates[0]['slot'] != slot
            or len({r['slot'] for r in candidates}) != len(candidates)
            or any(not math.isfinite(r['keep_logit']) for r in candidates)
            or candidates != sorted(candidates, key=lambda r: (-r['keep_logit'], r['slot']))
            or (len(candidates) > 1 and abs(candidates[0]['keep_logit'] - candidates[1]['keep_logit']) <= 1e-8)
            or slot in queue['selected_slots']):
        raise ValueError('Unique previously ranked rejected candidate required')
    return slot


def admit_one(raw, selected, queue):
    """Clone only the mask; all non-mask tensor objects stay identical."""
    slot = first_review_candidate(queue)
    mask, active = selected['atom_mask'], raw['atom_mask']
    if (mask.ndim != 2 or mask.shape[0] != 1 or mask.dtype != torch.bool
            or active.shape != mask.shape or active.dtype != torch.bool
            or active.device != mask.device or slot >= mask.shape[1]
            or bool((mask & ~active).any()) or not bool(active[0, slot]) or bool(mask[0, slot])
            or mask[0].nonzero().flatten().tolist() != queue['selected_slots']):
        raise ValueError('Queue must match one unchanged raw-active selected set')
    result = dict(selected, atom_mask=mask.clone())
    result['atom_mask'][0, slot] = True
    return result


def scene_state(trace):
    """Stable candidate IDs, excluding packed bond IDs and incident-port lists."""
    scene = trace['scene_before_fallback']
    lineage = trace['compiled_candidate_indices']
    if len(set(lineage)) != len(lineage) or len(lineage) != len(scene['atom_objects']):
        raise ValueError('Unique exact candidate lineage required')
    atoms = {}
    for i, atom in enumerate(scene['atom_objects']):
        if atom['id'] != i:
            raise ValueError('Packed scene atoms required')
        atoms[lineage[i]] = {k: v for k, v in atom.items() if k not in ('id', 'incident_ports')}
    edges = {}
    for bond in scene['bond_objects']:
        refs = bond['atom_refs']
        if len(refs) != 2 or any(type(i) is not int or not 0 <= i < len(lineage) for i in refs):
            raise ValueError('Invalid edge references')
        pair = tuple(sorted(lineage[i] for i in refs))
        if pair[0] == pair[1] or pair in edges:
            raise ValueError('Unique non-self edges required')
        fields = {k: v for k, v in bond.items() if k not in ('id', 'atom_refs', 'ports', 'stereo_atom_refs')}
        # Preserve endpoint orientation for directional/stereo bonds.
        if bond.get('direction', 'NONE') != 'NONE' or bond.get('stereo', 'STEREONONE') != 'STEREONONE':
            fields['ordered_slots'] = [lineage[i] for i in refs]
        fields['stereo_slots'] = (None if bond.get('stereo_atom_refs') is None
                                  else [lineage[i] for i in bond['stereo_atom_refs']])
        edges[pair] = fields
    return atoms, edges


def compare_scenes(before, after):
    old_atoms, old_edges = scene_state(before)
    new_atoms, new_edges = scene_state(after)
    common = old_atoms.keys() & new_atoms.keys()
    return dict(
        removed_slots=sorted(old_atoms.keys() - new_atoms.keys()),
        added_slots=sorted(new_atoms.keys() - old_atoms.keys()),
        changed_old_atoms=sorted(s for s in common if old_atoms[s] != new_atoms[s]),
        removed_edges=[list(p) for p in sorted(old_edges.keys() - new_edges.keys())],
        changed_old_edges=[list(p) for p in sorted(old_edges.keys() & new_edges.keys())
                           if old_edges[p] != new_edges[p]],
        added_edges=[dict(pair=list(p), **new_edges[p]) for p in sorted(new_edges.keys() - old_edges.keys())])
