"""Ground-truth-assisted supervision proposals, never an inference filter.

No model, checkpoint, optimizer, or released training labels are used here.
Membership means spatial eligibility under the existing radius, not proof of
correct element/bond identity. Ambiguous memberships remain explicitly open.
"""

import math

import torch

from tools import selector_node_semantics as semantics


def training_rows(manifest):
    """Do not silently promote validation-only rows into a new fitting use."""
    rows = manifest['training_samples']
    keys = [r['sample_key'] for r in rows]
    validation = {r['sample_key'] for r in manifest['validation_only_samples']}
    if (len(keys) != len(set(keys)) or set(keys) & validation
            or any(r.get('split') != 'train' or r.get('purpose') == 'diagnostic'
                   or r.get('fit_domain') not in ('witness', 'old_full_molecule') for r in rows)):
        raise ValueError('Only disjoint existing training/replay rows are permitted')
    return list(rows)


@torch.no_grad()
def memberships(geometry):
    """All within-radius relations; do not replace the original matcher."""
    candidates = geometry['candidates']
    ids = [c['slot'] for c in candidates]
    if not candidates or len(ids) != len(set(ids)) or not geometry['target_anchors']:
        raise ValueError('Nonempty unique candidates and targets required')
    points = torch.tensor([c['geometry'][:2] for c in candidates], dtype=torch.float32)
    anchors = torch.tensor(geometry['target_anchors'], dtype=torch.float32)
    if not torch.isfinite(points).all() or not torch.isfinite(anchors).all():
        raise ValueError('Nonfinite geometry')
    distances = torch.cdist(points, anchors) * geometry['image_size']
    return {c['slot']: [dict(target_id=int(t), distance_px=float(d))
                        for t, d in enumerate(distances[i]) if float(d) <= geometry['match_radius_px']]
            for i, c in enumerate(candidates)}


def endpoint_owner(point, bond, radius):
    """A clipped line end must be clearly closer to one chemical bond end."""
    a, b = math.dist(point, bond['begin']), math.dist(point, bond['end'])
    if abs(a - b) <= radius:
        return None
    return bond['begin_atom'] if a < b else bond['end_atom']


def ownership(point, glyphs, paths, bonds, radius):
    """Keep all nearby evidence so nearest-witness ties cannot hide conflicts."""
    glyph_hits = [dict(atom_id=g['atom_id'], text=g['text'], bbox=g['bbox'])
                  for g in glyphs if semantics.inside(point, g['bbox'])]
    tips = []
    for path in paths:
        bond = bonds[path['bond_id']]
        for tip in (path['points'][0], path['points'][-1]):
            distance = math.dist(point, tip)
            if distance > radius:
                continue
            offset = semantics.segment_distance(tip, bond['begin'], bond['end'])
            tips.append(dict(bond_id=path['bond_id'], node_index=path['node_index'],
                point=tip, distance_px=distance, axis_offset_px=offset,
                owner=endpoint_owner(tip, bond, radius),
                multiple_off_axis=path['drawing_type'] in ('DOUBLE', 'TRIPLE') and offset > radius))
    owners = {g['atom_id'] for g in glyph_hits}
    owners.update(t['owner'] for t in tips if t['owner'] is not None)
    kinds = []
    if glyph_hits:
        kinds.append('glyph_fragment')
    if any(t['multiple_off_axis'] for t in tips):
        kinds.append('multiple_bond_off_axis_tip')
    return dict(glyph_hits=glyph_hits, nearby_tips=tips, possible_owners=sorted(owners), kinds=kinds)


def propose(geometry, layers, glyphs, paths, unsupported):
    """Propose pairs only for unambiguous outside-radius stroke fragments."""
    atoms = sorted(layers['atoms'], key=lambda a: a['atom_id'])
    if [a['atom_id'] for a in atoms] != list(range(len(geometry['target_anchors']))):
        raise ValueError('Target-slot to source-atom identity must be explicit and contiguous')
    bonds = {b['bond_id']: b for b in layers['bonds']}
    if len(bonds) != len(layers['bonds']):
        raise ValueError('Duplicate source bond IDs')
    if any(b[k] not in range(len(atoms)) for b in bonds.values() for k in ('begin_atom', 'end_atom')):
        raise ValueError('Unknown source atom in bond')
    related = memberships(geometry)
    raw = {c['slot']: c for c in geometry['candidates']}
    nodes = []
    for target, atom in enumerate(atoms):
        possible = [s for s, members in related.items() if any(m['target_id'] == target for m in members)]
        unique = [s for s in possible if len(related[s]) == 1]
        positives = [s for s, c in raw.items() if c['raw_target'] == target]
        if len(positives) > 1 or any(s not in possible for s in positives):
            raise ValueError('Raw supervision conflicts with spatial eligibility')
        status = ('no_candidate' if not possible else 'ambiguous_membership' if len(unique) != len(possible)
                  else 'no_raw_representative' if not positives else 'reviewable')
        nodes.append(dict(target_id=target, element=atom['element'], visible_text=atom['visible_text'],
            label_text=atom.get('text', ''), possible_slots=possible, unique_slots=unique,
            raw_positive_slots=positives, alternative_slots=[s for s in unique if raw[s]['raw_target'] < 0],
            status=status))
    unknown_bonds = {r['bond_id'] for r in unsupported}
    candidates, pairs = [], []
    for slot, c in raw.items():
        members = related[slot]
        point = [v * geometry['image_size'] for v in c['geometry'][:2]]
        evidence = ownership(point, glyphs, paths, bonds, geometry['match_radius_px'])
        owner = None
        if members:
            status = 'ambiguous_node_membership' if len(members) > 1 else 'protected_node_candidate'
        elif c['raw_target'] >= 0:
            raise ValueError('A raw positive must never become a fragment negative')
        elif not evidence['kinds']:
            status = 'unresolved_location'
        elif len(evidence['possible_owners']) != 1 or any(t['owner'] is None for t in evidence['nearby_tips']):
            status = 'owner_conflict_or_ambiguous_tip'
        elif evidence['possible_owners'][0] not in range(len(nodes)):
            status = 'unknown_owner'
        elif any(t['bond_id'] in unknown_bonds for t in evidence['nearby_tips']):
            status = 'related_bond_has_unparsed_paths'
        else:
            owner = evidence['possible_owners'][0]
            node = nodes[owner]
            status = 'review_pair' if node['status'] == 'reviewable' else 'owner_' + node['status']
        entry = dict(slot=slot, raw_target=c['raw_target'], pixel_xy=point,
                     memberships=members, status=status, owner=owner, evidence=evidence)
        candidates.append(entry)
        if status == 'review_pair':
            node = nodes[owner]
            pairs.append(dict(fragment_slot=slot, owner_target=owner, kinds=evidence['kinds'],
                representative_slots=node['raw_positive_slots'],
                acceptable_unique_slots=node['unique_slots'],
                requires_review=True, training_authorized=False))
    return dict(nodes=nodes, candidates=candidates, proposed_pairs=pairs,
                unsupported_svg_nodes=unsupported, training_authorized=False, requires_review=True)


def coverage(geometry, slots, layers):
    mapping = semantics.assignment(geometry, slots)
    matched = sorted(t for t in mapping.values() if t >= 0)
    bonds = sorted(b['bond_id'] for b in layers['bonds']
                   if b['begin_atom'] in matched and b['end_atom'] in matched)
    return dict(matched_target_ids=matched, reachable_bond_ids=bonds,
                unmatched_selected=sum(t < 0 for t in mapping.values()), mapping=mapping)


def removal_check(geometry, proposal, record, layers):
    """GT-assisted counterfactual, NOT a changed prediction or improvement."""
    selected = {int(s) for s, value in record['metrics']['logits'].items() if value >= -.25}
    before = coverage(geometry, selected, layers)
    for key in ('matched_target_ids', 'reachable_bond_ids', 'unmatched_selected'):
        if before[key] != record['metrics'][key]:
            raise ValueError('Counterfactual baseline differs from saved original matcher')
    excluded = {p['fragment_slot'] for p in proposal['proposed_pairs']}
    by_slot = {c['slot']: c for c in proposal['candidates']}
    if any(by_slot[s]['memberships'] or by_slot[s]['raw_target'] >= 0 for s in excluded):
        raise ValueError('Attempted exclusion of a real/alternative node candidate')
    removed = selected & excluded
    after = coverage(geometry, selected - removed, layers)
    lost_atoms = sorted(set(before['matched_target_ids']) - set(after['matched_target_ids']))
    lost_bonds = sorted(set(before['reachable_bond_ids']) - set(after['reachable_bond_ids']))
    glyph_correct = [s for s, target in before['mapping'].items()
                     if target >= 0 and by_slot[s]['evidence']['glyph_hits']]
    replacements = [s for s, target in before['mapping'].items()
                    if target >= 0 and by_slot[s]['raw_target'] < 0]
    constraints = []
    for node in proposal['nodes']:
        if node['status'] != 'reviewable':
            continue
        retained = sorted(set(node['unique_slots']) & selected)
        constraints.append(dict(target_id=node['target_id'], selected_members=retained,
            state='missing' if not retained else 'single' if len(retained) == 1 else 'duplicate'))
    return dict(uses_ground_truth=True, not_model_predictions=True,
        removed_fragment_slots=sorted(removed),
        before={k:v for k,v in before.items() if k != 'mapping'},
        after={k:v for k,v in after.items() if k != 'mapping'},
        lost_target_ids=lost_atoms, lost_bond_ids=lost_bonds,
        gained_target_ids=sorted(set(after['matched_target_ids']) - set(before['matched_target_ids'])),
        original_glyph_matched_slots=glyph_correct, original_valid_replacement_slots=replacements,
        removed_correct_glyph_slots=sorted(set(glyph_correct) & removed),
        removed_valid_replacement_slots=sorted(set(replacements) & removed),
        node_constraints=constraints,
        original_matcher_nonregression=not lost_atoms and not lost_bonds,
        spatial_eligibility_preserved=all(not by_slot[s]['memberships'] for s in removed))
