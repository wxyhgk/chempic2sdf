"""Saved-data error inventory, not a matcher, label generator, or graph repairer.

Keep three separate facts: local coordinate correspondence, receipt-proven
matched membership, and prediction. No unavailable float cache is reconstructed.
"""

from collections import Counter
from itertools import combinations
import math

from tools.pair_local_repair import edge_map

BOND_TYPES = ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')
IMAGE_SIZE = 768
QMAX = 1023
RADIUS = 6
# Saved predicted anchors were rounded to q1023. Targets were already quantized
# by the historical adapter; they must NOT receive a second quantization error.
EPSILON = math.sqrt(2) * IMAGE_SIZE / (2 * QMAX) + 1e-4


def adapted_anchor(center):
    """Existing target convention: source pixels / 767 -> rounded q1023."""
    if len(center) != 2 or any(not math.isfinite(v) or not 0 <= v <= 767 for v in center):
        raise ValueError('Finite original 768px source coordinates required')
    return [round(v / 767 * QMAX) for v in center]


def correspondence(atoms, lineage, targets):
    if (not atoms or len(atoms) != len(lineage) or len(set(lineage)) != len(lineage)
            or any(type(s) is not int or s < 0 for s in lineage)
            or [a['id'] for a in atoms] != list(range(len(atoms)))
            or [a['atom_id'] for a in targets] != list(range(len(targets))) or len(targets) < 2):
        raise ValueError('Unique packed candidate lineage and ordered source target IDs required')
    target_xy = [[v * IMAGE_SIZE / QMAX for v in adapted_anchor(a['center'])] for a in targets]
    if len(set(map(tuple, target_xy))) != len(target_xy):
        raise ValueError('Target coordinate collision: cannot certify local ownership')
    distances = {}
    for slot, atom in zip(lineage, atoms):
        if (len(atom['anchor']) != 2 or any(type(v) is not int or not 0 <= v <= QMAX
                                          for v in atom['anchor'])):
            raise ValueError('Integer q1023 saved anchors required')
        point = [v * IMAGE_SIZE / QMAX for v in atom['anchor']]
        distances[slot] = [math.dist(point, xy) for xy in target_xy]
    output = []
    for slot in lineage:
        ds = distances[slot]
        nearest = sorted(range(len(targets)), key=ds.__getitem__)
        target = nearest[0]
        d = ds[target]
        other_candidate = min((values[target] for s, values in distances.items() if s != slot),
                              default=math.inf)
        separation = min(math.dist(target_xy[target], xy) for t, xy in enumerate(target_xy) if t != target)
        reasons = []
        if ds[nearest[1]] - d <= 2 * EPSILON:
            reasons.append('target_tie_or_quantization_overlap')
        if other_candidate - d <= 2 * EPSILON:
            reasons.append('not_quantization_robust_reciprocal_nearest')
        if d + EPSILON >= separation / 2:
            reasons.append('outside_local_separation_ball')
        output.append(dict(slot=slot, nearest_target=target, distance_in_match_metric=d,
            distance_bounds=[max(0, d - EPSILON), d + EPSILON],
            radius_status=('robust_inside' if d + EPSILON <= RADIUS else
                           'robust_outside' if d - EPSILON > RADIUS else 'quantization_boundary'),
            possible_radius_target_ids=[t for t, distance in enumerate(ds) if distance - EPSILON <= RADIUS],
            diagnostic_target=target if not reasons else None, unresolved_reasons=reasons))
    return output, distances


def receipt_membership(pairs, sampled_rows, coverage, locations):
    """Known sampled endpoints prove membership, not an unsaved float distance.

If these endpoints exhaust the recorded matched count, the remaining selected
slots are proven unmatched. A matched slot's target can be derived only when
exactly one target is within the radius including quantization uncertainty.
"""
    labels = {}
    selected = {r['slot'] for r in locations}
    for row in sampled_rows:
        rid, label = row['row_id'], row['label']
        if type(rid) is not int or not 0 <= rid < len(pairs) or rid in labels or type(label) is not bool:
            raise ValueError('Unique exact sampled row IDs and boolean labels required')
        if len(pairs[rid]) != 2 or not set(pairs[rid]) <= selected:
            raise ValueError('Sampled endpoints outside saved selected lineage')
        labels[rid] = label
    known = {s for rid in labels for s in pairs[rid]}
    matched_count = coverage['selected_atoms'] - coverage['unmatched_selected_atoms']
    if (coverage['selected_atoms'] != len(selected) or not 0 <= matched_count <= len(selected)
            or len(known) > matched_count or sum(labels.values()) != coverage['reachable_positive_pairs']):
        raise ValueError('Sampler/coverage mismatch (the trial sampled every reachable positive)')
    exhausted = len(known) == matched_count
    result = []
    for r in locations:
        slot = r['slot']
        status = ('matched' if slot in known else 'unmatched' if exhausted else 'unresolved')
        possible = r['possible_radius_target_ids']
        if status == 'matched' and not possible:
            raise ValueError('Receipt-proven match impossible under saved anchor bounds')
        target = possible[0] if status == 'matched' and len(possible) == 1 else None
        result.append(dict(slot=slot, status=status, target_id=target,
            target_evidence='receipt_membership_and_unique_radius_target' if target is not None else None))
    mapping = {r['slot']: r['target_id'] for r in result if r['target_id'] is not None}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Receipt-derived assignments are not one-to-one')
    return result, labels


def literal_text_status(target, atom):
    expected = target['text']
    predicted = '' if atom['text'] is None else atom['text']['value']
    if expected == predicted:
        return 'exact_literal'
    if not predicted and target['visible_text']:
        return 'visible_text_not_routed_to_ocr'
    if (set((expected, predicted)) == {'HN', 'NH'} and target['element'] == 'N'
            and atom['chemical']['element'] == 'N' and atom['chemical']['explicit_h_count'] == 1
            and target['explicit_h_count'] + target['implicit_h_count'] == 1):
        return 'hydrogen_layout_equivalent_not_element_error'
    return 'literal_mismatch'


def component_members(lineage, edges, association):
    neighbors = {s: set() for s in lineage}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining, result = set(lineage), []
    while remaining:
        queue = [min(remaining)]
        remaining.remove(queue[0])
        members = []
        while queue:
            slot = queue.pop()
            members.append(slot)
            new = neighbors[slot] & remaining
            remaining -= new
            queue.extend(sorted(new))
        result.append(dict(slots=sorted(members), diagnostic_targets=sorted(
            association[s] for s in members if s in association),
            unresolved_slots=sorted(s for s in members if s not in association)))
    return result


def inventory(layers, records, coverage, sampled_rows):
    before, after = records
    trace = before['trace']
    atoms, lineage = trace['scene_before_fallback']['atom_objects'], trace['compiled_candidate_indices']
    if (lineage != after['trace']['compiled_candidate_indices']
            or before['pairs'] != after['pairs']
            or before['metrics']['candidate_label_sha256'] != after['metrics']['candidate_label_sha256']
            or before['metrics']['text_sha256'] != after['metrics']['text_sha256']):
        raise ValueError('Saved upstream/label/text lineage changed across phases')
    locations, distances = correspondence(atoms, lineage, layers['atoms'])
    membership, labels = receipt_membership(before['pairs'], sampled_rows, coverage, locations)
    loc = {r['slot']: r for r in locations}
    mem = {r['slot']: r for r in membership}
    diagnostic = {r['slot']: r['diagnostic_target'] for r in locations if r['diagnostic_target'] is not None}
    if len(set(diagnostic.values())) != len(diagnostic):
        raise ValueError('Diagnostic ownership is not one-to-one')
    inverse = {t: s for s, t in diagnostic.items()}
    official = {r['slot']: r['target_id'] for r in membership if r['target_id'] is not None}
    official_inverse = {t: s for s, t in official.items()}
    if any(diagnostic.get(s) != t for s, t in official.items()):
        raise ValueError('Local diagnostic and receipt-derived target identities disagree')
    by_slot = dict(zip(lineage, atoms))
    pairs = {tuple(p): i for i, p in enumerate(before['pairs'])}
    if len(pairs) != len(before['pairs']) or set(pairs) != set(combinations(sorted(lineage), 2)):
        raise ValueError('Saved pair rows must enumerate every selected unordered pair once')
    target_edges = {tuple(sorted((b['begin_atom'], b['end_atom']))): b for b in layers['bonds']}
    for rid, label in labels.items():
        a, b = before['pairs'][rid]
        if a in official and b in official:
            if (tuple(sorted((official[a], official[b]))) in target_edges) != label:
                raise ValueError('Receipt-derived correspondence contradicts exact sampled pair labels')
    nodes = []
    for target in layers['atoms']:
        tid = target['atom_id']
        slot = inverse.get(tid)
        nearest = min(lineage, key=lambda s: distances[s][tid])
        row = dict(target_id=tid, expected_element=target['element'], expected_text=target['text'],
            source_center=target['center'], adapted_q1023=adapted_anchor(target['center']),
            candidate_slot=slot, receipt_matched_slot=official_inverse.get(tid),
            nearest_selected_slot=nearest, nearest_distance_in_match_metric=distances[nearest][tid],
            no_selected_within_radius_even_with_rounding=all(ds[tid] - EPSILON > RADIUS for ds in distances.values()),
            status='no_robust_local_selected_correspondence' if slot is None else
                'matched' if mem[slot]['status'] == 'matched' else 'selected_but_' + mem[slot]['status'],
            incident_target_bonds=[b['bond_id'] for b in layers['bonds'] if tid in (b['begin_atom'], b['end_atom'])])
        if slot is not None:
            atom = by_slot[slot]
            row.update(location=loc[slot], receipt_membership=mem[slot], predicted_role=atom['role'],
                predicted_element=atom['chemical']['element'],
                predicted_text=None if atom['text'] is None else atom['text']['value'],
                text_status=literal_text_status(target, atom),
                stable_chemical_mismatches=[field for field in ('element', 'formal_charge', 'isotope', 'radical_electrons')
                    if atom['chemical'][field] != target[field]],
                expected_chiral_tag=target['chiral_tag'],
                predicted_chiral_tag=atom['chemical']['canonical_chiral_tag'],
                stereo_annotation_missing=target['chiral_tag'] != 'CHI_UNSPECIFIED'
                    and atom['chemical']['canonical_chiral_tag'] == 'CHI_UNSPECIFIED')
        nodes.append(row)
    phases = {}
    for step, record in zip((0, 50), records):
        scene = record['trace']['scene_before_fallback']
        if [{k: v for k, v in a.items() if k != 'incident_ports'} for a in scene['atom_objects']] != [
                {k: v for k, v in a.items() if k != 'incident_ports'} for a in atoms]:
            raise ValueError('Atom attributes changed; shared node inventory would be invalid')
        edges = edge_map(scene, lineage)
        if (len(record['presence_logits']) != len(pairs) or len(record['chemical_logits']) != len(pairs)
                or any(not math.isfinite(z) for z in record['presence_logits'])
                or any(len(z) != 4 or any(not math.isfinite(v) for v in z) for z in record['chemical_logits'])):
            raise ValueError('Complete finite saved pair logits required')
        def prediction(pair):
            rid = pairs[pair]
            z, types = record['presence_logits'][rid], record['chemical_logits'][rid]
            probability = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            present = probability >= .85
            if present != (pair in edges):
                raise ValueError('Saved scene / raw presence threshold mismatch')
            edge = edges.get(pair)
            return dict(row_id=rid, pair=list(pair), presence_logit=z, probability=probability, present=present,
                raw_type=BOND_TYPES[max(range(4), key=types.__getitem__)], chemical_logits=types,
                scene_type=None if edge is None else edge['chemical_type'],
                scene_direction=None if edge is None else edge['direction'],
                scene_stereo=None if edge is None else edge['stereo'],
                sampled_label=labels.get(rid), sampled_label_available=rid in labels)
        bonds = []
        receipt_typed = []
        for bond in layers['bonds']:
            a, b = bond['begin_atom'], bond['end_atom']
            reachable = a in official_inverse and b in official_inverse
            row = dict(bond_id=bond['bond_id'], target_endpoints=[a, b], expected_type=bond['chemical_type'],
                receipt_derived_reachable=reachable,
                expected_drawing_direction=bond['drawing']['direction'], expected_stereo=bond['stereo'])
            if a not in inverse or b not in inverse:
                row.update(status='missing_local_selected_endpoint', prediction=None,
                    missing_target_endpoints=[t for t in (a, b) if t not in inverse])
            else:
                pred = prediction(tuple(sorted((inverse[a], inverse[b]))))
                row.update(prediction=pred, status='missing_edge' if not pred['present'] else
                    'wrong_raw_type' if pred['raw_type'] != bond['chemical_type'] else 'present_raw_type_correct',
                    raw_type_mismatch=pred['raw_type'] != bond['chemical_type'],
                    stereo_annotation_missing=bond['stereo'] != 'STEREONONE' and pred['scene_stereo'] in (None, 'STEREONONE'))
                if reachable and pred['present'] and pred['raw_type'] == bond['chemical_type']:
                    receipt_typed.append(bond['bond_id'])
            bonds.append(row)
        if len(official) == coverage['selected_atoms'] - coverage['unmatched_selected_atoms']:
            if (sum(b['receipt_derived_reachable'] for b in bonds) != coverage['reachable_positive_pairs']
                    or sorted(receipt_typed) != record['metrics']['correct_typed_bond_ids']):
                raise ValueError('Receipt-derived coverage / typed IDs fail historical cross-check')
        extra = []
        for pair in sorted(edges):
            if all(s in diagnostic for s in pair):
                targets = sorted(diagnostic[s] for s in pair)
                if tuple(targets) in target_edges:
                    continue
                evidence = 'diagnostic_nonedge'
            else:
                targets, evidence = None, 'unresolved_target_correspondence'
            pred = prediction(pair)
            extra.append(dict(**pred, diagnostic_target_endpoints=targets, geometric_status=evidence,
                official_membership_unknown=any(mem[s]['status'] != 'matched' for s in pair),
                confirmed_sampled_negative=pred['sampled_label_available'] and pred['sampled_label'] is False))
        phases[str(step)] = dict(target_bonds=bonds, extra_or_unresolved_edges=extra,
            components=component_members(lineage, edges, diagnostic),
            counts=dict(Counter(b['status'] for b in bonds)), historical_metrics=record['metrics'])
    return dict(nodes=nodes, selected_correspondence=locations, receipt_membership=membership,
        receipt_derived_mapping_complete=len(official) == coverage['selected_atoms'] - coverage['unmatched_selected_atoms'],
        coverage=coverage, phases=phases, original_float_matching_replayed=False,
        diagnostic_correspondence_is_training_label=False, raw_proposal_absence_cause_available=False,
        stereo_scope='Missing saved annotations only; no CIP/tag-orientation equivalence certification')
