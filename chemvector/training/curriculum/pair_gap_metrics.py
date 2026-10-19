"""Post-prediction metrics for the fixed mixed-origin pair-gap diagnostic.

No training admission, matching intervention, reference OCR or model repair.
"""

from itertools import combinations

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .pair_readiness import geometry_negatives
from .supervision_contract import pair_targets_with_unknown
from .upstream_diagnostic import labeled_coverage

BOND_STAGES = ('raw_correspondence_gap', 'selection_correspondence_gap',
               'reachable_missing', 'reachable_wrong_type', 'reachable_correct')


def validate_sample(sample, row):
    """Preserve actual admission flags; permit only the two explicit origins."""
    meta = sample['meta']
    expected = {'new_quarantine': ('quarantine', False, 'unchecked'),
                'existing_training_control': ('train', True, 'checked')}
    if row.get('origin') not in expected:
        raise ValueError('Unknown diagnostic origin')
    if (sample.get('purpose') != 'diagnostic' or sample['key'] != row['sample_key']
            or meta['molecule_id'] != row['molecule_id']
            or sample['graph']['canonical_smiles'] != row['canonical_smiles']
            or meta['profile_id'] != 'reference'
            or (meta['split'], meta['training_authorized'], meta['external_overlap_status']) != expected[row['origin']]):
        raise ValueError('Diagnostic sample differs from its fixed origin or identity')


@torch.no_grad()
def measure(sample, row, batch, raw, structure, trace, config, threshold):
    validate_sample(sample, row)
    if not 0 < threshold < 1:
        raise ValueError('Expected fixed probability threshold')
    atoms = structure['atoms']
    if bool((atoms['atom_mask'] & ~raw['atom_mask']).any()):
        raise ValueError('Selected candidates must be a subset of raw candidates')
    coverage = labeled_coverage(raw, atoms, batch, config)
    _, raw_matched = match_atom_targets(raw, batch, config)
    mapping, _ = match_atom_targets(atoms, batch, config)
    contract = pair_targets_with_unknown(atoms, mapping, batch)
    if not torch.equal(contract['pair_indices'], structure['pairs']):
        raise ValueError('Metrics must consume the exact predicted pair enumeration')
    probabilities = structure['output']['presence_logits'].sigmoid()
    chemical_logits = structure['output']['chemical_logits']
    chemistry = structure['chemistry']  # Existing aromatic handling, unchanged.
    if (probabilities.shape != (len(structure['pairs']),)
            or chemical_logits.shape != (len(probabilities), len(BOND_TYPES))
            or chemistry.shape != probabilities.shape
            or not bool(torch.isfinite(probabilities).all())
            or not bool(torch.isfinite(chemical_logits).all())
            or bool(((chemistry < 0) | (chemistry >= len(BOND_TYPES))).any())):
        raise ValueError('Invalid predicted pair outputs')
    selected = atoms['atom_mask'][0].nonzero().flatten().tolist()
    to_candidate = {int(mapping[0, c]): c for c in selected if int(mapping[0, c]) >= 0}
    pair_rows = {tuple(pair): i for i, pair in enumerate(structure['pairs'].tolist())}
    slots = batch['atom_ids'][0]
    truth = {a['atom_id']: a for a in sample['graph']['atoms']}
    visual = {a['atom_id']: a for a in sample['layers']['atoms']}
    roles = {r['candidate']: r for r in trace['roles']}
    ocr = {r['candidate']: r for r in trace['ocr']}
    if len(roles) != len(trace['roles']) or set(roles) != set(selected):
        raise ValueError('Role trace must cover exactly selected candidates')
    if len(ocr) != len(trace['ocr']) or set(ocr) != {c for c in selected if roles[c]['text']}:
        raise ValueError('OCR trace must cover exactly open text gates')
    scene = trace.get('scene_before_fallback')
    compiled = trace.get('compiled_candidate_indices', [])
    if scene is not None and len(compiled) != len(scene['atom_objects']):
        raise ValueError('Compiled scene mapping differs')
    scene_atoms = {} if scene is None else dict(zip(compiled, scene['atom_objects']))
    fields = ('element', 'formal_charge', 'isotope', 'radical_electrons')
    atom_rows = []
    for slot, identity in enumerate(slots):
        candidate = to_candidate.get(slot)
        a, v = truth[identity], visual[identity]
        actual = None if candidate not in scene_atoms else scene_atoms[candidate]['chemical']
        delta = None if actual is None else {f: {'expected': a[f], 'predicted': actual[f]}
                                           for f in fields if a[f] != actual[f]}
        atom_rows.append({'slot': slot, 'atom_id': identity, 'element': a['element'],
            'candidate': candidate, **coverage['target_details'][slot],
            'expected_text': v['text'], 'visible_text': v['visible_text'],
            'text_gate': None if candidate is None else roles[candidate]['text'],
            'predicted_text': None if candidate not in ocr else ocr[candidate]['text'],
            'basic_field_differences': delta})

    def pair_record(left, right):
        candidates = [to_candidate.get(left), to_candidate.get(right)]
        index = None if None in candidates else pair_rows[tuple(sorted(candidates))]
        return {'target_slots': [left, right], 'candidates': candidates,
            'elements': '-'.join(sorted((truth[slots[left]]['element'], truth[slots[right]]['element']))),
            'reachable': index is not None,
            'raw_endpoints_matched': bool(raw_matched[0, left] and raw_matched[0, right]),
            'presence_probability': None if index is None else float(probabilities[index]),
            'present': False if index is None else bool(probabilities[index] >= threshold),
            'predicted_type': None if index is None else BOND_TYPES[int(chemistry[index])],
            'raw_type_argmax': None if index is None else BOND_TYPES[int(chemical_logits[index].argmax())]}

    bonds, target_pairs = [], set()
    for i in range(int(batch['bond_mask'].sum())):
        left, right = batch['bond_atom_refs'][0, i].tolist()
        target_pairs.add(tuple(sorted((left, right))))
        record = pair_record(left, right)
        record['expected_type'] = BOND_TYPES[int(batch['bond_chemical'][0, i])]
        if not record['reachable']:
            stage = 'selection_correspondence_gap' if record['raw_endpoints_matched'] else 'raw_correspondence_gap'
        elif not record['present']:
            stage = 'reachable_missing'
        else:
            stage = 'reachable_correct' if record['predicted_type'] == record['expected_type'] else 'reachable_wrong_type'
        bonds.append({**record, 'stage': stage})
    geometry = geometry_negatives(sample['graph'], sample['layers'])
    oso = {tuple(sorted(p['pair'])) for p in geometry['intervening_nonedge_pairs']
           if any(w['elements'] == ['O', 'S', 'O'] for w in p['witnesses'])}
    nonedges = [{**pair_record(i, j), 'oso_intervening': tuple(sorted((slots[i], slots[j]))) in oso}
                for i, j in combinations(range(len(slots)), 2) if (i, j) not in target_pairs]
    unknown = [{'candidates': structure['pairs'][i].tolist(),
                'target_slots': mapping[0, structure['pairs'][i]].tolist(),
                'presence_probability': float(probabilities[i])}
               for i in range(len(probabilities)) if bool(contract['unknown_mask'][i] and probabilities[i] >= threshold)]
    return {'coverage': coverage, 'atoms': atom_rows, 'bonds': bonds, 'nonedges': nonedges,
        'unknown_present_pairs': unknown, 'pair_contract': contract['coverage'],
        'bond_stages': {k: sum(b['stage'] == k for b in bonds) for k in BOND_STAGES},
        'candidate_bijection': coverage['target_atoms'] == coverage['selected_matched'] == coverage['selected_atoms']}


def summarize(cases):
    if not cases or len({r['key'] for r in cases}) != len(cases):
        raise ValueError('Nonempty distinct diagnostic cases required')
    result = {}
    for origin in ('all', 'new_quarantine', 'existing_training_control'):
        rows = [r for r in cases if origin == 'all' or r['origin'] == origin]
        bonds = [b for r in rows for b in r['metrics']['bonds']]
        nonedges = [b for r in rows for b in r['metrics']['nonedges']]
        atoms = [a for r in rows for a in r['metrics']['atoms']]
        motifs = {}
        for name in sorted({b['elements'] + ':' + b['expected_type'] for b in bonds}):
            values = [b for b in bonds if b['elements'] + ':' + b['expected_type'] == name]
            motifs[name] = {'targets': len(values), **{k: sum(b['stage'] == k for b in values) for k in BOND_STAGES}}
        negative_groups = {}
        for name in ('all', 'N-N', 'N-S', 'O-S-O'):
            values = [b for b in nonedges if name == 'all' or
                      (b['oso_intervening'] if name == 'O-S-O' else b['elements'] == name)]
            negative_groups[name] = {'targets': len(values), 'reachable': sum(b['reachable'] for b in values),
                                     'false_positive': sum(b['present'] for b in values)}
        result[origin] = {'images': len(rows),
            **{k: sum(r['prediction'][k] for r in rows) for k in ('exact', 'valid', 'fallback')},
            'candidate_bijection_images': sum(r['metrics']['candidate_bijection'] for r in rows),
            'coverage': {k: sum(r['metrics']['coverage'][k] for r in rows) for k in (
                'target_atoms', 'raw_matched', 'selected_matched', 'selected_atoms',
                'unmatched_selected', 'target_bonds', 'reachable_positive_pairs')},
            'bond_stages': {k: sum(b['stage'] == k for b in bonds) for k in BOND_STAGES},
            'bond_motifs': motifs, 'nonedge_groups': negative_groups,
            'unknown_present_pairs': sum(len(r['metrics']['unknown_present_pairs']) for r in rows),
            'visible_text_targets': sum(a['visible_text'] for a in atoms),
            'unmatched_visible_targets': sum(a['visible_text'] and a['candidate'] is None for a in atoms),
            'closed_visible_gates': sum(a['visible_text'] and a['text_gate'] is False for a in atoms),
            'opened_implicit_gates': sum(not a['visible_text'] and a['text_gate'] is True for a in atoms),
            'active_literal_mismatches': sum(a['visible_text'] and a['text_gate'] is True
                and a['expected_text'] != a['predicted_text'] for a in atoms),
            'basic_field_mismatch_atoms': sum(bool(a['basic_field_differences']) for a in atoms)}
    return result
