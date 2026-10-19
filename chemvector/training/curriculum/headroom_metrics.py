"""Post-prediction training-sample diagnostics, never oracle-fed inference."""

from collections import Counter
from hashlib import sha256

import torch

SEED = '20260918-initial-headroom-v1'
SAMPLE_COUNT = 32


def select_training_rows(manifest):
    """Hash order, distinct molecules/identities; no model scores or replacement."""
    all_rows = manifest['samples']
    keys = [r['source'] + ':' + r['sample_key'] for r in all_rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate released sample keys')
    candidates = [r for r in all_rows if r['split'] == 'train' and r['domain'] == 'full_molecule']
    candidates.sort(key=lambda r: (sha256((SEED + ':' + r['source'] + ':' + r['sample_key']).encode()).hexdigest(),
                                   r['source'], r['sample_key']))
    result, molecules, identities = [], set(), set()
    for row in candidates:
        if row['molecule_id'] in molecules or row['canonical_smiles'] in identities:
            continue
        result.append(dict(row))
        molecules.add(row['molecule_id'])
        identities.add(row['canonical_smiles'])
        if len(result) == SAMPLE_COUNT:
            return result
    raise ValueError('Fewer than 32 distinct admitted full-molecule training identities')


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., right - left) * max(0., bottom - top)
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    bb = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.


@torch.no_grad()
def measure(sample, batch, raw, structure, trace, config, pair_threshold):
    """All truth use is after the unmodified initial prediction has completed."""
    from chemvector.bond_object import BOND_TYPES
    from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
    from chemvector.train_direct_atom_pair_bond import match_atom_targets, build_pair_targets
    from .upstream_diagnostic import labeled_coverage

    if (sample.get('purpose') != 'diagnostic' or sample['meta']['split'] != 'train'
            or sample['meta']['external_overlap_status'] != 'checked'):
        raise ValueError('Headroom accepts only admitted training images in diagnostic mode')
    atoms = structure['atoms']
    coverage = labeled_coverage(raw, atoms, batch, config)
    mapping, matched = match_atom_targets(atoms, batch, config)
    pairs, rows = enumerate_unordered_atom_pairs(atoms['atom_mask'])
    if not torch.equal(pairs, structure['pairs']):
        raise ValueError('Scored pairs differ from actual prediction pairs')
    targets = build_pair_targets(pairs, rows, mapping, batch, atoms)
    present = structure['output']['presence_logits'].sigmoid() >= pair_threshold
    positive = targets['presence']
    true_positive = int((present & positive).sum())
    wrong_kind = present & positive & (structure['chemistry'] != targets['chemical'])
    pair_errors = []
    for i in range(len(pairs)):
        expected, predicted = bool(positive[i]), bool(present[i])
        if expected != predicted or bool(wrong_kind[i]):
            pair_errors.append({'candidates': pairs[i].tolist(),
                'target_slots': mapping[0, pairs[i]].tolist(),
                'kind': 'wrong_chemical' if expected and predicted else 'missing' if expected else 'extra',
                'probability': float(structure['output']['presence_logits'][i].sigmoid()),
                'predicted_chemical': BOND_TYPES[int(structure['chemistry'][i])] if predicted else None,
                'expected_chemical': BOND_TYPES[int(targets['chemical'][i])] if expected else None})
    pair = {'target_bonds': coverage['target_bonds'], 'reachable_bonds': int(positive.sum()),
            'true_positive': true_positive, 'extra_edges': int((present & ~positive).sum()),
            'missing_reachable_edges': int((~present & positive).sum()),
            'unreachable_bonds': coverage['target_bonds'] - int(positive.sum()),
            'missing_total_edges': coverage['target_bonds'] - true_positive,
            'wrong_chemical_edges': int(wrong_kind.sum()), 'errors': pair_errors}
    bijective = coverage['selected_matched'] == coverage['selected_atoms'] == coverage['target_atoms']
    pair['exact_on_fixed_candidates'] = (bijective and pair['extra_edges'] == 0
        and pair['missing_total_edges'] == 0 and pair['wrong_chemical_edges'] == 0)

    roles = {r['candidate']: r for r in trace['roles']}
    ocr = {r['candidate']: r for r in trace['ocr']}
    selected_ids = atoms['atom_mask'][0].nonzero().flatten().tolist()
    if set(roles) != set(selected_ids):
        raise ValueError('Role trace does not cover exactly the selected candidates')
    target_to_candidate = {int(mapping[0, c]): c for c in selected_ids if int(mapping[0, c]) >= 0}
    visual = {a['atom_id']: a for a in sample['layers']['atoms']}
    text_rows = []
    for slot, atom_id in enumerate(batch['atom_ids'][0]):
        target = visual[atom_id]
        candidate = target_to_candidate.get(slot)
        gate = roles[candidate]['text'] if candidate is not None else None
        decoded = ocr.get(candidate)
        iou = None
        if decoded is not None and target['bbox'] is not None:
            x, y = decoded['center']; w, h = decoded['half_extent']
            iou = box_iou([(x-w)*768, (y-h)*768, (x+w)*768, (y+h)*768], target['bbox'])
        text_rows.append({'target_slot': slot, 'atom_id': atom_id, 'candidate': candidate,
            'visible_text': target['visible_text'], 'expected_text': target['text'],
            'gate': gate, 'predicted_text': None if decoded is None else decoded['text'],
            'literal_match': bool(decoded is not None and decoded['text'] == target['text'])
                             if target['visible_text'] else gate is False,
            'bbox_iou': iou})
    text = {'visible_targets': sum(r['visible_text'] for r in text_rows),
            'missing_visible_candidates': sum(r['visible_text'] and r['candidate'] is None for r in text_rows),
            'closed_visible_gates': sum(r['visible_text'] and r['candidate'] is not None and not r['gate'] for r in text_rows),
            'opened_implicit_gates': sum(not r['visible_text'] and r['gate'] is True for r in text_rows),
            'active_literal_mismatches': sum(r['visible_text'] and r['gate'] is True and not r['literal_match'] for r in text_rows),
            'exact_literal_labels': sum(r['visible_text'] and r['literal_match'] for r in text_rows),
            'unmatched_selected_text_gates': sum(roles[c]['text'] and int(mapping[0, c]) < 0 for c in selected_ids),
            'rows': text_rows}
    text['roles_exact'] = bijective and all(r['gate'] == r['visible_text'] for r in text_rows)
    text['literal_path_exact'] = text['roles_exact'] and all(r['literal_match'] for r in text_rows)

    # These stable atomic fields cannot be equated with full chemical identity.
    fields = ('element', 'formal_charge', 'isotope', 'radical_electrons')
    truth_atoms = {a['atom_id']: a for a in sample['graph']['atoms']}
    scene = trace.get('scene_before_fallback')
    ids = trace.get('compiled_candidate_indices', [])
    differences = []
    mapped_count = 0
    if scene is not None:
        if len(ids) != len(scene['atom_objects']):
            raise ValueError('Compiled scene/candidate mapping mismatch')
        for prediction, candidate in zip(scene['atom_objects'], ids):
            slot = int(mapping[0, candidate])
            if slot < 0:
                continue
            expected = truth_atoms[batch['atom_ids'][0][slot]]
            mapped_count += 1
            delta = {f: {'expected': expected[f], 'predicted': prediction['chemical'][f]}
                     for f in fields if expected[f] != prediction['chemical'][f]}
            if delta:
                differences.append({'candidate': candidate, 'target_slot': slot, 'fields': delta})
    stereo = (any(a['chiral_tag'] != 'CHI_UNSPECIFIED' for a in truth_atoms.values())
              or any(b['stereo'] != 'STEREONONE' for b in sample['graph']['bonds']))
    attributes = {'fields': fields, 'differences': differences, 'scene_available': scene is not None,
        'basic_fields_exact': bijective and mapped_count == coverage['target_atoms'] and not differences,
        'target_has_stereo': stereo,
        'not_checked_as_basic_fields': ['hydrogen count', 'aromaticity', 'no_implicit', 'stereochemistry']}
    return {'coverage': coverage, 'candidate_bijection': bijective, 'pair': pair,
            'text': text, 'attributes': attributes}


def classify(record):
    """Non-exclusive observed errors; no oracle repair or SDF recovery claim."""
    m, exact = record['metrics'], record['prediction']['exact']
    c, t, p, a = m['coverage'], m['text'], m['pair'], m['attributes']
    isolated_pair = (not exact and m['candidate_bijection'] and t['literal_path_exact']
                     and a['basic_fields_exact'] and not a['target_has_stereo'] and not p['exact_on_fixed_candidates'])
    return {'baseline_correct_control': bool(exact),
        'raw_candidate_gap': c['raw_matched'] < c['target_atoms'],
        'selection_gap': c['raw_matched'] == c['target_atoms'] and c['selected_matched'] < c['target_atoms'],
        'extra_selected_candidates': c['unmatched_selected'] > 0,
        'pair_task_error': not p['exact_on_fixed_candidates'],
        'role_entry_error': bool(t['closed_visible_gates'] or t['opened_implicit_gates'] or t['unmatched_selected_text_gates']),
        'active_text_difference': t['active_literal_mismatches'] > 0,
        'pair_isolation_candidate': isolated_pair,
        'text_interface_candidate': bool(not exact and m['candidate_bijection'] and p['exact_on_fixed_candidates']
                                          and t['roles_exact'] and t['active_literal_mismatches']),
        'bbox_training_justified': False}


def summarize(records):
    flags = Counter(k for r in records for k, value in r['flags'].items() if value)
    keys = ('target_atoms', 'target_bonds', 'raw_matched', 'selected_matched',
            'selected_atoms', 'unmatched_selected', 'reachable_positive_pairs')
    return {'images': len(records), 'baseline_exact': sum(r['prediction']['exact'] for r in records),
        'baseline_valid': sum(r['prediction']['valid'] for r in records),
        'baseline_fallback': sum(r['prediction']['fallback'] for r in records),
        'candidate_bijection_images': sum(r['metrics']['candidate_bijection'] for r in records),
        'pair_exact_images': sum(r['metrics']['pair']['exact_on_fixed_candidates'] for r in records),
        'literal_text_path_exact_images': sum(r['metrics']['text']['literal_path_exact'] for r in records),
        'coverage_totals': {k: sum(r['metrics']['coverage'][k] for r in records) for k in keys},
        'nonexclusive_flags': {k: flags[k] for k in records[0]['flags']},
        'flag_keys': {k: [r['key'] for r in records if r['flags'][k]] for k in records[0]['flags']},
        'training_authorized': False,
        'interpretation': 'Opportunity screening only; neither a measured oracle SDF ceiling nor evidence that optimization will improve validation'}
