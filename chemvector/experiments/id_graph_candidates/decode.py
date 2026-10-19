"""One-edit candidate generation without truth, image reads, or learned ranking."""

from copy import deepcopy
from dataclasses import dataclass
import math

from rdkit import Chem

from chemvector.ocr import parse_atom_display_text
from chemvector.experiments.id_graph.graph import to_sdf
from chemvector.experiments.id_graph_prediction.adapter import BOND_TYPES, from_prediction


@dataclass(frozen=True)
class Budget:
    attribute_edits: int = 64
    delete_edges: int = 4
    retype_edges: int = 4

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.attribute_edits, self.delete_edges, self.retype_edges)):
            raise ValueError('Candidate budgets must be nonnegative integers')


def compiled(graph):
    try:
        sdf = to_sdf(graph)
        mol = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=True)
        if mol is None:
            raise ValueError('Invalid SDF readback')
        mol = Chem.RemoveHs(mol)
        stereo = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        Chem.RemoveStereochemistry(mol)
        core = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return {'valid': True, 'core': core, 'stereo': stereo, 'error': None}
    except Exception as error:
        return {'valid': False, 'core': None, 'stereo': None,
                'error': {'type': type(error).__name__, 'message': str(error)}}


def attribute_actions(graph, evidence, limit):
    actions = []
    for atom in sorted(graph['atoms'], key=lambda a: a['id']):
        item = evidence.get(atom['id'])
        if item is None or item['error'] or not item['candidates']:
            continue
        best = item['candidates'][0]['score']
        for candidate in item['candidates']:
            parsed = parse_atom_display_text(candidate['text'])
            if parsed is None or any(parsed[f] != atom[f] for f in ('element', 'isotope', 'radical_electrons')):
                continue
            values = {f: parsed[f] for f in ('formal_charge', 'explicit_h_count')}
            values['no_implicit'] = parsed['explicit_h_count'] > 0
            if all(atom[f] == value for f, value in values.items()):
                continue
            actions.append({'kind': 'ATTRIBUTE', 'atom_id': atom['id'], 'values': values,
                            'evidence': {'object_key': item['object_key'], 'crop_sha256': item['crop_sha256'],
                                         'model_id': item['model_id'], 'text': candidate['text'],
                                         'rank': candidate['rank'], 'score_kind': candidate['score_kind'],
                                         'ctc_log_probability': candidate['score'],
                                         'within_crop_logp_gap': best - candidate['score']}})
    # This is an enumeration budget, NOT a calibrated cross-crop ranking.
    # Breadth-first by beam rank prevents one atom from monopolizing the budget.
    actions.sort(key=lambda a: (a['evidence']['rank'], a['atom_id']))
    return actions[:limit], len(actions)


def bond_actions(raw, graph, threshold, budget):
    selected = {b['id'] for b in graph['bonds']}
    pairs = [p for p in raw['pairs'] if p['id'] in selected]
    deletion = sorted(pairs, key=lambda p: (p['presence_logit'], p['id']))
    margin = lambda p: sorted(p['chemical_logits'], reverse=True)[0] - sorted(p['chemical_logits'], reverse=True)[1]
    retyping = sorted(pairs, key=lambda p: (margin(p), p['id']))
    result = []
    for pair in deletion[:budget.delete_edges]:
        result.append({'kind': 'DELETE', 'bond_id': pair['id'],
                       'evidence': {'presence_logit': pair['presence_logit'],
                                    'presence_margin_to_frozen_threshold': pair['presence_logit'] - threshold}})
    for pair in retyping[:budget.retype_edges]:
        logits = pair['chemical_logits']
        chosen = max(range(4), key=logits.__getitem__)
        for index in sorted(range(4), key=lambda i: (-logits[i], i)):
            if index == chosen:
                continue
            result.append({'kind': 'RETYPE', 'bond_id': pair['id'], 'bond_type': BOND_TYPES[index],
                           'evidence': {'chemical_logits': list(logits), 'top_two_margin': margin(pair),
                                        'class_logit_gap': logits[chosen] - logits[index]}})
    return result


def apply_action(graph, action):
    changed = deepcopy(graph)
    if action['kind'] == 'ATTRIBUTE':
        atom = next(a for a in changed['atoms'] if a['id'] == action['atom_id'])
        if set(action['values']) != {'formal_charge', 'explicit_h_count', 'no_implicit'}:
            raise ValueError('Attribute edit exceeded its contract')
        atom.update(action['values'])
    elif action['kind'] == 'DELETE':
        if not any(b['id'] == action['bond_id'] for b in changed['bonds']):
            raise ValueError('Unknown deletion edge')
        changed['bonds'] = [b for b in changed['bonds'] if b['id'] != action['bond_id']]
    elif action['kind'] == 'RETYPE':
        bond = next(b for b in changed['bonds'] if b['id'] == action['bond_id'])
        if action['bond_type'] not in BOND_TYPES:
            raise ValueError('Unknown bond type')
        bond.update(type=action['bond_type'], aromatic=action['bond_type'] == 'AROMATIC')
    else:
        raise ValueError('Unsupported edit')
    if action['kind'] == 'DELETE':
        neighbors = {a['id']: [] for a in changed['atoms']}
        for bond in sorted(changed['bonds'], key=lambda b: b['id']):
            first, second = bond['atom_ids']
            neighbors[first].append(second)
            neighbors[second].append(first)
        for atom in changed['atoms']:
            atom['stereo']['neighbor_ids'] = neighbors[atom['id']]
    if [a['id'] for a in graph['atoms']] != [a['id'] for a in changed['atoms']] or changed['observations'] != graph['observations']:
        raise ValueError('Edit modified identities or input observations')
    return changed


def decode(raw, graph, evidence, threshold, budget=Budget()):
    if not math.isfinite(threshold):
        raise ValueError('Invalid frozen threshold')
    reconstructed, _ = from_prediction(raw, threshold)
    if reconstructed != graph:
        raise ValueError('Raw prediction does not reproduce the frozen ID graph')
    atoms = {a['id']: a for a in graph['atoms']}
    original_text = {r['id']: r['text'] for r in raw['ocr']}
    for key, item in evidence.items():
        if key not in atoms or item['object_key'] != f"{raw['sample_key']}:p{key:03d}":
            raise ValueError('OCR evidence belongs to a different candidate')
        if item['raw_greedy_text'] != original_text.get(key):
            raise ValueError('Evidence greedy string differs from saved prediction')
    original = deepcopy((raw, graph, evidence))
    baseline = compiled(graph)
    result = {'sample_key': raw['sample_key'], 'baseline': baseline, 'groups': [], 'attempts': [],
              'automatic_replacement': False, 'disposition': 'KEEP_VALID_BASELINE',
              'missing_beam_records': sum(bool(e['error']) for e in evidence.values()),
              'attribute_eligible': 0, 'attribute_truncated': 0}
    if baseline['valid']:
        return result
    attributes, total = attribute_actions(graph, evidence, budget.attribute_edits)
    result['attribute_eligible'] = total
    result['attribute_truncated'] = total - len(attributes)
    groups = {}
    for action in attributes + bond_actions(raw, graph, threshold, budget):
        changed = apply_action(graph, action)
        outcome = compiled(changed)
        result['attempts'].append({'action': action, **outcome})
        if not outcome['valid']:
            continue
        key = outcome['stereo']
        if key not in groups:
            groups[key] = {'core': outcome['core'], 'stereo': key, 'representative_graph': changed, 'actions': []}
        groups[key]['actions'].append(action)
    result['groups'] = [groups[key] for key in sorted(groups)]
    result['disposition'] = ('ABSTAIN_NO_VALID_CANDIDATE' if not groups else
                             'REVIEW_MULTIPLE_CHEMICAL_GRAPHS' if len(groups) > 1 else 'REVIEW_SINGLE_UNPROVEN_CANDIDATE')
    if (raw, graph, evidence) != original:
        raise ValueError('Candidate generation mutated inputs')
    return result
