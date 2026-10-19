"""Truth-free additional edit families; original candidate set is always retained."""

from copy import deepcopy
from dataclasses import dataclass

from chemvector.ocr import parse_atom_display_text
from chemvector.experiments.id_graph_candidates.decode import Budget, compiled, decode as replay


@dataclass(frozen=True)
class ExpansionBudget:
    element_edits: int = 64
    delete_nodes: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.element_edits, self.delete_nodes)):
            raise ValueError('Budgets must be nonnegative integers')


def element_actions(graph, evidence, limit):
    actions = []
    for atom in sorted(graph['atoms'], key=lambda a: a['id']):
        item = evidence.get(atom['id'])
        if item is None or item['error']:
            continue
        for candidate in item['candidates']:
            parsed = parse_atom_display_text(candidate['text'])
            if (parsed is None or parsed['element'] in ('H', atom['element']) or atom['element'] == 'H'
                    or any(parsed[f] != atom[f] for f in ('isotope', 'radical_electrons'))):
                continue
            values = {f: parsed[f] for f in ('element', 'formal_charge', 'explicit_h_count')}
            values['no_implicit'] = parsed['explicit_h_count'] > 0
            actions.append({'kind': 'ELEMENT', 'atom_id': atom['id'], 'values': values,
                            'evidence': {'object_key': item['object_key'], 'crop_sha256': item['crop_sha256'],
                                         'model_id': item['model_id'], 'text': candidate['text'],
                                         'rank': candidate['rank'], 'score_kind': candidate['score_kind'],
                                         'ctc_log_probability': candidate['score']}})
    actions.sort(key=lambda a: (a['evidence']['rank'], a['atom_id']))
    return actions[:limit], len(actions)


def apply_edit(graph, action):
    changed = deepcopy(graph)
    atoms = {a['id']: a for a in changed['atoms']}
    key = action['atom_id']
    if key not in atoms:
        raise ValueError('Unknown candidate atom')
    if action['kind'] == 'ELEMENT':
        if set(action['values']) != {'element', 'formal_charge', 'explicit_h_count', 'no_implicit'}:
            raise ValueError('Element edit exceeded its contract')
        if action['values']['element'] in ('H', '*') or atoms[key]['element'] == 'H':
            raise ValueError('Element branch does not add/remove heavy atoms')
        atoms[key].update(action['values'])
        if changed['bonds'] != graph['bonds']:
            raise ValueError('Element edit modified connectivity')
    elif action['kind'] == 'DELETE_NODE':
        if len(atoms) <= 1:
            raise ValueError('Never propose an empty molecule')
        if atoms[key]['element'] == 'H':
            raise ValueError('Only heavy-node deletion is in scope')
        changed['atoms'] = [a for a in changed['atoms'] if a['id'] != key]
        changed['bonds'] = [b for b in changed['bonds'] if key not in b['atom_ids']]
        neighbors = {a['id']: [] for a in changed['atoms']}
        for bond in sorted(changed['bonds'], key=lambda b: b['id']):
            first, second = bond['atom_ids']
            neighbors[first].append(second)
            neighbors[second].append(first)
        for atom in changed['atoms']:
            atom['stereo']['neighbor_ids'] = neighbors[atom['id']]
    else:
        raise ValueError('Unknown extension action')
    if changed['observations'] != graph['observations']:
        raise ValueError('Original evidence pool must remain immutable')
    # Observations intentionally retain evidence for all original proposals,
    # including a rejected node; only the active semantic graph is edited.
    return changed


def extend(raw, graph, evidence, previous, threshold, budget=ExpansionBudget()):
    check = replay(raw, graph, evidence, threshold, Budget(0, 0, 0))
    if previous['sample_key'] != raw['sample_key'] or previous['baseline'] != check['baseline']:
        raise ValueError('Frozen candidate baseline does not replay')
    protected = deepcopy((raw, graph, evidence, previous))
    result = {'sample_key': raw['sample_key'], 'baseline': check['baseline'],
              'groups': deepcopy(previous['groups']), 'extension_attempts': [],
              'previous_attempt_count': len(previous['attempts']), 'automatic_replacement': False,
              'element_eligible': 0, 'element_truncated': 0, 'deletions_eligible': 0, 'deletions_truncated': 0,
              'baseline_replayed': True, 'old_candidate_set_preserved': True}
    if check['baseline']['valid']:
        if result['groups']:
            raise ValueError('Valid baseline should have no shadow edits')
        return result
    if any(a['stereo']['tag'] != 'CHI_UNSPECIFIED' for a in graph['atoms']):
        raise ValueError('Frozen unspecified-stereo baseline required')
    elements, total = element_actions(graph, evidence, budget.element_edits)
    result['element_eligible'], result['element_truncated'] = total, total - len(elements)
    nodes = [a for a in sorted(graph['atoms'], key=lambda a: a['id']) if a['element'] != 'H'] if len(graph['atoms']) > 1 else []
    result['deletions_eligible'] = len(nodes)
    result['deletions_truncated'] = max(0, len(nodes) - budget.delete_nodes)
    deletes = [{'kind': 'DELETE_NODE', 'atom_id': a['id'],
                'removed_bond_ids': [b['id'] for b in graph['bonds'] if a['id'] in b['atom_ids']],
                'evidence': {'source': 'all_active_heavy_nodes_in_stable_id_order',
                             'note': 'enumeration only, not evidence this atom is false'}}
               for a in nodes[:budget.delete_nodes]]
    groups = {g['stereo']: g for g in result['groups']}
    if len(groups) != len(result['groups']):
        raise ValueError('Duplicate frozen chemical groups')
    for action in elements + deletes:
        changed = apply_edit(graph, action)
        outcome = compiled(changed)
        result['extension_attempts'].append({'action': action, **outcome})
        if not outcome['valid']:
            continue
        key = outcome['stereo']
        if key not in groups:
            groups[key] = {'core': outcome['core'], 'stereo': key, 'representative_graph': changed, 'actions': []}
        groups[key]['actions'].append(action)
    result['groups'] = [groups[key] for key in sorted(groups)]
    if not {g['stereo'] for g in previous['groups']}.issubset(groups):
        raise ValueError('Extension dropped old candidates')
    if protected != (raw, graph, evidence, previous):
        raise ValueError('Extension mutated input artifacts')
    return result
