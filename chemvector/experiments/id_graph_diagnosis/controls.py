"""Replace known attributes/induced edges on mutually unique geometric matches."""

from copy import deepcopy

from chemvector.experiments.id_graph.graph import ATOM_FIELDS, to_molecule
from chemvector.experiments.id_graph_prediction.scoring import geometry_matches

FIELDS = {
    'none': (),
    'identity': ('element', 'formal_charge', 'isotope', 'radical_electrons'),
    'hydrogen': ('explicit_h_count', 'no_implicit'),
    'aromatic': ('aromatic',),
    'full': ATOM_FIELDS,
}


def intervene(graph, truth, mode, replace_bonds=False):
    if mode not in FIELDS:
        raise ValueError('Unknown attribute intervention')
    if any(a['stereo']['tag'] != 'CHI_UNSPECIFIED' for a in graph['atoms']) or any(
            b['stereo'] != 'STEREONONE' for b in graph['bonds']):
        raise ValueError('This diagnostic requires the frozen stereo-unspecified prediction baseline')
    result = deepcopy(graph)
    _, safe, ambiguity = geometry_matches(graph, truth)
    targets = {a['id']: a for a in truth['atoms']}
    for atom in result['atoms']:
        if atom['id'] in safe:
            target = targets[safe[atom['id']]]
            for field in FIELDS[mode]:
                atom[field] = deepcopy(target[field])
    inverse = {j: i for i, j in safe.items()}
    eligible = [b for b in truth['bonds'] if all(i in inverse for i in b['atom_ids'])]
    retained_boundary = [b for b in graph['bonds'] if not all(i in safe for i in b['atom_ids'])]
    if replace_bonds:
        # Replace only the induced matched subgraph; never delete edges incident
        # to unmatchable/ambiguous nodes or silently add a missing atom.
        result['bonds'] = deepcopy(retained_boundary)
        next_id = max((b['id'] for b in graph['bonds']), default=-1) + 1
        for offset, bond in enumerate(eligible):
            item = deepcopy(bond)
            item.update(id=next_id + offset, atom_ids=[inverse[i] for i in bond['atom_ids']],
                        stereo='STEREONONE', stereo_atom_ids=[])
            result['bonds'].append(item)
        neighbors = {a['id']: [] for a in result['atoms']}
        for bond in sorted(result['bonds'], key=lambda b: b['id']):
            first, second = bond['atom_ids']
            neighbors[first].append(second)
            neighbors[second].append(first)
        for atom in result['atoms']:
            atom['stereo']['neighbor_ids'] = neighbors[atom['id']]
    if [a['id'] for a in result['atoms']] != [a['id'] for a in graph['atoms']] or result['observations'] != graph['observations']:
        raise ValueError('Intervention changed identities or observations')
    return result, {'safe_matched_atoms': len(safe), **ambiguity,
                    'complete_safe_bijection': len(safe) == len(graph['atoms']) == len(truth['atoms']),
                    'eligible_truth_bonds': len(eligible), 'unavailable_truth_bonds': len(truth['bonds']) - len(eligible),
                    'preserved_boundary_bonds': len(retained_boundary)}


def normalized_bonds(graph):
    mol = to_molecule(graph)
    ids = [a['id'] for a in sorted(graph['atoms'], key=lambda a: a['id'])]
    return {frozenset((ids[b.GetBeginAtomIdx()], ids[b.GetEndAtomIdx()])): str(b.GetBondType()) for b in mol.GetBonds()}


def representation_check(graph, truth):
    _, safe, _ = geometry_matches(graph, truth)
    raw_truth = {frozenset(b['atom_ids']): b['type'] for b in truth['bonds']}
    raw = []
    for bond in graph['bonds']:
        if not all(i in safe for i in bond['atom_ids']):
            continue
        pair = frozenset(safe[i] for i in bond['atom_ids'])
        if pair in raw_truth and bond['type'] != raw_truth[pair]:
            raw.append({'prediction_atom_ids': bond['atom_ids'], 'truth_atom_ids': sorted(pair),
                        'predicted_type': bond['type'], 'truth_type': raw_truth[pair]})
    if not raw:
        return {'raw_differences': [], 'resolved': 0, 'persistent': 0, 'unresolved': 0}
    try:
        predicted_normal = normalized_bonds(graph)
        truth_normal = normalized_bonds(truth)
    except Exception as error:
        return {'raw_differences': raw, 'resolved': 0, 'persistent': 0, 'unresolved': len(raw),
                'normalization_error': str(error)}
    for row in raw:
        row['normalized_prediction'] = predicted_normal[frozenset(row['prediction_atom_ids'])]
        row['normalized_truth'] = truth_normal[frozenset(row['truth_atom_ids'])]
        row['equivalent_after_normalization'] = row['normalized_prediction'] == row['normalized_truth']
    resolved = sum(r['equivalent_after_normalization'] for r in raw)
    return {'raw_differences': raw, 'resolved': resolved, 'persistent': len(raw) - resolved, 'unresolved': 0}
