"""Truth is used here only, after predictions are saved. No repair or fallback."""

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from chemvector.experiments.id_graph.graph import to_sdf
from chemvector.experiments.id_graph.roundtrip import signature


def geometry_matches(graph, truth, radius=6.):
    positions = lambda g: {p['atom_id']: p['xy'] for p in g['observations']['atom_positions']}
    pred, target = positions(graph), positions(truth)
    pids, tids = sorted(pred), sorted(target)
    if not pids or not tids:
        return {}, {}, {'predicted_ambiguous': 0, 'truth_ambiguous': 0, 'eligible_pairs': 0}
    distances = np.linalg.norm(np.asarray([pred[i] for i in pids])[:, None] - np.asarray([target[i] for i in tids])[None], axis=-1)
    allowed = distances <= radius
    # Maximum cardinality first, then minimum distance. Labels never break ties.
    penalty = (min(distances.shape) + 1) * (radius + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, distances, penalty))
    mapping = {pids[r]: tids[c] for r, c in zip(rows, cols) if allowed[r, c]}
    safe = {pids[r]: tids[c] for r, c in zip(rows, cols)
            if allowed[r, c] and allowed[r].sum() == 1 and allowed[:, c].sum() == 1}
    return mapping, safe, {'predicted_ambiguous': int((allowed.sum(1) > 1).sum()),
                          'truth_ambiguous': int((allowed.sum(0) > 1).sum()),
                          'eligible_pairs': int(allowed.sum())}


def component_errors(graph, truth, radius=6.):
    mapping, safe, ambiguity = geometry_matches(graph, truth, radius)
    atoms = {a['id']: a for a in graph['atoms']}
    targets = {a['id']: a for a in truth['atoms']}
    fields = ('element', 'formal_charge', 'isotope', 'radical_electrons')
    disagreements = {name: sum(atoms[i][name] != targets[j][name] for i, j in safe.items()) for name in fields}
    # H/aromatic flags can have equivalent encodings; do not call these chemical errors.
    representation = {name: sum(atoms[i][name] != targets[j][name] for i, j in safe.items())
                      for name in ('explicit_h_count', 'no_implicit', 'aromatic')}
    predicted_bonds = {frozenset(safe[i] for i in b['atom_ids']): b['type'] for b in graph['bonds']
                       if all(i in safe for i in b['atom_ids'])}
    safe_targets = set(safe.values())
    truth_bonds = {frozenset(b['atom_ids']): b['type'] for b in truth['bonds']
                  if set(b['atom_ids']).issubset(safe_targets)}
    common = predicted_bonds.keys() & truth_bonds.keys()
    return {'predicted_atoms': len(atoms), 'truth_atoms': len(targets), 'matched_atoms': len(mapping),
            'unmatched_predicted_atoms': len(atoms) - len(mapping),
            'unmatched_truth_atoms': len(targets) - len(mapping),
            'unambiguous_matched_atoms': len(safe), **ambiguity,
            'attribute_errors': disagreements, 'representation_differences': representation,
            'atoms_with_attribute_error': sum(any(atoms[i][f] != targets[j][f] for f in fields) for i, j in safe.items()),
            'eligible_predicted_bonds': len(predicted_bonds), 'eligible_truth_bonds': len(truth_bonds),
            'excluded_predicted_bonds': len(graph['bonds']) - len(predicted_bonds),
            'excluded_truth_bonds': len(truth['bonds']) - len(truth_bonds),
            'missing_bonds': len(truth_bonds.keys() - predicted_bonds.keys()),
            'extra_bonds': len(predicted_bonds.keys() - truth_bonds.keys()),
            'wrong_bond_types': sum(predicted_bonds[p] != truth_bonds[p] for p in common),
            'correct_bonds': sum(predicted_bonds[p] == truth_bonds[p] for p in common)}


def compile_score(graph, expected):
    row = {'sdf_valid': False, 'core_graph_exact': False, 'stereo_graph_exact': False,
           'compile_error': None, 'carbon_fallback': False}
    try:
        sdf = to_sdf(graph)
        mol = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=True)
        row.update(sdf_valid=mol is not None, predicted_core=signature(mol, False), predicted_stereo=signature(mol))
        row['core_graph_exact'] = row['predicted_core'] == expected['core']
        row['stereo_graph_exact'] = row['predicted_stereo'] == expected['stereo']
        return row, sdf
    except Exception as error:
        row['compile_error'] = {'type': type(error).__name__, 'message': str(error)}
        return row, None
