"""Scoring is separate from candidate generation; truth never selects an edit."""

from collections import Counter

from chemvector.experiments.id_graph.graph import to_molecule
from chemvector.experiments.id_graph_prediction.scoring import geometry_matches


def normalized_edges(graph):
    mol = to_molecule(graph)
    ids = sorted(a['id'] for a in graph['atoms'])
    return {frozenset((ids[b.GetBeginAtomIdx()], ids[b.GetEndAtomIdx()])): str(b.GetBondType()) for b in mol.GetBonds()}


def edge_agreement(graph, truth):
    _, safe, _ = geometry_matches(graph, truth)
    predicted = {frozenset(safe[i] for i in b['atom_ids']): b for b in graph['bonds'] if all(i in safe for i in b['atom_ids'])}
    expected = {frozenset(b['atom_ids']): b for b in truth['bonds'] if all(i in safe.values() for i in b['atom_ids'])}
    common = predicted.keys() & expected.keys()
    result = Counter(common_edges=len(common), missing_edges=len(expected.keys() - predicted.keys()),
                     extra_edges=len(predicted.keys() - expected.keys()),
                     excluded_predicted_edges=len(graph['bonds']) - len(predicted),
                     excluded_truth_edges=len(truth['bonds']) - len(expected))
    try:
        pn, tn = normalized_edges(graph), normalized_edges(truth)
    except Exception:
        pn = tn = None
    for pair in common:
        raw_equal = predicted[pair]['type'] == expected[pair]['type']
        raw = 'raw_equal' if raw_equal else 'raw_different'
        if pn is None:
            result[f'{raw}_normalized_unresolved'] += 1
        else:
            normalized_equal = pn[frozenset(predicted[pair]['atom_ids'])] == tn[pair]
            result[f"{raw}_normalized_{'equal' if normalized_equal else 'different'}"] += 1
    return dict(result)


def score_candidates(result, expected):
    baseline = result['baseline']
    baseline_core = baseline['valid'] and baseline['core'] == expected['core']
    baseline_stereo = baseline['valid'] and baseline['stereo'] == expected['stereo']
    correct = [g for g in result['groups'] if g['core'] == expected['core']]
    wrong = [g for g in result['groups'] if g['core'] != expected['core']]
    return {'sample_key': result['sample_key'], 'baseline_core_exact': baseline_core,
            'baseline_stereo_exact': baseline_stereo, 'baseline_valid': baseline['valid'],
            'operational_core_exact': baseline_core, 'operational_stereo_exact': baseline_stereo,
            'automatic_replacements': 0, 'selected_candidate': None,
            'disposition': result['disposition'], 'attempts': len(result['attempts']),
            'legal_candidate_groups': len(result['groups']), 'candidate_contains_correct_core': bool(correct),
            'baseline_or_candidate_core_coverage': baseline_core or bool(correct),
            'candidate_contains_wrong_core': bool(wrong),
            'candidate_only_wrong_core': bool(wrong) and not correct,
            'multiple_groups': len(result['groups']) > 1,
            'single_group_wrong_core': len(result['groups']) == 1 and bool(wrong),
            'attribute_truncated': result['attribute_truncated'], 'missing_beam_records': result['missing_beam_records'],
            'correct_coverage_by_edit': {kind: any(any(a['kind'] == kind for a in g['actions']) for g in correct)
                                         for kind in ('ATTRIBUTE', 'DELETE', 'RETYPE')}}


def summarize(rows):
    if len(rows) != 200 or len({r['sample_key'] for r in rows}) != 200:
        raise ValueError('Require full 200-image candidate denominator')
    fields = ('baseline_core_exact', 'baseline_stereo_exact', 'baseline_valid', 'operational_core_exact',
              'operational_stereo_exact', 'automatic_replacements', 'candidate_contains_correct_core',
              'baseline_or_candidate_core_coverage', 'candidate_contains_wrong_core', 'candidate_only_wrong_core',
              'multiple_groups', 'single_group_wrong_core', 'attempts', 'legal_candidate_groups',
              'attribute_truncated', 'missing_beam_records')
    return {'denominator': 200, 'counts': {f: sum(r[f] for r in rows) for f in fields},
            'dispositions': dict(Counter(r['disposition'] for r in rows)),
            'correct_coverage_by_edit_nonexclusive': {kind: sum(r['correct_coverage_by_edit'][kind] for r in rows)
                                                     for kind in ('ATTRIBUTE', 'DELETE', 'RETYPE')},
            'training_performed': False, 'promotion_authorized': False,
            'candidate_ranking_performed': False, 'candidate_coverage_is_not_accuracy': True}
