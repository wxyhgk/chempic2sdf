"""Truth-only scoring of four nested, separately reported candidate sets."""

from collections import Counter

OLD_KINDS = {'ATTRIBUTE', 'DELETE', 'RETYPE'}
ARMS = {'v1': OLD_KINDS, 'v1_element': OLD_KINDS | {'ELEMENT'},
        'v1_node_delete': OLD_KINDS | {'DELETE_NODE'}, 'union': OLD_KINDS | {'ELEMENT', 'DELETE_NODE'}}


def heavy_inventory(graph):
    return Counter((a['element'], a['isotope']) for a in graph['atoms'] if a['element'] != 'H')


def score(result, expected):
    rows = []
    baseline = result['baseline']
    for arm, allowed in ARMS.items():
        groups = [g for g in result['groups'] if any(a['kind'] in allowed for a in g['actions'])]
        correct = [g for g in groups if g['core'] == expected['core']]
        wrong = [g for g in groups if g['core'] != expected['core']]
        rows.append({'sample_key': result['sample_key'], 'arm': arm,
                     'baseline_valid': baseline['valid'],
                     'operational_core_exact': baseline['valid'] and baseline['core'] == expected['core'],
                     'covered_core': (baseline['valid'] and baseline['core'] == expected['core']) or bool(correct),
                     'new_correct_candidate': bool(correct), 'legal_groups': len(groups),
                     'has_legal_candidate': bool(groups), 'wrong_only': bool(wrong) and not correct,
                     'has_wrong_candidate': bool(wrong), 'multiple_groups': len(groups) > 1,
                     'single_wrong': len(groups) == 1 and bool(wrong),
                     'automatic_replacement': False})
    return rows


def summarize(rows, partitions):
    by = {(r['sample_key'], r['arm']): r for r in rows}
    keys = {p['sample_key'] for p in partitions}
    if len(keys) != 200 or len(partitions) != 200 or len(rows) != 800 or len(by) != 800:
        raise ValueError('Require 200 distinct samples in all four arms')
    if any((k, a) not in by for k in keys for a in ARMS):
        raise ValueError('Incomplete arm keys')
    fields = ('operational_core_exact', 'covered_core', 'new_correct_candidate', 'legal_groups',
              'has_legal_candidate', 'wrong_only', 'has_wrong_candidate', 'multiple_groups', 'single_wrong')
    report = {'denominator': 200, 'arms': {a: {f: sum(by[k, a][f] for k in keys) for f in fields} for a in ARMS},
              'transitions_from_v1': {a: {'gained_correct_coverage': sum(not by[k, 'v1']['covered_core'] and by[k, a]['covered_core'] for k in keys),
                                         'lost_correct_coverage': sum(by[k, 'v1']['covered_core'] and not by[k, a]['covered_core'] for k in keys)}
                                      for a in ARMS if a != 'v1'},
              'invalid_inventory_partition': dict(Counter(p['inventory_category'] for p in partitions if not p['baseline_valid'])),
              'automatic_replacements': 0, 'candidate_ranking_performed': False, 'training_performed': False,
              'promotion_authorized': False}
    return report
