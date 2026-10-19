"""Fixed validation selection and paired summaries; no inference or fitting."""

from collections import Counter, defaultdict

ARMS = ('initial', 'presence_delta')
PARITY_FIELDS = ('predicted_smiles', 'expected_smiles', 'valid', 'exact', 'fallback')
VARIANTS = {'original', 'rdkit', 'coordgen', 'stress'}


def sample_key(row):
    return row['source'] + ':' + row['sample_key']


def fixed_validation_rows(manifest, saved):
    rows = [r for r in manifest['samples'] if r['split'] == 'validation']
    keys = [sample_key(r) for r in rows]
    if (len(set(keys)) != len(rows)
            or Counter(r['domain'] for r in rows) != {'full_molecule': 800, 'curriculum': 144}):
        raise ValueError('Keep all original 800 full-molecule and 144 curriculum validation images')
    groups = defaultdict(list)
    for row in rows:
        if row['domain'] == 'full_molecule':
            if row['source'] != 'fresh_dev':
                raise ValueError('Only the existing original fresh-dev source is allowed')
            groups[row['molecule_id']].append(row)
    if len(groups) != 200 or any(len(v) != 4 or {r['variant'] for r in v} != VARIANTS for v in groups.values()):
        raise ValueError('Expected original 200 molecules with all four variants')
    old = {r['key']: r for r in saved}
    if len(old) != len(saved) or set(old) != set(keys):
        raise ValueError('Saved baseline must contain exactly the same validation identities')
    for row in rows:
        previous = old[sample_key(row)]
        if previous['domain'] != row['domain'] or previous['expected_smiles'] != row['canonical_smiles']:
            raise ValueError('Saved validation truth/domain differs from the fixed manifest')
    return rows, old


def parity_differences(prediction, previous):
    return {k: {'saved': previous[k], 'cpu': prediction[k]}
            for k in PARITY_FIELDS if previous[k] != prediction[k]}


def paired_summary(cases):
    if not cases or len({r['key'] for r in cases}) != len(cases):
        raise ValueError('Nonempty unique paired cases required')
    counts = {arm: {k: sum(c['arms'][arm][k] for c in cases) for k in ('exact', 'valid', 'fallback')}
              for arm in ARMS}
    for value in counts.values():
        value['sdf_exact_rate'] = value['exact'] / len(cases)
    rescued = sum(not c['arms']['initial']['exact'] and c['arms']['presence_delta']['exact'] for c in cases)
    broken = sum(c['arms']['initial']['exact'] and not c['arms']['presence_delta']['exact'] for c in cases)
    return {'images': len(cases), 'arms': counts, 'rescued': rescued, 'broken': broken,
        'net_exact_change': rescued - broken,
        'valid_to_invalid': sum(c['arms']['initial']['valid'] and not c['arms']['presence_delta']['valid'] for c in cases),
        'invalid_to_valid': sum(not c['arms']['initial']['valid'] and c['arms']['presence_delta']['valid'] for c in cases),
        'historical_baseline_parity_mismatches': sum(bool(c['historical_differences']) for c in cases)}


def summarize(cases):
    by_domain = {d: [r for r in cases if r['domain'] == d] for d in ('full_molecule', 'curriculum')}
    groups = defaultdict(list)
    for row in by_domain['full_molecule']:
        groups[row['molecule_id']].append(row)
    return {'domains': {d: paired_summary(v) for d, v in by_domain.items()},
        'full_molecule_styles': {s: paired_summary([r for r in by_domain['full_molecule'] if r['style'] == s])
                                for s in sorted(VARIANTS)},
        'curriculum_styles': {s: paired_summary([r for r in by_domain['curriculum'] if r['style'] == s])
                              for s in sorted({r['style'] for r in by_domain['curriculum']})},
        'full_molecule_groups': {'molecules': len(groups), 'variants_per_molecule': 4,
            'all_four_exact': {a: sum(all(r['arms'][a]['exact'] for r in v) for v in groups.values()) for a in ARMS},
            'any_style_rescued': sum(any(not r['arms']['initial']['exact'] and r['arms']['presence_delta']['exact']
                                         for r in v) for v in groups.values()),
            'any_style_broken': sum(any(r['arms']['initial']['exact'] and not r['arms']['presence_delta']['exact']
                                        for r in v) for v in groups.values())}}
