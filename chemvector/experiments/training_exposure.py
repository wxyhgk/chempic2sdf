"""Metadata-only, conservative exposure components; never grant admission.

Known histories are LOWER BOUNDS. Missing events do not prove a component was
unobserved. No input here can certify a new independent challenge cohort.
"""

from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256

from chemvector.training.curriculum.admission import identities


def sample_key(row):
    return row['source'] + ':' + row['sample_key']


def build_ledger(train_rows, design_rows, optimizer_events):
    """Union molecule, declared/computed core group, and connectivity prefix.

    design_rows carry ``evidence_id``. optimizer_events map admitted train keys
    to historical run IDs, independently from design exposure. All external
    diagnostic rows only exclude components; they never enter the train pool.
    """
    if not train_rows or any(r.get('split') != 'train' for r in train_rows):
        raise ValueError('Published training rows only')
    keys = [sample_key(r) for r in train_rows]
    if len(set(keys)) != len(keys) or set(optimizer_events) - set(keys):
        raise ValueError('Duplicate training key or optimizer event outside the published train pool')
    if any(not r.get('evidence_id') for r in design_rows):
        raise ValueError('Every diagnostic exclusion needs explicit evidence')
    identity = lru_cache(None)(identities)
    all_rows = list(train_rows) + list(design_rows)
    parents = list(range(len(all_rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parents[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(all_rows):
        ids = identity(row['canonical_smiles'])
        tags = [('molecule', row['molecule_id']), ('group', ids['group_key']),
                ('connectivity', ids['connectivity_prefix'])]
        if row.get('group_key'): tags.append(('group', row['group_key']))
        for tag in tags:
            if tag in seen: union(i, seen[tag])
            else: seen[tag] = i
    members, evidence = defaultdict(list), defaultdict(set)
    for i, row in enumerate(all_rows):
        component = find(i)
        if i < len(train_rows): members[component].append(row)
        else: evidence[component].add(row['evidence_id'])
    components = []
    for component, rows in members.items():
        member_keys = sorted(sample_key(r) for r in rows)
        optimizer_keys = [k for k in member_keys if k in optimizer_events]
        design = sorted(evidence[component])
        status = ('known_design_exposure' if design else
                  'optimizer_history_only_design_unknown' if optimizer_keys else 'history_unknown')
        components.append(dict(component_id=sha256('\n'.join(member_keys).encode()).hexdigest(),
            training_rows=len(rows), keys=member_keys,
            molecule_ids=sorted({r['molecule_id'] for r in rows}),
            group_keys=sorted({r['group_key'] for r in rows}),
            design_evidence=design, optimizer_updated_keys=optimizer_keys,
            optimizer_runs=sorted({run for k in optimizer_keys for run in optimizer_events[k]}),
            reference_style_rows=sum(r['sample_key'].endswith('__reference') for r in rows),
            exposure_status=status, certified_design_unexposed=False,
            model_unseen_established=False, challenge_eligible=False))
    components.sort(key=lambda c: c['component_id'])
    return dict(schema_version='training-exposure-ledger-0.1', components=components,
        counts=dict(training_rows=len(train_rows), components=len(components),
            by_status_components=dict(Counter(c['exposure_status'] for c in components)),
            by_status_rows={status: sum(c['training_rows'] for c in components if c['exposure_status'] == status)
                            for status in sorted({c['exposure_status'] for c in components})},
            certified_design_unexposed_components=0, selected_challenge_images=0),
        exposure_history_complete=False, independent_challenge_ready=False,
        grouping='transitive union: molecule ID OR declared/core group OR connectivity prefix',
        interpretation='Training-side design exposure audit, not a model-unseen generalization test',
        limitations=['Absence from audited events is unknown, never proof of no prior observation',
                     'Optimizer exposure and human/rule-design exposure are different histories',
                     'Reference style is metadata, not visually verified clean input',
                     'Historical train/dev separation inherited; no new evaluation-identity query'])
