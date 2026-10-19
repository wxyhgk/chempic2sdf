"""Synthetic metadata only; no image/evaluation identity access."""

from copy import deepcopy

import pytest

from chemvector.experiments import training_exposure as exposure


def row(name, *, group=None, molecule=None, smiles=None, style='reference'):
    return dict(source='train', sample_key=name + '__' + style,
        molecule_id=molecule or name, group_key=group or name,
        canonical_smiles=smiles or name, split='train')


def fake_ids(monkeypatch, mapping=None):
    mapping = mapping or {}
    monkeypatch.setattr(exposure, 'identities', lambda s: mapping.get(s,
        dict(group_key='core:' + s, connectivity_prefix='conn:' + s)))


def test_transitive_union_propagates_exposure_across_all_identity_channels(monkeypatch):
    fake_ids(monkeypatch, {'c': dict(group_key='core:c', connectivity_prefix='shared'),
                          'd': dict(group_key='core:d', connectivity_prefix='shared')})
    rows = [row('a', molecule='m'), row('b', molecule='m', group='g'),
            row('c', group='g'), row('d'), row('unrelated')]
    design = [dict(row('a', molecule='m'), evidence_id='fixed58')]
    saved = deepcopy(rows)
    ledger = exposure.build_ledger(rows, design, {})
    exposed = [c for c in ledger['components'] if c['design_evidence']]
    assert len(exposed) == 1 and exposed[0]['training_rows'] == 4
    assert exposed[0]['exposure_status'] == 'known_design_exposure'
    assert ledger['counts']['by_status_rows']['history_unknown'] == 1
    assert rows == saved


def test_computed_core_excludes_stereo_or_isotope_variant_with_different_declared_group(monkeypatch):
    fake_ids(monkeypatch, {s: dict(group_key='shared-core', connectivity_prefix=s) for s in ('a', 'b')})
    ledger = exposure.build_ledger([row('a')], [dict(row('b'), evidence_id='diagnostic')], {})
    assert ledger['components'][0]['design_evidence'] == ['diagnostic']


def test_optimizer_seen_is_not_automatically_rule_seen_and_unknown_never_certified(monkeypatch):
    fake_ids(monkeypatch)
    rows = [row('a'), row('b')]
    ledger = exposure.build_ledger(rows, [], {'train:a__reference': ['old_run']})
    assert ledger['counts']['by_status_rows'] == {'history_unknown': 1, 'optimizer_history_only_design_unknown': 1}
    assert ledger['counts']['selected_challenge_images'] == 0
    assert all(not c['challenge_eligible'] and not c['certified_design_unexposed'] for c in ledger['components'])


def test_order_invariant_component_ids_and_style_variants_stay_together(monkeypatch):
    fake_ids(monkeypatch)
    rows = [row('a'), row('a', style='serif'), row('b')]
    a = exposure.build_ledger(rows, [], {})
    b = exposure.build_ledger(list(reversed(rows)), [], {})
    assert a == b
    assert sorted(c['training_rows'] for c in a['components']) == [1, 2]


@pytest.mark.parametrize('bad', ['dev', 'duplicate', 'unmapped_update', 'missing_evidence', 'empty'])
def test_fail_closed_metadata_errors(monkeypatch, bad):
    fake_ids(monkeypatch)
    rows, design, updates = [row('a')], [], {}
    if bad == 'dev': rows[0]['split'] = 'validation'
    elif bad == 'duplicate': rows += deepcopy(rows)
    elif bad == 'unmapped_update': updates = {'unmapped': ['run']}
    elif bad == 'missing_evidence': design = [row('b')]
    else: rows = []
    with pytest.raises(ValueError): exposure.build_ledger(rows, design, updates)
