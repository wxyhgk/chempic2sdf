"""Synthetic CPU chemistry only; no corpus, model, OCR, checkpoint or GPU."""

from copy import deepcopy

import pytest
from rdkit import Chem
import torch

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.components.atoms.chemistry import aromatic_transactions as tx
from chemvector.evaluation.audits.aromatic_saved_replay import reconstruct


def scene(edges, *, elements=None):
    count = max(max(a, b) for a, b, _ in edges) + 1
    atoms = [dict(id=i, anchor=[40 + i % 10 * 90, 40 + i // 10 * 90], role='IMPLICIT', text=None,
        chemical=dict(element=(elements or {}).get(i, 'C'), formal_charge=0, isotope=0, aromatic=False,
            explicit_h_count=0, no_implicit=False, radical_electrons=0, canonical_chiral_tag='CHI_UNSPECIFIED'),
        incident_ports=[]) for i in range(count)]
    bonds = []
    for i, (a, b, label) in enumerate(edges):
        ports = [list(atoms[a]['anchor']), list(atoms[b]['anchor'])]
        bonds.append(dict(id=i, atom_refs=[a, b], ports=ports, chemical_type=label,
                          display_type=label, direction='NONE', stereo='STEREONONE', bbox=None))
        for endpoint, atom in enumerate((a, b)):
            atoms[atom]['incident_ports'].append(dict(bond_id=i, endpoint=endpoint,
                coordinate=list(ports[endpoint]), source='ANCHOR'))
    return dict(schema_version='chemscene-0.2', molecule_id='synthetic', units='q1023',
                canvas=dict(width=768, height=768), atom_objects=atoms, bond_objects=bonds)


def ring(size=6, offset=0, label='SINGLE'):
    return [(offset + i, offset + (i + 1) % size, label) for i in range(size)]


def proposal(start=0, size=6): return dict(bond_ids=list(range(start, start + size)))


def test_valid_promotion_is_accepted_without_mutating_inputs_or_atom_attributes():
    raw = scene(ring()); original = deepcopy(raw)
    changed, record = tx.guarded_promotions(raw, [proposal()])
    assert raw == original and changed['atom_objects'] == original['atom_objects']
    assert record['transactions'][0]['action'] == 'accepted' and record['compiler_calls'] == 2
    assert all(b['chemical_type'] == 'AROMATIC' for b in changed['bond_objects'])
    assert Chem.MolToSmiles(mol_from_chemscene_v0_2(changed)) == 'c1ccccc1'
    assert record['chemistry_validity_is_not_identity']  # Not evidence that this was benzene in an image.


def test_invalid_nitrogen_ring_promotion_is_rolled_back_without_adding_h():
    raw = scene(ring(5), elements={4: 'N'})
    changed, record = tx.guarded_promotions(raw, [proposal(size=5)])
    assert changed == raw and record['local_rollback'] and not record['use_legacy']
    assert record['transactions'][0]['action'] == 'rolled_back'
    assert record['transactions'][0]['before_scene_sha256'] == record['transactions'][0]['after_scene_sha256']
    assert changed['atom_objects'][4]['chemical']['explicit_h_count'] == 0


def test_later_failed_transaction_preserves_accepted_ring_and_outside_bonds():
    raw = scene(ring() + ring(5, 6), elements={10: 'N'})
    changed, record = tx.guarded_promotions(raw, [proposal(), proposal(6, 5)])
    assert [r['action'] for r in record['transactions']] == ['accepted', 'rolled_back']
    assert all(b['chemical_type'] == 'AROMATIC' for b in changed['bond_objects'][:6])
    assert changed['bond_objects'][6:] == raw['bond_objects'][6:]
    assert changed['atom_objects'] == raw['atom_objects']


def test_ring_order_permutation_and_member_order_do_not_change_result():
    raw = scene(ring() + ring(5, 6), elements={10: 'N'})
    a = tx.guarded_promotions(raw, [proposal(), proposal(6, 5)])
    b = tx.guarded_promotions(raw, [dict(bond_ids=list(reversed(proposal(6, 5)['bond_ids']))), proposal()])
    assert a == b


def test_fused_rings_are_one_atomic_transaction():
    # Shared edge 4--5; two six-member rings.
    raw = scene(ring() + [(5, 6, 'SINGLE'), (6, 7, 'SINGLE'), (7, 8, 'SINGLE'),
                          (8, 9, 'SINGLE'), (9, 4, 'SINGLE')])
    groups = tx.transaction_groups(raw, [proposal(), dict(bond_ids=[4, 6, 7, 8, 9, 10])])
    assert len(groups) == 1 and len(groups[0]['bond_ids']) == 11
    _, record = tx.guarded_promotions(raw, [proposal(), dict(bond_ids=[4, 6, 7, 8, 9, 10])])
    assert record['compiler_calls'] == 2 and len(record['transactions']) == 1


def test_spiro_rings_sharing_only_atom_also_grouped():
    raw = scene(ring() + [(5, 6, 'SINGLE'), (6, 7, 'SINGLE'), (7, 8, 'SINGLE'),
                          (8, 9, 'SINGLE'), (9, 10, 'SINGLE'), (10, 5, 'SINGLE')])
    groups = tx.transaction_groups(raw, [proposal(), proposal(6)])
    assert len(groups) == 1 and len(groups[0]['rings']) == 2


def test_preexisting_invalid_does_not_search_and_requires_legacy_route():
    raw = scene(ring() + [(6, i, 'SINGLE') for i in range(7, 12)])  # Pentavalent C.
    changed, record = tx.guarded_promotions(raw, [proposal()])
    assert changed == raw and record['status'] == 'preexisting_invalid_use_legacy'
    assert record['compiler_calls'] == 1 and not record['transactions'] and record['use_legacy']


def test_compiler_mutation_on_failure_cannot_pollute_current_scene():
    raw = scene(ring()); original = deepcopy(raw); calls = []
    def compiler(private):
        calls.append(True)
        private['atom_objects'][0]['chemical']['element'] = 'P'
        if len(calls) == 2: raise ValueError('synthetic failed sanitize after mutation')
        return object()
    changed, record = tx.guarded_promotions(raw, [proposal()], compiler=compiler)
    assert changed == raw == original and record['local_rollback']


def test_noop_promotion_never_erases_existing_raw_aromatic_bonds():
    raw = scene(ring(label='AROMATIC'))
    changed, record = tx.guarded_promotions(raw, [proposal()])
    assert changed == raw and record['compiler_calls'] == 1
    assert record['transactions'][0]['action'] == 'noop'


def test_no_proposal_leaves_multi_fragment_scene_unchanged():
    raw = scene(ring() + [(6, 7, 'DOUBLE')])
    changed, record = tx.guarded_promotions(raw, [])
    assert changed == raw and record['transactions'] == []
    assert len(Chem.GetMolFrags(mol_from_chemscene_v0_2(changed))) == 2


@pytest.mark.parametrize('ids', [[0, 1, 2], [0, 1, 2, 3, 3], [0, 1, 2, 3, 99], [False, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
def test_malformed_or_incomplete_ring_is_rejected(ids):
    with pytest.raises(ValueError): tx.guarded_promotions(scene(ring()), [dict(bond_ids=ids)])


def test_duplicate_proposals_and_budgets_rejected(monkeypatch):
    with pytest.raises(ValueError): tx.guarded_promotions(scene(ring()), [proposal(), proposal()])
    monkeypatch.setattr(tx, 'MAX_TRANSACTIONS', 1)
    with pytest.raises(ValueError): tx.guarded_promotions(scene(ring() + ring(6, 6)), [proposal(), proposal(6)])
    monkeypatch.setattr(tx, 'MAX_PROPOSALS', 0)
    with pytest.raises(ValueError): tx.guarded_promotions(scene(ring()), [proposal()])


def saved_inputs():
    pairs = torch.tensor(sorted(tuple(sorted((a, b))) for a, b, _ in ring()), dtype=torch.int64)
    presence = torch.full((6,), 5.)
    # Raw SINGLE, but aromatic probability above .40 satisfies the fixed rule.
    chemical = torch.tensor([[2., -3., -3., 1.8]] * 6)
    promoted = scene([(a, b, 'AROMATIC') for a, b in pairs.tolist()])
    return promoted, list(range(6)), pairs, presence, chemical


def test_saved_reconstruction_exactly_reproduces_legacy_and_recovers_raw_types():
    args = saved_inputs(); before = deepcopy(args[0])
    raw, proposals, receipt = reconstruct(*args)
    assert args[0] == before and receipt['legacy_scene_exact'] and not receipt['use_legacy']
    assert all(b['chemical_type'] == b['display_type'] == 'SINGLE' for b in raw['bond_objects'])
    assert len(proposals) == 1
    guarded, _ = tx.guarded_promotions(raw, proposals)
    assert guarded == before


@pytest.mark.parametrize('change', ['scene_type', 'mapping', 'nan', 'dtype', 'pairs', 'grad', 'atom_aromatic'])
def test_reconstruction_fails_closed_on_drift_or_ambiguous_inputs(change):
    s, mapping, p, z, c = saved_inputs()
    if change == 'scene_type': s['bond_objects'][0]['chemical_type'] = 'SINGLE'
    elif change == 'mapping': mapping[1] = mapping[0]
    elif change == 'nan': z[0] = float('nan')
    elif change == 'dtype': c = c.double()
    elif change == 'pairs': p[0] = p[1]
    elif change == 'grad': c.requires_grad_()
    else: s['atom_objects'][0]['chemical']['aromatic'] = True
    with pytest.raises(ValueError): reconstruct(s, mapping, p, z, c)


def test_detached_cpu_pipeline_does_not_consume_rng_or_mutate_tensors():
    args = saved_inputs(); originals = [t.clone() for t in args[2:]]
    rng = torch.random.get_rng_state().clone()
    raw, proposals, _ = reconstruct(*args); tx.guarded_promotions(raw, proposals)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(originals, args[2:]))


def two_ring_case():
    from tools.audit_aromatic_transactions import ChemistryBudget, legacy_prediction
    edges = sorted(tuple(sorted((a, b))) for a, b, _ in ring() + ring(5, 6))
    s = scene([(a, b, 'AROMATIC') for a, b in edges], elements={10: 'N'})
    prediction, _ = legacy_prediction(s, ChemistryBudget())
    identity = Chem.MolToSmiles(Chem.MolFromMolBlock(prediction['molblock']))
    return dict(trace=dict(scene_before_fallback=s, compiled_candidate_indices=list(range(11))),
        pairs=[list(e) for e in edges], presence_logits=[5.] * len(edges),
        chemical_logits=[[2., -3., -3., 1.8]] * len(edges), prediction=prediction,
        metrics=dict(expected_smiles=identity, predicted_smiles=identity, sdf_valid=True,
                     sdf_exact=True, fallback=prediction['fallback']))


def test_saved_comparison_keeps_other_ring_and_reports_local_correction_not_hidden_fallback():
    from tools.audit_aromatic_transactions import ChemistryBudget, compare_case
    case = two_ring_case(); original = deepcopy(case)
    result = compare_case(case, ChemistryBudget())
    assert case == original and result['reconstruction']['legacy_scene_exact']
    assert result['legacy']['global_fallback_attempted']
    assert result['legacy']['global_demoted_bonds'] == 11
    assert not result['guarded']['global_fallback_attempted'] and result['journal']['local_rollback']
    assert result['any_correction_warning']  # Do not hide intervention by redefining fallback.
    assert len(result['outside_rejected_aromatic_bond_ids_preserved']) == 6
    # This toy expected label deliberately equals the old output. Valid guard
    # output can change identity, which must be visible instead of called success.
    assert result['legacy']['sdf_exact'] and not result['guarded']['sdf_exact']


def test_scoring_target_cannot_change_decoder_prediction():
    from tools.audit_aromatic_transactions import ChemistryBudget, compare_case
    case = two_ring_case(); a = compare_case(case, ChemistryBudget())
    case['metrics'].update(expected_smiles='C', sdf_exact=False)
    b = compare_case(case, ChemistryBudget())
    assert a['prediction'] == b['prediction'] and a['guarded_scene'] == b['guarded_scene']
    assert a['journal'] == b['journal']


def test_saved_prediction_drift_stops_before_comparing_alternate_decoder():
    from tools.audit_aromatic_transactions import ChemistryBudget, compare_case
    case = two_ring_case(); case['prediction']['fallback'] = False
    with pytest.raises(ValueError, match='exactly reproduced'): compare_case(case, ChemistryBudget())


def test_compiler_budget_exhaustion_is_not_a_chemistry_rollback():
    from tools.audit_aromatic_transactions import ChemistryBudget
    with pytest.raises(TimeoutError):
        tx.guarded_promotions(scene(ring()), [proposal()], compiler=ChemistryBudget(maximum=1).compile)


def test_summary_preserves_denominators_and_detects_identity_damage():
    from tools.audit_aromatic_transactions import summarize
    records = [dict(arm=arm, key=str(i), group='witness' if i < 26 else 'old_full_molecule',
        legacy=dict(sdf_exact=True, sdf_valid=True, fallback=False, global_fallback_attempted=False),
        guarded=dict(sdf_exact=True, sdf_valid=True, fallback=False, global_fallback_attempted=False),
        journal=dict(local_rollback=False, use_legacy=False), any_correction_warning=False)
        for arm in ('initial', 'control50', 'replay50') for i in range(58)]
    records[-1]['guarded']['sdf_exact'] = False
    result = summarize(records)
    assert result['replay50']['all']['protection_failures'] == ['57']
    assert result['initial']['all']['images'] == 58
    with pytest.raises(ValueError): summarize(records[:-1])
