"""Synthetic fixed-selection/delta/paired-inference guards; no real model load."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from rdkit import Chem
import torch

from chemvector.training.curriculum import presence_regression as audit
from chemvector.training.curriculum import presence_regression_metrics as metrics
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.presence_probe import frozen_state_digest


def selection():
    full = [dict(source='fresh_dev', sample_key=f'fresh_dev_{i:06d}_{s}', molecule_id=f'mol{i}',
                 split='validation', domain='full_molecule', variant=s, canonical_smiles='C')
            for i in range(200) for s in sorted(metrics.VARIANTS)]
    curriculum = [dict(source='curriculum', sample_key=f'curr{i}', molecule_id=f'c{i}',
                       split='validation', domain='curriculum', canonical_smiles='C') for i in range(144)]
    rows = curriculum + full
    previous = [dict(key=metrics.sample_key(r), domain=r['domain'], expected_smiles='C') for r in rows]
    return {'samples': rows + [dict(split='train', source='never_load', sample_key='x')]}, previous


def test_selection_retains_entire_original_validation_in_order():
    manifest, saved = selection()
    rows, previous = metrics.fixed_validation_rows(manifest, saved)
    assert rows == manifest['samples'][:-1] and len(previous) == 944


@pytest.mark.parametrize('change', ['drop', 'duplicate', 'train', 'source', 'variant', 'molecule',
                                   'saved_drop', 'saved_duplicate', 'truth', 'domain'])
def test_selection_rejects_changed_denominators_or_historical_baseline(change):
    manifest, saved = selection()
    if change == 'drop': manifest['samples'].pop(0)
    elif change == 'duplicate': manifest['samples'][0] = manifest['samples'][1]
    elif change == 'train': manifest['samples'][0]['split'] = 'train'
    elif change == 'source': manifest['samples'][144]['source'] = 'sealed_test'
    elif change == 'variant': manifest['samples'][144]['variant'] = 'other'
    elif change == 'molecule': manifest['samples'][144]['molecule_id'] = 'replacement'
    elif change == 'saved_drop': saved.pop()
    elif change == 'saved_duplicate': saved[0] = saved[1]
    elif change == 'truth': saved[0]['expected_smiles'] = 'N'
    elif change == 'domain': saved[0]['domain'] = 'full_molecule'
    with pytest.raises(ValueError):
        metrics.fixed_validation_rows(manifest, saved)


def synthetic_delta():
    torch.manual_seed(11)
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(2, 3)
    model.pair = torch.nn.Module()
    model.pair.presence = torch.nn.Linear(3, 1)
    model.pair.chemical = torch.nn.Linear(3, 4)
    model.eval().requires_grad_(False)
    modified = deepcopy(model)
    with torch.no_grad():
        modified.pair.presence.weight.add_(.25)
        modified.pair.presence.bias.sub_(.1)
    delta = dict(schema_version='chemvector-presence-readout-delta-0.1', base_initial_sha256=audit.INITIAL_SHA,
                 optimizer_steps=200, not_standalone_model=True, deployed=False, resume_supported=False,
                 weight=modified.pair.presence.weight.tolist(), bias=modified.pair.presence.bias.tolist())
    trial = dict(model_state_before=state_digest(model), model_state_after=state_digest(modified),
                 frozen_state_before=frozen_state_digest(model), frozen_state_after=frozen_state_digest(modified))
    return model, delta, trial


def test_delta_is_restored_in_independent_frozen_copy_and_only_readout_differs():
    model, delta, trial = synthetic_delta()
    restored = audit.patched_model(model, delta, trial)
    assert state_digest(model) == trial['model_state_before']
    assert state_digest(restored) == trial['model_state_after']
    assert {k for k,v in model.state_dict().items() if not torch.equal(v,restored.state_dict()[k])} == {
        'pair.presence.weight', 'pair.presence.bias'}
    assert all(not p.requires_grad and p.grad is None for p in restored.parameters())
    assert all(not m.training for m in restored.modules())
    assert all(a.data_ptr() != b.data_ptr() for a,b in zip(model.parameters(),restored.parameters()))


@pytest.mark.parametrize('change', ['shape','nan','base','steps','deployed','resume','standalone','initial_state','final_state','frozen'])
def test_bad_delta_never_mutates_initial(change):
    model, delta, trial = synthetic_delta()
    before = state_digest(model)
    if change == 'shape': delta['weight'] = [[1.]]
    elif change == 'nan': delta['weight'][0][0] = float('nan')
    elif change == 'base': delta['base_initial_sha256'] = '0'*64
    elif change == 'steps': delta['optimizer_steps'] = 201
    elif change == 'deployed': delta['deployed'] = True
    elif change == 'resume': delta['resume_supported'] = True
    elif change == 'standalone': delta['not_standalone_model'] = False
    elif change == 'initial_state': trial['model_state_before'] = 'wrong'
    elif change == 'final_state': trial['model_state_after'] = 'wrong'
    elif change == 'frozen': trial['frozen_state_after'] = 'wrong'
    with pytest.raises(ValueError): audit.patched_model(model, delta, trial)
    assert state_digest(model) == before


def prediction(exact=False, valid=True):
    return dict(exact=exact,valid=valid,fallback=False,predicted_smiles='C' if valid else None,expected_smiles='C')


def test_paired_counts_keep_invalid_and_separate_validity_from_strict_regression():
    pairs = [(prediction(),prediction(True)), (prediction(True),prediction()),
             (prediction(),prediction(valid=False)), (prediction(valid=False),prediction())]
    cases = [dict(key=str(i),arms=dict(zip(metrics.ARMS,p)),historical_differences={}) for i,p in enumerate(pairs)]
    cases[0]['historical_differences'] = {'exact': {'saved': True, 'cpu': False}}
    counts = metrics.paired_summary(cases)
    assert counts['images']==4 and counts['rescued']==counts['broken']==1 and counts['net_exact_change']==0
    assert counts['valid_to_invalid']==counts['invalid_to_valid']==1
    assert counts['historical_baseline_parity_mismatches']==1
    assert counts['arms']['initial']['sdf_exact_rate']==.25
    with pytest.raises(ValueError): metrics.paired_summary(cases+cases[:1])


def test_cpu_historical_mismatch_is_reported_not_silently_replaced():
    old=prediction(True); fresh=prediction()
    assert metrics.parity_differences(fresh,old)=={'exact':{'saved':True,'cpu':False}}
    assert fresh['exact'] is False and old['exact'] is True


def test_full_molecule_and_curriculum_scores_are_never_pooled():
    manifest,_=selection()
    cases=[dict(key=metrics.sample_key(r),domain=r['domain'],molecule_id=r['molecule_id'],
                style=r.get('variant','reference'),historical_differences={},
                arms={a:prediction(r['domain']=='curriculum') for a in metrics.ARMS})
           for r in manifest['samples'] if r['split']=='validation']
    result=metrics.summarize(cases)
    assert result['domains']['full_molecule']['arms']['initial']['exact']==0
    assert result['domains']['curriculum']['arms']['initial']['exact']==144
    assert result['full_molecule_groups']['molecules']==200
    assert all(v['images']==200 for v in result['full_molecule_styles'].values())


def test_cached_pair_path_shares_upstream_but_never_accepts_truth(monkeypatch):
    import chemvector.training.curriculum.continuation as continuation
    calls=[]
    atom={'x':torch.ones(1)}; hidden=torch.ones(1,3); block=Chem.MolToMolBlock(Chem.MolFromSmiles('C'))
    def forward(*args):
        assert not torch.is_grad_enabled()
        calls.append('upstream')
        return (hidden,None,None,None,None,None,atom)
    monkeypatch.setattr(continuation,'structure_forward',forward)
    class Predictor:
        provenance={'atom_threshold':-.25,'pair_threshold':.85}
        def from_predicted_atoms(self,pair,features,atoms):
            assert features is hidden and atoms is atom
            return {'atoms':atoms,'pairs':torch.tensor([[0,1]]),'output':{
                'hidden':hidden,'chemical_logits':torch.ones(1,4),'presence_logits':torch.tensor([pair])}}
        def from_structure(self,structure,role,bbox,pixels,image,*,trace):
            trace.update(roles=[],ocr=[])
            return {'molblock':block,'fallback':False,'error':None}
        def __call__(self,*args):
            calls.append('direct_replay')
            return {'molblock':block,'fallback':False,'error':None}
    models={a:SimpleNamespace(pair=p,role=None,bbox=None) for a,p in zip(metrics.ARMS,[0.,5.])}
    preds, traces, signature, changed=audit.paired_predictions(models,None,Predictor(),None,None,replay=True)
    assert changed==1 and len(signature)==64 and set(preds)==set(metrics.ARMS)
    assert calls==['upstream','direct_replay','direct_replay']


def test_check_only_never_runs_and_real_cli_is_bounded_cpu_without_training(monkeypatch):
    calls=[]
    monkeypatch.setattr(audit,'preflight',lambda: [dict(images=944)])
    def run(command,*,output,budget):
        calls.append((command,output,budget))
        return {'status':'completed'}
    monkeypatch.setattr(audit,'run_budgeted',run)
    assert audit.main(['--check-only'])==0 and not calls
    assert audit.main([])==0 and len(calls)==1
    assert calls[0][2].wall_seconds==1800 and calls[0][2].gpu_devices==()


def test_changed_input_binding_rejected(tmp_path):
    p=tmp_path/'input.json'; p.write_text('{}')
    audit.verify_inputs({str(p):audit.file_digest(p)})
    with pytest.raises(ValueError): audit.verify_inputs({str(p):'0'*64})
