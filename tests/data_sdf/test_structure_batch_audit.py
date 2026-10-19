import json

import pytest

from Data.sdf.structure_expansion_v1 import batch
from tools import audit_structure_expansion_batch as checker


def test_cross_batch_conflicts_differ_from_same_identity_duplicate():
    rows=[dict(sample_key='a',molecule_id='x',rgb_sha256='r'),dict(sample_key='b',molecule_id='y',rgb_sha256='s')]
    result=checker.compare_prior(rows,{'old':('rgb_sha256',{'r':{'x'},'s':{'z'}})})['old']
    assert result['matching_images']==2
    assert result['different_identity_pixel_matches']==[dict(sample_key='b',molecule_id='y',prior_molecule_ids=['z'])]


def test_real_micro_audit_target_adapter_and_tamper_rejection(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    entries,_=batch.load_fixed_candidates(smoke=True)
    selected=[min(entries,key=lambda e:(e['heavy_atoms'],e['molecule_id']))]
    fixed=lambda **kw:(selected,{})
    monkeypatch.setattr(batch,'load_fixed_candidates',fixed)
    monkeypatch.setattr(checker,'load_fixed_candidates',fixed)
    monkeypatch.setattr(batch,'bounded_map',lambda fn,jobs,**kw:(fn(j) for j in jobs))
    monkeypatch.setattr(checker,'prior_indices',lambda:({},{}))
    dataset=tmp_path/'data'; batch.run(dataset,workers=1)
    output=tmp_path/'audit'; output.mkdir(); checker.audit(dataset,output)
    report=json.loads((output/'report.json').read_bytes())
    assert report['integrity_passed'] and report['verified_SDF_and_label_contracts']==8
    assert report['target_panel_passed']==report['target_panel_images']==1
    assert report['target_panel_failures']==[] and not report['all_images_collated']
    assert report['training_loader_and_collator_rejected_quarantine']==1
    assert not report['training_authorized'] and report['model_forwards']==0
    key=json.loads((dataset/'samples.json').read_bytes())[0]['sample_key']
    (dataset/'samples'/(key+'.png')).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Generated file changed'):
        checker.audit(dataset,tmp_path/'failed')
    assert not (tmp_path/'failed'/'report.json').exists()


def test_cpu_only(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU-only'):
        checker.audit(tmp_path/'data',tmp_path/'out')
