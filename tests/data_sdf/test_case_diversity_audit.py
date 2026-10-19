import json

import pytest

from Data.sdf.case_diversity_v1 import build as builder
from Data.sdf.case_diversity_v1.catalog import features
from Data.sdf.case_diversity_v1.profiles import profiles
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from tools.audit_case_diversity_v1 import worker


def test_full_plan_accounting_keeps_rejected_and_pending_distinct():
    es = [dict(molecule_id='a')]; ps = profiles()
    rows = [dict(molecule_id='a', profile_id=p.profile_id) for p in ps]
    builder.accounting(es, ps, rows[:-1], rows[-1:])
    with pytest.raises(ValueError): builder.accounting(es, ps, rows[:-1], [])
    with pytest.raises(ValueError): builder.accounting(es, ps, rows, rows[-1:])


def test_micro_bundle_independent_audit_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    es = merged_records([dict(record_id='test', name='test', smiles='CF', stage='atoms_bonds', concepts=['fluoride'])])
    for e in es:
        e['source_kind']='template_chemistry';e['diversity_features']=features(e)
    monkeypatch.setattr(builder, 'pool', lambda: (es, dict(synthetic_test=True)))
    monkeypatch.setattr(builder, 'select', lambda entries, count: entries)
    monkeypatch.setattr(builder, 'bindings', lambda: {})
    monkeypatch.setattr(builder, 'bounded_map', lambda function, jobs, **kw: (function(job) for job in jobs))
    dataset=tmp_path/'dataset'; output=tmp_path/'audit'; output.mkdir()
    builder.run(dataset, workers=1)
    worker(dataset, output)
    result=json.loads((output/'report.json').read_text())
    assert result['verified_SDF_and_label_contracts']==16 and not result['training_authorized']
    assert result['model_forwards']==0 and result['current_display_parser_unsupported']['retained']=={}
    assert result['target_grid_collision_samples']==[]
    with pytest.raises(FileExistsError):builder.run(dataset, workers=1)
    # Tampering is a failed audit, never an ordinary quality rejection.
    sample=json.loads((dataset/'samples.json').read_text())[0]['sample_key']
    (dataset/'samples'/(sample+'.png')).write_bytes(b'tampered')
    failed=tmp_path/'failed_audit';failed.mkdir()
    with pytest.raises(ValueError,match='Generated file changed'):worker(dataset,failed)
    assert not (failed/'report.json').exists()


def test_cpu_requirement(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError):builder.run(tmp_path/'dataset')
    with pytest.raises(ValueError):worker(tmp_path/'dataset',tmp_path/'audit')
