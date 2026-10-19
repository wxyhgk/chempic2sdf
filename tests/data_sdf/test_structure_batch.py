from collections import Counter
from dataclasses import asdict
import json

import pytest

from Data.sdf.structure_expansion_v1 import batch


def test_fixed_manifest_and_preview_subset_are_bound_without_reenumeration():
    entries, sources = batch.load_fixed_candidates()
    small, _ = batch.load_fixed_candidates(smoke=True)
    assert len(entries) == 1534 and len(small) == 8
    assert all(e in entries for e in small)
    batch.verify_hashes(sources)


def test_profiles_have_four_fonts_and_sizes_but_only_two_orientations():
    ps = batch.selected_profiles()
    assert tuple(p.profile_id for p in ps) == batch.PROFILE_IDS
    assert set(Counter(p.font_family for p in ps).values()) == {2}
    assert {p.font_size for p in ps} == {12,18,24,32}
    assert {p.rotation_degrees for p in ps} == {0,180}
    assert all(p.width == p.height == 768 for p in ps)


def test_bad_hash_is_not_an_ordinary_quality_rejection(tmp_path):
    path = tmp_path/'bound'
    path.write_text('value')
    batch.verify_hashes({path:batch.digest(path)})
    with pytest.raises(ValueError,match='Bound input changed'):
        batch.verify_hashes({path:'0'*64})


@pytest.fixture
def micro_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    es, _ = batch.load_fixed_candidates(smoke=True)
    es = [min(es, key=lambda e:(e['heavy_atoms'],e['molecule_id']))]
    monkeypatch.setattr(batch,'load_fixed_candidates',lambda **kwargs:(es,{}))
    monkeypatch.setattr(batch,'bounded_map',lambda fn,jobs,**kw:(fn(j) for j in jobs))
    output = tmp_path/'dataset'
    batch.run(output,workers=1)
    return output


def test_real_micro_batch_preserves_complete_denominator_and_refuses_overwrite(micro_batch):
    summary = json.loads((micro_batch/'summary.json').read_bytes())
    assert summary['complete'] and summary['planned_images'] == 8
    assert summary['accepted_images'] + summary['rejected_images'] == 8
    assert not summary['training_authorized'] and summary['accepted_images'] > 0
    assert json.loads((micro_batch/'profiles.json').read_bytes()) == [asdict(p) for p in batch.selected_profiles()]
    with pytest.raises(FileExistsError):
        batch.run(micro_batch,workers=1)


def test_cpu_only(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU-only'):
        batch.run(tmp_path/'dataset')
