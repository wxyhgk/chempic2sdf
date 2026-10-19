import json

import pytest

from Data.sdf.case_diversity_v1.catalog import pool
from Data.sdf.structure_expansion_v1.catalog import new_candidates
from Data.sdf.structure_expansion_v1 import preview


def test_preview_is_fixed_and_covers_required_contexts():
    es, _ = new_candidates(pool()[0])
    selected = preview.select_preview(es)
    assert selected == preview.select_preview(list(reversed(es)))
    assert len(selected) == len({e['molecule_id'] for e in selected}) == 32
    assert set(preview.CONCEPTS) <= {c for e in selected for c in e['concepts']}
    with pytest.raises(ValueError):
        preview.select_preview(es, 5)


def test_micro_preview_preserves_denominator_and_never_overwrites(tmp_path, monkeypatch):
    es, _ = new_candidates(pool()[0])
    e = min(es, key=lambda e: (e['heavy_atoms'], e['molecule_id']))
    monkeypatch.setattr(preview, 'bounded_map', lambda fn,jobs,**kw: (fn(j) for j in jobs))
    output = tmp_path/'preview'
    summary = preview.render_preview([e], output)
    assert summary['planned_images'] == summary['accepted_images'] + summary['rejected_images'] == 4
    assert summary['accepted_images'] > 0 and not summary['training_authorized']
    assert summary == json.loads((output/'summary.json').read_text())
    with pytest.raises(FileExistsError):
        preview.render_preview([e], output)
