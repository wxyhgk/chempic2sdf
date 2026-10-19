"""Synthetic evidence tests; no production model or data writes."""

from copy import deepcopy

import pytest
import torch

from tools import selector_node_semantics as s
from tools import selector_node_semantics_gallery as gallery
from chemvector.training.curriculum.selector_retention_worker import assess
from tests.chemvector.test_curriculum_continuation import tiny
from tests.chemvector.test_selector_retention_trial import cached
from chemvector.training.curriculum.selector_full_loss_evidence import geometry


@pytest.mark.parametrize('value', ['M 0,0 L 2,3', 'M1e1 0 L +2.0,-3.5 L 4 6'])
def test_explicit_lines_only(value):
    assert s.linear_points(value)


@pytest.mark.parametrize('value', ['M0 0 L2 2 Z', 'm0 0 l2 2', 'M0 0 C1 1 2 2 3 3', 'M0 0 L1 1 M2 2 L3 3', 'M0 0', 'junk M0 0 L1 1'])
def test_unsupported_paths_not_guessed(value):
    assert s.linear_points(value) is None


def candidate(x, y, raw=-1, distance=10):
    return dict(geometry=[x/768,y/768,0,0,0,0,0,0], raw_target=raw, nearest_distance_px=distance)


def test_geometry_buckets_never_turn_true_text_atom_negative():
    glyphs = [dict(atom_id=0, text='N', bbox=[0,0,20,20])]
    a = s.location(candidate(10,10,raw=0,distance=1),768,6,glyphs,[])
    b = s.location(candidate(10,10,raw=-1,distance=10),768,6,glyphs,[])
    c = s.location(candidate(10,10,raw=-1,distance=2),768,6,glyphs,[])
    assert a['bucket'] == 'raw_positive' and a['glyph_bbox_hits']
    assert b['bucket'] == 'glyph_bbox_outside_target_radius'
    assert c['bucket'] == 'near_target_unassigned'


def test_off_axis_multiple_stroke_endpoint_and_unknown():
    paths = [dict(points=[(0,10),(100,10)], bond_id=0,node_index=1,chemical_type='TRIPLE',
                  drawing_type='TRIPLE',begin=(0,0),end=(100,0))]
    assert s.location(candidate(1,11),768,6,[],paths)['bucket'] == 'multiple_bond_off_axis_endpoint'
    assert s.location(candidate(50,50),768,6,[],paths)['bucket'] == 'other_or_unresolved'
    assert s.segment_distance((10,2),(0,0),(0,0)) == pytest.approx(104**.5)


def test_linear_svg_attribution_and_transform_rejection():
    svg = '<svg viewBox="0 0 768 768"><path class="bond-0" d="M0 0 L20 0"/><path class="bond-0" d="M0 2 L20 2 Z"/></svg>'
    layers = dict(bonds=[dict(bond_id=0,svg_node_indices=[0,1],chemical_type='DOUBLE',drawing=dict(bond_type='DOUBLE'),begin=[0,0],end=[20,0])])
    paths, unknown = s.strokes(svg,layers)
    assert len(paths)==1 and len(unknown)==1
    with pytest.raises(ValueError,match='transform'):
        s.strokes(svg.replace('<svg ', '<svg transform="scale(2)" '),layers)


def test_replay_exactly_reproduces_existing_matcher(cached):
    model,config,cache,_=cached
    saved=geometry(cache,config)
    # Production image_size contract is checked by audit; synthetic fixture is64px.
    record=assess(model,config,None,cache,device='cpu')
    locations={c['slot']:s.location(c,config.image_size,config.atom_match_radius_px,[],[]) for c in saved['candidates']}
    states=s.decisions(saved,record,locations)
    assert sum(r['selected'] for r in states)==record['metrics']['selected_atoms']
    changed=deepcopy(record)
    changed['metrics']['selected_atoms']+=1
    with pytest.raises(ValueError,match='assessment changed'):
        s.decisions(saved,changed,locations)


def test_empty_assignment_and_no_mutation():
    geo=dict(candidates=[dict(slot=0,geometry=[.5,.5,0,0,0,0,0,0])], target_anchors=[[.5,.5]],image_size=768,match_radius_px=6)
    before=deepcopy(geo)
    assert s.assignment(geo,[])=={}
    assert s.assignment(geo,[0])=={0:0}
    assert geo==before


def test_gallery_overlay_is_not_source_edit(tmp_path):
    from PIL import Image
    source=tmp_path/'source.png'
    Image.new('RGB',(768,768),'white').save(source)
    before=source.read_bytes()
    case=dict(sample_key='synthetic',canonical_smiles='C',target_anchors=[[.5,.5]],
              phases=dict(initial=dict(candidates=[])))
    name=gallery.render(tmp_path,case,source)
    gallery.index(tmp_path,[(case,name)])
    assert source.read_bytes()==before
    assert (tmp_path/name).is_file() and (tmp_path/'index.html').is_file()
