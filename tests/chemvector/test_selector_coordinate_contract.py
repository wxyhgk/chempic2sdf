"""Synthetic coordinate and old-supervision checks; no model/checkpoint load."""

from copy import deepcopy
import math

import pytest
import torch

from chemvector.training.curriculum.data import _point
from chemvector.components.atoms.objects.spatial_atom_object import spatial_atom_object_loss
from tools import selector_coordinate_contract as c


def fixture(center=(100.1,100.1), prediction=(94.3,100.1)):
    q=torch.tensor(c.quantized_anchor(center),dtype=torch.float32).tolist()
    xy=[v/767 for v in prediction]
    candidate=dict(slot=0,grid_id=int(xy[1]*192)*192+int(xy[0]*192),geometry=xy+[0]*6,raw_target=-1)
    geo=dict(image_size=768,match_radius_px=6,target_anchors=[q],candidates=[candidate])
    layers=dict(atoms=[dict(atom_id=0,element='N',text='N',visible_text=True,center=list(center),bbox=[90,90,110,110])],bonds=[])
    return geo,layers


@pytest.mark.parametrize('point',[(0,0),(767,767),(100.1,300.6),(383.94,419.94)])
def test_adapter_quantization_reproduces_real_function(point):
    expected=(torch.tensor(_point(point))/1023).tolist()
    actual=torch.tensor(c.quantized_anchor(point),dtype=torch.float32).tolist()
    assert actual==expected
    assert c.source_xy([v/767 for v in point])==pytest.approx(point)
    assert math.dist(c.source_xy(actual),point)<=math.sqrt(2)*767/(2*1023)+1e-4


def test_same_normalized_supervision_has_zero_anchor_offset_loss():
    target=torch.tensor([[[v/1023 for v in _point((100.1,300.6))]]])
    cell=target[0,0]*192
    ix,iy=cell.long().tolist()
    offset=cell-cell.floor()
    logits=torch.zeros(1,2,192,192)
    logits[0,:,iy,ix]=torch.logit(offset)
    output=dict(heatmap_logits=torch.full((1,1,192,192),-2.),anchor_offset_logits=logits,
        role_logits=torch.zeros(1,2,192,192),text_centers=torch.zeros(1,2,192,192),
        text_extents=torch.zeros(1,2,192,192),count_logits=torch.zeros(1,2))
    targets=dict(atom_anchors=target,atom_mask=torch.tensor([[True]]),atom_roles=torch.zeros(1,1,dtype=torch.long),
        atom_text_mask=torch.tensor([[False]]),atom_text_center=torch.zeros(1,1,2),atom_text_bbox=torch.zeros(1,1,4))
    loss=spatial_atom_object_loss(output,targets)
    assert float(loss['anchor_offset_loss'])<1e-12
    decoded=(cell.floor()+logits[0,:,iy,ix].sigmoid())/192
    assert torch.equal(decoded,target[0,0])


def test_q1023_can_change_boundary_membership_without_relabeling():
    geo,layers=fixture()
    before=deepcopy((geo,layers))
    result=c.target_diagnostics(geo,layers)
    target=result['targets'][0]
    assert target['legacy_no_candidate']
    assert target['unquantized_model_nearest_px']<6
    assert target['span_only_nearest_px']>6
    assert target['q1023_shift_model_px']<.531
    assert (geo,layers)==before


def test_wrong_anchor_or_cell_fails_closed():
    geo,layers=fixture()
    geo['candidates'][0]['grid_id']+=2
    with pytest.raises(ValueError,match='saved cell'):
        c.target_diagnostics(geo,layers)
    geo,layers=fixture()
    geo['target_anchors'][0][0]+=.01
    with pytest.raises(ValueError,match='adapter quantization'):
        c.target_diagnostics(geo,layers)


def test_cell_bound_is_infimum_not_actual_prediction():
    candidate=dict(grid_id=25*192+25)
    assert c.cell_lower_bound(candidate,[102/768,102/768])==0
    assert c.cell_lower_bound(candidate,[110/768,102/768])==pytest.approx(6)
    assert c.cell_lower_bound(candidate,[112/768,102/768])==pytest.approx(8)


def test_glyph_center_is_not_chemical_anchor():
    geo,layers=fixture(prediction=(100.1,100.1))
    layers['atoms'][0]['bbox']=[100,90,130,110]
    target=c.target_diagnostics(geo,layers)['targets'][0]
    assert target['anchor_to_glyph_union_center_px']>14
    assert target['direct_nearest_px']<1
    assert not target['legacy_no_candidate']


def pair_fixture():
    geo,layers=fixture(center=(90,100),prediction=(103.4*767/768,100))
    glyphs=[dict(atom_id=0,text='r',bbox=[103.3,90,110,110])]
    proposal=dict(proposed_pairs=[dict(fragment_slot=0,owner_target=0,kinds=['glyph_fragment'])])
    return geo,layers,glyphs,proposal


def test_source_projection_can_change_bbox_membership():
    geo,layers,glyphs,proposal=pair_fixture()
    row=c.pair_robustness(geo,layers,glyphs,[],[],proposal)[0]
    assert row['old_overlay_xy'][0]>=glyphs[0]['bbox'][0]
    assert row['source_xy'][0]<glyphs[0]['bbox'][0]
    assert not row['coordinate_stable']
    assert 'source_projection_owner_not_confirmed' in row['review_reasons']
    assert not row['training_authorized'] and row['requires_review']


def test_stable_evidence_still_does_not_authorize_training():
    geo,layers,glyphs,proposal=pair_fixture()
    glyphs[0]['bbox'][0]=102
    row=c.pair_robustness(geo,layers,glyphs,[],[],proposal)[0]
    assert row['coordinate_stable']
    assert not row['training_authorized'] and row['requires_review']


def test_quantization_sensitive_pair_is_held_even_if_glyph_owner_stable():
    geo,layers=fixture()
    proposal=dict(proposed_pairs=[dict(fragment_slot=0,owner_target=0,kinds=['glyph_fragment'])])
    glyphs=[dict(atom_id=0,text='N',bbox=[90,90,110,110])]
    row=c.pair_robustness(geo,layers,glyphs,[],[],proposal)[0]
    assert row['source_evidence']['possible_owners']==[0]
    assert row['unquantized_members']==[0]
    assert row['review_reasons']==['q1023_membership_sensitive']


def test_corrected_overlay_does_not_edit_source(tmp_path,monkeypatch):
    from PIL import Image
    from tools import audit_selector_coordinate_contract as audit
    geo,layers=fixture()
    target=c.target_diagnostics(geo,layers)['targets'][0]
    source=tmp_path/'source.png'
    Image.new('RGB',(768,768),'white').save(source)
    previous=source.read_bytes()
    monkeypatch.setattr(audit,'OUTPUT',tmp_path)
    name=audit.overlay(source,'synthetic',target)
    audit.gallery([('synthetic',target,name)],[])
    assert source.read_bytes()==previous
    assert (tmp_path/name).is_file() and (tmp_path/'index.html').is_file()


def test_serialized_proposal_comparison_accepts_only_container_conversion():
    from tools.audit_selector_coordinate_contract import proposal_equal
    previous=dict(candidates=[dict(point=[1.25,2.5],slot=1)])
    reconstructed=dict(candidates=[dict(point=(1.25,2.5),slot=1)])
    assert proposal_equal(previous,reconstructed)
    reconstructed['candidates'][0]['point']=(1.25,2.500001)
    assert not proposal_equal(previous,reconstructed)
    assert not proposal_equal({},reconstructed)
