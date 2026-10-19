from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from tools import pair_local_evidence as local


def node(x, y, *, box=None, text=None, element='C', charge=0):
    return dict(xy=[x,y], text_box=box, text=text,
                chemical=dict(element=element, formal_charge=charge, radical_electrons=0, explicit_h_count=0))


def image(lines):
    im = Image.new('L', (160,160), 255)
    draw = ImageDraw.Draw(im)
    for segment in lines: draw.line(segment, fill=0, width=2)
    return np.asarray(im)


def test_continuous_and_broken_strokes_are_distinguished_without_targets():
    a,b = node(20,80),node(140,80)
    whole = local.pixel_support(image([(20,80,140,80)]),a,b)
    broken = local.pixel_support(image([(20,80,50,80),(90,80,140,80)]),a,b)
    assert whole['available'] and not whole['gap_warning']
    assert broken['gap_warning'] and broken['profiles']['wide']['max_blank_run_px'] > 30


def test_legitimate_endpoint_text_gap_is_excluded():
    a,b = node(20,80,box=[10,70,55,90],text='NH'), node(140,80)
    pixels=image([(58,80,140,80)])
    assert not local.pixel_support(pixels,a,b)['gap_warning']
    no_box=deepcopy(a); no_box['text_box']=None
    assert local.pixel_support(pixels,no_box,b)['gap_warning']


def test_long_diagonal_text_trim_has_json_native_flags():
    a=node(30,100,box=[20,80,70,110],text='NH');b=node(140,40)
    result=local.pixel_support(image([]),a,b)
    assert type(result['gap_warning']) is bool
    assert type(result['start_trim_px']) is float
    json.dumps(result,allow_nan=False)


def test_parallel_double_bond_does_not_require_centerline_ink():
    result = local.pixel_support(image([(20,73,140,73),(20,87,140,87)]),node(20,80),node(140,80))
    assert result['profiles']['narrow']['ink_coverage']==0
    assert result['profiles']['wide']['ink_coverage']==1 and not result['gap_warning']


def test_no_pixel_evidence_when_boxes_cover_segment():
    result=local.pixel_support(image([]),node(20,80,box=[10,60,100,100]),node(60,80))
    assert not result['available'] and not result['gap_warning']


def test_out_of_image_is_unknown_not_automatically_a_gap():
    result=local.pixel_support(image([]),node(20,3),node(140,3))
    assert not result['available'] and result['reason']=='ribbon_outside_source'


def test_direction_reversal_preserves_gap_measurement():
    pixels=image([(20,80,50,80),(90,80,140,80)])
    a,b=node(20,80),node(140,80)
    x,y=local.pixel_support(pixels,a,b),local.pixel_support(pixels,b,a)
    assert x['profiles']==y['profiles']


def graph():
    return dict(nodes={0:node(20,80),1:node(140,80),2:node(60,80),3:node(100,80)},
                edges={(0,2):'SINGLE',(2,3):'SINGLE',(1,3):'SINGLE'})


def test_alternate_path_excludes_the_queried_edge_and_obeys_hop_bound():
    g=graph();g['edges'][(0,1)]='SINGLE'
    assert local.corridor_path(g,(0,1),[2,3],4)==[0,2,3,1]
    assert local.corridor_path(g,(0,1),[2,3],2) is None
    assert local.corridor_path(g,(0,1),[2],4) is None


def test_third_candidate_or_path_alone_is_never_a_deletion_or_combined_warning():
    g=graph(); pixels=image([(20,80,140,80)])
    result=local.inspect_pair(pixels,g,(0,1),'SINGLE')
    assert result['flags']['alternate_corridor_path']
    assert not result['flags']['combined_review'] and result['action']=='none'


def test_carbon_conflict_does_not_single_out_the_guilty_edge():
    g=graph()
    g['nodes'].update({4:node(20,30),5:node(20,130),6:node(40,110)})
    g['edges'].update({(0,1):'SINGLE',(0,4):'SINGLE',(0,5):'SINGLE',(0,6):'SINGLE'})
    # Any of five incident edges participates in the same carbon conflict.
    for pair in [(0,1),(0,2),(0,4)]:
        assert local.carbon_conflicts(g,pair,'SINGLE')[0]['conflict']
    result=local.inspect_pair(image([(20,80,140,80)]),g,(0,1),'SINGLE')
    assert result['flags']['combined_review'] and result['action']=='none'


@pytest.mark.parametrize('element,charge', [('F',0),('N',0),('C',1)])
def test_no_naive_valence_veto_for_other_elements_or_charged_carbon(element,charge):
    g=graph();g['nodes'][0]=node(20,80,element=element,charge=charge)
    g['edges'][(0,2)]='TRIPLE'
    result=local.carbon_conflicts(g,(0,1),'TRIPLE')[0]
    assert not result['eligible_neutral_carbon'] and not result['conflict']


def test_aromatic_order_is_not_naively_summed_to_four_point_five():
    g=graph();g['edges']={(0,1):'AROMATIC',(0,2):'AROMATIC',(0,3):'AROMATIC'}
    result=local.carbon_conflicts(g,(0,1),'AROMATIC')[0]
    assert result['nonaromatic_order_plus_explicit_h'] is None and not result['conflict']


def test_no_mutation_of_raster_or_prediction_graph():
    g=graph();snap=deepcopy(g);pixels=image([(20,80,140,80)]);before=pixels.copy()
    result=local.inspect_pair(pixels,g,(0,1),'SINGLE')
    assert g==snap and np.array_equal(pixels,before)
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('pair', [(0,0),(1,0),(0,99),(False,1)])
def test_invalid_pair_rejected(pair):
    with pytest.raises(ValueError):local.inspect_pair(image([]),graph(),pair,'SINGLE')


def test_prediction_adapter_uses_packed_lineage_not_slot_position():
    trace=dict(compiled_candidate_indices=[4,7],scene_before_fallback=dict(
        atom_objects=[dict(id=i,anchor=[100+i*200,100],text=None,chemical={'element':'C'}) for i in range(2)],
        bond_objects=[dict(atom_refs=[0,1],chemical_type='SINGLE')]))
    result=local.prediction_graph(trace,(768,768))
    assert set(result['nodes'])=={4,7} and result['edges']=={(4,7):'SINGLE'}
    trace['compiled_candidate_indices']=[4,4]
    with pytest.raises(ValueError):local.prediction_graph(trace,(768,768))
