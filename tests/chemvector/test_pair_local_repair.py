from copy import deepcopy

import pytest
from Data.sdf.chemscene_v0_2 import parse_chemscene_v0_2
from tools import pair_local_repair as repair


def scene(edges=None, atoms=6):
    edges = edges if edges is not None else [(0,i) for i in range(1,6)]+[(1,2)]
    result=dict(schema_version='chemscene-0.2',units='q1023',molecule_id='synthetic',
        canvas={'width':768,'height':768},atom_objects=[],bond_objects=[])
    for i in range(atoms):
        result['atom_objects'].append(dict(id=i,anchor=[100+i*100,100+i*20],role='IMPLICIT',text=None,
            chemical=dict(element='C',formal_charge=0,isotope=0,aromatic=False,explicit_h_count=0,
                          no_implicit=False,radical_electrons=0,canonical_chiral_tag='CHI_UNSPECIFIED'),incident_ports=[]))
    for i,(a,b) in enumerate(edges):
        ports=[result['atom_objects'][j]['anchor'][:] for j in (a,b)]
        result['bond_objects'].append(dict(id=i,atom_refs=[a,b],ports=ports,display_type='SINGLE',
            chemical_type='SINGLE',direction='NONE',stereo='STEREONONE',bbox=None))
        for endpoint,j in enumerate((a,b)):
            result['atom_objects'][j]['incident_ports'].append(dict(bond_id=i,endpoint=endpoint,
                coordinate=ports[endpoint][:],source='ANCHOR'))
    parse_chemscene_v0_2(result)
    return result


def test_delete_copy_reindexes_ports_preserves_atoms_and_kept_bonds():
    original=scene();snapshot=deepcopy(original);lineage=[3,7,11,15,19,23]
    fixed=repair.remove_edges(original,lineage,[(3,7)])
    assert original==snapshot and len(fixed['bond_objects'])==5
    parse_chemscene_v0_2(fixed)
    assert (3,7) not in repair.edge_map(fixed,lineage)
    for old,new in zip(original['atom_objects'],fixed['atom_objects']):
        assert old['chemical']==new['chemical'] and old['anchor']==new['anchor']


def test_invalid_scene_restored_without_fragmentation_but_no_target_identity():
    original=scene();snapshot=deepcopy(original)
    result=repair.explore(original,list(range(6)),[(0,1)])
    assert not result['baseline']['valid']
    assert result['candidates'][0]['compilation']['valid']
    assert result['decision']['candidate_index']==0
    assert result['decision']['does_not_establish_correct_molecule']
    assert original==snapshot


def test_valid_original_not_replaced_even_if_an_edit_compiles():
    original=scene([(0,1),(1,2),(0,2)],atoms=3)
    result=repair.explore(original,[0,1,2],[(0,2)])
    assert result['baseline']['valid'] and result['candidates'][0]['compilation']['valid']
    assert result['decision']['action']=='keep'


def test_fragmenting_valid_candidate_rejected():
    original=scene([(0,i) for i in range(1,6)])
    result=repair.explore(original,list(range(6)),[(0,1)])
    assert result['candidates'][0]['compilation']['valid']
    assert result['candidates'][0]['scene_components']==2
    assert result['decision']['action']=='keep'


def test_two_equally_small_valid_candidates_abstain():
    result=repair.explore(scene(),list(range(6)),[(0,1),(0,2)])
    assert len(result['candidates'])==3
    assert result['decision']['reason']=='ambiguous_minimal_candidates'


def test_more_than_two_warnings_do_not_trigger_combinatorial_search():
    assert repair.deletion_subsets([(0,1),(0,2),(0,3)])==[]


@pytest.mark.parametrize('pairs',[[(0,0)],[(1,0)],[(False,1)],[(0,1),(0,1)]])
def test_invalid_warnings_rejected(pairs):
    with pytest.raises(ValueError):repair.deletion_subsets(pairs)


@pytest.mark.parametrize('change',['double','stereo','direction','chiral'])
def test_stereo_and_non_single_edits_refused(change):
    item=scene()
    if change=='double':item['bond_objects'][0]['chemical_type']='DOUBLE'
    elif change=='stereo':item['bond_objects'][0]['stereo']='STEREOANY'
    elif change=='direction':item['bond_objects'][0]['direction']='BEGINWEDGE'
    else:item['atom_objects'][0]['chemical']['canonical_chiral_tag']='CHI_TETRAHEDRAL_CCW'
    with pytest.raises(ValueError):repair.remove_edges(item,list(range(6)),[(0,1)])


def test_absent_edge_and_wrong_lineage_rejected():
    with pytest.raises(ValueError):repair.remove_edges(scene(),list(range(6)),[(4,5)])
    with pytest.raises(ValueError):repair.edge_map(scene(),[0]*6)


def test_selected_candidate_does_not_depend_on_external_scoring_fields():
    result=repair.explore(scene(),list(range(6)),[(0,1)])
    row=result['candidates'][0]
    for target_score in (True,False):
        row['sdf_exact']=target_score;row['known_true_edges_removed']=100 if target_score else 0
        assert repair.select_candidate(result['baseline'],[row],atom_count=6,components_before=1)['candidate_index']==0


def test_unknown_chemistry_is_not_rewritten_to_force_validity():
    item=scene();item['atom_objects'][0]['chemical']['element']='F'
    result=repair.explore(item,list(range(6)),[(0,1)])
    assert not result['candidates'][0]['compilation']['valid']
    assert result['decision']['action']=='keep'
    assert result['candidates'][0]['scene']['atom_objects'][0]['chemical']['element']=='F'


@pytest.mark.parametrize('change',['fallback','atom_count','annotations','compiled_fragments'])
def test_selection_refuses_additional_chemistry_or_structure_damage(change):
    result=repair.explore(scene(),list(range(6)),[(0,1)])
    row=result['candidates'][0]
    if change=='fallback':row['compilation']['fallback']=True
    elif change=='atom_count':row['compilation']['atom_count']-=1
    elif change=='annotations':row['annotations_preserved']=False
    else:row['compilation']['compiled_components']=2
    assert repair.select_candidate(result['baseline'],[row],atom_count=6,components_before=1)['action']=='keep'


def test_text_ports_are_preserved_instead_of_replaced_with_atom_anchors():
    item=scene()
    atom=item['atom_objects'][0]
    atom.update(role='TEXT',text=dict(value='C',center=atom['anchor'][:],bbox=[90,90,130,130]))
    retained=item['bond_objects'][1]
    retained['ports'][0]=[130,106]
    atom['incident_ports'][1].update(coordinate=[130,106],source='TEXT_BBOX_RAY')
    fixed=repair.remove_edges(item,list(range(6)),[(0,1)])
    assert fixed['bond_objects'][0]['ports'][0]==[130,106]
    assert fixed['atom_objects'][0]['incident_ports'][0]['source']=='TEXT_BBOX_RAY'
