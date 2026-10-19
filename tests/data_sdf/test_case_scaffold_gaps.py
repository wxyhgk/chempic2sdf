from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from tools.plan_case_scaffold_gaps import missing_representatives


def test_one_representative_per_uncovered_scaffold_without_acyclic_fake_group():
    es = merged_records([dict(record_id=str(i),name=str(i),smiles=s,stage='atoms_bonds',concepts=[])
                         for i,s in enumerate(('c1ccccc1','Cc1ccccc1','c1ccncc1','CCc1ccncc1','CCCC'))])
    selected=[e for e in es if e['canonical_smiles']=='c1ccccc1']
    result=missing_representatives(es,selected)
    assert len(result)==1 and result[0]['available_structures']==2
    assert result[0]['entry']['canonical_smiles']=='c1ccncc1'
    assert result==missing_representatives(list(reversed(es)),selected)
