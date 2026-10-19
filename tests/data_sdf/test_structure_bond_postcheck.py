"""Synthetic saved predictions only; no checkpoint loading."""

from types import SimpleNamespace

import pytest

from tools import audit_structure_bond_diagnostic as audit


def case():
    return dict(candidates=dict(raw_count=2,selected_count=2,candidates=[
        dict(candidate=i,geometry=[.1+i*.1,.2,0,0,0,0,0,0],selected=True,selection_logit=.5)
        for i in range(2)]),pair_predictions=dict(pairs=[[0,1]],presence_logits=[10.],
        chemical_logits=[[10.,0.,0.,0.]],promoted_chemical_ids=[0]))


def test_restore_only_required_metric_tensors():
    raw,selected,predicted=audit.restore_for_metrics(case(),SimpleNamespace(maximum_atoms=4),-.25,.85)
    assert raw["atom_geometry"].shape==(1,4,8)
    assert raw["atom_mask"].sum()==selected["atom_mask"].sum()==2
    assert predicted["pairs"].tolist()==[[0,1]]
    assert "atom_features" not in raw


@pytest.mark.parametrize("mutation",["slot","selection","raw_count","selected_count","pair","logits","promotion"])
def test_changed_saved_contract_rejected(mutation):
    value=case()
    if mutation=="slot": value["candidates"]["candidates"][1]["candidate"]=0
    elif mutation=="selection": value["candidates"]["candidates"][0]["selection_logit"]=-1.
    elif mutation=="raw_count": value["candidates"]["raw_count"]=3
    elif mutation=="selected_count": value["candidates"]["selected_count"]=1
    elif mutation=="pair": value["pair_predictions"]["pairs"]=[[1,0]]
    elif mutation=="logits": value["pair_predictions"]["presence_logits"]=[float("nan")]
    else: value["pair_predictions"]["promoted_chemical_ids"]=[1]
    with pytest.raises(ValueError): audit.restore_for_metrics(value,SimpleNamespace(maximum_atoms=4),-.25,.85)
